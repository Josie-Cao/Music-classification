import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Statistical analysis imports
from scipy import stats
from scipy.stats import ttest_rel, f_oneway, friedmanchisquare
import itertools

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# AUGMENTATION CLASSES
# ============================================================================

class SpecAugment:
    def __init__(self, freq_mask_param=20, time_mask_param=40, num_freq_masks=1, num_time_masks=1, mask_value=0.0):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
    
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        tensor = tensor.clone()
        _, freq_dim, time_dim = tensor.shape
        
        for _ in range(self.num_freq_masks):
            freq_mask_width = np.random.randint(1, min(self.freq_mask_param, freq_dim) + 1)
            freq_start = np.random.randint(0, freq_dim - freq_mask_width + 1)
            tensor[:, freq_start:freq_start + freq_mask_width, :] = self.mask_value
        
        for _ in range(self.num_time_masks):
            time_mask_width = np.random.randint(1, min(self.time_mask_param, time_dim) + 1)
            time_start = np.random.randint(0, time_dim - time_mask_width + 1)
            tensor[:, :, time_start:time_start + time_mask_width] = self.mask_value
        
        return tensor

class SpectrogramNoise:
    def __init__(self, noise_factor=0.01, probability=0.5):
        self.noise_factor = noise_factor
        self.probability = probability
    
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if np.random.random() < self.probability:
            noise = torch.randn_like(tensor) * self.noise_factor
            tensor = tensor + noise
            tensor = torch.clamp(tensor, -2.0, 2.0)
        
        return tensor

class AudioFriendlyTransform:
    def __init__(self, use_spec_augment=True, use_noise=True, spec_augment_params=None, noise_params=None):
        self.transforms = []
        
        if use_spec_augment:
            spec_params = spec_augment_params or {}
            self.transforms.append(SpecAugment(**spec_params))
        
        if use_noise:
            noise_params = noise_params or {}
            self.transforms.append(SpectrogramNoise(**noise_params))
    
    def __call__(self, tensor):
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

class RandomTemporalCrop:
    def __init__(self, crop_ratio=0.7):
        self.crop_ratio = crop_ratio
    
    def __call__(self, img):
        width, height = img.size
        crop_width = int(width * self.crop_ratio)
        
        max_start = width - crop_width
        start_x = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        cropped_img = img.crop((start_x, 0, start_x + crop_width, height))
        resized_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)
        
        return resized_img

# ============================================================================
# DATASET AND MODEL CLASSES
# ============================================================================

class MusicGenreDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class MusicGenreCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MusicGenreCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 16 * 24, 1024),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        
        self.output = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        
        return x

# ============================================================================
# DATA PREPARATION FOR CROSS-VALIDATION
# ============================================================================

def prepare_cv_data(spectrograms_dir, n_folds=5):
    """Prepare data for cross-validation"""
    genres = sorted([d for d in os.listdir(spectrograms_dir) 
                    if os.path.isdir(os.path.join(spectrograms_dir, d))])
    
    print(f"Found {len(genres)} genres: {genres}")

    genre_to_idx = {genre: i for i, genre in enumerate(genres)}
    
    all_images = []
    all_labels = []

    for genre in genres:
        genre_dir = os.path.join(spectrograms_dir, genre)
        for img_name in os.listdir(genre_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(genre_dir, img_name)
                all_images.append(img_path)
                all_labels.append(genre_to_idx[genre])

    print(f"Total images: {len(all_images)}")
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_splits = list(skf.split(all_images, all_labels))
    
    return all_images, all_labels, cv_splits, genre_to_idx, {v: k for k, v in genre_to_idx.items()}

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=30, fold_num=1, method_name=""):
    """Train model with early stopping"""
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Training {method_name} - Fold {fold_num}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{num_epochs}: Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%')
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'  Early stopping triggered after {epoch+1} epochs')
            break
            
    # Load best model
    model.load_state_dict(best_model_weights)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 
                  'train_accs': train_accs, 'val_accs': val_accs, 'best_val_loss': best_val_loss}

def evaluate_model(model, test_loader, criterion, idx_to_genre):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    class_names = [idx_to_genre[i] for i in range(len(idx_to_genre))]
    classification_rep = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_loss, accuracy, classification_rep, cm, all_preds, all_labels

# ============================================================================
# CROSS-VALIDATION STATISTICAL ANALYSIS
# ============================================================================

def paired_t_test_analysis(cv_results):
    """Perform paired t-tests between methods"""
    methods = list(cv_results.keys())
    n_methods = len(methods)
    
    # Extract accuracy scores for each method across folds
    method_scores = {}
    for method in methods:
        method_scores[method] = [fold_result['test_accuracy'] for fold_result in cv_results[method]]
    
    # Perform pairwise t-tests
    pairwise_results = {}
    n_comparisons = n_methods * (n_methods - 1) // 2
    alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
    
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            method1, method2 = methods[i], methods[j]
            scores1, scores2 = method_scores[method1], method_scores[method2]
            
            # Paired t-test
            statistic, p_value = ttest_rel(scores1, scores2)
            
            # Effect size (Cohen's d for paired samples)
            diff_scores = np.array(scores1) - np.array(scores2)
            cohens_d = np.mean(diff_scores) / np.std(diff_scores, ddof=1)
            
            pairwise_results[f"{method1}_vs_{method2}"] = {
                'statistic': statistic,
                'p_value': p_value,
                'p_value_corrected': p_value * n_comparisons,
                'significant': p_value < alpha_corrected,
                'cohens_d': abs(cohens_d),
                'effect_interpretation': (
                    'Large' if abs(cohens_d) >= 0.8 else 
                    'Medium' if abs(cohens_d) >= 0.5 else 
                    'Small' if abs(cohens_d) >= 0.2 else 
                    'Negligible'
                ),
                'mean_diff': np.mean(diff_scores)
            }
    
    return pairwise_results

def anova_analysis(cv_results):
    """Perform one-way ANOVA"""
    methods = list(cv_results.keys())
    
    # Extract accuracy scores for each method
    all_scores = []
    for method in methods:
        scores = [fold_result['test_accuracy'] for fold_result in cv_results[method]]
        all_scores.append(scores)
    
    # One-way ANOVA
    f_statistic, p_value = f_oneway(*all_scores)
    
    # Friedman test (non-parametric alternative)
    friedman_statistic, friedman_p = friedmanchisquare(*all_scores)
    
    return {
        'anova': {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'friedman': {
            'statistic': friedman_statistic,
            'p_value': friedman_p,
            'significant': friedman_p < 0.05
        }
    }

def comprehensive_cv_analysis(cv_results):
    """Perform comprehensive cross-validation statistical analysis"""
    print("\n" + "="*80)
    print("CROSS-VALIDATION STATISTICAL ANALYSIS")
    print("="*80)
    
    methods = list(cv_results.keys())
    
    # 1. Descriptive Statistics
    print("\n1. DESCRIPTIVE STATISTICS (5-Fold Cross-Validation)")
    print("-" * 55)
    
    stats_data = []
    for method in methods:
        accuracies = [fold['test_accuracy'] for fold in cv_results[method]]
        losses = [fold['test_loss'] for fold in cv_results[method]]
        
        stats_data.append({
            'Method': method,
            'Mean_Accuracy': np.mean(accuracies),
            'Std_Accuracy': np.std(accuracies, ddof=1),
            'Min_Accuracy': np.min(accuracies),
            'Max_Accuracy': np.max(accuracies),
            'Mean_Loss': np.mean(losses),
            'Std_Loss': np.std(losses, ddof=1)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False, float_format='%.4f'))
    
    # 2. ANOVA Analysis
    print("\n2. ANALYSIS OF VARIANCE (ANOVA)")
    print("-" * 35)
    anova_results = anova_analysis(cv_results)
    
    print(f"One-way ANOVA:")
    print(f"  F-statistic: {anova_results['anova']['f_statistic']:.4f}")
    print(f"  p-value: {anova_results['anova']['p_value']:.4f}")
    print(f"  Significant: {'Yes' if anova_results['anova']['significant'] else 'No'}")
    
    print(f"\nFriedman Test (non-parametric):")
    print(f"  Chi-square statistic: {anova_results['friedman']['statistic']:.4f}")
    print(f"  p-value: {anova_results['friedman']['p_value']:.4f}")
    print(f"  Significant: {'Yes' if anova_results['friedman']['significant'] else 'No'}")
    
    # 3. Pairwise Comparisons
    print("\n3. PAIRWISE COMPARISONS (Paired t-tests)")
    print("-" * 45)
    pairwise_results = paired_t_test_analysis(cv_results)
    
    for comparison, result in pairwise_results.items():
        significance = "***" if result['p_value_corrected'] < 0.001 else "**" if result['p_value_corrected'] < 0.01 else "*" if result['p_value_corrected'] < 0.05 else ""
        print(f"{comparison}:")
        print(f"  p-value (Bonferroni corrected): {result['p_value_corrected']:.4f} {significance}")
        print(f"  Effect size (Cohen's d): {result['cohens_d']:.3f} ({result['effect_interpretation']})")
        print(f"  Mean difference: {result['mean_diff']:.4f}")
    
    return {
        'descriptive_stats': stats_df,
        'anova_results': anova_results,
        'pairwise_results': pairwise_results
    }

# ============================================================================
# CROSS-VALIDATION VISUALIZATION
# ============================================================================

def plot_cv_comparison(cv_results, statistical_results):
    """Create comprehensive cross-validation comparison visualization"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    methods = list(cv_results.keys())
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsteelblue', 'lightpink']
    n_folds = len(cv_results[methods[0]])
    
    # Subplot 1: Cross-Validation Accuracy Scores
    axes[0, 0].set_title('Cross-Validation Accuracy Scores', fontsize=14, fontweight='bold')
    
    for i, method in enumerate(methods):
        accuracies = [fold['test_accuracy'] for fold in cv_results[method]]
        folds = list(range(1, len(accuracies) + 1))
        axes[0, 0].plot(folds, accuracies, 'o-', label=method, color=colors[i], linewidth=2, markersize=8)
    
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, n_folds + 1))
    
    # Subplot 2: Box Plot of Accuracy Distribution
    accuracy_data = []
    for method in methods:
        accuracies = [fold['test_accuracy'] for fold in cv_results[method]]
        accuracy_data.append(accuracies)
    
    bp = axes[0, 1].boxplot(accuracy_data, labels=[m.replace(' ', '\n') for m in methods], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0, 1].set_title('Accuracy Distribution Across Folds', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=0, labelsize=8)
    
    # Subplot 3: Mean Accuracy with Error Bars
    means = [np.mean([fold['test_accuracy'] for fold in cv_results[method]]) for method in methods]
    stds = [np.std([fold['test_accuracy'] for fold in cv_results[method]], ddof=1) for method in methods]
    
    bars = axes[0, 2].bar(range(len(methods)), means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    axes[0, 2].set_title('Mean Accuracy ± Standard Deviation', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_xticks(range(len(methods)))
    axes[0, 2].set_xticklabels([m.replace(' ', '\n') for m in methods], rotation=0, fontsize=8)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005, 
                       f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Statistical Significance Heatmap
    n_methods = len(methods)
    p_value_matrix = np.ones((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i != j:
                comparison_key = f"{method1}_vs_{method2}" if i < j else f"{method2}_vs_{method1}"
                if comparison_key in statistical_results['pairwise_results']:
                    p_val = statistical_results['pairwise_results'][comparison_key]['p_value_corrected']
                    p_value_matrix[i, j] = p_val
    
    im = axes[1, 0].imshow(p_value_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    cbar = plt.colorbar(im, ax=axes[1, 0], label='p-value (Bonferroni corrected)')
    axes[1, 0].set_xticks(range(n_methods))
    axes[1, 0].set_xticklabels([m.replace(' ', '\n') for m in methods], rotation=45, ha='right')
    axes[1, 0].set_yticks(range(n_methods))
    axes[1, 0].set_yticklabels(methods)
    axes[1, 0].set_title('Statistical Significance Matrix\n(Paired t-test p-values)', fontsize=14, fontweight='bold')
    
    # Add significance markers
    for i in range(n_methods):
        for j in range(n_methods):
            if i != j and p_value_matrix[i, j] < 0.05:
                axes[1, 0].text(j, i, '*', ha='center', va='center', fontsize=20, fontweight='bold', color='white')
    
    # Subplot 5: Effect Sizes
    effect_comparisons = list(statistical_results['pairwise_results'].keys())
    effect_values = [statistical_results['pairwise_results'][comp]['cohens_d'] for comp in effect_comparisons]
    
    bars = axes[1, 1].bar(range(len(effect_values)), effect_values, alpha=0.7)
    axes[1, 1].set_title('Effect Sizes (Cohen\'s d)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Cohen\'s d')
    axes[1, 1].set_xticks(range(len(effect_comparisons)))
    # Create shorter labels for effect comparisons
    short_labels = []
    for comp in effect_comparisons:
        # Split comparison and create abbreviated labels
        parts = comp.replace('_vs_', ' vs ').split(' vs ')
        if len(parts) == 2:
            # Abbreviate method names
            abbr1 = parts[0][:8] + '...' if len(parts[0]) > 8 else parts[0]
            abbr2 = parts[1][:8] + '...' if len(parts[1]) > 8 else parts[1]
            short_labels.append(f'{abbr1}\nvs\n{abbr2}')
        else:
            short_labels.append(comp[:10] + '...' if len(comp) > 10 else comp)
    
    axes[1, 1].set_xticklabels(short_labels, rotation=0, ha='center', fontsize=8)
    
    # Color bars based on effect size
    for bar, effect_size in zip(bars, effect_values):
        if effect_size < 0.2:
            bar.set_color('lightgray')  # Small effect
        elif effect_size < 0.5:
            bar.set_color('orange')     # Medium effect
        else:
            bar.set_color('red')        # Large effect
    
    # Add effect size thresholds
    axes[1, 1].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    
    # Subplot 6: Training Loss Convergence
    axes[1, 2].set_title('Training Loss Convergence (Fold 1)', fontsize=14, fontweight='bold')
    
    for i, method in enumerate(methods):
        if cv_results[method][0]['history']:  # Check if history exists
            train_losses = cv_results[method][0]['history']['train_losses']
            epochs = range(1, len(train_losses) + 1)
            axes[1, 2].plot(epochs, train_losses, label=method, color=colors[i], linewidth=2)
    
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Training Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Subplot 7: Average Confusion Matrix for Best Method
    best_method = max(methods, key=lambda x: np.mean([fold['test_accuracy'] for fold in cv_results[x]]))
    
    # Average confusion matrices across folds
    avg_cm = np.mean([fold['confusion_matrix'] for fold in cv_results[best_method]], axis=0)
    
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[2, 0])
    axes[2, 0].set_title(f'Average Confusion Matrix - {best_method}', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Predicted')
    axes[2, 0].set_ylabel('Actual')
    
    # Subplot 8: Variance Comparison
    variances = [np.var([fold['test_accuracy'] for fold in cv_results[method]], ddof=1) for method in methods]
    
    bars = axes[2, 1].bar(range(len(methods)), variances, alpha=0.7, color=colors)
    axes[2, 1].set_title('Accuracy Variance Across Folds', fontsize=14, fontweight='bold')
    axes[2, 1].set_ylabel('Variance')
    axes[2, 1].set_xticks(range(len(methods)))
    axes[2, 1].set_xticklabels([m.replace(' ', '\n') for m in methods], rotation=0)
    
    # Add value labels
    for bar, var in zip(bars, variances):
        axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                       f'{var:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 9: Summary Statistics Table
    axes[2, 2].axis('off')
    
    # Create summary statistics table with shortened method names
    summary_data = []
    method_abbreviations = {
        'Baseline (No Aug)': 'Baseline',
        'Traditional Aug': 'Traditional',
        'Temporal Cropping': 'Temporal',
        'SpectrogramNoise': 'Spectrogram\nNoise',
        'SpecAugment': 'SpecAugment'
    }
    
    for method in methods:
        accuracies = [fold['test_accuracy'] for fold in cv_results[method]]
        # Use abbreviated name or fallback to original
        short_name = method_abbreviations.get(method, method[:10] + '...' if len(method) > 10 else method)
        summary_data.append([
            short_name,
            f"{np.mean(accuracies):.3f}",
            f"{np.std(accuracies, ddof=1):.4f}",
            f"{np.min(accuracies):.3f}",
            f"{np.max(accuracies):.3f}"
        ])
    
    table = axes[2, 2].table(cellText=summary_data,
                           colLabels=['Method', 'Mean', 'Std Dev', 'Min', 'Max'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0.3, 1, 0.65])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.3, 2.0)
    
    # Adjust column widths - make first column wider
    cellDict = table.get_celld()
    for i in range(len(summary_data) + 1):  # +1 for header
        cellDict[(i, 0)].set_width(0.35)  # Method column wider
        for j in range(1, 5):  # Other columns
            cellDict[(i, j)].set_width(0.15)
    
    # Color the best performance
    best_method_idx = np.argmax([np.mean([fold['test_accuracy'] for fold in cv_results[method]]) for method in methods])
    table[(best_method_idx + 1, 1)].set_facecolor('lightgreen')  # Best mean
    
    # Add ANOVA results
    anova_text = f"ANOVA Results:\n"
    anova_text += f"F-statistic: {statistical_results['anova_results']['anova']['f_statistic']:.3f}\n"
    anova_text += f"p-value: {statistical_results['anova_results']['anova']['p_value']:.4f}\n"
    anova_text += f"Significant: {'Yes' if statistical_results['anova_results']['anova']['significant'] else 'No'}"
    
    axes[2, 2].text(0.5, 0.12, anova_text, ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   transform=axes[2, 2].transAxes)
    
    axes[2, 2].set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.0)
    plt.savefig('crossvalidation_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# RESULT MANAGEMENT FUNCTIONS
# ============================================================================

def save_cv_results(cv_results, filename='cv_results_partial.pkl'):
    """Save cross-validation results to file"""
    with open(filename, 'wb') as f:
        pickle.dump(cv_results, f)
    print(f"Results saved to {filename}")

def load_cv_results(filename='cv_results_partial.pkl'):
    """Load cross-validation results from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cv_results = pickle.load(f)
        print(f"Loaded existing results from {filename}")
        return cv_results
    else:
        print(f"No existing results file found: {filename}")
        return {}

def merge_cv_results(existing_results, new_results):
    """Merge existing and new CV results"""
    merged = existing_results.copy()
    merged.update(new_results)
    return merged

# ============================================================================
# MAIN CROSS-VALIDATION COMPARISON FUNCTION
# ============================================================================

def run_cv_comparison():
    """Run comprehensive 5-fold cross-validation comparison - complete fresh training"""
    print("="*80)
    print("CNN CROSS-VALIDATION COMPARISON - RIGOROUS STATISTICAL ANALYSIS")
    print("5-fold cross-validation with multiple training runs per method")
    print("Expected completion time: 15-25 hours")
    print("="*80)
    
    # Data preparation
    spectrograms_dir = "Data/images_original/"
    all_images, all_labels, cv_splits, genre_to_idx, idx_to_genre = prepare_cv_data(spectrograms_dir, n_folds=5)
    
    # Define augmentation strategies
    augmentation_configs = {
        'Baseline (No Aug)': {
            'train_transform': transforms.Compose([
                transforms.Resize((256, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        },
        'Traditional Aug': {
            'train_transform': transforms.Compose([
                transforms.Resize((256, 384)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        },
        'Temporal Cropping': {
            'train_transform': transforms.Compose([
                transforms.Resize((256, 384)),
                RandomTemporalCrop(crop_ratio=0.7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        },
        'SpectrogramNoise': {
            'train_transform': transforms.Compose([
                transforms.Resize((256, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AudioFriendlyTransform(
                    use_spec_augment=False,
                    use_noise=True,
                    noise_params={'noise_factor': 0.004, 'probability': 0.15}
                )
            ])
        },
        'SpecAugment': {
            'train_transform': transforms.Compose([
                transforms.Resize((256, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AudioFriendlyTransform(
                    use_spec_augment=True,
                    use_noise=False,
                    spec_augment_params={
                        'freq_mask_param': 10, 
                        'time_mask_param': 20, 
                        'num_freq_masks': 1, 
                        'num_time_masks': 1
                    }
                )
            ])
        }
    }
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize results structure
    cv_results = {method: [] for method in augmentation_configs.keys()}
    
    # Perform cross-validation for all methods
    for method_name, config in augmentation_configs.items():
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION: {method_name}")
        print(f"{'='*70}")
        
        # Reset seed for each method to ensure fair comparison
        set_seed(42)
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
            # Reset seed for each fold to ensure consistent training conditions
            set_seed(42)
            print(f"\nFold {fold_idx + 1}/5")
            print("-" * 20)
            
            # Split training data into train and validation
            train_val_images = [all_images[i] for i in train_val_indices]
            train_val_labels = [all_labels[i] for i in train_val_indices]
            test_images = [all_images[i] for i in test_indices]
            test_labels = [all_labels[i] for i in test_indices]
            
            # Further split train_val into train and validation (8:1 ratio to match 8/1/1 overall split)
            train_images, val_images, train_labels_fold, val_labels = train_test_split(
                train_val_images, train_val_labels, test_size=1/9, random_state=42, 
                stratify=train_val_labels
            )
            
            # Create datasets
            train_dataset = MusicGenreDataset(train_images, train_labels_fold, transform=config['train_transform'])
            val_dataset = MusicGenreDataset(val_images, val_labels, transform=test_transform)
            test_dataset = MusicGenreDataset(test_images, test_labels, transform=test_transform)
            
            # Create data loaders with controlled randomness
            batch_size = 32
            # Create generator for reproducible shuffle
            generator = torch.Generator()
            generator.manual_seed(42)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=4, generator=generator)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # Initialize model
            model = MusicGenreCNN(num_classes=len(genre_to_idx)).to(device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=False)
            
            # Train model
            model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=200, patience=30, fold_num=fold_idx + 1, method_name=method_name
            )
            
            # Evaluate on test set
            test_loss, test_accuracy, classification_rep, cm, all_preds, all_labels_test = evaluate_model(
                model, test_loader, criterion, idx_to_genre
            )
            
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            
            # Store results
            cv_results[method_name].append({
                'fold': fold_idx + 1,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'confusion_matrix': cm,
                'predictions': all_preds,
                'true_labels': all_labels_test,
                'classification_report': classification_rep,
                'history': history
            })
    
    # All methods have been trained
    
    # Perform comprehensive statistical analysis
    statistical_results = comprehensive_cv_analysis(cv_results)
    
    # Create comprehensive visualization
    plot_cv_comparison(cv_results, statistical_results)
    
    # Save detailed results
    cv_summary_data = []
    for method in cv_results.keys():
        for fold_result in cv_results[method]:
            cv_summary_data.append({
                'Method': method,
                'Fold': fold_result['fold'],
                'Test_Accuracy': fold_result['test_accuracy'],
                'Test_Loss': fold_result['test_loss']
            })
    
    cv_summary_df = pd.DataFrame(cv_summary_data)
    cv_summary_df.to_csv('crossvalidation_detailed_results.csv', index=False)
    
    # Save statistical analysis results
    statistical_results['descriptive_stats'].to_csv('crossvalidation_summary_stats.csv', index=False)
    
    # Save pairwise comparison results
    pairwise_data = []
    for comparison, result in statistical_results['pairwise_results'].items():
        pairwise_data.append({
            'Comparison': comparison,
            'p_value': result['p_value'],
            'p_value_corrected': result['p_value_corrected'],
            'significant': result['significant'],
            'cohens_d': result['cohens_d'],
            'effect_interpretation': result['effect_interpretation'],
            'mean_difference': result['mean_diff']
        })
    
    pairwise_df = pd.DataFrame(pairwise_data)
    pairwise_df.to_csv('crossvalidation_pairwise_tests.csv', index=False)
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Detailed results saved to 'crossvalidation_detailed_results.csv'")
    print(f"Summary statistics saved to 'crossvalidation_summary_stats.csv'")
    print(f"Pairwise tests saved to 'crossvalidation_pairwise_tests.csv'")
    print(f"Comprehensive visualization saved to 'crossvalidation_comparison_results.png'")
    
    # Print final summary
    print(f"\nFINAL SUMMARY:")
    for method in cv_results.keys():
        accuracies = [fold['test_accuracy'] for fold in cv_results[method]]
        print(f"{method}: {np.mean(accuracies):.4f} ± {np.std(accuracies, ddof=1):.4f}")
    
    return cv_results, statistical_results

if __name__ == "__main__":
    # Run complete cross-validation comparison from scratch
    cv_results, statistical_results = run_cv_comparison()