#!/usr/bin/env python3
"""
Pretrained Models Comparison for Music Classification

This script compares 3 different model architectures:
- CNN Baseline (trained from scratch)  
- ResNet-18 (with conservative transfer learning)
- Vision Transformer (with 2-stage fine-tuning)

Uses modular components for clean, maintainable code.
"""

import os
import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import modular components
from models import MusicGenreCNN, OptimizedResNet18MusicClassifier, ViTMusicClassifier, is_transformers_available
from data import MusicGenreDataset, prepare_cv_data, get_model_specific_transforms, print_data_summary  
from training import (
    TrainingConfig, get_cnn_config, get_resnet_config, get_vit_config,
    CNNTrainer, ResNetTrainer, ViTTrainer
)
from analysis import evaluate_model
from utils import set_seed, get_device_info, format_time, print_experiment_header


def run_model_comparison(config_type='quick'):
    """
    Run complete model comparison experiment
    
    Args:
        config_type: 'quick' for testing, 'full' for complete experiment
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    print("Pretrained Models Comparison for Music Classification")
    print("=" * 80)
    
    # Print device info
    device_info = get_device_info()
    print(f"Device: {device_info['device_name']}")
    print(f"CUDA available: {device_info['cuda_available']}")
    
    # Prepare data
    print(f"\nPreparing cross-validation data...")
    data_dir = "Data/images_original/"
    n_folds = 3 if config_type == 'quick' else 5
    
    all_images, all_labels, cv_splits, genre_to_idx, idx_to_genre = prepare_cv_data(
        data_dir, n_folds=n_folds
    )
    print_data_summary(all_images, all_labels, cv_splits, idx_to_genre)
    
    # Define models to compare
    models_to_test = {
        'CNN Baseline': {
            'config_func': get_cnn_config,
            'model_class': MusicGenreCNN,
            'trainer_class': CNNTrainer
        },
        'ResNet-18': {
            'config_func': get_resnet_config,
            'model_class': OptimizedResNet18MusicClassifier,
            'trainer_class': ResNetTrainer
        }
    }
    
    # Add ViT if transformers is available
    if is_transformers_available():
        models_to_test['ViT'] = {
            'config_func': get_vit_config,
            'model_class': ViTMusicClassifier,
            'trainer_class': ViTTrainer
        }
    else:
        print("⚠️  Transformers not available. ViT will be skipped.")
    
    # Results storage
    all_results = []
    
    # Run experiments for each model
    for model_name, model_info in models_to_test.items():
        print(f"\n" + "="*80)
        print(f"TRAINING {model_name.upper()}")
        print("="*80)
        
        # Get model-specific configuration
        if config_type == 'quick':
            from training.config import get_quick_test_config
            config = get_quick_test_config(model_info['config_func']().model_type)
        else:
            config = model_info['config_func']()
        
        config.num_classes = len(genre_to_idx)
        
        print_experiment_header(f"{model_name} Training", config.to_dict())
        
        # Run cross-validation
        fold_results = []
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(cv_splits):
            print(f"\n{'='*20} FOLD {fold_idx + 1}/{len(cv_splits)} {'='*20}")
            
            start_time = time.time()
            
            # Split train/val
            train_val_images = [all_images[i] for i in train_val_indices]
            train_val_labels = [all_labels[i] for i in train_val_indices]
            
            train_images, val_images, train_labels, val_labels = train_test_split(
                train_val_images, train_val_labels, 
                test_size=1/9, random_state=42, stratify=train_val_labels
            )
            
            test_images = [all_images[i] for i in test_indices]
            test_labels = [all_labels[i] for i in test_indices]
            
            # Get transforms
            train_transform, test_transform = get_model_specific_transforms(config.model_type)
            
            # Create datasets
            train_dataset = MusicGenreDataset(train_images, train_labels, train_transform)
            val_dataset = MusicGenreDataset(val_images, val_labels, test_transform)
            test_dataset = MusicGenreDataset(test_images, test_labels, test_transform)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
            
            # Create model
            model = model_info['model_class'](num_classes=config.num_classes).to(device)
            
            # Create trainer
            trainer = model_info['trainer_class'](config, device)
            
            # Train model
            criterion = nn.CrossEntropyLoss()
            training_results = trainer.train_single_fold(
                model, train_loader, val_loader, criterion, fold_idx + 1
            )
            
            # Evaluate on test set
            test_loss, test_acc, _, _, _, _ = evaluate_model(
                model, test_loader, criterion, idx_to_genre, device
            )
            
            fold_time = time.time() - start_time
            
            # Store results
            fold_result = {
                'Model': model_name,
                'Fold': fold_idx + 1,
                'Test_Accuracy': test_acc * 100,  # Convert to percentage
                'Training_Time': fold_time
            }
            fold_results.append(fold_result)
            all_results.append(fold_result)
            
            print(f"Fold {fold_idx + 1} Results:")
            print(f"  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Training Time: {format_time(fold_time)}")
        
        # Print model summary
        avg_acc = np.mean([r['Test_Accuracy'] for r in fold_results])
        std_acc = np.std([r['Test_Accuracy'] for r in fold_results])
        total_time = sum([r['Training_Time'] for r in fold_results])
        
        print(f"\n{model_name} Summary:")
        print(f"  Average Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  Total Training Time: {format_time(total_time)}")
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_file = f"pretrained_comparison_{'quick' if config_type == 'quick' else 'full'}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Print final summary
    print(f"\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for model_name in results_df['Model'].unique():
        model_results = results_df[results_df['Model'] == model_name]
        avg_acc = model_results['Test_Accuracy'].mean()
        std_acc = model_results['Test_Accuracy'].std()
        print(f"{model_name:15}: {avg_acc:.2f}% ± {std_acc:.2f}%")
    
    return results_df


if __name__ == "__main__":
    # Check command line arguments for quick vs full test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick test configuration...")
        results = run_model_comparison('quick')
    else:
        print("Running full experiment...")
        print("(Use --quick flag for faster testing)")
        results = run_model_comparison('full')
    
    print(f"\n🎉 Experiment completed successfully!")