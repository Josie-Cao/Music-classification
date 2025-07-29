import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(
    model: torch.nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    criterion: torch.nn.Module, 
    idx_to_genre: Dict[int, str],
    device: torch.device
) -> Tuple[float, float, Dict, np.ndarray, List[int], List[int]]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        criterion: Loss function
        idx_to_genre: Mapping from index to genre name
        device: Device to run evaluation on
        
    Returns:
        tuple: (test_loss, accuracy, classification_report, confusion_matrix, predictions, labels)
    """
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
    classification_rep = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_loss, accuracy, classification_rep, cm, all_preds, all_labels


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device_info() -> Dict[str, any]:
    """Get information about available computing devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    return info


def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def save_results_summary(results: Dict, filename: str = 'experiment_summary.txt'):
    """Save experiment results summary to file"""
    with open(filename, 'w') as f:
        f.write("Music Classification Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results summary saved to {filename}")


def print_experiment_header(experiment_name: str, config: Dict):
    """Print formatted experiment header"""
    print("=" * 80)
    print(f"{experiment_name.upper()}")
    print("=" * 80)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("=" * 80)