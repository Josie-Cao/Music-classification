import random
import numpy as np
import torch
import time
from typing import Dict


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


def count_parameters(model: torch.nn.Module) -> tuple:
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


def print_experiment_header(experiment_name: str, config_dict: Dict):
    """Print formatted experiment header"""
    print("=" * 80)
    print(f"{experiment_name.upper()}")
    print("=" * 80)
    
    print("Configuration:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    print("=" * 80)