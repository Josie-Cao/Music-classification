from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    
    # Training hyperparameters
    num_epochs: int = 200
    patience: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Model architecture
    dropout_rate: float = 0.5
    conv_channels: List[int] = None
    fc_units: List[int] = None
    
    # Data parameters
    input_size: tuple = (256, 384)
    num_classes: int = 10
    n_folds: int = 5
    random_state: int = 42
    
    # Data loading
    num_workers: int = 4
    
    # Scheduler parameters
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Augmentation parameters
    spec_augment_params: Dict[str, Any] = None
    noise_params: Dict[str, Any] = None
    temporal_crop_ratio: float = 0.7
    
    # Paths
    data_dir: str = "Data/images_original/"
    
    def __post_init__(self):
        """Set default values for complex fields"""
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]
        
        if self.fc_units is None:
            self.fc_units = [1024, 256]
        
        if self.spec_augment_params is None:
            self.spec_augment_params = {
                'freq_mask_param': 10, 
                'time_mask_param': 20, 
                'num_freq_masks': 1, 
                'num_time_masks': 1
            }
        
        if self.noise_params is None:
            self.noise_params = {
                'noise_factor': 0.004, 
                'probability': 0.15
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'conv_channels': self.conv_channels,
            'fc_units': self.fc_units,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'num_workers': self.num_workers,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'min_lr': self.min_lr,
            'spec_augment_params': self.spec_augment_params,
            'noise_params': self.noise_params,
            'temporal_crop_ratio': self.temporal_crop_ratio,
            'data_dir': self.data_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Predefined configurations
def get_quick_config() -> TrainingConfig:
    """Configuration for quick testing"""
    return TrainingConfig(
        num_epochs=50,
        patience=10,
        batch_size=16,
        n_folds=3
    )


def get_production_config() -> TrainingConfig:
    """Configuration for production training"""
    return TrainingConfig(
        num_epochs=300,
        patience=50,
        batch_size=32,
        learning_rate=0.0005,
        weight_decay=1e-4,
        dropout_rate=0.6
    )


def get_lightweight_config() -> TrainingConfig:
    """Configuration for lightweight model"""
    return TrainingConfig(
        conv_channels=[16, 32, 64, 128],
        fc_units=[512, 128],
        dropout_rate=0.3,
        batch_size=64
    )


def get_deep_config() -> TrainingConfig:
    """Configuration for deeper model"""
    return TrainingConfig(
        conv_channels=[32, 64, 128, 256, 512],
        fc_units=[2048, 512, 256],
        dropout_rate=0.6,
        learning_rate=0.0005,
        batch_size=16  # Smaller batch size for larger model
    )