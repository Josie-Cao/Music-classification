from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class TrainingConfig:
    """Configuration class for pretrained models training parameters"""
    
    # Basic training parameters
    num_epochs: int = 200
    patience: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Data parameters
    input_size: tuple = (224, 224)  # Default for pretrained models
    num_classes: int = 10
    n_folds: int = 5
    random_state: int = 42
    num_workers: int = 4
    
    # Model-specific parameters
    model_type: str = 'cnn'  # 'cnn', 'resnet', 'vit'
    dropout_rate: float = 0.5
    
    # Transfer learning parameters (for ResNet and ViT)
    use_pretrained: bool = True
    freeze_backbone: bool = False
    
    # Scheduler parameters
    scheduler_type: str = 'plateau'  # 'plateau', 'cosine'
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Overfitting monitoring
    overfitting_threshold: float = 20.0  # Accuracy gap threshold (%)
    max_consecutive_overfitting: int = 3
    
    # Paths
    data_dir: str = "Data/images_original/"
    
    # Stage-specific configurations (for multi-stage training)
    stage_configs: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        """Set model-specific defaults"""
        if self.model_type == 'cnn':
            self.input_size = (256, 384)  # CNN baseline uses larger input
        elif self.model_type in ['resnet', 'vit']:
            self.input_size = (224, 224)  # Standard for pretrained models
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'num_workers': self.num_workers,
            'model_type': self.model_type,
            'dropout_rate': self.dropout_rate,
            'use_pretrained': self.use_pretrained,
            'freeze_backbone': self.freeze_backbone,
            'scheduler_type': self.scheduler_type,
            'scheduler_factor': self.scheduler_factor,
            'scheduler_patience': self.scheduler_patience,
            'min_lr': self.min_lr,
            'overfitting_threshold': self.overfitting_threshold,
            'max_consecutive_overfitting': self.max_consecutive_overfitting,
            'data_dir': self.data_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k != 'stage_configs'})


def get_cnn_config() -> TrainingConfig:
    """Get configuration optimized for CNN baseline training"""
    return TrainingConfig(
        model_type='cnn',
        num_epochs=200,
        patience=30,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.5,
        input_size=(256, 384),
        scheduler_type='plateau'
    )


def get_resnet_config() -> TrainingConfig:
    """Get configuration optimized for ResNet-18 transfer learning"""
    config = TrainingConfig(
        model_type='resnet',
        num_epochs=150,  # Shorter training due to pretrained weights
        patience=20,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.6,
        input_size=(224, 224),
        use_pretrained=True,
        scheduler_type='plateau'
    )
    
    # Define 3-stage training configuration
    config.stage_configs = {
        'stage1': {
            'name': 'Stage 1: Classifier Only',
            'epochs': 30,
            'learning_rate': 0.001,
            'freeze_backbone': True,
            'layers_to_freeze': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        },
        'stage2': {
            'name': 'Stage 2: + Layer 4',
            'epochs': 50,
            'learning_rate': 0.0005,
            'freeze_backbone': False,
            'layers_to_freeze': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        },
        'stage3': {
            'name': 'Stage 3: All Layers',
            'epochs': 70,
            'learning_rate': 0.0001,
            'freeze_backbone': False,
            'layers_to_freeze': []
        }
    }
    
    return config


def get_vit_config() -> TrainingConfig:
    """Get configuration optimized for Vision Transformer fine-tuning"""
    config = TrainingConfig(
        model_type='vit',
        num_epochs=100,  # ViT typically needs fewer epochs
        patience=15,
        batch_size=16,  # Smaller batch size for ViT
        learning_rate=0.0001,  # Lower learning rate for fine-tuning
        weight_decay=1e-4,
        input_size=(224, 224),
        use_pretrained=True,
        scheduler_type='cosine'  # Cosine annealing works well with ViT
    )
    
    # Define 2-stage training configuration
    config.stage_configs = {
        'stage1': {
            'name': 'Stage 1: Classifier Head Only',
            'epochs': 30,
            'learning_rate': 0.001,
            'freeze_backbone': True
        },
        'stage2': {
            'name': 'Stage 2: Full Fine-tuning',
            'epochs': 70,
            'learning_rate': 0.0001,
            'freeze_backbone': False
        }
    }
    
    return config


def get_quick_test_config(model_type: str = 'cnn') -> TrainingConfig:
    """Get configuration for quick testing (reduced epochs and folds)"""
    base_configs = {
        'cnn': get_cnn_config,
        'resnet': get_resnet_config,
        'vit': get_vit_config
    }
    
    if model_type not in base_configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config = base_configs[model_type]()
    
    # Reduce for quick testing
    config.num_epochs = 50
    config.patience = 10
    config.n_folds = 3
    config.batch_size = 16
    
    # Adjust stage configs if they exist
    if config.stage_configs:
        for stage_name, stage_config in config.stage_configs.items():
            stage_config['epochs'] = max(10, stage_config['epochs'] // 3)
    
    return config