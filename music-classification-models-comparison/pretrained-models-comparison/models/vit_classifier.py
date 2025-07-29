import torch
import torch.nn as nn

try:
    from transformers import ViTForImageClassification, ViTConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ViTMusicClassifier(nn.Module):
    """
    Vision Transformer based music classifier
    
    Uses a pretrained ViT-Base model with 2-stage fine-tuning for music genre classification.
    Requires the transformers library to be installed.
    """
    
    def __init__(self, num_classes=10):
        super(ViTMusicClassifier, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required for ViT. "
                "Install with: pip install transformers"
            )
        
        # Load pretrained ViT
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Store number of classes
        self.num_classes = num_classes
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits
    
    def freeze_backbone(self):
        """Freeze the backbone (everything except classifier head)"""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Return model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_params()
        
        return {
            'model_name': 'Vision Transformer (ViT-Base)',
            'num_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'ViT-Base-Patch16-224 pretrained model',
            'transfer_learning': '2-stage fine-tuning (head first, then full model)'
        }


def create_vit_classifier(num_classes: int = 10) -> ViTMusicClassifier:
    """
    Factory function to create ViT classifier
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        ViTMusicClassifier instance
        
    Raises:
        ImportError: If transformers library is not available
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers library is required for ViT. "
            "Install with: pip install transformers"
        )
    
    return ViTMusicClassifier(num_classes=num_classes)


def get_vit_stage_configs():
    """
    Get configurations for different training stages of ViT fine-tuning
    
    Returns:
        dict: Configuration for each training stage
    """
    return {
        'stage1': {
            'name': 'Stage 1: Classifier Head Only',
            'freeze_backbone': True,
            'description': 'Only train the classification head'
        },
        'stage2': {
            'name': 'Stage 2: Full Fine-tuning',
            'freeze_backbone': False,
            'description': 'Fine-tune entire ViT model'
        }
    }


def is_transformers_available() -> bool:
    """Check if transformers library is available"""
    return TRANSFORMERS_AVAILABLE