import torch
import torch.nn as nn
import torchvision.models as models


class OptimizedResNet18MusicClassifier(nn.Module):
    """
    ResNet-18 based music classifier with conservative transfer learning
    
    This model uses a pretrained ResNet-18 backbone with a custom classifier head
    and supports progressive unfreezing for conservative transfer learning.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.6):
        super(OptimizedResNet18MusicClassifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace classifier with high dropout (keep original structure)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone_layers(self, layers_to_freeze):
        """
        Freeze specified layers for conservative transfer learning
        
        Args:
            layers_to_freeze: List of layer names to freeze
        """
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """Return model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = self.get_trainable_params()
        
        return {
            'model_name': 'ResNet-18 Transfer Learning',
            'num_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Pretrained ResNet-18 with custom classifier',
            'transfer_learning': 'Conservative 3-stage progressive unfreezing'
        }


def create_resnet18_classifier(num_classes: int = 10, dropout_rate: float = 0.6) -> OptimizedResNet18MusicClassifier:
    """Factory function to create ResNet-18 classifier"""
    return OptimizedResNet18MusicClassifier(num_classes=num_classes, dropout_rate=dropout_rate)


def get_resnet_stage_configs():
    """
    Get configurations for different training stages of ResNet transfer learning
    
    Returns:
        dict: Configuration for each training stage
    """
    return {
        'stage1': {
            'name': 'Stage 1: Classifier Only',
            'layers_to_freeze': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
            'description': 'Only train the custom classifier head'
        },
        'stage2': {
            'name': 'Stage 2: + Layer 4',
            'layers_to_freeze': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'],
            'description': 'Unfreeze layer4 + classifier'
        },
        'stage3': {
            'name': 'Stage 3: All Layers',
            'layers_to_freeze': [],
            'description': 'Fine-tune entire network'
        }
    }