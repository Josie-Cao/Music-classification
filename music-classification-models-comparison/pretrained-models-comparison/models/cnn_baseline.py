import torch
import torch.nn as nn


class MusicGenreCNN(nn.Module):
    """
    CNN Baseline model for music genre classification
    
    This is the same architecture as used in the augmentation comparison experiment,
    serving as a baseline for comparing with pretrained models.
    """
    
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
    
    def get_model_info(self):
        """Return model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CNN Baseline',
            'num_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Custom 4-layer CNN trained from scratch'
        }


def create_cnn_baseline(num_classes: int = 10) -> MusicGenreCNN:
    """Factory function to create CNN baseline model"""
    return MusicGenreCNN(num_classes=num_classes)