import torch
import torch.nn as nn
from typing import List, Optional


class MusicGenreCNN(nn.Module):
    """
    Configurable CNN model for music genre classification
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for fully connected layers
        conv_channels: List of channel sizes for convolutional layers
        fc_units: List of units for fully connected layers
        input_size: Expected input size (height, width)
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        conv_channels: List[int] = [32, 64, 128, 256],
        fc_units: List[int] = [1024, 256],
        input_size: tuple = (256, 384)
    ):
        super(MusicGenreCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.conv_channels = conv_channels
        self.fc_units = fc_units
        self.input_size = input_size
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB input
        
        for out_channels in conv_channels:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_layers.append(conv_block)
            in_channels = out_channels
        
        # Calculate flattened size after convolutions
        self.flattened_size = self._calculate_flattened_size()
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_units = self.flattened_size
        
        for units in fc_units:
            fc_block = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(prev_units, units),
                nn.ReLU()
            )
            self.fc_layers.append(fc_block)
            prev_units = units
        
        # Output layer
        self.output = nn.Linear(prev_units, num_classes)
        
    def _calculate_flattened_size(self):
        """Calculate the size after convolution and pooling operations"""
        h, w = self.input_size
        for _ in self.conv_channels:
            h = h // 2  # MaxPool2d with stride=2
            w = w // 2
        return self.conv_channels[-1] * h * w
    
    def forward(self, x):
        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # Output layer
        x = self.output(x)
        
        return x
    
    def get_model_info(self):
        """Return model configuration information"""
        return {
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'conv_channels': self.conv_channels,
            'fc_units': self.fc_units,
            'input_size': self.input_size,
            'flattened_size': self.flattened_size,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_default_cnn(num_classes: int = 10) -> MusicGenreCNN:
    """Create CNN with default architecture (matches original)"""
    return MusicGenreCNN(
        num_classes=num_classes,
        dropout_rate=0.5,
        conv_channels=[32, 64, 128, 256],
        fc_units=[1024, 256],
        input_size=(256, 384)
    )


def create_lightweight_cnn(num_classes: int = 10) -> MusicGenreCNN:
    """Create a lighter CNN model"""
    return MusicGenreCNN(
        num_classes=num_classes,
        dropout_rate=0.3,
        conv_channels=[16, 32, 64, 128],
        fc_units=[512, 128],
        input_size=(256, 384)
    )


def create_deep_cnn(num_classes: int = 10) -> MusicGenreCNN:
    """Create a deeper CNN model"""
    return MusicGenreCNN(
        num_classes=num_classes,
        dropout_rate=0.6,
        conv_channels=[32, 64, 128, 256, 512],
        fc_units=[2048, 512, 256],
        input_size=(256, 384)
    )