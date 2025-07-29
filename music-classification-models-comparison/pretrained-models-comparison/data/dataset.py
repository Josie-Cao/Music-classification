import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Callable


class MusicGenreDataset(Dataset):
    """
    Dataset class for music genre classification using spectrogram images
    
    This dataset loads spectrogram images and their corresponding genre labels,
    applying optional transforms for data augmentation and normalization.
    """
    
    def __init__(
        self, 
        image_paths: List[str], 
        labels: List[int], 
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_paths: List of paths to spectrogram images
            labels: List of corresponding genre labels (integers)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Validate input
        if len(image_paths) != len(labels):
            raise ValueError(f"Number of images ({len(image_paths)}) and labels ({len(labels)}) must match")
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            tuple: (image_tensor, label) where image_tensor is the transformed image
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image as RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations if provided
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image of expected size if loading fails
            if self.transform:
                dummy_image = Image.new('RGB', (384, 256), color='black')  # (width, height)
                image = self.transform(dummy_image)
            else:
                image = torch.zeros(3, 256, 384)  # (channels, height, width)
            
            return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset"""
        from collections import Counter
        return Counter(self.labels)
    
    def get_sample_info(self) -> dict:
        """Get basic information about the dataset"""
        return {
            'total_samples': len(self),
            'num_classes': len(set(self.labels)),
            'class_distribution': dict(self.get_class_distribution())
        }