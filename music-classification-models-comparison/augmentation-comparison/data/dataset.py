import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable


class MusicGenreDataset(Dataset):
    """
    Dataset class for music genre classification using spectrogram images
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
            labels: List of corresponding genre labels
            transform: Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Validate input
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            tuple: (image, label) where image is the transformed spectrogram
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
            # Return a black image of the expected size if loading fails
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
    
    def get_sample_paths_by_class(self, class_idx: int) -> List[str]:
        """Get all image paths for a specific class"""
        return [path for path, label in zip(self.image_paths, self.labels) if label == class_idx]


def create_balanced_subset(
    image_paths: List[str],
    labels: List[int],
    samples_per_class: int
) -> tuple:
    """
    Create a balanced subset with equal samples per class
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        samples_per_class: Number of samples to keep per class
        
    Returns:
        tuple: (subset_paths, subset_labels)
    """
    from collections import defaultdict
    import random
    
    # Group paths by class
    class_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        class_paths[label].append(path)
    
    # Sample from each class
    subset_paths = []
    subset_labels = []
    
    for class_idx, paths in class_paths.items():
        if len(paths) >= samples_per_class:
            selected_paths = random.sample(paths, samples_per_class)
        else:
            selected_paths = paths  # Use all available if not enough
        
        subset_paths.extend(selected_paths)
        subset_labels.extend([class_idx] * len(selected_paths))
    
    return subset_paths, subset_labels


def validate_dataset_paths(image_paths: List[str]) -> tuple:
    """
    Validate that all image paths exist and are readable
    
    Args:
        image_paths: List of paths to validate
        
    Returns:
        tuple: (valid_paths, invalid_paths)
    """
    valid_paths = []
    invalid_paths = []
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                # Try to open the image to ensure it's valid
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
            except Exception:
                invalid_paths.append(path)
        else:
            invalid_paths.append(path)
    
    return valid_paths, invalid_paths