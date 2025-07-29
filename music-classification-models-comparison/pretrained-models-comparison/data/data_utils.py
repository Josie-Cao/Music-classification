import os
from typing import List, Tuple, Dict
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


def prepare_cv_data(spectrograms_dir: str, n_folds: int = 5) -> Tuple[List[str], List[int], List[Tuple], Dict[str, int], Dict[int, str]]:
    """
    Prepare data for cross-validation
    
    Args:
        spectrograms_dir: Directory containing genre subdirectories with images
        n_folds: Number of folds for cross-validation
        
    Returns:
        tuple: (all_images, all_labels, cv_splits, genre_to_idx, idx_to_genre)
    """
    # Get all genre directories
    genres = sorted([d for d in os.listdir(spectrograms_dir) 
                    if os.path.isdir(os.path.join(spectrograms_dir, d))])
    
    print(f"Found {len(genres)} genres: {genres}")

    # Create genre mappings
    genre_to_idx = {genre: i for i, genre in enumerate(genres)}
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}
    
    # Collect all images and labels
    all_images = []
    all_labels = []

    for genre in genres:
        genre_dir = os.path.join(spectrograms_dir, genre)
        genre_images = []
        
        for img_name in os.listdir(genre_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(genre_dir, img_name)
                genre_images.append(img_path)
        
        # Add to collections
        all_images.extend(genre_images)
        all_labels.extend([genre_to_idx[genre]] * len(genre_images))
        
        print(f"  {genre}: {len(genre_images)} images")

    print(f"Total images: {len(all_images)}")
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_splits = list(skf.split(all_images, all_labels))
    
    return all_images, all_labels, cv_splits, genre_to_idx, idx_to_genre


def get_data_transforms(input_size: tuple = (224, 224)):
    """
    Get data transforms for training and testing
    
    Args:
        input_size: Target size for images (height, width)
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def get_model_specific_transforms(model_type: str):
    """
    Get model-specific transforms
    
    Args:
        model_type: Type of model ('cnn', 'resnet', 'vit')
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    if model_type.lower() == 'cnn':
        # CNN baseline uses larger input size
        return get_data_transforms(input_size=(256, 384))
    elif model_type.lower() in ['resnet', 'vit']:
        # Pretrained models typically use 224x224
        return get_data_transforms(input_size=(224, 224))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_data_summary(
    all_images: List[str], 
    all_labels: List[int], 
    cv_splits: List[Tuple], 
    idx_to_genre: Dict[int, str]
):
    """Print a comprehensive summary of the dataset"""
    from collections import Counter
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    # Basic statistics
    print(f"Total samples: {len(all_images)}")
    print(f"Number of classes: {len(set(all_labels))}")
    print(f"Cross-validation folds: {len(cv_splits)}")
    
    # Genre distribution
    print(f"\nGenre distribution:")
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    for genre_idx, count in sorted(label_counts.items()):
        genre_name = idx_to_genre[genre_idx]
        percentage = (count / total_samples) * 100
        print(f"  {genre_name}: {count} samples ({percentage:.1f}%)")
    
    # CV validation
    print(f"\nCross-validation statistics:")
    fold_sizes = [len(test_indices) for _, test_indices in cv_splits]
    avg_test_size = sum(fold_sizes) / len(fold_sizes)
    print(f"  Average test size per fold: {avg_test_size:.1f}")
    print(f"  Test sizes: {fold_sizes}")
    
    print("="*60)