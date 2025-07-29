import os
from typing import List, Tuple, Dict
from sklearn.model_selection import StratifiedKFold


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
    idx_to_genre = {i: genre for genre, i in genre_to_idx.items()}
    
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


def get_genre_statistics(all_labels: List[int], idx_to_genre: Dict[int, str]) -> Dict[str, dict]:
    """
    Get statistics about genre distribution
    
    Args:
        all_labels: List of all labels
        idx_to_genre: Mapping from index to genre name
        
    Returns:
        dict: Statistics for each genre
    """
    from collections import Counter
    import numpy as np
    
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    
    statistics = {}
    for genre_idx, count in label_counts.items():
        genre_name = idx_to_genre[genre_idx]
        statistics[genre_name] = {
            'count': count,
            'percentage': (count / total_samples) * 100,
            'index': genre_idx
        }
    
    return statistics


def validate_cv_splits(cv_splits: List[Tuple], all_labels: List[int]) -> Dict[str, any]:
    """
    Validate cross-validation splits for balance and correctness
    
    Args:
        cv_splits: List of (train_indices, test_indices) tuples
        all_labels: List of all labels
        
    Returns:
        dict: Validation results
    """
    import numpy as np
    from collections import Counter
    
    n_folds = len(cv_splits)
    total_samples = len(all_labels)
    unique_labels = set(all_labels)
    
    # Check each fold
    fold_stats = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_labels = [all_labels[i] for i in train_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_distribution': dict(train_dist),
            'test_distribution': dict(test_dist),
            'train_classes': set(train_labels),
            'test_classes': set(test_labels)
        })
    
    # Overall validation
    avg_test_size = np.mean([stats['test_size'] for stats in fold_stats])
    std_test_size = np.std([stats['test_size'] for stats in fold_stats])
    
    # Check if all classes are represented in each fold
    all_classes_in_train = all(unique_labels.issubset(stats['train_classes']) for stats in fold_stats)
    all_classes_in_test = all(unique_labels.issubset(stats['test_classes']) for stats in fold_stats)
    
    validation_results = {
        'n_folds': n_folds,
        'total_samples': total_samples,
        'avg_test_size': avg_test_size,
        'std_test_size': std_test_size,
        'test_size_cv': std_test_size / avg_test_size if avg_test_size > 0 else 0,
        'all_classes_in_train': all_classes_in_train,
        'all_classes_in_test': all_classes_in_test,
        'fold_stats': fold_stats,
        'is_valid': all_classes_in_train and std_test_size / avg_test_size < 0.1  # CV < 10%
    }
    
    return validation_results


def print_data_summary(
    all_images: List[str], 
    all_labels: List[int], 
    cv_splits: List[Tuple], 
    idx_to_genre: Dict[int, str]
):
    """Print a comprehensive summary of the dataset"""
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    # Basic statistics
    print(f"Total samples: {len(all_images)}")
    print(f"Number of classes: {len(set(all_labels))}")
    print(f"Cross-validation folds: {len(cv_splits)}")
    
    # Genre distribution
    print(f"\nGenre distribution:")
    genre_stats = get_genre_statistics(all_labels, idx_to_genre)
    for genre, stats in genre_stats.items():
        print(f"  {genre}: {stats['count']} samples ({stats['percentage']:.1f}%)")
    
    # CV validation
    print(f"\nCross-validation validation:")
    cv_validation = validate_cv_splits(cv_splits, all_labels)
    print(f"  Average test size: {cv_validation['avg_test_size']:.1f} ± {cv_validation['std_test_size']:.1f}")
    print(f"  Test size CV: {cv_validation['test_size_cv']:.3f}")
    print(f"  All classes in train: {cv_validation['all_classes_in_train']}")
    print(f"  All classes in test: {cv_validation['all_classes_in_test']}")
    print(f"  Validation passed: {'✓' if cv_validation['is_valid'] else '✗'}")
    
    print("="*60)