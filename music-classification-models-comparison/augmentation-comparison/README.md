# Data Augmentation Techniques Comparison

This experiment compares 5 different data augmentation strategies for music genre classification using CNN models.

## Augmentation Methods Compared

- **Baseline (No Augmentation)**: Standard training without data augmentation
- **Traditional Augmentation**: Classical computer vision augmentations (horizontal flip, affine transforms)
- **Temporal Cropping**: Audio-specific temporal cropping with resizing
- **Spectrogram Noise**: Adding controlled noise to spectrograms
- **SpecAugment**: Frequency and time masking specifically designed for audio spectrograms


## Output Files

- `crossvalidation_detailed_results.csv`: Detailed results for each fold
- `crossvalidation_summary_stats.csv`: Summary statistics for each method
- `crossvalidation_pairwise_tests.csv`: Statistical test results
- `crossvalidation_comparison_results.png`: Comprehensive visualization

