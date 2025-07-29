# Music Classification Models Comparison

This repository contains comprehensive experiments comparing different approaches to music genre classification using deep learning models.

## Project Overview

This project includes two independent but related experiments:

### 1. Data Augmentation Techniques Comparison (`augmentation-comparison/`)
Compares 5 different data augmentation strategies for music genre classification:
- Baseline (No Augmentation)
- Traditional Computer Vision Augmentations
- Temporal Cropping (Audio-specific)
- Spectrogram Noise Addition
- SpecAugment (Frequency/Time Masking)

### 2. Pretrained Models Comparison (`pretrained-models-comparison/`)
Compares 3 different model architectures:
- CNN Baseline (trained from scratch)
- ResNet-18 (with conservative transfer learning)
- Vision Transformer (ViT-Base with staged fine-tuning)

## Features

- **Rigorous Statistical Analysis**: 5-fold cross-validation with ANOVA and paired t-tests
- **Comprehensive Visualization**: Detailed plots and confusion matrices
- **Robust Training**: Early stopping, learning rate scheduling, overfitting prevention
- **Audio-Specific Techniques**: Custom augmentations designed for spectrograms

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn scipy pillow
```

For pretrained models comparison, also install:
```bash
pip install transformers
```

### Compatibility
- Python >= 3.7
- Works with recent versions of all dependencies
- CUDA support optional but recommended for faster training

## Usage

### Data Augmentation Comparison
```bash
cd augmentation-comparison
python main.py
```

### Pretrained Models Comparison
```bash
cd pretrained-models-comparison
python main.py
```

## Data Structure

Both experiments expect data in the following structure:
```
Data/
└── images_original/
    ├── genre1/
    │   ├── spectrogram1.png
    │   └── ...
    ├── genre2/
    └── ...
```
## Results

Each experiment generates:
- Detailed CSV results for each fold
- Summary statistics and statistical test results
- High-quality visualizations
- Comprehensive performance analysis

## Statistical Methods

- **Cross-Validation**: Stratified 5-fold CV for robust evaluation
- **ANOVA**: Tests for significant differences between methods
- **Paired t-tests**: Pairwise comparisons with Bonferroni correction
- **Effect Sizes**: Cohen's d for practical significance assessment

## Model Architectures

### CNN Baseline
- 4-layer convolutional neural network
- Batch normalization and dropout
- ~2.5M parameters

### ResNet-18 (Conservative Transfer Learning)
- Pretrained ImageNet backbone
- 3-stage progressive fine-tuning
- Overfitting prevention strategies

### Vision Transformer
- ViT-Base-Patch16-224 pretrained model
- 2-stage fine-tuning approach
- Automatic fallback if transformers unavailable

## Citation

If you use this code in your research, please cite:

```bibtex
@software{music_classification_models_comparison,
  title={Music Classification Models Comparison},
  author={Josephine Cao},
  year={2024},
  url={https://github.com/yourusername/music-classification-models-comparison}
}
```
