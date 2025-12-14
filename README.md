# Music Classification Models Comparison

This repository contains experiments comparing different approaches to music genre classification using deep learning models.

## Project Overview

This project includes two experiments:

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
- **Statistical Analysis**: 5-fold cross-validation with ANOVA and paired t-tests
- **Visualization**: Plots and confusion matrices
- **Training**: Early stopping, learning rate scheduling, overfitting prevention
- **Audio-Specific Techniques**: Custom augmentations designed for spectrograms


## Model Architectures

### CNN Baseline
- 4-layer convolutional neural network
- Batch normalization and dropout

### ResNet-18 (Conservative Transfer Learning)
- Pretrained ImageNet backbone
- 3-stage progressive fine-tuning
- Overfitting prevention strategies

### Vision Transformer
- ViT-Base-Patch16-224 pretrained model
- 2-stage fine-tuning approach

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
