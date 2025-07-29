# Pretrained Models Comparison

This experiment compares 3 different model architectures for music genre classification, including CNN baseline, ResNet-18, and Vision Transformer (ViT).

## Models Compared

- **CNN Baseline**: Custom 4-layer CNN trained from scratch
- **ResNet-18 (Conservative)**: Pretrained ResNet-18 with conservative 3-stage transfer learning
- **Vision Transformer**: Pretrained ViT-Base with 2-stage fine-tuning

## Key Features

- Conservative transfer learning to prevent overfitting
- Multi-stage training strategies
- Overfitting monitoring and prevention
- Differential learning rates for different model components

## Usage

```bash
cd pretrained-models-comparison
python main.py
```

## Output Files

- `pretrained_comparison_detailed_results.csv`: Detailed results for each fold
- `pretrained_comparison_summary_stats.csv`: Summary statistics for each model
- `pretrained_comparison_pairwise_tests.csv`: Statistical test results
- `pretrained_models_comparison_results.png`: Comprehensive visualization


See the main project README for dependencies and setup instructions.