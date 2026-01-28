# Deep Learning for Image Classification

**Machine Learning Exercise 3.2 - Deep Learning for Image Tasks**

## Overview

This project implements multiple machine learning approaches for image classification:

1. **Shallow Learning** - SVM and Random Forest with traditional feature extraction
2. **CNN** - Convolutional Neural Network (custom architecture)
3. **ViT** - Vision Transformer (pretrained from HuggingFace)

### Datasets

| Dataset | Images | Classes | Size | Channels |
|---------|--------|---------|------|----------|
| **FashionMNIST** | 70,000 | 10 | 28×28 | Grayscale |
| **CIFAR-10** | 60,000 | 10 | 32×32 | RGB |

Data is loaded using TensorFlow/Keras (matching group members' approach).

### Metrics

The following metrics are computed (as required by the assignment):

- **Overall Accuracy**: Proportion of correct predictions
- **Macro-averaged F1-Score**: Unweighted mean of F1 scores across all classes
- **AUC-ROC**: Area Under the ROC Curve (One-vs-Rest, Macro-averaged)

## Project Structure

```
ml_project/
├── utils.py            # Timer, compute_metrics, set_seed, constants
├── data.py             # Data loading (TensorFlow → PyTorch)
├── plots.py            # Visualization functions
├── shallow.py          # Shallow learning (SVM, Random Forest)
├── cnn.py              # CNN model and training
├── vit.py              # Vision Transformer model and training
├── main.sh             # Bash script to run all approaches
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── outputs/            # Results directory (created at runtime)
    ├── shallow_results_*.json
    ├── confusion_matrices_*.txt/json
    ├── cnn_results_*.json
    ├── cnn_results_*.csv
    ├── vit_results_*.json
    ├── vit_results_*.csv
    └── *.png           # Visualization plots
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Run Individual Models

```bash
# Shallow Learning (SVM + Random Forest)
python shallow.py --dataset fashion --best
python shallow.py --dataset cifar --max-samples 10000
python shallow.py --dataset fashion --vocab-size 300

# CNN
python cnn.py --dataset fashionmnist
python cnn.py --dataset cifar10 --grid-search
python cnn.py --dataset cifar10 --dry-run

# ViT
python vit.py --dataset fashionmnist
python vit.py --dataset cifar10 --grid-search
python vit.py --dataset cifar10 --dry-run
```

### Run All Models via main.sh

```bash
# Make executable
chmod +x main.sh

# Run with default settings
./main.sh

# Run with specific dataset
./main.sh --dataset cifar10

# Quick test run
./main.sh --dry-run

# Full grid search
./main.sh --grid-search
```

### Command Line Arguments

**shallow.py** arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Dataset: `fashion` or `cifar` |
| `--best` | False | Use pre-optimized parameters (skip GridSearch) |
| `--max-samples` | None | Tune on N samples, train on full dataset |
| `--vocab-size` | 200 | Size of visual vocabulary for ORB+BoW |
| `--svm-c` | None | Manual override: SVM C parameter |
| `--svm-gamma` | None | Manual override: SVM gamma parameter |
| `--rf-n-estimators` | None | Manual override: Random Forest n_estimators |
| `--rf-max-depth` | None | Manual override: Random Forest max_depth |

**cnn.py** and **vit.py** share similar arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | fashionmnist | Dataset: `fashionmnist` or `cifar10` |
| `--grid-search` | False | Enable grid search over hyperparameters |
| `--dry-run` | False | Quick test with small data subset |
| `--epochs` | 30 (CNN) / 50 (ViT) | Number of training epochs |
| `--batch-size` | 64 (CNN) / 32 (ViT) | Batch size |
| `--lr` | 1e-3 (CNN) / 2e-5 (ViT) | Learning rate |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | ./outputs | Output directory |
| `--quiet` | False | Reduce output verbosity |
| `--no-plots` | False | Skip visualization generation |

CNN-specific arguments:
| `--num-filters` | 32 | Number of filters in first conv layer |
| `--dropout` | 0.5 | Dropout rate |
| `--no-augment` | False | Disable data augmentation |

## Module Descriptions

### utils.py

Contains shared utilities:

```python
from utils import Timer, timer, compute_metrics, set_seed
from utils import FASHIONMNIST_CLASSES, CIFAR10_CLASSES
from utils import CIFAR10_MEAN, CIFAR10_STD, FASHIONMNIST_MEAN, FASHIONMNIST_STD

# Timer usage
timer.start("training")
# ... training code ...
elapsed = timer.stop("training")
print(timer.format(elapsed))  # "2m 30s"

# Metrics computation
metrics = compute_metrics(y_true, y_pred, y_prob)
# Returns: {"accuracy": 0.92, "macro_f1": 0.91, "roc_auc": 0.99}
```

### shallow.py

Shallow learning with traditional ML:

```python
from shallow import run_shallow_task

# Run with best parameters (fastest)
results = run_shallow_task(
    dataset_name="cifar",
    use_best_params=True
)

# Run with grid search
results = run_shallow_task(
    dataset_name="fashion",
    max_samples=10000,  # Tune on 10k, train on full
    vocab_size=200
)
```

Implements:
- **Feature extraction**: 3-channel color histograms and ORB+BoW
- **Models**: SVM (RBF kernel) and Random Forest
- **Metrics**: Accuracy, Macro F1, AUC-ROC per experiment

### data.py

Data loading functions:

```python
from data import get_loaders, get_raw_data

# For CNN (returns PyTorch DataLoaders)
train_loader, val_loader, test_loader, classes = get_loaders(
    dataset="cifar10",
    batch_size=64,
    augment=True,
    dry_run=False
)

# For ViT (returns raw numpy arrays)
(x_train, y_train), (x_val, y_val), (x_test, y_test), classes = get_raw_data(
    dataset="cifar10",
    dry_run=False
)
```

### plots.py

Visualization functions:

```python
from plots import (
    plot_confusion_matrix,
    plot_training_history,
    plot_metrics_comparison,
    plot_runtime_comparison,
    generate_all_visualizations
)

# Generate all plots for results
generate_all_visualizations(
    results=results,
    dataset="cifar10",
    classes=classes,
    output_dir="./outputs",
    model_name="cnn",
    best_y_true=y_true,
    best_y_pred=y_pred,
    best_history=history
)
```

## Output Files

### JSON Results

```json
{
  "model": "CNN",
  "dataset": "fashionmnist",
  "timestamp": "20240128_143022",
  "total_script_time_seconds": 245.3,
  "total_script_time_formatted": "4m 5s",
  "num_experiments": 32,
  "best_result": {
    "parameters": {
      "epochs": 30,
      "batch_size": 64,
      "lr": 0.001,
      "num_filters": 64,
      "dropout": 0.3,
      "augment": true
    },
    "test_metrics": {
      "accuracy": 0.9234,
      "macro_f1": 0.9221,
      "roc_auc": 0.9912
    },
    "training_time_seconds": 45.2
  },
  "all_results": [...]
}
```

### CSV Results

| rank | epochs | batch_size | lr | ... | test_accuracy | test_macro_f1 | test_roc_auc | training_time_seconds |
|------|--------|------------|-----|-----|---------------|---------------|--------------|----------------------|
| 1 | 30 | 64 | 0.001 | ... | 0.9234 | 0.9221 | 0.9912 | 45.2 |

### Visualizations

| File | Description |
|------|-------------|
| `{model}_confusion_matrix_{dataset}.png` | Normalized confusion matrix |
| `{model}_training_history_{dataset}.png` | Loss/Accuracy/F1 over epochs |
| `{model}_metrics_comparison_{dataset}.png` | Comparison across configurations |
| `{model}_runtime_comparison_{dataset}.png` | Training time comparison |

## Model Architectures

### CNN

```
Input (C × H × W)
    │
    ├── Conv2d(C, 32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    │
    ├── Conv2d(32, 64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    │
    ├── Conv2d(64, 128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
    │
    ├── Flatten
    │
    ├── Linear(*, 128) + ReLU + Dropout
    │
    └── Linear(128, 10)
        │
        Output (10 classes)
```

### ViT

Uses pretrained `google/vit-base-patch16-224-in21k` from HuggingFace:
- Patch size: 16×16
- Input resolution: 224×224 (images are resized)
- ~85.8M parameters
- Fine-tuned classification head

## Grid Search

### CNN Grid

```python
param_grid = {
    "epochs": [30],
    "batch_size": [64, 128],
    "lr": [1e-3, 5e-4],
    "num_filters": [32, 64],
    "dropout": [0.3, 0.5],
    "augment": [True, False],
}
# Total: 32 combinations
```

### ViT Grid

```python
param_grid = {
    "epochs": [50],
    "batch_size": [16, 32],
    "lr": [2e-5, 5e-5],
}
# Total: 4 combinations
```

## Expected Results

### FashionMNIST

| Model | Accuracy | Macro F1 | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| SVM (Histogram) | 0.84-0.87 | 0.84-0.87 | 0.97-0.98 | ~1-3 min |
| RF (Histogram) | 0.85-0.88 | 0.85-0.88 | 0.98-0.99 | ~1-2 min |
| SVM (ORB+BoW) | 0.82-0.85 | 0.82-0.85 | 0.96-0.97 | ~2-4 min |
| RF (ORB+BoW) | 0.83-0.86 | 0.83-0.86 | 0.97-0.98 | ~1-3 min |
| CNN | 0.91-0.93 | 0.91-0.93 | 0.99+ | ~2-5 min |
| ViT | 0.93-0.95 | 0.93-0.95 | 0.99+ | ~10-15 min |

### CIFAR-10

| Model | Accuracy | Macro F1 | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| SVM (Histogram) | 0.35-0.42 | 0.35-0.42 | 0.75-0.82 | ~3-6 min |
| RF (Histogram) | 0.38-0.45 | 0.38-0.45 | 0.78-0.85 | ~2-4 min |
| SVM (ORB+BoW) | 0.28-0.35 | 0.28-0.35 | 0.68-0.75 | ~4-8 min |
| RF (ORB+BoW) | 0.30-0.37 | 0.30-0.37 | 0.70-0.77 | ~2-5 min |
| CNN | 0.78-0.85 | 0.78-0.85 | 0.97-0.98 | ~3-8 min |
| ViT | 0.95-0.98 | 0.95-0.98 | 0.99+ | ~15-30 min |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python cnn.py --batch-size 32
python vit.py --batch-size 8
```

### HuggingFace Model Download Issues

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Or use offline mode (if model is already cached)
export TRANSFORMERS_OFFLINE=1
```

### Slow CPU Training

```bash
# Use dry-run for testing
python cnn.py --dry-run
python vit.py --dry-run
```

## Report Structure (Suggested)

Based on the group discussion:

1. **Overview of Approaches**
2. **Metrics / Setup**
3. **CNN Methodology/Architecture**
4. **CNN Results**
5. **ViT Methodology/Architecture**
6. **ViT Results**
7. **Metrics Comparison** (3 Metrics / Confusion Matrix)
8. **Runtime Comparison**
9. **Patterns / Analysis**
10. **Best Choice / Conclusion**

## References

- Dataset sources:
  - FashionMNIST: Xiao et al., 2017
  - CIFAR-10: Krizhevsky et al., 2009
- ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020
- HuggingFace Transformers: https://huggingface.co/transformers/

## Authors

ML Exercise 3.2 Group

## License

Academic use only - TU Wien Machine Learning Course
