# CNN Image Classification

**Machine Learning Exercise 3.2 - Deep Learning for Image Tasks**

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification on two benchmark datasets. It is part of a group project comparing different deep learning approaches (CNN, RNN, Traditional ML).

### Datasets

| Dataset | Images | Classes | Size | Channels |
|---------|--------|---------|------|----------|
| **FashionMNIST** | 70,000 | 10 | 28×28 | Grayscale |
| **CIFAR-10** | 60,000 | 10 | 32×32 | RGB |

### Metrics

The following metrics are computed (as required by the assignment):

- **Overall Accuracy**: Proportion of correct predictions
- **Macro-averaged F1-Score**: Unweighted mean of F1 scores across all classes
- **AUC-ROC**: Area Under the ROC Curve (One-vs-Rest, Macro-averaged)

## Project Structure

```
cnn_project/
├── cnn.py              # Main CNN implementation
├── main.sh             # Bash script to run all approaches
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── outputs/            # Results directory (created at runtime)
    ├── cnn_results_*.json
    ├── cnn_results_*.csv
    ├── confusion_matrix_*.png
    ├── training_history_*.png
    ├── metrics_comparison_*.png
    └── runtime_comparison_*.png
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

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
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Quick Start

```bash
# Single training run
python cnn.py --dataset fashionmnist

# Grid search for hyperparameter tuning
python cnn.py --dataset cifar10 --grid-search

# Quick test (dry-run mode)
python cnn.py --dataset cifar10 --dry-run
```

### Using main.sh (Run All Approaches)

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

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | fashionmnist | Dataset: `fashionmnist` or `cifar10` |
| `--grid-search` | False | Enable grid search over hyperparameters |
| `--dry-run` | False | Quick test with 500 samples, 3 epochs |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--num-filters` | 32 | Number of filters in first conv layer |
| `--dropout` | 0.5 | Dropout rate |
| `--no-augment` | False | Disable data augmentation |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | ./outputs | Output directory |
| `--quiet` | False | Reduce output verbosity |
| `--no-plots` | False | Skip visualization generation |

## Architecture

### CNN Model

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

### Data Pipeline

1. **Data Source**: TensorFlow/Keras datasets (matching group members' approach)
2. **Preprocessing**: Normalization with dataset-specific mean/std
3. **Augmentation** (training only):
   - Random horizontal flip
   - Random crop with padding
4. **Training**: PyTorch with GPU acceleration (if available)

### Optimizations

- **Mixed Precision Training (AMP)**: ~2x speedup on GPU
- **torch.compile()**: Additional ~10-30% speedup (PyTorch 2.0+)
- **Optimized DataLoader**: `pin_memory=True`, `persistent_workers=True`

## Output Files

### JSON Results

```json
{
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

| rank | epochs | batch_size | lr | num_filters | dropout | augment | test_accuracy | test_macro_f1 | test_roc_auc | training_time_seconds |
|------|--------|------------|-----|-------------|---------|---------|---------------|---------------|--------------|----------------------|
| 1 | 30 | 64 | 0.001 | 64 | 0.3 | True | 0.9234 | 0.9221 | 0.9912 | 45.2 |
| 2 | 30 | 128 | 0.001 | 64 | 0.5 | True | 0.9198 | 0.9185 | 0.9901 | 38.7 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Visualizations

| File | Description |
|------|-------------|
| `confusion_matrix_*.png` | Normalized confusion matrix for best model |
| `training_history_*.png` | Loss and metrics over epochs |
| `metrics_comparison_*.png` | Comparison of accuracy, F1, AUC across configurations |
| `runtime_comparison_*.png` | Training time comparison |

## Grid Search

Default hyperparameter grid:

```python
param_grid = {
    "epochs": [30],
    "batch_size": [64, 128],
    "lr": [1e-3, 5e-4],
    "num_filters": [32, 64],
    "dropout": [0.3, 0.5],
    "augment": [True, False],
}
# Total: 2 × 2 × 2 × 2 × 2 = 32 combinations
```

To modify, edit the `param_grid` dictionary in `cnn.py` (line ~560).

## Expected Results

### FashionMNIST (30 epochs)

| Metric | Typical Range |
|--------|---------------|
| Accuracy | 0.91 - 0.93 |
| Macro F1 | 0.91 - 0.93 |
| ROC-AUC | 0.99+ |

### CIFAR-10 (30 epochs)

| Metric | Typical Range |
|--------|---------------|
| Accuracy | 0.78 - 0.85 |
| Macro F1 | 0.78 - 0.85 |
| ROC-AUC | 0.97 - 0.98 |

## Runtime Estimates

| Configuration | FashionMNIST | CIFAR-10 |
|--------------|--------------|----------|
| Single run (CPU) | ~10-15 min | ~15-20 min |
| Single run (GPU) | ~2-3 min | ~3-5 min |
| Grid search 32 configs (GPU) | ~1-2 hours | ~2-3 hours |
| Dry-run | ~30 sec | ~30 sec |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python cnn.py --batch-size 32

# Or modify grid search to use smaller batches
```

### TensorFlow/PyTorch Conflict

```bash
# If you get import errors, try:
pip uninstall tensorflow torch torchvision
pip install torch torchvision
pip install tensorflow
```

### Slow CPU Training

```bash
# Use dry-run for testing
python cnn.py --dry-run

# Or reduce epochs
python cnn.py --epochs 10
```

## Integration with Group Project

This script is designed to work alongside other approaches. The `main.sh` script can run all three approaches sequentially:

1. **CNN** (this script): `cnn.py`
2. **Approach 2**: Add your script (e.g., `rnn.py`)
3. **Approach 3**: Add your script (e.g., `traditional.py`)

To add another approach:

1. Create your Python script with similar CLI arguments
2. Edit `main.sh` to include your script
3. Run `./main.sh` to execute all approaches

## Report Structure (Suggested)

Based on the group discussion:

1. **Overview of Approaches**
2. **Metrics / Setup**
3. **CNN Methodology/Architecture** (this approach)
4. **CNN Results**
5. **[Other Approach] Methodology**
6. **[Other Approach] Results**
7. **Metrics Comparison** (3 Metrics / Confusion Matrix)
8. **Runtime Comparison**
9. **Patterns / Analysis**
10. **Best Choice / Conclusion**

## References

- Dataset sources:
  - FashionMNIST: Xiao et al., 2017
  - CIFAR-10: Krizhevsky et al., 2009
- Normalization values: PyTorch community standard
- Architecture inspired by: VGG, ResNet concepts

## Authors

ML Exercise 3.2 Group

- Ben Schill (12347303)
- Ege Özbaran (12433722)
- Felix Kapfer (12429669)

