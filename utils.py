#!/usr/bin/env python3
"""
Utility functions and classes for ML Exercise 3.2

Contains:
- Timer: Class for measuring execution time
- compute_metrics: Function to compute accuracy, macro-F1, and ROC-AUC
- set_seed: Function to set random seeds for reproducibility
- Config constants: Dataset classes and normalization values
"""

import time
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

FASHIONMNIST_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# ----------------------------------------------------------------------------
# Data normalization values for CIFAR-10.
# These mean and std values are empirically computed over the CIFAR-10 training
# set and are commonly used in the literature and PyTorch community.
#
# Source:
#   - CIFAR-10 dataset
#     https://pytorch.org/vision/stable/transforms.html
#     https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
# 
# Value Used for Normalization:
#   - https://github.com/pytorch/examples
#   - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#   - https://pytorch.org/vision/stable/transforms.html
#   - https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
# 
# The values are not computed here but reused as standard reference statistics
# ----------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)





# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name):
        """Start a named timer."""
        self.start_times[name] = time.time()
    
    def stop(self, name):
        """Stop a named timer and record elapsed time."""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            return elapsed
        return 0
    
    def get(self, name):
        """Get elapsed time for a named timer."""
        return self.times.get(name, 0)
    
    def get_all(self):
        """Get all recorded times."""
        return self.times.copy()
    
    def format(self, seconds):
        """Format seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


# Global timer instance
timer = Timer()


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all metrics: Accuracy, Macro-F1, ROC-AUC.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary with accuracy, macro_f1, and roc_auc
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics




# ============================================================================
# Set Seed for reproducibility
# ============================================================================

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
