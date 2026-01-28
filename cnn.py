# !/usr/bin/env python3


# Import general libaries
import csv
import json
import time
import random
import warnings
import itertools
import argparse

from datetime import datetime
from pathlib import Path

# Import ML libaries
import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as F

from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets, transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Libaries for visualisation
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Agg')  # Non-interactive backend

# Remove warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Set the configs
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
#
# The values are not computed here but reused as standard reference statistics
# in the function get_loaders
# ----------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD  = (0.3530,)


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
# Helper function to calculate the metrics
# ============================================================================
def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all metrics.
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
# DATA LOADING (TensorFlow/Keras)
# ============================================================================

def load_tensorflow_data(dataset):
    """
    Load data using TensorFlow/Keras datasets.
    
    This matches the approach used by group members:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    
    Returns numpy arrays.

    """
    if dataset == "fashionmnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        classes = FASHIONMNIST_CLASSES
        # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        classes = CIFAR10_CLASSES
        # y comes as (N, 1), flatten to (N,)
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    return (x_train, y_train), (x_test, y_test), classes


def normalize_data(x, mean, std):
    """Apply channel-wise normalization."""
    mean = np.array(mean).reshape(1, 1, 1, -1)
    std = np.array(std).reshape(1, 1, 1, -1)
    return (x - mean) / std


def apply_augmentation(x, dataset):
    """
    Apply simple data augmentation (random flip, random crop).
    Works on numpy arrays.
    """
    augmented = []
    pad = 4
    
    for img in x:
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1)
        
        # Random crop with padding
        if dataset == "fashionmnist":
            h, w = 28, 28
        else:
            h, w = 32, 32
        
        # Pad image
        if len(img.shape) == 3:
            padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        else:
            padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
        
        # Random crop
        start_h = random.randint(0, 2 * pad)
        start_w = random.randint(0, 2 * pad)
        img = padded[start_h:start_h+h, start_w:start_w+w]
        
        augmented.append(img)
    
    return np.array(augmented)


def get_loaders(dataset, batch_size=64, augment=True, val_split=0.1, seed=42,
                dry_run=False, num_workers=4):
    """
    Load dataset using TensorFlow/Keras and create PyTorch DataLoaders.
    
    Data source: TensorFlow/Keras datasets (matching group members' approach)
    Training: PyTorch (for GPU acceleration and model compatibility)
    """
    
    # Load data via TensorFlow
    (x_train, y_train), (x_test, y_test), classes = load_tensorflow_data(dataset)
    
    # Get normalization values
    if dataset == "fashionmnist":
        mean, std = FASHIONMNIST_MEAN, FASHIONMNIST_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    
    # Dry-run mode: use small subset
    if dry_run:
        np.random.seed(seed)
        train_idx = np.random.permutation(len(x_train))[:500]
        test_idx = np.random.permutation(len(x_test))[:100]
        x_train, y_train = x_train[train_idx], y_train[train_idx]
        x_test, y_test = x_test[test_idx], y_test[test_idx]
        print(f"[DRY-RUN] Train: {len(x_train)}, Test: {len(x_test)}")
    
    # Train/Val split
    np.random.seed(seed)
    n_val = int(len(x_train) * val_split)
    indices = np.random.permutation(len(x_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    
    # Apply augmentation to training data
    if augment:
        x_train = apply_augmentation(x_train, dataset)
    
    # Normalize all data
    x_train = normalize_data(x_train, mean, std)
    x_val = normalize_data(x_val, mean, std)
    x_test = normalize_data(x_test, mean, std)
    
    # Convert to PyTorch format: (N, H, W, C) -> (N, C, H, W)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_val = np.transpose(x_val, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).long()
    )
    
    # DataLoader settings
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": num_workers if use_cuda else 0,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0 and use_cuda,
    }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    
    if not dry_run:
        print(f"[DATA] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, classes


# ============================================================================
# MODEL
# ============================================================================

class CNN(torch.nn.Module):
    """CNN for image classification."""
    
    def __init__(self, in_channels=1, num_classes=10, num_filters=32, dropout=0.5):
        super().__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, num_filters, 3, padding=1),
            torch.nn.BatchNorm2d(num_filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(num_filters, num_filters*2, 3, padding=1),
            torch.nn.BatchNorm2d(num_filters*2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(num_filters*2, num_filters*4, 3, padding=1),
            torch.nn.BatchNorm2d(num_filters*4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )
        
        self.classifier = None
        self.dropout = torch.nn.Dropout(dropout)
        self.num_filters = num_filters
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        if self.classifier is None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(x.size(1), 128),
                torch.nn.ReLU(inplace=True),
                self.dropout,
                torch.nn.Linear(128, self.num_classes)
            ).to(x.device)
        
        return self.classifier(x)


def get_model(dataset, num_filters=32, dropout=0.5, compile_model=True):
    """Create and optionally compile model."""
    in_channels = 1 if dataset == "fashionmnist" else 3
    model = CNN(in_channels=in_channels, num_classes=10, 
                num_filters=num_filters, dropout=dropout)
    
    # torch.compile for PyTorch 2.0+ (used for speedup)
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] Model compiled with torch.compile()")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch_amp(model, loader, optimizer, scaler, device):
    """Train one epoch with mixed precision (GPU)."""
    model.train()
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type="cuda"):
            output = model(data)
            loss = F.cross_entropy(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * data.size(0)
    
    return total_loss / len(loader.dataset)


def train_epoch_cpu(model, loader, optimizer, device):
    """Train one epoch on CPU (no AMP)."""
    model.train()
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """Evaluate model and return metrics + predictions."""
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        if use_amp:
            with autocast(device_type="cuda"):
                output = model(data)
                loss = F.cross_entropy(output, target)
        else:
            output = model(data)
            loss = F.cross_entropy(output, target)
        
        total_loss += loss.item() * data.size(0)
        probs = F.softmax(output, dim=1)
        preds = probs.argmax(dim=1)
        
        all_y.extend(target.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
    
    metrics = compute_metrics(all_y, all_pred, np.array(all_prob))
    metrics["loss"] = total_loss / len(loader.dataset)
    
    return metrics, np.array(all_y), np.array(all_pred), np.array(all_prob)


def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                device="cuda", patience=5, verbose=True):
    """Train model with AMP and early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    best_f1 = 0
    best_metrics = None
    best_state = None
    no_improve = 0
    history = []
    
    # Start training timer
    timer.start("training")
    
    iterator = range(1, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for epoch in iterator:
        if use_amp:
            train_loss = train_epoch_amp(model, train_loader, optimizer, scaler, device)
        else:
            train_loss = train_epoch_cpu(model, train_loader, optimizer, device)
        
        val_metrics, _, _, _ = evaluate(model, val_loader, device, use_amp)
        scheduler.step()
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        })
        
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if verbose:
            iterator.set_postfix({"loss": f"{train_loss:.4f}", "val_f1": f"{val_metrics['macro_f1']:.4f}"})
    
    # Stop training timer
    training_time = timer.stop("training")
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics, history, training_time


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix saved: {save_path}")


def plot_training_history(history, save_path):
    """Plot training history (loss and metrics)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = [h['epoch'] for h in history]
    
    # Loss
    axes[0].plot(epochs, [h['train_loss'] for h in history], 'b-', label='Train Loss')
    axes[0].plot(epochs, [h['val_loss'] for h in history], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, [h['val_accuracy'] for h in history], 'g-', label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, [h['val_macro_f1'] for h in history], 'm-', label='Val Macro-F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Validation Macro F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Training history saved: {save_path}")


def plot_metrics_comparison(results, save_path):
    """Plot metrics comparison across different configurations."""
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    labels = [f"#{i+1}" for i in range(min(10, len(results)))]
    results_subset = results[:10]
    
    metrics = ['accuracy', 'macro_f1', 'roc_auc']
    titles = ['Accuracy', 'Macro F1', 'ROC-AUC']
    colors = ['steelblue', 'forestgreen', 'coral']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        values = [r['test_metrics'][metric] for r in results_subset]
        bars = axes[idx].bar(labels, values, color=color, edgecolor='black', alpha=0.8)
        axes[idx].set_xlabel('Configuration Rank')
        axes[idx].set_ylabel(title)
        axes[idx].set_title(f'Test {title} Comparison')
        axes[idx].set_ylim(min(values) * 0.95, max(values) * 1.02)
        axes[idx].grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Metrics comparison saved: {save_path}")


def plot_runtime_comparison(results, save_path):
    """Plot runtime comparison across configurations."""
    if len(results) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = [f"#{i+1}" for i in range(min(10, len(results)))]
    results_subset = results[:10]
    
    times = [r.get('training_time', 0) for r in results_subset]
    
    bars = ax.bar(labels, times, color='teal', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Configuration Rank')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Runtime Comparison')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Runtime comparison saved: {save_path}")


def generate_all_visualizations(results, dataset, classes, output_dir, 
                                 best_y_true=None, best_y_pred=None, best_history=None):
    """Generate all visualization plots."""
    output_dir = Path(output_dir)
    
    # Confusion matrix for best model
    if best_y_true is not None and best_y_pred is not None:
        plot_confusion_matrix(
            best_y_true, best_y_pred, classes,
            output_dir / f"confusion_matrix_{dataset}.png",
            title=f"Confusion Matrix - {dataset.upper()} (Best Model)"
        )
    
    # Training history for best model
    if best_history is not None:
        plot_training_history(
            best_history,
            output_dir / f"training_history_{dataset}.png"
        )
    
    # Metrics comparison (if grid search)
    if len(results) > 1:
        plot_metrics_comparison(
            results,
            output_dir / f"metrics_comparison_{dataset}.png"
        )
        
        plot_runtime_comparison(
            results,
            output_dir / f"runtime_comparison_{dataset}.png"
        )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_experiment(dataset, params, device, dry_run=False, verbose=True):
    """Run a single experiment."""
    set_seed(params.get("seed", 42))
    
    train_loader, val_loader, test_loader, classes = get_loaders(
        dataset=dataset,
        batch_size=params["batch_size"],
        augment=params["augment"],
        seed=params.get("seed", 42),
        dry_run=dry_run
    )
    
    # Don't compile in dry-run (compilation overhead not worth it)
    model = get_model(
        dataset=dataset,
        num_filters=params["num_filters"],
        dropout=params["dropout"],
        compile_model=not dry_run and device.type == "cuda"
    )
    
    epochs = min(3, params["epochs"]) if dry_run else params["epochs"]
    
    best_val_metrics, history, training_time = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=params["lr"],
        device=device,
        patience=params.get("patience", 5),
        verbose=verbose
    )
    
    # Test evaluation
    test_metrics, y_true, y_pred, y_prob = evaluate(
        model, test_loader, device, use_amp=device.type == "cuda"
    )
    
    return {
        "params": params,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "training_time": training_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "classes": classes,
    }


def grid_search(dataset, param_grid, device, dry_run=False, verbose=True):
    """Run grid search over hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n{'='*60}")
    print(f"Grid Search: {len(combinations)} combinations")
    print(f"Dataset: {dataset}")
    print(f"Optimizations: AMP={'cuda' in str(device)}, torch.compile={'cuda' in str(device)}")
    if dry_run:
        print("[DRY-RUN MODE]")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combinations)}] {params}")
        
        try:
            result = run_experiment(dataset, params, device, dry_run=dry_run, verbose=verbose)
            result["combo_id"] = i
            results.append(result)
            
            print(f"  -> Test Acc: {result['test_metrics']['accuracy']:.4f}, "
                  f"F1: {result['test_metrics']['macro_f1']:.4f}, "
                  f"AUC: {result['test_metrics']['roc_auc']:.4f}, "
                  f"Time: {result['training_time']:.1f}s")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by test Macro-F1 (descending)
    results.sort(key=lambda x: x["test_metrics"]["macro_f1"], reverse=True)
    
    return results


def save_results(results, dataset, output_dir="./outputs", dry_run=False, total_time=0):
    """Save results as JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else ""
    
    best = results[0] if results else None
    
    # JSON export
    json_data = {
        "dataset": dataset,
        "timestamp": timestamp,
        "dry_run": dry_run,
        "total_script_time_seconds": total_time,
        "total_script_time_formatted": timer.format(total_time),
        "num_experiments": len(results),
        "best_result": {
            "parameters": best["params"] if best else None,
            "test_metrics": best["test_metrics"] if best else None,
            "val_metrics": best["val_metrics"] if best else None,
            "training_time_seconds": best.get("training_time", 0) if best else 0,
        },
        "all_results": [
            {
                "rank": i + 1,
                "parameters": r["params"],
                "test_accuracy": r["test_metrics"]["accuracy"],
                "test_macro_f1": r["test_metrics"]["macro_f1"],
                "test_roc_auc": r["test_metrics"]["roc_auc"],
                "val_accuracy": r["val_metrics"]["accuracy"],
                "val_macro_f1": r["val_metrics"]["macro_f1"],
                "val_roc_auc": r["val_metrics"]["roc_auc"],
                "training_time_seconds": r.get("training_time", 0),
            }
            for i, r in enumerate(results)
        ]
    }
    
    json_path = output_dir / f"cnn_results_{dataset}_{timestamp}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[OUTPUT] JSON saved: {json_path}")
    
    # CSV export
    csv_path = output_dir / f"cnn_results_{dataset}_{timestamp}{suffix}.csv"
    
    if results:
        param_keys = list(results[0]["params"].keys())
        fieldnames = ["rank"] + param_keys + [
            "test_accuracy", "test_macro_f1", "test_roc_auc",
            "val_accuracy", "val_macro_f1", "val_roc_auc",
            "training_time_seconds"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, r in enumerate(results):
                row = {"rank": i + 1}
                row.update(r["params"])
                row.update({
                    "test_accuracy": round(r["test_metrics"]["accuracy"], 4),
                    "test_macro_f1": round(r["test_metrics"]["macro_f1"], 4),
                    "test_roc_auc": round(r["test_metrics"]["roc_auc"], 4),
                    "val_accuracy": round(r["val_metrics"]["accuracy"], 4),
                    "val_macro_f1": round(r["val_metrics"]["macro_f1"], 4),
                    "val_roc_auc": round(r["val_metrics"]["roc_auc"], 4),
                    "training_time_seconds": round(r.get("training_time", 0), 2),
                })
                writer.writerow(row)
    
    print(f"[OUTPUT] CSV saved: {csv_path}")
    
    return json_path, csv_path


def print_best_results(results, n=5):
    """Display top n results."""
    print(f"\n{'='*60}")
    print(f"TOP {min(n, len(results))} RESULTS (by Test Macro-F1)")
    print(f"{'='*60}")
    
    for i, r in enumerate(results[:n], 1):
        print(f"\n#{i}")
        print(f"  Parameters: {r['params']}")
        print(f"  Test:  Acc={r['test_metrics']['accuracy']:.4f}, "
              f"F1={r['test_metrics']['macro_f1']:.4f}, "
              f"AUC={r['test_metrics']['roc_auc']:.4f}")
        print(f"  Training time: {r.get('training_time', 0):.1f}s")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CNN Image Classification")
    parser.add_argument("--dataset", type=str, default="fashionmnist",
                        choices=["fashionmnist", "cifar10"])
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Start total timer
    timer.start("total")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("CNN IMAGE CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("[OPTIMIZATIONS ENABLED] AMP + torch.compile")
    else:
        print("[INFO] Running on CPU")
    
    if args.dry_run:
        print("[DRY-RUN MODE]")
    
    # Run experiments
    if args.grid_search:
        param_grid = {
            "epochs": [30],
            "batch_size": [64, 128],
            "lr": [1e-3, 5e-4],
            "num_filters": [32, 64],
            "dropout": [0.3, 0.5],
            "augment": [True, False],
            "patience": [5],
            "seed": [args.seed],
        }
        
        results = grid_search(args.dataset, param_grid, device, args.dry_run, not args.quiet)
    else:
        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_filters": args.num_filters,
            "dropout": args.dropout,
            "augment": not args.no_augment,
            "patience": 5,
            "seed": args.seed,
        }
        
        print(f"\nTraining with: {params}")
        result = run_experiment(args.dataset, params, device, args.dry_run, not args.quiet)
        results = [result]
    
    # Stop total timer
    total_time = timer.stop("total")
    
    # Save results
    output_dir = Path(args.output_dir)
    json_path, csv_path = save_results(results, args.dataset, args.output_dir, args.dry_run, total_time)
    
    # Generate visualizations
    if not args.no_plots and results:
        best_result = results[0]
        generate_all_visualizations(
            results=results,
            dataset=args.dataset,
            classes=best_result["classes"],
            output_dir=args.output_dir,
            best_y_true=best_result.get("y_true"),
            best_y_pred=best_result.get("y_pred"),
            best_history=best_result.get("history")
        )
    
    # Print summary
    print_best_results(results)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if results:
        best = results[0]
        print(f"Dataset:        {args.dataset}")
        print(f"Best Test Acc:  {best['test_metrics']['accuracy']:.4f}")
        print(f"Best Test F1:   {best['test_metrics']['macro_f1']:.4f}")
        print(f"Best Test AUC:  {best['test_metrics']['roc_auc']:.4f}")
    print(f"Total runtime:  {timer.format(total_time)}")
    print(f"Output dir:     {args.output_dir}")
    print(f"{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
