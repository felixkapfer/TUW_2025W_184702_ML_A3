#!/usr/bin/env python3
"""
Visualization functions for ML Exercise 3.2

Contains functions to create:
- Confusion matrices
- Training history plots
- Metrics comparison plots
- Runtime comparison plots
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes, save_path, title="Confusion Matrix"):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save the plot
        title: Plot title
    """
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
    """
    Plot training history (loss and metrics).
    
    Args:
        history: List of dictionaries with training history
        save_path: Path to save the plot
    """
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
    """
    Plot metrics comparison across different configurations.

    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
    """
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
        values = [r["test_metrics"][metric] for r in results_subset]

        finite = [v for v in values if np.isfinite(v)]
        if len(finite) == 0:
            y_min, y_max = 0.0, 1.0
        else:
            y_min, y_max = min(finite) * 0.95, max(finite) * 1.02

        bars = axes[idx].bar(labels, values, color=color, edgecolor='black', alpha=0.8)
        axes[idx].set_xlabel('Configuration Rank')
        axes[idx].set_ylabel(title)
        axes[idx].set_title(f'Test {title} Comparison')
        axes[idx].set_ylim(y_min, y_max)
        axes[idx].grid(True, axis='y', alpha=0.3)

        # Add value labels on bars (skip NaN/Inf)
        for bar, val in zip(bars, values):
            if not np.isfinite(val):
                continue
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Metrics comparison saved: {save_path}")





def plot_runtime_comparison(results, save_path):
    """
    Plot runtime comparison across configurations.
    
    Args:
        results: List of result dictionaries
        save_path: Path to save the plot
    """
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





def generate_all_visualizations(results, dataset, classes, output_dir, model_name="cnn",
                                 best_y_true=None, best_y_pred=None, best_history=None):
    """
    Generate all visualization plots.
    
    Args:
        results: List of result dictionaries
        dataset: Dataset name ('fashionmnist' or 'cifar10')
        classes: List of class names
        output_dir: Directory to save plots
        model_name: Model name for file naming (e.g., 'cnn', 'vit')
        best_y_true: True labels from best model
        best_y_pred: Predicted labels from best model
        best_history: Training history from best model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix for best model
    if best_y_true is not None and best_y_pred is not None:
        plot_confusion_matrix(
            best_y_true, best_y_pred, classes,
            output_dir / f"{model_name}_confusion_matrix_{dataset}.png",
            title=f"Confusion Matrix - {dataset.upper()} ({model_name.upper()} Best Model)"
        )
    
    # Training history for best model
    if best_history is not None:
        plot_training_history(
            best_history,
            output_dir / f"{model_name}_training_history_{dataset}.png"
        )
    
    # Metrics comparison (if grid search)
    if len(results) > 1:
        plot_metrics_comparison(
            results,
            output_dir / f"{model_name}_metrics_comparison_{dataset}.png"
        )
        
        plot_runtime_comparison(
            results,
            output_dir / f"{model_name}_runtime_comparison_{dataset}.png"
        )
