#!/usr/bin/env python3
"""
Data loading utilities for ML Exercise 3.2

Contains functions to load CIFAR-10 and FashionMNIST datasets
using TensorFlow/Keras and create PyTorch DataLoaders.
"""

import random
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    FASHIONMNIST_CLASSES, CIFAR10_CLASSES,
    FASHIONMNIST_MEAN, FASHIONMNIST_STD,
    CIFAR10_MEAN, CIFAR10_STD
)


# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')



# Load the datasets
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




# Helper function to normalize the data for cnn
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



# Used for CNN
def get_loaders(dataset, batch_size=64, augment=True, val_split=0.1, seed=42,
                dry_run=False, num_workers=4):
    """
    Load dataset using TensorFlow/Keras and create PyTorch DataLoaders.
    
    Data source: TensorFlow/Keras datasets (matching group members' approach)
    Training: PyTorch (for GPU acceleration and model compatibility)
    
    Args:
        dataset: 'fashionmnist' or 'cifar10'
        batch_size: Batch size for DataLoaders
        augment: Whether to apply data augmentation
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        dry_run: If True, use only a small subset of data
        num_workers: Number of workers for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, classes
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





# Used for vit
def get_raw_data(dataset, val_split=0.1, seed=42, dry_run=False):
    """
    Load raw numpy data for use with ViT processor.
    
    Returns data in (N, H, W, C) format with values in [0, 255] range.
    
    Args:
        dataset: 'fashionmnist' or 'cifar10'
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
        dry_run: If True, use only a small subset of data
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), classes
    """
    if dataset == "fashionmnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        classes = FASHIONMNIST_CLASSES
        # Convert grayscale to RGB: (N, 28, 28) -> (N, 28, 28, 3)
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        classes = CIFAR10_CLASSES
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
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
    
    if not dry_run:
        print(f"[DATA] Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), classes
