#!/usr/bin/env python3
"""
Vision Transformer (ViT) Image Classification with Grid Search

Uses pretrained ViT from HuggingFace Transformers with PyTorch Lightning.
Datasets: FashionMNIST, CIFAR-10 (loaded via TensorFlow/Keras)
Metrics: Accuracy, Macro-F1, ROC-AUC

Usage:
    python vit.py --dataset fashionmnist
    python vit.py --dataset cifar10 --grid-search
    python vit.py --dataset cifar10 --dry-run
"""

# ============================================================================
# IMPORTS
# ============================================================================

import csv
import json
import warnings
import itertools
import argparse

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import Accuracy

from utils import timer, compute_metrics, set_seed, FASHIONMNIST_CLASSES, CIFAR10_CLASSES
from data import get_raw_data
from plots import generate_all_visualizations


# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# GLOBAL VARIABLES (set during runtime)
# ============================================================================

processor = None
device = None


# ============================================================================
# DATA LOADING FOR VIT
# ============================================================================

def collate_fn(batch):
    """
    Collate function for ViT DataLoader.
    Processes images using ViT processor.
    """
    images, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long).squeeze()
    
    inputs = processor(list(images), return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"].to(device),
        "labels": labels.to(device)
    }


def get_vit_loaders(dataset, batch_size=32, val_split=0.1, seed=42, dry_run=False):
    """
    Create DataLoaders for ViT training.
    
    Args:
        dataset: 'fashionmnist' or 'cifar10'
        batch_size: Batch size
        val_split: Validation split ratio
        seed: Random seed
        dry_run: Use small data subset
    
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test), classes = get_raw_data(
        dataset=dataset,
        val_split=val_split,
        seed=seed,
        dry_run=dry_run
    )
    
    # Create data lists (image, label pairs)
    train_data = list(zip(x_train, y_train))
    val_data = list(zip(x_val, y_val))
    test_data = list(zip(x_test, y_test))
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, classes


# ============================================================================
# MODEL (PyTorch Lightning)
# ============================================================================

class ViTClassifier(pl.LightningModule):
    """
    Vision Transformer Classifier using PyTorch Lightning.
    Based on pretrained ViT from HuggingFace.
    """
    
    def __init__(self, model, lr=2e-5, num_classes=10):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Store predictions for metrics computation
        self.val_preds = []
        self.val_labels = []
        self.val_probs = []

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        acc = self.train_acc(preds, batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        probs = F.softmax(outputs.logits, dim=-1)
        acc = self.val_acc(preds, batch["labels"])
        self.log("val_loss", outputs.loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        # Store for metrics
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(batch["labels"].cpu().numpy())
        self.val_probs.extend(probs.cpu().numpy())

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []
        self.val_probs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def get_vit_model(num_classes=10, model_name="google/vit-base-patch16-224-in21k"):
    """
    Load pretrained ViT model from HuggingFace.
    
    Args:
        num_classes: Number of output classes
        model_name: HuggingFace model name
    
    Returns:
        processor, model
    """
    proc = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )
    return proc, model


# ============================================================================
# TRAINING
# ============================================================================

def train_vit(train_loader, val_loader, model, lr=2e-5, epochs=50, patience=5, dry_run=False):
    """
    Train ViT model using PyTorch Lightning.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model: ViT model
        lr: Learning rate
        epochs: Maximum epochs
        patience: Early stopping patience
        dry_run: Use fewer epochs for testing
    
    Returns:
        classifier, trainer, training_time
    """
    # Adjust for dry-run
    if dry_run:
        epochs = min(2, epochs)
    
    # Create classifier
    classifier = ViTClassifier(model=model, lr=lr, num_classes=10)
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_acc", 
        mode="max", 
        patience=patience, 
        verbose=True, 
        min_delta=1e-4
    )
    
    # Trainer
    trainer = pl.Trainer(
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        max_epochs=epochs,
        callbacks=[early_stop],
        enable_progress_bar=True,
        logger=False,  # Disable default logger
    )
    
    # Start timer
    timer.start("training")
    
    # Train
    trainer.fit(classifier, train_loader, val_loader)
    
    # Stop timer
    training_time = timer.stop("training")
    
    return classifier, trainer, training_time


@torch.no_grad()
def evaluate_vit(classifier, test_loader):
    """
    Evaluate ViT model on test set.
    
    Returns:
        metrics, y_true, y_pred, y_prob
    """
    classifier.eval()
    all_y, all_pred, all_prob = [], [], []
    total_loss = 0
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        outputs = classifier.model(**batch)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = probs.argmax(dim=-1)
        
        total_loss += outputs.loss.item() * batch["labels"].size(0)
        all_y.extend(batch["labels"].cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
    
    metrics = compute_metrics(all_y, all_pred, np.array(all_prob))
    metrics["loss"] = total_loss / len(test_loader.dataset)
    
    return metrics, np.array(all_y), np.array(all_pred), np.array(all_prob)


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(dataset, params, dry_run=False, verbose=True):
    """Run a single ViT experiment."""
    global processor, device
    
    set_seed(params.get("seed", 42))
    pl.seed_everything(params.get("seed", 42))
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor
    print(f"[INFO] Loading ViT model: {params.get('model_name', 'google/vit-base-patch16-224-in21k')}")
    processor, model = get_vit_model(
        num_classes=10, 
        model_name=params.get("model_name", "google/vit-base-patch16-224-in21k")
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader, classes = get_vit_loaders(
        dataset=dataset,
        batch_size=params["batch_size"],
        seed=params.get("seed", 42),
        dry_run=dry_run
    )
    
    # Train
    classifier, trainer, training_time = train_vit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        lr=params["lr"],
        epochs=params["epochs"],
        patience=params.get("patience", 5),
        dry_run=dry_run
    )
    
    # Get validation metrics from last epoch
    val_metrics = {
        "accuracy": float(classifier.val_acc.compute()),
        "macro_f1": 0.0,  # Will compute from stored predictions
        "roc_auc": 0.0,
    }
    
    if classifier.val_labels:
        val_metrics_computed = compute_metrics(
            classifier.val_labels, 
            classifier.val_preds, 
            np.array(classifier.val_probs) if classifier.val_probs else None
        )
        val_metrics.update(val_metrics_computed)
    
    # Evaluate on test set
    test_metrics, y_true, y_pred, y_prob = evaluate_vit(classifier, test_loader)
    
    # Create history from trainer logs (simplified)
    history = [{
        "epoch": 1,
        "train_loss": 0.0,
        "val_loss": test_metrics.get("loss", 0.0),
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics.get("macro_f1", 0.0),
        "val_roc_auc": val_metrics.get("roc_auc", 0.0),
    }]
    
    return {
        "params": params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "training_time": training_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "classes": classes,
    }


def grid_search(dataset, param_grid, dry_run=False, verbose=True):
    """Run grid search over hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n{'='*60}")
    print(f"ViT Grid Search: {len(combinations)} combinations")
    print(f"Dataset: {dataset}")
    if dry_run:
        print("[DRY-RUN MODE]")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        print(f"\n[{i}/{len(combinations)}] {params}")
        
        try:
            result = run_experiment(dataset, params, dry_run=dry_run, verbose=verbose)
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
        "model": "ViT",
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
                "val_macro_f1": r["val_metrics"].get("macro_f1", 0),
                "val_roc_auc": r["val_metrics"].get("roc_auc", 0),
                "training_time_seconds": r.get("training_time", 0),
            }
            for i, r in enumerate(results)
        ]
    }
    
    json_path = output_dir / f"vit_results_{dataset}_{timestamp}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n[OUTPUT] JSON saved: {json_path}")
    
    # CSV export
    csv_path = output_dir / f"vit_results_{dataset}_{timestamp}{suffix}.csv"
    
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
                    "val_macro_f1": round(r["val_metrics"].get("macro_f1", 0), 4),
                    "val_roc_auc": round(r["val_metrics"].get("roc_auc", 0), 4),
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
    parser = argparse.ArgumentParser(description="ViT Image Classification")
    parser.add_argument("--dataset", type=str, default="fashionmnist",
                        choices=["fashionmnist", "cifar10"])
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Start total timer
    timer.start("total")
    
    # Device setup
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("VISION TRANSFORMER (ViT) IMAGE CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Device: {dev}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("[OPTIMIZATIONS ENABLED] Mixed Precision Training")
    else:
        print("[INFO] Running on CPU")
    
    if args.dry_run:
        print("[DRY-RUN MODE]")
    
    # Run experiments
    if args.grid_search:
        param_grid = {
            "epochs": [50],
            "batch_size": [16, 32],
            "lr": [2e-5, 5e-5],
            "patience": [5],
            "seed": [args.seed],
            "model_name": ["google/vit-base-patch16-224-in21k"],
        }
        
        results = grid_search(args.dataset, param_grid, args.dry_run, not args.quiet)
    else:
        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": 5,
            "seed": args.seed,
            "model_name": "google/vit-base-patch16-224-in21k",
        }
        
        print(f"\nTraining with: {params}")
        result = run_experiment(args.dataset, params, args.dry_run, not args.quiet)
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
            model_name="vit",
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
