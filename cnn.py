#!/usr/bin/env python3
"""
CNN Image Classification with Grid Search

Datasets: FashionMNIST, CIFAR-10 (loaded via TensorFlow/Keras)
Metrics: Accuracy, Macro-F1, ROC-AUC

Usage:
    python cnn.py --dataset fashionmnist
    python cnn.py --dataset cifar10 --grid-search
    python cnn.py --dataset cifar10 --dry-run
"""


# Import general libaries
import csv
import json
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


# Import own classes
from utils import timer, compute_metrics, set_seed
from data import get_loaders
from plots import generate_all_visualizations


matplotlib.use('Agg')  # Non-interactive backend

# Remove warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
# EXPERIMENT RUNNER
# ============================================================================

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
        "model": "CNN",
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
            model_name="cnn",
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
