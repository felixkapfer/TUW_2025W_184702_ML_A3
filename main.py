# !/usr/bin/env python3


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
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score




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
#   - CIFAR-10 dataset: Krizhevsky et al., 2009
#   - PyTorch practice, e.g.:
#     https://pytorch.org/vision/stable/transforms.html
#     https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
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
# Load the data
# ============================================================================
def get_loaders(dataset, batch_size=64, augment=True, val_split=0.1, seed=42,
                dry_run=False, num_workers=4):
    """
    Data loader utility for CIFAR-10 / FashionMNIST.

    This function is based on common PyTorch training pipelines as used in
    tutorials and example repositories (e.g. torchvision examples).
    The structure was adapted and extended for this project
    (dry-run mode, validation split, augmentation control).

    Original inspiration:
      - https://github.com/pytorch/examples
      - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
      - https://pytorch.org/vision/stable/transforms.html
      - https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch

    Modifications by the author:
      - Added dry-run mode
      - Explicit train/val split with fixed seed
      - Dataset-specific transforms and normalization
      - Unified interface for CIFAR-10 and FashionMNIST
    """
    
    # if dataset == "fashionmnist":
    #     mean, std, size = (0.2860,), (0.3530,), 28
    #     Dataset = datasets.FashionMNIST
    #     classes = FASHIONMNIST_CLASSES
    # else:
    #     mean, std, size = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32
    #     Dataset = datasets.CIFAR10
    #     classes = CIFAR10_CLASSES


    if dataset == "fashionmnist":
        mean, std, size = FASHIONMNIST_MEAN, FASHIONMNIST_STD, 28
        Dataset = datasets.FashionMNIST
        classes = FASHIONMNIST_CLASSES
    elif dataset == "cifar10":
        mean, std, size = CIFAR10_MEAN, CIFAR10_STD, 32
        Dataset = datasets.CIFAR10
        classes = CIFAR10_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    
    # Transforms
    if augment:
        train_tf = T.Compose([
            T.RandomCrop(size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    else:
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    
    # Load datasets
    train_full = Dataset("./data", train=True, download=True, transform=train_tf)
    test_data = Dataset("./data", train=False, download=True, transform=test_tf)
    
    # Dry-run mode
    if dry_run:
        n = 500
        gen = torch.Generator().manual_seed(seed)
        train_idx = torch.randperm(len(train_full), generator=gen)[:n].tolist()
        test_idx = torch.randperm(len(test_data), generator=gen)[:100].tolist()
        
        train_full = Subset(train_full, train_idx)
        test_data = Subset(test_data, test_idx)
        
        n_val = 50
        train_data = Subset(train_full, list(range(n - n_val)))
        val_full = Dataset("./data", train=True, download=False, transform=test_tf)
        val_data = Subset(val_full, train_idx[n - n_val:n])
        print(f"[DRY-RUN] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    else:
        n_val = int(len(train_full) * val_split)
        n_train = len(train_full) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_idx, val_idx = random_split(range(len(train_full)), [n_train, n_val], generator=gen)
        
        train_data = Subset(train_full, train_idx.indices)
        val_full = Dataset("./data", train=True, download=False, transform=test_tf)
        val_data = Subset(val_full, val_idx.indices)
    
    # Optimized DataLoader settings
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": num_workers if use_cuda else 0,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0 and use_cuda,
    }
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, classes




# ============================================================================
# MODEL
# ============================================================================

class CNN(torch.nn.Module):
    """Optimized CNN."""
    
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
    
    # torch.compile for PyTorch 2.0+ (significant speedup)
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] Model compiled with torch.compile()")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    
    return model





# ============================================================================
# TRAINING (with AMP)
# ============================================================================

def train_epoch_amp(model, loader, optimizer, scaler, device):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0
    
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
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
    """Evaluate model."""
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
    return metrics





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
    
    iterator = range(1, epochs + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Training")
    
    for epoch in iterator:
        if use_amp:
            train_loss = train_epoch_amp(model, train_loader, optimizer, scaler, device)
        else:
            train_loss = train_epoch_cpu(model, train_loader, optimizer, device)
        
        val_metrics = evaluate(model, val_loader, device, use_amp)
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
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics, history


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for consistent input sizes


def run_experiment(dataset, params, device, dry_run=False, verbose=True):
    """Run single experiment."""
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
    
    best_val_metrics, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=params["lr"],
        device=device,
        patience=params.get("patience", 5),
        verbose=verbose
    )
    
    test_metrics = evaluate(model, test_loader, device, use_amp=device.type == "cuda")
    
    return {
        "params": params,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history
    }


def grid_search(dataset, param_grid, device, dry_run=False, verbose=True):
    """Run grid search."""
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
                  f"AUC: {result['test_metrics']['roc_auc']:.4f}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    results.sort(key=lambda x: x["test_metrics"]["macro_f1"], reverse=True)
    return results


def save_results(results, dataset, output_dir="./outputs", dry_run=False):
    """Save results as JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_dryrun" if dry_run else "_fast"
    
    best = results[0] if results else None
    
    json_data = {
        "dataset": dataset,
        "timestamp": timestamp,
        "dry_run": dry_run,
        "optimized": True,
        "num_experiments": len(results),
        "best_result": {
            "parameters": best["params"] if best else None,
            "test_metrics": best["test_metrics"] if best else None,
            "val_metrics": best["val_metrics"] if best else None,
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
            }
            for i, r in enumerate(results)
        ]
    }
    
    json_path = output_dir / f"results_{dataset}_{timestamp}{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON saved: {json_path}")
    
    csv_path = output_dir / f"results_{dataset}_{timestamp}{suffix}.csv"
    
    if results:
        param_keys = list(results[0]["params"].keys())
        fieldnames = ["rank"] + param_keys + [
            "test_accuracy", "test_macro_f1", "test_roc_auc",
            "val_accuracy", "val_macro_f1", "val_roc_auc"
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
                })
                writer.writerow(row)
    
    print(f"CSV saved: {csv_path}")
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


def main():
    parser = argparse.ArgumentParser(description="OPTIMIZED CNN Classification")
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
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"[OPTIMIZATIONS ENABLED] AMP + torch.compile")
    else:
        print("[WARNING] Running on CPU - for best performance use GPU")
    
    if args.dry_run:
        print("[DRY-RUN MODE]")
    
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
        save_results(results, args.dataset, args.output_dir, args.dry_run)
        print_best_results(results)
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
        save_results(results, args.dataset, args.output_dir, args.dry_run)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Test Accuracy:  {result['test_metrics']['accuracy']:.4f}")
        print(f"Test Macro-F1:  {result['test_metrics']['macro_f1']:.4f}")
        print(f"Test ROC-AUC:   {result['test_metrics']['roc_auc']:.4f}")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()