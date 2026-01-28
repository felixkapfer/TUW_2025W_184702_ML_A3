

import argparse
import sys
from pathlib import Path

import time
from typing import Tuple, Dict

# Data manipulation and numerical computing
import numpy as np

# Machine learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV

# Image processing and dataset loading
import cv2
import tensorflow as tf
from keras.datasets import cifar10, fashion_mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# BEST PARAMETERS (found from Gridsearchcv runs))
BEST_PARAMS = {
    'fashion': {
        'svm': {'C': 1, 'gamma': 'scale'},
        'random_forest': {'max_depth': 40, 'n_estimators': 200}
    },
    'cifar': {
        'svm': {'C': 1, 'gamma': 'scale'},
        'random_forest': {'max_depth': 20, 'n_estimators': 200}
    }
}


def load_and_preprocess_data(dataset_name: str, max_samples: int = None, 
                              return_full: bool = False) -> Tuple:
    
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} dataset...")
    print(f"{'='*60}")
    
    if dataset_name.lower() == 'cifar':
        (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
    elif dataset_name.lower() == 'fashion':
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'cifar' or 'fashion'")
    
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    
    if dataset_name.lower() == 'fashion':
        print("Converting Fashion-MNIST: 28x28 grayscale → 32x32 RGB")
        X_train_resized = []
        X_test_resized = []
        
        for img in X_train_full:
            resized = cv2.resize(img, (32, 32))
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            X_train_resized.append(rgb_img)
        
        for img in X_test:
            resized = cv2.resize(img, (32, 32))
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            X_test_resized.append(rgb_img)
        
        X_train_full = np.array(X_train_resized)
        X_test = np.array(X_test_resized)
    
    print(f" Full training samples: {X_train_full.shape[0]}")
    print(f" Test samples: {X_test.shape[0]}")
    print(f" Image shape: {X_train_full.shape[1:]}")
    print(f" Number of classes: {len(np.unique(y_train_full))}")
    
    # Create subset if needed
    if max_samples is not None and max_samples < len(X_train_full):
        print(f"✓ Creating subset: {max_samples} samples for hyperparameter tuning")
        indices = np.random.choice(len(X_train_full), max_samples, replace=False)
        X_train_subset = X_train_full[indices]
        y_train_subset = y_train_full[indices]
        
        if return_full:
            return X_train_subset, X_train_full, X_test, y_train_subset, y_train_full, y_test
        else:
            return X_train_subset, X_test, y_train_subset, y_test
    else:
        # No subset needed, use full dataset
        if return_full:
            return X_train_full, X_train_full, X_test, y_train_full, y_train_full, y_test
        else:
            return X_train_full, X_test, y_train_full, y_test


def extract_histogram_features(images: np.ndarray, bins_per_channel: int = 8) -> np.ndarray:
    """Extract 3-channel color histogram features from images."""
    print(f"\n{'='*60}")
    print(f"Extracting 3-channel histogram features...")
    print(f"{'='*60}")
    print(f"Bins per channel: {bins_per_channel}")
    print(f"Total feature dimensions: {bins_per_channel ** 3}")
    
    features = []
    total_images = len(images)
    
    for i, img in enumerate(images):
        hist = cv2.calcHist(
            [img], [0, 1, 2], None,
            [bins_per_channel] * 3,
            [0, 256] * 3
        )
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        features.append(hist)
        
        if (i + 1) % 5000 == 0 or (i + 1) == total_images:
            print(f"  Processed {i + 1}/{total_images} images...", end='\r')
    
    print(f"\n  Feature extraction complete!")
    features = np.array(features)
    print(f" Feature matrix shape: {features.shape}")
    
    return features


def extract_orb_bow_features(train_images: np.ndarray, test_images: np.ndarray, 
                              vocab_size: int = 200, n_features: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ORB + Bag of Visual Words (BoW) features from images.
    
    This method uses:
    1. ORB feature detector to extract keypoints and descriptors
    2. K-Means clustering to build a visual vocabulary
    3. Vector quantization to represent images as histograms of visual words
    """
    print(f"\n{'='*60}")
    print(f"Extracting ORB + Bag of Visual Words features...")
    print(f"{'='*60}")
    print(f"ORB features per image: {n_features}")
    print(f"Visual vocabulary size: {vocab_size}")
    
    # Step A: Feature Detection using ORB
    print(f"\nStep A: Detecting ORB keypoints and descriptors...")
    print("  Upscaling images to 64x64 for better feature detection...")
    
    # ORB works better on grayscale images with sensitive parameters
    orb = cv2.ORB_create(
        nfeatures=1000,       # Increased for more keypoints
        scaleFactor=1.1,      # Finer pyramid scale
        nlevels=8,
        edgeThreshold=2,      # Very sensitive to edges
        patchSize=7,          # Smaller patch for 64x64 images
        fastThreshold=5       # Low threshold for FAST detector
    )
    
    all_train_descriptors = []
    train_image_descriptors = []
    
    for i, img in enumerate(train_images):
        # Upscale to 64x64 for better feature detection
        img_upscaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # Convert to grayscale for ORB (it works better on grayscale)
        gray_img = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)
        
        if descriptors is not None:
            all_train_descriptors.append(descriptors)
            train_image_descriptors.append(descriptors)
        else:
            train_image_descriptors.append(np.zeros((1, 32), dtype=np.float32))
        
        if (i + 1) % 5000 == 0 or (i + 1) == len(train_images):
            print(f"  Train: {i + 1}/{len(train_images)} images...", end='\r')
    
    print(f"\n Extracted descriptors from {len(train_images)} training images")
    print(f"  Images with keypoints: {len(all_train_descriptors)}/{len(train_images)}")
    
    # Check if we have enough descriptors
    if len(all_train_descriptors) == 0:
        print("\n WARNING: No ORB keypoints detected in any images!")
        print("   This can happen with simple/low-texture images like Fashion-MNIST.")
        print("   Returning zero features for ORB+BoW...\n")
        
        # Return zero features
        train_features = np.zeros((len(train_images), vocab_size), dtype=np.float32)
        test_features = np.zeros((len(test_images), vocab_size), dtype=np.float32)
        
        return train_features, test_features
    
    # Step B: Build Visual Vocabulary using K-Means
    print(f"\nStep B: Building visual vocabulary (clustering descriptors from {len(all_train_descriptors)} images)...")
    
    all_descriptors_matrix = np.vstack(all_train_descriptors)
    # Ensure float32 dtype for consistency
    all_descriptors_matrix = all_descriptors_matrix.astype(np.float32)
    total_descriptors = all_descriptors_matrix.shape[0]
    print(f"  Total descriptors: {total_descriptors}")
    
    # Robust fallback: adjust vocab_size based on descriptor count
    if total_descriptors < vocab_size:
        # Use half of total descriptors if we don't have enough
        actual_vocab_size = max(10, total_descriptors // 2)
        print(f" Adjusting vocab_size from {vocab_size} to {actual_vocab_size}")
        print(f"     (Using {total_descriptors} descriptors, clustering into {actual_vocab_size} words)")
    else:
        actual_vocab_size = vocab_size
    
    kmeans = MiniBatchKMeans(
        n_clusters=actual_vocab_size,
        random_state=42,
        batch_size=min(1000, actual_vocab_size),
        verbose=0,
        max_iter=100
    )
    
    kmeans.fit(all_descriptors_matrix)
    print(f" Visual vocabulary created with {actual_vocab_size} visual words")
    
    # Step C: Generate BoW histograms for training images
    print(f"\nStep C: Generating BoW histograms...")
    train_features = []
    
    for i, descriptors in enumerate(train_image_descriptors):
        if descriptors.shape[0] > 0 and descriptors.shape[1] == 32:
            # Convert descriptors to float32 for consistent dtype
            descriptors_float = descriptors.astype(np.float32)
            labels = kmeans.predict(descriptors_float)
            hist, _ = np.histogram(labels, bins=actual_vocab_size, range=(0, actual_vocab_size))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-7)
        else:
            hist = np.zeros(actual_vocab_size, dtype=np.float32)
        
        train_features.append(hist)
        
        if (i + 1) % 5000 == 0 or (i + 1) == len(train_image_descriptors):
            print(f"  Train BoW: {i + 1}/{len(train_image_descriptors)} images...", end='\r')
    
    train_features = np.array(train_features)
    print(f"\n✓ Training features shape: {train_features.shape}")
    
    # Process test images
    print(f"\nProcessing test images...")
    test_features = []
    
    for i, img in enumerate(test_images):
        # Upscale to 64x64 for consistent feature detection
        img_upscaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # Convert to grayscale for ORB
        gray_img = cv2.cvtColor(img_upscaled, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)
        
        if descriptors is not None and descriptors.shape[0] > 0:
            # Convert descriptors to float32 for consistent dtype
            descriptors_float = descriptors.astype(np.float32)
            labels = kmeans.predict(descriptors_float)
            hist, _ = np.histogram(labels, bins=actual_vocab_size, range=(0, actual_vocab_size))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-7)
        else:
            hist = np.zeros(actual_vocab_size, dtype=np.float32)
        
        test_features.append(hist)
        
        if (i + 1) % 2000 == 0 or (i + 1) == len(test_images):
            print(f"  Test BoW: {i + 1}/{len(test_images)} images...", end='\r')
    
    test_features = np.array(test_features)
    print(f"\n Test features shape: {test_features.shape}")
    print(f"ORB+BoW feature extraction complete!")
    
    return train_features, test_features


def train_svm(X_train: np.ndarray, y_train: np.ndarray, 
              best_params: Dict = None, run_gridsearch: bool = True) -> Tuple[SVC, Dict, float]:
    """
    Train Support Vector Machine with RBF kernel.
    
    Args:
        X_train: Training features
        y_train: Training labels
        best_params: If provided, use these parameters directly (skip GridSearchCV)
        run_gridsearch: If False and best_params is None, raise error
    
    Returns:
        Trained model, parameters used, training time
    """
    print(f"\n{'='*60}")
    
    if best_params is not None:
        # Use provided best parameters
        print(f"Training SVM with predefined parameters (RBF kernel)...")
        print(f"{'='*60}")
        print(f"Using parameters: {best_params}")
        
        start_time = time.time()
        model = SVC(kernel='rbf', probability=False, random_state=42, verbose=False, 
                    cache_size=1000, max_iter=-1, **best_params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"✓ SVM training completed in {training_time:.2f} seconds")
        return model, best_params, training_time
    
    elif run_gridsearch:
        # Run GridSearchCV
        print("Training SVM with GridSearchCV (RBF kernel)...")
        print(f"{'='*60}")
        
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.1, 0.01]
        }
        
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: 3")
        
        start_time = time.time()
        svm_base = SVC(kernel='rbf', probability=False, random_state=42, verbose=False, 
                       cache_size=1000, max_iter=-1)
        grid_search = GridSearchCV(
            estimator=svm_base,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f" SVM GridSearch completed in {training_time:.2f} seconds")
        print(f" Best parameters: {grid_search.best_params_}")
        print(f" Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, training_time
    
    else:
        raise ValueError("Either best_params must be provided or run_gridsearch must be True")


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        best_params: Dict = None, run_gridsearch: bool = True) -> Tuple[RandomForestClassifier, Dict, float]:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        best_params: If provided, use these parameters directly (skip GridSearchCV)
        run_gridsearch: If False and best_params is None, raise error
    
    Returns:
        Trained model, parameters used, training time
    """
    print(f"\n{'='*60}")
    
    if best_params is not None:
        # Use provided best parameters
        print(f"Training Random Forest with predefined parameters...")
        print(f"{'='*60}")
        print(f"Using parameters: {best_params}")
        
        start_time = time.time()
        model = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0, **best_params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f" Random Forest training completed in {training_time:.2f} seconds")
        return model, best_params, training_time
    
    elif run_gridsearch:
        # Run GridSearchCV
        print("Training Random Forest with GridSearchCV...")
        print(f"{'='*60}")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 40, None]
        }
        
        print(f"Parameter grid: {param_grid}")
        print(f"Cross-validation folds: 3")
        
        start_time = time.time()
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f" Random Forest GridSearch completed in {training_time:.2f} seconds")
        print(f" Best parameters: {grid_search.best_params_}")
        print(f" Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, training_time
    
    else:
        raise ValueError("Either best_params must be provided or run_gridsearch must be True")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   model_name: str, num_classes: int = 10, best_params: Dict = None,
                   training_time: float = None) -> Dict:
    """Evaluate model and calculate all required metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    
    # Get scores for AUC-ROC calculation
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, 'decision_function'):
        # SVM without probability=True uses decision_function
        y_scores = model.decision_function(X_test)
        # For binary classification, decision_function returns shape (n_samples,)
        # For multiclass, it returns (n_samples, n_classes)
        if len(y_scores.shape) == 1:
            # Binary case - not applicable for our datasets
            raise ValueError(f"{model_name}: Binary classification not supported")
        y_proba = y_scores  # Use decision scores directly for AUC
    else:
        raise ValueError(f"{model_name} does not support probability or decision function predictions")
    
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    y_test_binarized = label_binarize(y_test, classes=range(num_classes))
    auc_roc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
    
    # Compute confusion matrix and per-class metrics
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_f1 = f1_score(y_test, y_pred, average=None)
    
    print(f"\n{'─'*60}")
    print(f"{'METRIC':<30} {'VALUE':>30}")
    print(f"{'─'*60}")
    if best_params:
        print(f"{'Best Parameters:':<30} {str(best_params):>30}")
    if training_time is not None:
        print(f"{'Training Time (seconds):':<30} {training_time:>29.2f}")
    print(f"{'Overall Accuracy:':<30} {accuracy:>29.4f}")
    print(f"{'Macro-Averaged F1-Score:':<30} {f1_macro:>29.4f}")
    print(f"{'AUC-ROC (One-vs-Rest):':<30} {auc_roc:>29.4f}")
    print(f"{'Inference Time (seconds):':<30} {inference_time:>29.2f}")
    print(f"{'─'*60}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auc_roc': auc_roc,
        'inference_time': inference_time,
        'training_time': training_time,
        'best_params': best_params,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'per_class_f1': per_class_f1.tolist()
    }


def save_confusion_matrices_to_file(results: Dict, dataset_name: str):
    """
    Save confusion matrices and per-class metrics to a text file in outputs folder.
    
    Args:
        results: Dictionary containing experiment results with confusion matrices
        dataset_name: Name of the dataset ('fashion' or 'cifar')
    """
    import json
    from pathlib import Path
    
    # Create outputs directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Define class names
    if dataset_name.lower() == 'fashion':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name.lower() == 'cifar':
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                      'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    else:
        class_names = [f'Class {i}' for i in range(10)]
    
    # Prepare output file
    output_file = output_dir / f"confusion_matrices_{dataset_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CONFUSION MATRICES AND PER-CLASS ANALYSIS: {dataset_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # Process each experiment
        experiments = {
            'Histogram Features': results['experiment_1_histogram'],
            'ORB+BoW Features': results['experiment_2_orb_bow']
        }
        
        for exp_name, exp_data in experiments.items():
            f.write(f"\n{'#'*80}\n")
            f.write(f"# {exp_name}\n")
            f.write(f"{'#'*80}\n\n")
            
            # Process each classifier
            classifiers = {
                'SVM': exp_data['svm'],
                'Random Forest': exp_data['random_forest']
            }
            
            for clf_name, clf_data in classifiers.items():
                f.write(f"\n{'-'*80}\n")
                f.write(f"{clf_name} ({exp_name})\n")
                f.write(f"{'-'*80}\n\n")
                
                # Overall metrics
                f.write(f"Overall Accuracy: {clf_data['accuracy']:.4f}\n")
                f.write(f"Macro F1-Score:   {clf_data['f1_macro']:.4f}\n")
                f.write(f"AUC-ROC:          {clf_data['auc_roc']:.4f}\n\n")
                
                # Confusion Matrix
                cm = np.array(clf_data['confusion_matrix'])
                f.write("Confusion Matrix (rows=true, cols=predicted):\n")
                f.write("    ")
                for name in class_names:
                    f.write(f"{name[:8]:>8} ")
                f.write("\n")
                
                for i, true_class in enumerate(class_names):
                    f.write(f"{true_class[:8]:>8} ")
                    for j in range(len(class_names)):
                        f.write(f"{cm[i, j]:>8} ")
                    f.write("\n")
                
                f.write("\n")
                
                # Per-class metrics
                f.write("Per-Class Metrics:\n")
                f.write(f"{'Class':<15} {'Accuracy':>10} {'F1-Score':>10} {'Support':>10}\n")
                f.write("-" * 47 + "\n")
                
                per_class_acc = clf_data['per_class_accuracy']
                per_class_f1 = clf_data['per_class_f1']
                
                for i, class_name in enumerate(class_names):
                    support = int(cm[i, :].sum())
                    f.write(f"{class_name:<15} {per_class_acc[i]:>10.4f} {per_class_f1[i]:>10.4f} {support:>10}\n")
                
                f.write("\n")
        
        # Summary comparison
        f.write(f"\n{'='*80}\n")
        f.write("PER-CLASS ACCURACY COMPARISON\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Class':<15} {'Hist-SVM':>12} {'Hist-RF':>12} {'BoW-SVM':>12} {'BoW-RF':>12} {'Best':>15}\n")
        f.write("-" * 80 + "\n")
        
        methods = [
            results['experiment_1_histogram']['svm'],
            results['experiment_1_histogram']['random_forest'],
            results['experiment_2_orb_bow']['svm'],
            results['experiment_2_orb_bow']['random_forest']
        ]
        method_names = ['Hist-SVM', 'Hist-RF', 'BoW-SVM', 'BoW-RF']
        
        for i, class_name in enumerate(class_names):
            accs = [m['per_class_accuracy'][i] for m in methods]
            best_idx = np.argmax(accs)
            best_method = method_names[best_idx]
            
            f.write(f"{class_name:<15} ")
            for acc in accs:
                f.write(f"{acc:>12.4f} ")
            f.write(f"{best_method:>15}\n")
        
        f.write("\n")
    
    print(f"✓ Confusion matrices saved to {output_file}")
    
    # Also save as JSON for programmatic access
    json_file = output_dir / f"confusion_matrices_{dataset_name}.json"
    cm_data = {
        'dataset': dataset_name,
        'class_names': class_names,
        'experiments': {}
    }
    
    for exp_name, exp_data in experiments.items():
        cm_data['experiments'][exp_name] = {
            'svm': {
                'confusion_matrix': exp_data['svm']['confusion_matrix'],
                'per_class_accuracy': exp_data['svm']['per_class_accuracy'],
                'per_class_f1': exp_data['svm']['per_class_f1'],
                'overall_accuracy': exp_data['svm']['accuracy']
            },
            'random_forest': {
                'confusion_matrix': exp_data['random_forest']['confusion_matrix'],
                'per_class_accuracy': exp_data['random_forest']['per_class_accuracy'],
                'per_class_f1': exp_data['random_forest']['per_class_f1'],
                'overall_accuracy': exp_data['random_forest']['accuracy']
            }
        }
    
    with open(json_file, 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    print(f"✓ Confusion matrices JSON saved to {json_file}")


def run_shallow_task(dataset_name: str, max_samples: int = None, vocab_size: int = 200,
                     use_best_params: bool = False, manual_svm_params: Dict = None,
                     manual_rf_params: Dict = None) -> Dict:
    """
    Main function to run shallow learning pipeline with TWO feature extraction methods.
    
    This implements a two-stage workflow:
    1. If use_best_params=False and max_samples is set: Tune on subset, then train on full dataset
    2. If use_best_params=True: Use hardcoded best params, train directly on full dataset
    
    Args:
        dataset_name: 'cifar' or 'fashion'
        max_samples: If specified (without use_best_params), use for tuning subset
        vocab_size: Size of visual vocabulary for ORB+BoW (default: 200)
        use_best_params: If True, skip GridSearchCV and use hardcoded best parameters
        manual_svm_params: Manual override for SVM parameters (optional)
        manual_rf_params: Manual override for RF parameters (optional)
    
    Returns:
        Dictionary containing all results for both experiments
    """
    overall_start = time.time()
    
    print("\n" + "="*70)
    print(f"SHALLOW LEARNING PIPELINE: {dataset_name.upper()}")
    print(f"Running TWO experiments with different feature extraction methods")
    print("="*70)
    
    # Determine execution mode
    if use_best_params:
        mode = "Direct Full Train (Optimized)"
        print(f"\n MODE: {mode}")
        print("   Using pre-optimized parameters from comprehensive GridSearchCV")
        print("   Training directly on full 60k dataset")
    elif max_samples is not None:
        mode = "Tuned + Full Train"
        print(f"\n MODE: {mode}")
        print(f"   Stage 1: GridSearchCV on {max_samples} samples")
        print(f"   Stage 2: Train with best params on full 60k dataset")
    else:
        mode = "Standard GridSearchCV (Full Dataset)"
        print(f"\n MODE: {mode}")
        print("   Running GridSearchCV on full dataset")
    
    # Step 1: Load and preprocess data
    if max_samples is not None and not use_best_params:
        # Two-stage: need both subset and full dataset
        X_train_subset, X_train_full, X_test, y_train_subset, y_train_full, y_test = \
            load_and_preprocess_data(dataset_name, max_samples, return_full=True)
        num_classes = len(np.unique(y_train_full))
    else:
        # Single-stage: only need full dataset
        X_train_full, X_test, y_train_full, y_test = \
            load_and_preprocess_data(dataset_name, max_samples=None, return_full=False)
        X_train_subset = None
        y_train_subset = None
        num_classes = len(np.unique(y_train_full))
    
    
    # EXPERIMENT 1: 3-Channel Color Histogram Features

    print("\n" + "="*70)
    print("EXPERIMENT 1: 3-Channel Color Histogram Features")
    print("="*70)
    
    # Determine which parameters to use
    if use_best_params:
        # Use hardcoded best parameters
        svm_params_to_use = manual_svm_params if manual_svm_params else BEST_PARAMS[dataset_name]['svm']
        rf_params_to_use = manual_rf_params if manual_rf_params else BEST_PARAMS[dataset_name]['random_forest']
        print(f"Using best parameters for {dataset_name}:")
        print(f"  SVM: {svm_params_to_use}")
        print(f"  RF: {rf_params_to_use}")
    elif max_samples is not None:
        # Two-stage: will tune on subset, then retrain on full
        svm_params_to_use = None
        rf_params_to_use = None
    else:
        # Standard GridSearchCV on full dataset
        svm_params_to_use = None
        rf_params_to_use = None
    
    # Extract features for full dataset
    feature_start = time.time()
    X_train_full_hist = extract_histogram_features(X_train_full)
    X_test_hist = extract_histogram_features(X_test)
    hist_feature_time = time.time() - feature_start
    print(f"\n✓ Histogram feature extraction time: {hist_feature_time:.2f} seconds")
    
    # Extract features for subset if needed
    if X_train_subset is not None:
        print("\n--- Extracting features for tuning subset ---")
        subset_feature_start = time.time()
        X_train_subset_hist = extract_histogram_features(X_train_subset)
        print(f" Subset feature extraction time: {time.time() - subset_feature_start:.2f} seconds")
    
    # Train models with histogram features
    print("\n--- Training Models with Histogram Features ---")
    
    if X_train_subset is not None and svm_params_to_use is None:
        # Stage 1: Tune on subset
        print("\n[Stage 1: Hyperparameter Tuning on Subset]")
        _, svm_tuned_params, svm_tune_time = train_svm(X_train_subset_hist, y_train_subset, 
                                                        best_params=None, run_gridsearch=True)
        _, rf_tuned_params, rf_tune_time = train_random_forest(X_train_subset_hist, y_train_subset,
                                                                best_params=None, run_gridsearch=True)
        
        # Stage 2: Train on full dataset with tuned params
        print("\n[Stage 2: Training on Full Dataset with Tuned Parameters]")
        svm_hist, svm_hist_params, svm_hist_train_time = train_svm(X_train_full_hist, y_train_full,
                                                                     best_params=svm_tuned_params, run_gridsearch=False)
        rf_hist, rf_hist_params, rf_hist_train_time = train_random_forest(X_train_full_hist, y_train_full,
                                                                            best_params=rf_tuned_params, run_gridsearch=False)
        # Add tuning time to total
        svm_hist_train_time += svm_tune_time
        rf_hist_train_time += rf_tune_time
    else:
        # Single-stage training
        svm_hist, svm_hist_params, svm_hist_train_time = train_svm(X_train_full_hist, y_train_full,
                                                                     best_params=svm_params_to_use,
                                                                     run_gridsearch=(svm_params_to_use is None))
        rf_hist, rf_hist_params, rf_hist_train_time = train_random_forest(X_train_full_hist, y_train_full,
                                                                            best_params=rf_params_to_use,
                                                                            run_gridsearch=(rf_params_to_use is None))
    
    # Evaluate models with histogram features
    svm_hist_results = evaluate_model(svm_hist, X_test_hist, y_test, 
                                       "SVM (Histogram)", num_classes,
                                       best_params=svm_hist_params,
                                       training_time=svm_hist_train_time)
    rf_hist_results = evaluate_model(rf_hist, X_test_hist, y_test, 
                                      "Random Forest (Histogram)", num_classes,
                                      best_params=rf_hist_params,
                                      training_time=rf_hist_train_time)
    
    
    # EXPERIMENT 2: ORB + Bag of Visual Words Features
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: ORB + Bag of Visual Words (BoW) Features")
    print("="*70)
    
    # Extract features for full dataset
    feature_start = time.time()
    X_train_full_bow, X_test_bow = extract_orb_bow_features(X_train_full, X_test, vocab_size=vocab_size)
    bow_feature_time = time.time() - feature_start
    print(f"\n✓ ORB+BoW feature extraction time: {bow_feature_time:.2f} seconds")
    
    # Extract features for subset if needed
    if X_train_subset is not None:
        print("\n--- Extracting ORB+BoW features for tuning subset ---")
        subset_feature_start = time.time()
        X_train_subset_bow, _ = extract_orb_bow_features(X_train_subset, X_test[:100], vocab_size=vocab_size)
        print(f"✓ Subset feature extraction time: {time.time() - subset_feature_start:.2f} seconds")
    
    # Train models with BoW features
    print("\n--- Training Models with ORB+BoW Features ---")
    
    if X_train_subset is not None and svm_params_to_use is None:
        # Stage 1: Tune on subset
        print("\n[Stage 1: Hyperparameter Tuning on Subset]")
        _, svm_tuned_params_bow, svm_tune_time_bow = train_svm(X_train_subset_bow, y_train_subset,
                                                                 best_params=None, run_gridsearch=True)
        _, rf_tuned_params_bow, rf_tune_time_bow = train_random_forest(X_train_subset_bow, y_train_subset,
                                                                         best_params=None, run_gridsearch=True)
        
        # Stage 2: Train on full dataset with tuned params
        print("\n[Stage 2: Training on Full Dataset with Tuned Parameters]")
        svm_bow, svm_bow_params, svm_bow_train_time = train_svm(X_train_full_bow, y_train_full,
                                                                  best_params=svm_tuned_params_bow, run_gridsearch=False)
        rf_bow, rf_bow_params, rf_bow_train_time = train_random_forest(X_train_full_bow, y_train_full,
                                                                         best_params=rf_tuned_params_bow, run_gridsearch=False)
        # Add tuning time to total
        svm_bow_train_time += svm_tune_time_bow
        rf_bow_train_time += rf_tune_time_bow
    else:
        # Single-stage training
        svm_bow, svm_bow_params, svm_bow_train_time = train_svm(X_train_full_bow, y_train_full,
                                                                  best_params=svm_params_to_use,
                                                                  run_gridsearch=(svm_params_to_use is None))
        rf_bow, rf_bow_params, rf_bow_train_time = train_random_forest(X_train_full_bow, y_train_full,
                                                                         best_params=rf_params_to_use,
                                                                         run_gridsearch=(rf_params_to_use is None))
    
    # Evaluate models with BoW features
    svm_bow_results = evaluate_model(svm_bow, X_test_bow, y_test, 
                                      "SVM (ORB+BoW)", num_classes,
                                      best_params=svm_bow_params,
                                      training_time=svm_bow_train_time)
    rf_bow_results = evaluate_model(rf_bow, X_test_bow, y_test, 
                                     "Random Forest (ORB+BoW)", num_classes,
                                     best_params=rf_bow_params,
                                     training_time=rf_bow_train_time)
    
    # Calculate total runtime
    total_runtime = time.time() - overall_start
    
    
    # FINAL COMPARISON SUMMARY
   
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Mode: {mode}")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    
    # Create comparison table
    print(f"\n{'─'*70}")
    print(f"{'Method':<25} {'Model':<15} {'Acc':<8} {'F1':<8} {'AUC':<8}")
    print(f"{'─'*70}")
    
    # Histogram results
    print(f"{'3-Ch Histogram':<25} {'SVM':<15} {svm_hist_results['accuracy']:<8.4f} "
          f"{svm_hist_results['f1_macro']:<8.4f} {svm_hist_results['auc_roc']:<8.4f}")
    print(f"{'3-Ch Histogram':<25} {'Random Forest':<15} {rf_hist_results['accuracy']:<8.4f} "
          f"{rf_hist_results['f1_macro']:<8.4f} {rf_hist_results['auc_roc']:<8.4f}")
    
    # BoW results
    print(f"{'ORB + BoW':<25} {'SVM':<15} {svm_bow_results['accuracy']:<8.4f} "
          f"{svm_bow_results['f1_macro']:<8.4f} {svm_bow_results['auc_roc']:<8.4f}")
    print(f"{'ORB + BoW':<25} {'Random Forest':<15} {rf_bow_results['accuracy']:<8.4f} "
          f"{rf_bow_results['f1_macro']:<8.4f} {rf_bow_results['auc_roc']:<8.4f}")
    
    print(f"{'─'*70}")
    
    # Feature extraction time comparison
    print(f"\nFeature Extraction Time Comparison:")
    print(f"  3-Channel Histogram: {hist_feature_time:.2f} seconds")
    print(f"  ORB + BoW:          {bow_feature_time:.2f} seconds")
    print("="*70 + "\n")
    
    # Prepare final results dictionary
    results = {
        'dataset': dataset_name,
        'mode': mode,
        'total_runtime': total_runtime,
        'experiment_1_histogram': {
            'feature_extraction_time': hist_feature_time,
            'svm': svm_hist_results,
            'random_forest': rf_hist_results
        },
        'experiment_2_orb_bow': {
            'feature_extraction_time': bow_feature_time,
            'vocab_size': vocab_size,
            'svm': svm_bow_results,
            'random_forest': rf_bow_results
        }
    }
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Shallow Learning Image Classification using Traditional ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use pre-optimized parameters (fastest, recommended for production)
  python main.py --dataset fashion --best
  
  # Tune on 10k samples, then train on full 60k dataset
  python main.py --dataset cifar --max-samples 10000
  
  # Use best params with manual override
  python main.py --dataset fashion --best --svm-c 10
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cifar', 'fashion'],
        help='Dataset to use: cifar or fashion'
    )
    
    parser.add_argument(
        '--best',
        action='store_true',
        help='Use pre-optimized best parameters (skip GridSearchCV, train on full dataset)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='If --best not used: tune on this many samples, then train on full dataset. Default: use all for tuning'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=200,
        help='Size of visual vocabulary for ORB+BoW features. Default: 200'
    )
    
    parser.add_argument(
        '--svm-c',
        type=float,
        default=None,
        help='Manual override: SVM C parameter'
    )
    
    parser.add_argument(
        '--svm-gamma',
        type=str,
        default=None,
        help='Manual override: SVM gamma parameter (e.g., "scale", 0.1, 0.01)'
    )
    
    parser.add_argument(
        '--rf-n-estimators',
        type=int,
        default=None,
        help='Manual override: Random Forest n_estimators'
    )
    
    parser.add_argument(
        '--rf-max-depth',
        type=int,
        default=None,
        help='Manual override: Random Forest max_depth (use 0 for None)'
    )
    
    args = parser.parse_args()
    
    # Build manual parameter overrides if provided
    manual_svm_params = None
    manual_rf_params = None
    
    if args.svm_c is not None or args.svm_gamma is not None:
        manual_svm_params = {}
        if args.svm_c is not None:
            manual_svm_params['C'] = args.svm_c
        if args.svm_gamma is not None:
            # Try to convert to float, otherwise keep as string (e.g., 'scale')
            try:
                manual_svm_params['gamma'] = float(args.svm_gamma)
            except ValueError:
                manual_svm_params['gamma'] = args.svm_gamma
    
    if args.rf_n_estimators is not None or args.rf_max_depth is not None:
        manual_rf_params = {}
        if args.rf_n_estimators is not None:
            manual_rf_params['n_estimators'] = args.rf_n_estimators
        if args.rf_max_depth is not None:
            manual_rf_params['max_depth'] = None if args.rf_max_depth == 0 else args.rf_max_depth
    
    try:
        # Run the shallow learning task with both feature extraction methods
        results = run_shallow_task(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            vocab_size=args.vocab_size,
            use_best_params=args.best,
            manual_svm_params=manual_svm_params,
            manual_rf_params=manual_rf_params
        )
        
        # Optional: Save results to file
        import json
        mode_suffix = "_best" if args.best else ("_tuned" if args.max_samples else "_gridsearch")
        output_file = f"results_{args.dataset}{mode_suffix}.json"
        with open(output_file, 'w') as f:
            json.dump(results, indent=2, fp=f, default=str)
        print(f" Results saved to {output_file}")
        
        # Save confusion matrices to separate file in outputs folder
        save_confusion_matrices_to_file(results, args.dataset)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
