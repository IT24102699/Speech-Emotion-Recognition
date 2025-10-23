#!/usr/bin/env python3
"""
train_svm.py (robust)
Train an SVM on flattened features (scikit-learn). Includes label-shape checks and debug prints.

Usage:
    python train_svm.py --data ./data/ravdess_features_balanced.pt --out_dir ./models
"""
import os
import argparse
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import torch

SEED = 42

def load_data(path):
    d = torch.load(path, map_location='cpu')
    X = d.get('features', None)
    y = d.get('labels', None)
    meta = d.get('meta', {})
    if X is None or y is None:
        raise RuntimeError("The .pt file does not contain 'features' and 'labels' keys.")
    return X, y, meta

def ensure_1d_int_labels(y_tensor):
    """
    Convert torch tensor labels to a 1D numpy int array.
    Accepts shapes like (N,), (N,1), or torch.LongTensor.
    """
    y_np = y_tensor.numpy()
    # collapse to 1D
    y_np = y_np.reshape(-1)
    # convert to int
    if not np.issubdtype(y_np.dtype, np.integer):
        try:
            y_np = y_np.astype(int)
        except Exception:
            raise ValueError("Labels could not be cast to integer dtype.")
    return y_np

def print_label_distribution(y_np, name="dataset"):
    uniq, counts = np.unique(y_np, return_counts=True)
    print(f"Label distribution ({name}): total={len(y_np)}  classes={len(uniq)}")
    for u,c in zip(uniq, counts):
        print(f"  label {u}: {c}")
    return uniq, counts

def main(args):
    random.seed(SEED); np.random.seed(SEED)
    X, y, meta = load_data(args.data)
    N, T, D = X.shape
    print("Loaded features shape:", X.shape)

    # Convert labels robustly to 1D integer array
    try:
        y_np = ensure_1d_int_labels(y)
    except Exception as e:
        raise RuntimeError(f"Error converting labels to 1D ints: {e}")

    # Print overall distribution
    uniq, counts = print_label_distribution(y_np, name="whole dataset")
    if len(uniq) < 2:
        raise RuntimeError("Dataset has fewer than 2 unique classes. Cannot train a classifier. "
                           "Check your extractor/balancer outputs.")

    # Flatten features
    X_np = X.numpy().reshape(N, -1)
    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.2, stratify=y_np, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    print("\nAfter stratified split:")
    print_label_distribution(y_train, name="train")
    print_label_distribution(y_val, name="val")
    print_label_distribution(y_test, name="test")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training split contains fewer than 2 classes. This prevents SVM training. "
                           "Possible causes:\n"
                           " - The original labels were malformed (e.g., all same label)\n"
                           " - Stratified splitting failed because labels had wrong shape/type\n"
                           "Please run the debug script to inspect the dataset.")

    # Fit SVM
    print("\nTraining SVM (this may take some time depending on data size)...")
    start = time.time()
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', verbose=False)
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f} seconds.")

    # Evaluate on test
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"SVM TEST Acc: {acc:.4f}")
    print("Classification report:\n", classification_report(y_test, preds))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "svm.joblib")
    joblib.dump(clf, out_path)
    print("Saved SVM to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="./models")
    args = p.parse_args()
    main(args)
