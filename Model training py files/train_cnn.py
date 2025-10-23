#!/usr/bin/env python3
"""
train_cnn.py
Train a CNN classifier on preprocessed RAVDESS features saved as a .pt

Usage:
    python train_cnn.py --data ./data/ravdess_features_balanced.pt --out_dir ./models --epochs 30
"""
import os
import argparse
import random
import math
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
PATIENCE = 6

class CNN1DClassifier(nn.Module):
    def __init__(self, feat_dim, n_time, n_classes=8):
        super().__init__()
        # Input: x (batch, T, D) -> permute to (batch, D, T)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=feat_dim, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # halves time
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # halves time again
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)        # output (batch, 512, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # x: (batch, T, D)
        x = x.permute(0, 2, 1)  # -> (batch, D, T)
        x = self.convnet(x)     # -> (batch, 512, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_data(path):
    d = torch.load(path, map_location='cpu')
    X = d['features']  # (N, T, D) standardized
    y = d['labels']
    return X, y, d.get('meta', {})

def stratified_split(X, y, test_size=0.2, seed=SEED):
    Xn = X.numpy(); yn = y.numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(Xn, yn, test_size=test_size, stratify=yn, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return (torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.long),
            torch.from_numpy(X_val), torch.tensor(y_val, dtype=torch.long),
            torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.long))

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0; total = 0; correct = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0; total = 0; correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss/total, correct/total

def main(args):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X, y, meta = load_data(args.data)
    N, T, D = X.shape
    print("Loaded:", X.shape, "labels:", y.shape, "device:", DEVICE)
    Xtr, ytr, Xval, yval, Xte, yte = stratified_split(X, y)
    print("Splits: train", Xtr.shape[0], "val", Xval.shape[0], "test", Xte.shape[0])

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xval, yval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = CNN1DClassifier(feat_dim=D, n_time=T, n_classes=8).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = 0.0; best_path = os.path.join(args.out_dir, "cnn_best.pt"); os.makedirs(args.out_dir, exist_ok=True)
    patience = 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        print(f"[CNN] Epoch {epoch} TrainLoss {tr_loss:.4f} TrainAcc {tr_acc:.4f} ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'state_dict': model.state_dict(), 'meta': meta}, best_path)
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            print("Early stopping.")
            break

    # Evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)['state_dict'])
    test_ds = TensorDataset(Xte, yte)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"CNN TEST Acc: {test_acc:.4f}")
    save_path = os.path.join(args.out_dir, "cnn.pt")
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, save_path)
    print("Saved model:", save_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Balanced .pt file path (features standardized)")
    p.add_argument("--out_dir", default="./models", help="Output folder")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    args = p.parse_args()
    main(args)
