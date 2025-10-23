#!/usr/bin/env python3
"""
train_mlp.py
Train a simple MLP by flattening (time,feat) into a vector.

Usage:
    python train_mlp.py --data ./data/ravdess_features_balanced.pt
"""
import os
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PATIENCE = 8

class MLPClassifier(nn.Module):
    def __init__(self, feat_dim, n_time, n_classes=8):
        super().__init__()
        in_dim = feat_dim * n_time
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_data(path):
    d = torch.load(path, map_location='cpu')
    return d['features'], d['labels'], d.get('meta', {})

def stratified_split_flat(X, y, test_size=0.2, seed=SEED):
    Xn = X.numpy(); yn = y.numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(Xn, yn, test_size=test_size, stratify=yn, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return (torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.long),
            torch.from_numpy(X_val), torch.tensor(y_val, dtype=torch.long),
            torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.long))

def train_epoch(model, loader, opt, crit, device):
    model.train()
    tl=0.0; tot=0; corr=0
    for xb, yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        tl += loss.item()*xb.size(0)
        preds = out.argmax(dim=1)
        corr += (preds==yb).sum().item()
        tot += xb.size(0)
    return tl/tot, corr/tot

def eval_epoch(model, loader, crit, device):
    model.eval()
    tl=0.0; tot=0; corr=0
    with torch.no_grad():
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            out = model(xb)
            loss = crit(out, yb)
            tl += loss.item()*xb.size(0)
            preds = out.argmax(dim=1)
            corr += (preds==yb).sum().item()
            tot += xb.size(0)
    return tl/tot, corr/tot

def main(args):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X, y, meta = load_data(args.data)  # (N, T, D)
    N, T, D = X.shape
    print("Loaded:", X.shape)
    # Flatten for dataset (MLP will accept (N, T, D) and flatten internally)
    Xtr, ytr, Xval, yval, Xte, yte = stratified_split_flat(X, y)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size, shuffle=False)

    model = MLPClassifier(feat_dim=D, n_time=T, n_classes=8).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = 0.0; best_path = os.path.join(args.out_dir, "mlp_best.pt"); os.makedirs(args.out_dir, exist_ok=True)
    patience = 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, crit, DEVICE)
        print(f"[MLP] Epoch {epoch} TrainAcc {tr_acc:.4f} ValAcc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'state_dict': model.state_dict(), 'meta': meta}, best_path)
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)['state_dict'])
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = eval_epoch(model, test_loader, crit, DEVICE)
    print("MLP TEST Acc:", test_acc)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, os.path.join(args.out_dir, "mlp.pt"))
    print("Saved MLP model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    args = p.parse_args()
    main(args)
