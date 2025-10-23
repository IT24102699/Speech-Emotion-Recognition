#!/usr/bin/env python3
"""
train_lstm.py
Train an LSTM classifier on preprocessed RAVDESS features.

Usage:
    python train_lstm.py --data ./data/ravdess_features_balanced.pt --epochs 40
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
EPOCHS = 40
LR = 1e-3
PATIENCE = 6

class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden=128, n_layers=2, n_classes=8, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden,
                            num_layers=n_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)    # (batch, T, hidden*dirs)
        last = out.mean(dim=1)   # average over time (robust to right-padding zeros)
        return self.fc(last)

def load_data(path):
    d = torch.load(path, map_location='cpu')
    return d['features'], d['labels'], d.get('meta', {})

def stratified_split(X, y, test_size=0.2, seed=SEED):
    Xn = X.numpy(); yn = y.numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(Xn, yn, test_size=test_size, stratify=yn, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return (torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.long),
            torch.from_numpy(X_val), torch.tensor(y_val, dtype=torch.long),
            torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.long))

def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0.0; total = 0; correct = 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss/total, correct/total

def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss = 0.0; total = 0; correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            out = model(xb)
            loss = crit(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss/total, correct/total

def main(args):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    X, y, meta = load_data(args.data)
    N, T, D = X.shape
    print("Loaded:", X.shape, "device:", DEVICE)
    Xtr, ytr, Xval, yval, Xte, yte = stratified_split(X, y)
    train_ds = TensorDataset(Xtr, ytr); val_ds = TensorDataset(Xval, yval)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMClassifier(feat_dim=D, n_classes=8).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0; best_path = os.path.join(args.out_dir, "lstm_best.pt"); os.makedirs(args.out_dir, exist_ok=True)
    patience = 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, crit, DEVICE)
        print(f"[LSTM] Epoch {epoch} TrainAcc {tr_acc:.4f} ValAcc {val_acc:.4f}")
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
    test_ds = TensorDataset(Xte, yte)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = eval_epoch(model, test_loader, crit, DEVICE)
    print("LSTM TEST Acc:", test_acc)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, os.path.join(args.out_dir, "lstm.pt"))
    print("Saved LSTM model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    args = p.parse_args()
    main(args)
