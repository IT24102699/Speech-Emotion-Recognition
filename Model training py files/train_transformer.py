#!/usr/bin/env python3
"""
train_transformer.py
Train a Transformer encoder-based classifier on RAVDESS features.

Usage:
    python train_transformer.py --data ./data/ravdess_features_balanced.pt
"""
import os
import argparse
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 24
EPOCHS = 40
LR = 1e-3
PATIENCE = 6

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim, n_time, d_model=256, nhead=4, num_layers=3, n_classes=8):
        super().__init__()
        self.project = nn.Linear(feat_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model, max_len=n_time+10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.project(x)
        x = self.layernorm(x)
        x = self.pos(x)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        return self.fc(pooled)

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

def train_epoch(model, loader, optimizer, crit, device):
    model.train()
    tl=0.0; tot=0; corr=0
    for xb, yb in loader:
        xb=xb.to(device); yb=yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        optimizer.step()
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
    X, y, meta = load_data(args.data)
    N, T, D = X.shape
    print("Loaded:", X.shape, "device:", DEVICE)
    Xtr, ytr, Xval, yval, Xte, yte = stratified_split(X, y)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xval, yval), batch_size=args.batch_size, shuffle=False)

    model = TransformerClassifier(feat_dim=D, n_time=T, d_model=256, nhead=4, num_layers=3, n_classes=8).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = 0.0; best_path = os.path.join(args.out_dir, "transformer_best.pt"); os.makedirs(args.out_dir, exist_ok=True)
    patience = 0

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, crit, DEVICE)
        val_loss, val_acc = eval_epoch(model, val_loader, crit, DEVICE)
        print(f"[Transformer] Epoch {epoch} TrainAcc {tr_acc:.4f} ValAcc {val_acc:.4f}")
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
    print("Transformer TEST Acc:", test_acc)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, os.path.join(args.out_dir, "transformer.pt"))
    print("Saved Transformer model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="./models")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    args = p.parse_args()
    main(args)
