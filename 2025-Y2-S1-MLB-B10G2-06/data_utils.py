import torch
import os, re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_pt(path):
    d = torch.load(path, map_location='cpu')
    X = d['features']
    y = d['labels']
    meta = d.get('meta', {})
    return X, y, meta

def stratified_split(X, y, test_size=0.2, seed=42):
    Xn = X.numpy(); yn = y.numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(Xn, yn, test_size=test_size, stratify=yn, random_state=seed)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)
    return (torch.from_numpy(X_train), torch.tensor(y_train, dtype=torch.long),
            torch.from_numpy(X_validation), torch.tensor(y_validation, dtype=torch.long),
            torch.from_numpy(X_test), torch.tensor(y_test, dtype=torch.long))

def loss_accuracy_graphs(loss_accuracy_data):
    # Plotting the Graphs
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Training Details and Validation Details', fontsize=16)
    
    
    axes[0].plot(loss_accuracy_data["epochs"], loss_accuracy_data["train_loss"], "o--", label='Train Loss')
    axes[0].plot(loss_accuracy_data["epochs"], loss_accuracy_data["validation_loss"], "o--", label='Validation Loss')
    axes[0].set_title('Loss VS Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    axes[1].plot(loss_accuracy_data["epochs"], loss_accuracy_data["train_accuracy"], "o--", label='Train Accuracy')
    axes[1].plot(loss_accuracy_data["epochs"], loss_accuracy_data["validation_accuracy"], "o--", label='Validation Accuracy')
    axes[1].set_title('Accuracy vs Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
        
    plt.tight_layout()
    plt.show()

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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss/total, correct/total