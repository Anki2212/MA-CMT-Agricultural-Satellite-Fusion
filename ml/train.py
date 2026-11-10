"""
Training script for all four models (MA-CMT + 3 baselines).
Run: python ml/train.py --model macmt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset import (
    load_all_samples, compute_normalization, normalize,
    stratified_split, CropDataset, compute_class_weights,
    CLASS_NAMES, N_CLASSES,
)
from ml.models.model import build_model

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"

PROC_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for s1, s2, m1, m2, y in loader:
        s1, s2, m1, m2, y = s1.to(device), s2.to(device), m1.to(device), m2.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(s1, s2, m1, m2)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for s1, s2, m1, m2, y in loader:
        s1, s2, m1, m2, y = s1.to(device), s2.to(device), m1.to(device), m2.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(s1, s2, m1, m2)
            loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def train_model(
    model_name: str,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
    missing_rate: float = 0.0,
    missing_mode: str = "none",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    cache = str(PROC_DIR / "dataset_cache.pkl")
    arrays, labels, ids = load_all_samples(
        str(DATA_DIR / "features.tar.gz"),
        str(DATA_DIR / "labels.geojson"),
        cache_path=cache,
    )

    train_idx, val_idx, test_idx = stratified_split(arrays, labels, ids, seed=seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Normalization (train only) ───────────────────────────────────────────
    norm_path = str(PROC_DIR / "normalization.npz")
    if os.path.exists(norm_path):
        npz = np.load(norm_path)
        mean, std = npz["mean"], npz["std"]
        print("Loaded existing normalization stats")
    else:
        mean, std = compute_normalization(arrays[train_idx])
        np.savez(norm_path, mean=mean, std=std)
        print("Computed and saved normalization stats")

    arrays_norm = normalize(arrays, mean, std)

    # ── Datasets / Loaders ───────────────────────────────────────────────────
    ds_kwargs = dict(missing_mode=missing_mode, missing_rate=missing_rate, seed=seed)
    train_ds = CropDataset(arrays_norm[train_idx], labels[train_idx], **ds_kwargs)
    val_ds   = CropDataset(arrays_norm[val_idx],   labels[val_idx],   missing_mode="none")
    test_ds  = CropDataset(arrays_norm[test_idx],  labels[test_idx],  missing_mode="none")

    num_workers = 0  # Windows-safe
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(model_name, n_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}  Parameters: {n_params:,}")

    class_weights = compute_class_weights(labels[train_idx], N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = -1.0
    patience_ctr = 0
    history = []
    ckpt_path = str(CKPT_DIR / f"{model_name}.pt")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "va_loss": va_loss, "va_acc": va_acc})
        print(f"Ep {epoch:03d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_ctr = 0
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "class_names": CLASS_NAMES,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved (val_acc={va_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ──────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {te_acc:.4f}")

    # Save history
    result = {"model": model_name, "best_val_acc": best_val_acc,
              "test_acc": te_acc, "history": history}
    hist_path = str(PROC_DIR / f"history_{model_name}.json")
    with open(hist_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"History saved: {hist_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="macmt",
                        choices=["macmt", "s1only", "s2only", "concat"])
    parser.add_argument("--epochs",    type=int,   default=60)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--patience",  type=int,   default=10)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
