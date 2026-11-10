"""
Experiment runner: evaluates all 4 models across 5 missing-data scenarios
at missing rates [0, 10, 20, 30, 50]%.

Run after training all models:
    python ml/experiments/run_experiments.py

Results saved to results/experiment_results.json and results/*.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset import (
    load_all_samples, normalize, stratified_split,
    CropDataset, CLASS_NAMES, N_CLASSES,
)
from ml.models.model import build_model

DATA_DIR  = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
CKPT_DIR  = ROOT / "checkpoints"
RES_DIR   = ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

MISSING_RATES = [0.0, 0.10, 0.20, 0.30, 0.50]
MISSING_MODES = ["random", "s1_missing", "s2_missing", "consecutive"]
MODEL_NAMES   = ["macmt", "s1only", "s2only", "concat"]
BATCH_SIZE    = 512
SEED          = 42


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for s1, s2, m1, m2, y in loader:
        s1, s2, m1, m2 = s1.to(device), s2.to(device), m1.to(device), m2.to(device)
        logits = model(s1, s2, m1, m2)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro":  float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(N_CLASSES))).tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
    }


def load_checkpoint(model_name, device):
    ckpt_path = CKPT_DIR / f"{model_name}.pt"
    if not ckpt_path.exists():
        return None, None, None
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = build_model(model_name, n_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)
    return model, mean, std


def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    cache = str(PROC_DIR / "dataset_cache.pkl")
    arrays, labels, ids = load_all_samples(
        str(DATA_DIR / "features.tar.gz"),
        str(DATA_DIR / "labels.geojson"),
        cache_path=cache,
    )
    _, _, test_idx = stratified_split(arrays, labels, ids, seed=SEED)
    print(f"Test samples: {len(test_idx)}")

    all_results = {}

    # Complete (no missing) evaluation
    print("\n=== Complete observations (0% missing) ===")
    complete_results = {}
    for model_name in MODEL_NAMES:
        model, mean, std = load_checkpoint(model_name, device)
        if model is None:
            print(f"  [{model_name}] checkpoint not found, skipping")
            complete_results[model_name] = None
            continue
        arrays_norm = normalize(arrays, mean, std)
        ds = CropDataset(arrays_norm[test_idx], labels[test_idx], missing_mode="none")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        metrics = evaluate(model, loader, device)
        metrics.pop("preds"); metrics.pop("labels")
        complete_results[model_name] = metrics
        print(f"  [{model_name}] acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")
    all_results["complete"] = complete_results

    # Missing-rate sweep per mode
    print("\n=== Missing-rate experiments ===")
    for mode in MISSING_MODES:
        mode_results = {}
        for model_name in MODEL_NAMES:
            model, mean, std = load_checkpoint(model_name, device)
            if model is None:
                mode_results[model_name] = None
                continue
            arrays_norm = normalize(arrays, mean, std)
            rate_metrics = {}
            for rate in MISSING_RATES:
                ds = CropDataset(
                    arrays_norm[test_idx], labels[test_idx],
                    missing_mode=mode if rate > 0 else "none",
                    missing_rate=rate, seed=SEED + 1,
                )
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                m = evaluate(model, loader, device)
                # Save confusion matrix separately, keep rest in table
                cm = m.pop("confusion_matrix")
                m.pop("preds"); m.pop("labels")
                m["confusion_matrix"] = cm
                rate_metrics[str(int(rate * 100))] = m
                print(f"  [{model_name}] mode={mode} rate={int(rate*100)}% "
                      f"acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}")
            mode_results[model_name] = rate_metrics
        all_results[mode] = mode_results

    # Save JSON
    out_json = str(RES_DIR / "experiment_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save CSV summary
    rows = []
    for scenario, model_results in all_results.items():
        for mname, rate_data in model_results.items():
            if rate_data is None:
                continue
            if scenario == "complete":
                rows.append({"scenario": "complete", "missing_rate": 0,
                             "model": mname, **{k: v for k, v in rate_data.items()
                                                 if k != "confusion_matrix"}})
            else:
                for rate_str, metrics in rate_data.items():
                    rows.append({"scenario": scenario, "missing_rate": int(rate_str),
                                 "model": mname, **{k: v for k, v in metrics.items()
                                                     if k != "confusion_matrix"}})
    df = pd.DataFrame(rows)
    csv_path = str(RES_DIR / "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return all_results


if __name__ == "__main__":
    run_experiments()
