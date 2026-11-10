"""
FastAPI backend for MA-CMT web application.
Run: uvicorn backend.main:app --reload --port 8000
"""

import os
import sys
import json
import pickle
import io
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset import (
    load_all_samples, normalize, stratified_split,
    CropDataset, CLASS_NAMES, N_CLASSES,
    FEATURE_NAMES, N_MONTHS, N_S1, N_S2, simulate_missing,
)
from ml.models.model import build_model

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MA-CMT Research API",
    description="Deep Learning-Based Missing-Aware Multi-Sensor Fusion for Agricultural Satellite Time-Series",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ─────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
RES_DIR  = ROOT / "results"

_state: Dict[str, Any] = {
    "arrays": None,
    "labels": None,
    "ids": None,
    "mean": None,
    "std": None,
    "models": {},
    "test_idx": None,
    "train_idx": None,
    "val_idx": None,
    "data_loaded": False,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_data_if_needed():
    if _state["data_loaded"]:
        return
    cache = str(PROC_DIR / "dataset_cache.pkl")
    tar   = str(DATA_DIR / "features.tar.gz")
    gj    = str(DATA_DIR / "labels.geojson")
    arrays, labels, ids = load_all_samples(tar, gj, cache_path=cache)
    tr_idx, va_idx, te_idx = stratified_split(arrays, labels, ids, seed=42)

    norm_path = str(PROC_DIR / "normalization.npz")
    if os.path.exists(norm_path):
        npz = np.load(norm_path)
        mean, std = npz["mean"], npz["std"]
    else:
        from ml.dataset import compute_normalization
        mean, std = compute_normalization(arrays[tr_idx])
        np.savez(norm_path, mean=mean, std=std)

    _state.update({
        "arrays": arrays,
        "labels": labels,
        "ids": ids,
        "mean": mean,
        "std": std,
        "train_idx": tr_idx,
        "val_idx": va_idx,
        "test_idx": te_idx,
        "data_loaded": True,
    })


def _load_model_if_needed(model_name: str) -> Optional[torch.nn.Module]:
    if model_name in _state["models"]:
        return _state["models"][model_name]
    ckpt_path = CKPT_DIR / f"{model_name}.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    model = build_model(model_name, n_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    _state["models"][model_name] = model
    # Update mean/std from checkpoint if not loaded
    if _state["mean"] is None:
        _state["mean"] = np.array(ckpt["mean"], dtype=np.float32)
        _state["std"]  = np.array(ckpt["std"],  dtype=np.float32)
    return model


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "MA-CMT Research API"}


@app.get("/health")
def health():
    ckpts = {m: (CKPT_DIR / f"{m}.pt").exists()
             for m in ["macmt", "s1only", "s2only", "concat"]}
    exp_ready = (RES_DIR / "experiment_results.json").exists()
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoints": ckpts,
        "experiments_ready": exp_ready,
        "data_cache_exists": (PROC_DIR / "dataset_cache.pkl").exists(),
    }


# ── Dataset info ──────────────────────────────────────────────────────────────

@app.get("/dataset/info")
def dataset_info():
    _load_data_if_needed()
    s = _state
    labels = s["labels"]
    from collections import Counter
    class_counts = {CLASS_NAMES[i]: int(np.sum(labels == i)) for i in range(N_CLASSES)}
    return {
        "total_samples": int(len(labels)),
        "train_samples": int(len(s["train_idx"])),
        "val_samples":   int(len(s["val_idx"])),
        "test_samples":  int(len(s["test_idx"])),
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "class_distribution": class_counts,
        "n_months": N_MONTHS,
        "n_features": 13,
        "feature_names": FEATURE_NAMES,
        "s1_features": FEATURE_NAMES[:N_S1],
        "s2_features": FEATURE_NAMES[N_S1:],
        "sensors": {"S1": {"n_features": N_S1, "names": FEATURE_NAMES[:N_S1]},
                    "S2": {"n_features": N_S2, "names": FEATURE_NAMES[N_S1:]}},
    }


@app.get("/dataset/samples")
def list_samples(
    split: str = Query("test", enum=["train", "val", "test"]),
    class_filter: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    _load_data_if_needed()
    s = _state
    idx_key = f"{split}_idx"
    idx = s[idx_key]
    labels = s["labels"]
    ids    = s["ids"]

    if class_filter is not None:
        if class_filter not in CLASS_NAMES:
            raise HTTPException(400, f"Unknown class: {class_filter}")
        cls_id = CLASS_NAMES.index(class_filter)
        idx = idx[labels[idx] == cls_id]

    total = len(idx)
    page_idx = idx[offset:offset + limit]

    samples = [
        {
            "sample_index": int(i),
            "id": ids[i],
            "label": CLASS_NAMES[int(labels[i])],
            "label_index": int(labels[i]),
        }
        for i in page_idx
    ]
    return {"total": total, "offset": offset, "limit": limit, "samples": samples}


@app.get("/dataset/sample/{sample_index}")
def get_sample(sample_index: int):
    _load_data_if_needed()
    s = _state
    arrays = s["arrays"]
    labels = s["labels"]
    ids    = s["ids"]
    n = len(labels)
    if sample_index < 0 or sample_index >= n:
        raise HTTPException(404, f"Sample index {sample_index} out of range [0, {n})")

    arr = arrays[sample_index]  # (12, 13) raw
    label = int(labels[sample_index])

    # Format monthly observations
    monthly = []
    for t in range(N_MONTHS):
        monthly.append({
            "month": t + 1,
            "s1": {"VV": float(arr[t, 0]), "VH": float(arr[t, 1])},
            "s2": {FEATURE_NAMES[2 + i]: float(arr[t, 2 + i]) for i in range(N_S2)},
        })

    return {
        "sample_index": sample_index,
        "id": ids[sample_index],
        "label": CLASS_NAMES[label],
        "label_index": label,
        "monthly_observations": monthly,
        "raw_array_shape": list(arr.shape),
    }


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    sample_index: int = Field(..., ge=0, description="Index into the full dataset")
    model_name: str = Field("macmt", description="macmt | s1only | s2only | concat")
    missing_months_s1: List[int] = Field(default_factory=list,
                                          description="0-based month indices to mask for S1")
    missing_months_s2: List[int] = Field(default_factory=list,
                                          description="0-based month indices to mask for S2")


@app.post("/predict")
def predict(req: PredictionRequest):
    _load_data_if_needed()
    model = _load_model_if_needed(req.model_name)
    if model is None:
        raise HTTPException(503, f"Checkpoint for '{req.model_name}' not found. "
                                 "Train the model first using ml/train.py")

    s = _state
    arrays = s["arrays"]
    n = len(arrays)
    if req.sample_index < 0 or req.sample_index >= n:
        raise HTTPException(404, f"sample_index {req.sample_index} out of range")

    mean = s["mean"]
    std  = s["std"]
    arr  = arrays[req.sample_index]           # (12, 13) raw
    arr_norm = (arr - mean[None, :]) / std[None, :]  # (12, 13)

    s1 = arr_norm[:, :N_S1]   # (12, 2)
    s2 = arr_norm[:, N_S1:]   # (12, 11)

    # Build masks
    mask_s1 = np.ones(N_MONTHS, dtype=np.float32)
    mask_s2 = np.ones(N_MONTHS, dtype=np.float32)
    for m in req.missing_months_s1:
        if 0 <= m < N_MONTHS:
            mask_s1[m] = 0.0
    for m in req.missing_months_s2:
        if 0 <= m < N_MONTHS:
            mask_s2[m] = 0.0

    # Zero-out missing
    s1_in = s1 * mask_s1[:, None]
    s2_in = s2 * mask_s2[:, None]

    # Convert to tensors
    s1_t  = torch.from_numpy(s1_in[None]).to(DEVICE)
    s2_t  = torch.from_numpy(s2_in[None]).to(DEVICE)
    m1_t  = torch.from_numpy(mask_s1[None]).to(DEVICE)
    m2_t  = torch.from_numpy(mask_s2[None]).to(DEVICE)

    with torch.no_grad():
        logits = model(s1_t, s2_t, m1_t, m2_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx  = int(np.argmax(probs))
    true_label = int(s["labels"][req.sample_index])

    return {
        "sample_index": req.sample_index,
        "model": req.model_name,
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "true_class": CLASS_NAMES[true_label],
        "true_index": true_label,
        "correct": pred_idx == true_label,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(N_CLASSES)},
        "confidence": float(probs[pred_idx]),
        "missing_months_s1": req.missing_months_s1,
        "missing_months_s2": req.missing_months_s2,
    }


# ── Experiments ───────────────────────────────────────────────────────────────

@app.get("/experiments/results")
def experiment_results():
    res_path = RES_DIR / "experiment_results.json"
    if not res_path.exists():
        raise HTTPException(503, "Experiment results not found. "
                                 "Run ml/experiments/run_experiments.py first.")
    with open(str(res_path)) as f:
        return json.load(f)


@app.get("/experiments/summary")
def experiment_summary():
    res_path = RES_DIR / "experiment_results.json"
    if not res_path.exists():
        raise HTTPException(503, "Experiment results not found.")
    with open(str(res_path)) as f:
        all_results = json.load(f)

    # Build a flat summary table
    rows = []
    for scenario, model_results in all_results.items():
        for mname, rate_data in model_results.items():
            if rate_data is None:
                continue
            if scenario == "complete":
                rows.append({
                    "scenario": "complete", "model": mname, "missing_rate": 0,
                    "accuracy": rate_data.get("accuracy"),
                    "f1_macro": rate_data.get("f1_macro"),
                    "precision": rate_data.get("precision"),
                    "recall": rate_data.get("recall"),
                })
            else:
                for rate_str, m in rate_data.items():
                    rows.append({
                        "scenario": scenario, "model": mname,
                        "missing_rate": int(rate_str),
                        "accuracy": m.get("accuracy"),
                        "f1_macro": m.get("f1_macro"),
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                    })
    return {"summary": rows}


@app.get("/experiments/confusion_matrix")
def get_confusion_matrix(
    model: str = Query("macmt"),
    scenario: str = Query("complete"),
    missing_rate: int = Query(0),
):
    res_path = RES_DIR / "experiment_results.json"
    if not res_path.exists():
        raise HTTPException(503, "Experiment results not found.")
    with open(str(res_path)) as f:
        all_results = json.load(f)

    if scenario not in all_results:
        raise HTTPException(404, f"Scenario '{scenario}' not found.")
    if model not in all_results[scenario]:
        raise HTTPException(404, f"Model '{model}' not found.")

    data = all_results[scenario][model]
    if data is None:
        raise HTTPException(503, f"No data for model '{model}' in scenario '{scenario}'.")

    if scenario == "complete":
        cm = data.get("confusion_matrix", [])
    else:
        rate_str = str(missing_rate)
        if rate_str not in data:
            raise HTTPException(404, f"missing_rate={missing_rate} not found.")
        cm = data[rate_str].get("confusion_matrix", [])

    return {
        "model": model, "scenario": scenario, "missing_rate": missing_rate,
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm,
    }


@app.get("/experiments/plot")
def get_experiment_plot():
    plot_path = RES_DIR / "accuracy_vs_missing.png"
    if not plot_path.exists():
        raise HTTPException(404, "Plot image not found.")
    return FileResponse(str(plot_path), media_type="image/png")


@app.get("/models/status")
def models_status():
    statuses = {}
    for mname in ["macmt", "s1only", "s2only", "concat"]:
        ckpt = CKPT_DIR / f"{mname}.pt"
        if ckpt.exists():
            try:
                data = torch.load(str(ckpt), map_location="cpu")
                statuses[mname] = {
                    "available": True,
                    "epoch": data.get("epoch"),
                    "val_acc": data.get("val_acc"),
                }
            except Exception as e:
                statuses[mname] = {"available": False, "error": str(e)}
        else:
            statuses[mname] = {"available": False}
    return statuses
