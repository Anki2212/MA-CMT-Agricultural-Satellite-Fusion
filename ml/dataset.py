"""
CropHarvest Dataset Loader for MA-CMT project.
Loads features.tar.gz + labels.geojson, selects 13 features (S1+S2),
handles missing data simulation, normalization, and splitting.
"""

import json
import tarfile
import io
import os
import pickle
import numpy as np
import h5py
from collections import Counter
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

# ── Feature indices ──────────────────────────────────────────────────────────
S1_INDICES = [0, 1]           # VV, VH
S2_INDICES = list(range(2, 13))  # B2–B12 (11 bands)
FEATURE_INDICES = S1_INDICES + S2_INDICES   # 13 features total
N_MONTHS = 12
N_S1 = 2
N_S2 = 11
N_FEATURES = 13

FEATURE_NAMES = ["S1_VV", "S1_VH",
                 "S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7",
                 "S2_B8", "S2_B8A", "S2_B9", "S2_B11", "S2_B12"]

CLASS_NAMES = [
    "cereals", "other", "non_crop", "fruits_nuts", "vegetables_melons",
    "oilseeds", "leguminous", "root_tuber", "beverage_spice", "sugar"
]
N_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_all_samples(
    tar_path: str,
    geojson_path: str,
    top_k_classes: int = 10,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all matched labeled samples from the archive.

    Returns:
        arrays : (N, 12, 13) float32 – the 13 selected features
        labels : (N,) int64 – class indices
        sample_ids : list of '{index}_{dataset}' strings
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Reading labels.geojson …")
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    labeled = [
        feat["properties"]
        for feat in gj["features"]
        if feat["properties"]["classification_label"] is not None
    ]

    # Select top-k classes
    class_counter = Counter(p["classification_label"] for p in labeled)
    top_classes = [c for c, _ in class_counter.most_common(top_k_classes)]
    class_to_idx = {c: i for i, c in enumerate(top_classes)}

    print(f"Selected {top_k_classes} classes: {top_classes}")

    # Build lookup: '{index}_{dataset}' -> classification_label
    lookup = {}
    for p in labeled:
        if p["classification_label"] in class_to_idx:
            key = f"{p['index']}_{p['dataset']}"
            lookup[key] = class_to_idx[p["classification_label"]]

    print(f"Scanning archive for {len(lookup)} keys …")

    arrays_list = []
    labels_list = []
    ids_list = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name  # e.g. features/arrays/42_lem-brazil.h5
            if not name.endswith(".h5") or "arrays" not in name:
                continue
            basename = os.path.basename(name).replace(".h5", "")  # e.g. 42_lem-brazil
            if basename not in lookup:
                continue

            f = tar.extractfile(member)
            data = f.read()
            with h5py.File(io.BytesIO(data), "r") as hf:
                arr = hf["array"][()]  # (12, 18)

            # Select 13 features
            arr = arr[:, FEATURE_INDICES].astype(np.float32)   # (12, 13)

            # Clamp extreme values (Sentinel data can have outliers)
            arr = np.clip(arr, -100.0, 10000.0)

            arrays_list.append(arr)
            labels_list.append(lookup[basename])
            ids_list.append(basename)

    arrays = np.stack(arrays_list, axis=0)   # (N, 12, 13)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"Loaded {len(arrays)} samples, shape={arrays.shape}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving cache to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump((arrays, labels, ids_list), f)

    return arrays, labels, ids_list


def compute_normalization(arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training data. Shape: (13,)"""
    # arrays: (N, 12, 13)
    flat = arrays.reshape(-1, N_FEATURES)  # (N*12, 13)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(arrays: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arrays - mean[None, None, :]) / std[None, None, :]


def stratified_split(
    arrays: np.ndarray,
    labels: np.ndarray,
    ids: List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Tuple:
    """Return train / val / test index splits (stratified)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = np.arange(n)

    # Per-class split
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(N_CLASSES):
        cls_idx = idx[labels == cls]
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        n_train = int(n_cls * train_ratio)
        n_val   = int(n_cls * val_ratio)
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])

    return (
        np.array(train_idx), np.array(val_idx), np.array(test_idx)
    )


# ── Missing-data simulation ───────────────────────────────────────────────────

def simulate_missing(
    s1: np.ndarray,     # (B, 12, 2)
    s2: np.ndarray,     # (B, 12, 11)
    mode: str = "random",
    missing_rate: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate missing observations and return zeroed arrays + binary masks.

    Mask convention: 1 = available, 0 = missing.
    Returns: s1_out, s2_out, mask_s1 (B,12), mask_s2 (B,12)
    """
    B, T, _ = s1.shape
    rng = np.random.default_rng(seed)

    mask_s1 = np.ones((B, T), dtype=np.float32)
    mask_s2 = np.ones((B, T), dtype=np.float32)

    if missing_rate <= 0.0:
        return s1.copy(), s2.copy(), mask_s1, mask_s2

    if mode == "random":
        # Each (sample, month, sensor) independently missing
        mask_s1 = (rng.random((B, T)) > missing_rate).astype(np.float32)
        mask_s2 = (rng.random((B, T)) > missing_rate).astype(np.float32)

    elif mode == "s1_missing":
        mask_s1 = (rng.random((B, T)) > missing_rate).astype(np.float32)

    elif mode == "s2_missing":
        mask_s2 = (rng.random((B, T)) > missing_rate).astype(np.float32)

    elif mode == "consecutive":
        # Drop consecutive months
        n_drop = max(1, int(T * missing_rate))
        for b in range(B):
            start = rng.integers(0, T - n_drop + 1)
            mask_s1[b, start:start + n_drop] = 0.0
            mask_s2[b, start:start + n_drop] = 0.0

    else:
        raise ValueError(f"Unknown missing mode: {mode}")

    # Zero-out missing
    s1_out = s1 * mask_s1[:, :, None]
    s2_out = s2 * mask_s2[:, :, None]
    return s1_out, s2_out, mask_s1, mask_s2


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class CropDataset(Dataset):
    """
    Returns (s1, s2, mask_s1, mask_s2, label) tensors.

    s1:     (12, 2)   float32
    s2:     (12, 11)  float32
    mask_s1:(12,)     float32  1=present, 0=missing
    mask_s2:(12,)     float32
    label:  ()        int64
    """

    def __init__(
        self,
        arrays:  np.ndarray,   # (N, 12, 13) normalized
        labels:  np.ndarray,   # (N,)
        missing_mode: str = "none",
        missing_rate: float = 0.0,
        seed: int = 42,
    ):
        self.labels = labels

        # Split into S1 and S2
        s1 = arrays[:, :, :N_S1]   # (N, 12, 2)
        s2 = arrays[:, :, N_S1:]   # (N, 12, 11)

        # Simulate missing data
        if missing_mode == "none" or missing_rate <= 0.0:
            self.s1 = s1
            self.s2 = s2
            self.mask_s1 = np.ones((len(arrays), N_MONTHS), dtype=np.float32)
            self.mask_s2 = np.ones((len(arrays), N_MONTHS), dtype=np.float32)
        else:
            self.s1, self.s2, self.mask_s1, self.mask_s2 = simulate_missing(
                s1, s2, mode=missing_mode, missing_rate=missing_rate, seed=seed
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.s1[idx]),
            torch.from_numpy(self.s2[idx]),
            torch.from_numpy(self.mask_s1[idx]),
            torch.from_numpy(self.mask_s2[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted CE loss."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weights)
