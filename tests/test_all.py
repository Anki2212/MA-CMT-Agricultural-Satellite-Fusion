"""
Unit tests for preprocessing, missing masks, model inference and API.
Run: python -m pytest tests/ -v
"""

import sys
import numpy as np
import torch
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset import (
    simulate_missing, compute_normalization, normalize,
    CropDataset, CLASS_NAMES, N_CLASSES, N_MONTHS, N_S1, N_S2,
)
from ml.models.model import build_model, MODEL_REGISTRY


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_arrays():
    """(N, 12, 13) random float32 arrays."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 12, 13)).astype(np.float32)


@pytest.fixture
def dummy_labels():
    return np.tile(np.arange(N_CLASSES), 20).astype(np.int64)  # 200 samples balanced


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestNormalization:
    def test_mean_std_shape(self, dummy_arrays):
        mean, std = compute_normalization(dummy_arrays)
        assert mean.shape == (13,)
        assert std.shape  == (13,)

    def test_std_positive(self, dummy_arrays):
        _, std = compute_normalization(dummy_arrays)
        assert (std > 0).all()

    def test_normalize_output_shape(self, dummy_arrays):
        mean, std = compute_normalization(dummy_arrays)
        normed = normalize(dummy_arrays, mean, std)
        assert normed.shape == dummy_arrays.shape

    def test_normalized_approx_zero_mean(self, dummy_arrays):
        mean, std = compute_normalization(dummy_arrays)
        normed = normalize(dummy_arrays, mean, std)
        # Mean of normalized training data should be ~0
        assert abs(normed.reshape(-1, 13).mean(0).max()) < 0.1


# ── Missing mask tests ────────────────────────────────────────────────────────

class TestMissingSimulation:
    def _dummy_s1_s2(self, B=50):
        rng = np.random.default_rng(1)
        s1 = rng.standard_normal((B, N_MONTHS, N_S1)).astype(np.float32)
        s2 = rng.standard_normal((B, N_MONTHS, N_S2)).astype(np.float32)
        return s1, s2

    def test_zero_rate_no_change(self):
        s1, s2 = self._dummy_s1_s2()
        s1_out, s2_out, m1, m2 = simulate_missing(s1, s2, mode="random", missing_rate=0.0)
        np.testing.assert_array_equal(s1, s1_out)
        np.testing.assert_array_equal(s2, s2_out)
        assert m1.sum() == 50 * N_MONTHS
        assert m2.sum() == 50 * N_MONTHS

    def test_mask_values_binary(self):
        s1, s2 = self._dummy_s1_s2()
        _, _, m1, m2 = simulate_missing(s1, s2, mode="random", missing_rate=0.3)
        assert set(np.unique(m1)).issubset({0.0, 1.0})
        assert set(np.unique(m2)).issubset({0.0, 1.0})

    def test_missing_zeroes_data(self):
        s1, s2 = self._dummy_s1_s2(B=10)
        s1_out, s2_out, m1, m2 = simulate_missing(s1, s2, mode="random", missing_rate=0.5)
        # Where mask=0, output should be 0
        assert (s1_out[m1 == 0] == 0).all()
        assert (s2_out[m2 == 0] == 0).all()

    def test_s1_missing_mode_only_affects_s1(self):
        s1, s2 = self._dummy_s1_s2()
        _, _, m1, m2 = simulate_missing(s1, s2, mode="s1_missing", missing_rate=0.5)
        assert m2.sum() == 50 * N_MONTHS  # S2 unaffected

    def test_consecutive_mode(self):
        s1, s2 = self._dummy_s1_s2(B=5)
        _, _, m1, _ = simulate_missing(s1, s2, mode="consecutive", missing_rate=0.25)
        # Each sample should have ~25% contiguous months zeroed
        for b in range(5):
            row = m1[b]
            # Should have at least one zero
            assert row.min() == 0.0 or N_MONTHS <= 1


# ── Model inference tests ─────────────────────────────────────────────────────

class TestModelInference:
    B = 4
    T = N_MONTHS

    def _tensors(self):
        rng = np.random.default_rng(42)
        s1 = torch.from_numpy(rng.standard_normal((self.B, self.T, N_S1)).astype(np.float32))
        s2 = torch.from_numpy(rng.standard_normal((self.B, self.T, N_S2)).astype(np.float32))
        m1 = torch.ones(self.B, self.T)
        m2 = torch.ones(self.B, self.T)
        return s1, s2, m1, m2

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_output_shape(self, model_name):
        model = build_model(model_name, n_classes=N_CLASSES)
        model.eval()
        s1, s2, m1, m2 = self._tensors()
        with torch.no_grad():
            out = model(s1, s2, m1, m2)
        assert out.shape == (self.B, N_CLASSES), f"{model_name}: shape={out.shape}"

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_all_missing_no_nan(self, model_name):
        """Model should not produce NaN even when all months are missing."""
        model = build_model(model_name, n_classes=N_CLASSES)
        model.eval()
        s1 = torch.zeros(self.B, self.T, N_S1)
        s2 = torch.zeros(self.B, self.T, N_S2)
        m1 = torch.zeros(self.B, self.T)   # all missing
        m2 = torch.zeros(self.B, self.T)
        with torch.no_grad():
            out = model(s1, s2, m1, m2)
        assert not torch.isnan(out).any(), f"{model_name} produced NaN"

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_partial_missing(self, model_name):
        """50% missing should still give valid logits."""
        model = build_model(model_name, n_classes=N_CLASSES)
        model.eval()
        s1, s2, _, _ = self._tensors()
        m1 = (torch.rand(self.B, self.T) > 0.5).float()
        m2 = (torch.rand(self.B, self.T) > 0.5).float()
        s1 = s1 * m1.unsqueeze(-1)
        s2 = s2 * m2.unsqueeze(-1)
        with torch.no_grad():
            out = model(s1, s2, m1, m2)
        assert out.shape == (self.B, N_CLASSES)
        assert not torch.isnan(out).any()

    def test_macmt_gating_uses_mask(self):
        """MA-CMT output should change when masks change."""
        model = build_model("macmt", n_classes=N_CLASSES)
        model.eval()
        s1, s2, m1, m2 = self._tensors()
        with torch.no_grad():
            out_full = model(s1, s2, m1, m2)
            m1_half = m1.clone()
            m1_half[:, :6] = 0.0  # first 6 months missing
            s1_half = s1 * m1_half.unsqueeze(-1)
            out_half = model(s1_half, s2, m1_half, m2)
        assert not torch.allclose(out_full, out_half), "Output unchanged with masking"


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestCropDataset:
    def test_length(self, dummy_arrays, dummy_labels):
        ds = CropDataset(dummy_arrays, dummy_labels)
        assert len(ds) == 200

    def test_item_shapes(self, dummy_arrays, dummy_labels):
        ds = CropDataset(dummy_arrays, dummy_labels)
        s1, s2, m1, m2, y = ds[0]
        assert s1.shape == (N_MONTHS, N_S1)
        assert s2.shape == (N_MONTHS, N_S2)
        assert m1.shape == (N_MONTHS,)
        assert m2.shape == (N_MONTHS,)
        assert y.dtype == torch.int64

    def test_missing_mode(self, dummy_arrays, dummy_labels):
        ds = CropDataset(dummy_arrays, dummy_labels,
                         missing_mode="random", missing_rate=0.5)
        # At least some masks should have zeros
        total_missing = sum(ds.mask_s1[i].sum() < N_MONTHS for i in range(len(ds)))
        assert total_missing > 0

    def test_full_missing_no_error(self, dummy_arrays, dummy_labels):
        """Dataset with all-missing months should not raise."""
        ds = CropDataset(dummy_arrays, dummy_labels,
                         missing_mode="random", missing_rate=0.99)
        s1, s2, m1, m2, y = ds[0]
        assert not torch.isnan(s1).any()
