"""
MA-CMT: Missing-Aware Cross-Modal Temporal Transformer
and three baseline models.

All models accept:
    s1      (B, 12, 2)   float32  – S1 features (zeroed where missing)
    s2      (B, 12, 11)  float32  – S2 features (zeroed where missing)
    mask_s1 (B, 12)      float32  – 1 = present, 0 = missing
    mask_s2 (B, 12)      float32  – 1 = present, 0 = missing

Returns logits (B, n_classes).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility blocks ────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence length T."""
    def __init__(self, d_model: int, max_len: int = 12, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder operating over the time axis."""
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, D)
        src_key_padding_mask: (B, T) bool, True = ignore (all-missing months)
        Returns: (B, T, D)
        """
        x = self.pos_enc(x)
        if src_key_padding_mask is not None:
            # Safety: PyTorch TransformerEncoder produces NaN when ALL tokens
            # of a sample are masked. Force at least one token visible per row.
            all_masked = src_key_padding_mask.all(dim=1, keepdim=True)  # (B,1)
            safe_mask = src_key_padding_mask & ~all_masked
        else:
            safe_mask = None
        return self.encoder(x, src_key_padding_mask=safe_mask)


def build_temporal_mask(mask_s1: torch.Tensor,
                         mask_s2: torch.Tensor) -> torch.Tensor:
    """
    Returns bool key-padding mask (B, T) where True = both sensors missing.
    Transformer ignores these positions.
    """
    both_missing = (mask_s1 < 0.5) & (mask_s2 < 0.5)   # (B, T)
    return both_missing   # True means ignore


# ── MA-CMT ───────────────────────────────────────────────────────────────────

class MACMT(nn.Module):
    """
    Missing-Aware Cross-Modal Temporal Transformer (proposed model).

    Architecture:
      1. Separate S1 / S2 linear encoders  (B,T,D)
      2. Availability-guided gated cross-modal fusion per timestep
      3. Temporal Transformer Encoder
      4. Mean pooling  → FC classification head
    """

    def __init__(
        self,
        n_s1: int = 2,
        n_s2: int = 11,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 256,
        n_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-modal encoders
        self.s1_enc = nn.Sequential(
            nn.Linear(n_s1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.s2_enc = nn.Sequential(
            nn.Linear(n_s2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Gated cross-modal fusion
        # Gate input: [h_s1, h_s2, mask_s1, mask_s2] → 2D gate
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model + 2, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Linear(d_model, d_model)

        # Temporal encoder
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(
        self,
        s1: torch.Tensor,       # (B, T, 2)
        s2: torch.Tensor,       # (B, T, 11)
        mask_s1: torch.Tensor,  # (B, T)
        mask_s2: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:

        B, T, _ = s1.shape

        # Encode each modality
        h_s1 = self.s1_enc(s1)   # (B, T, D)
        h_s2 = self.s2_enc(s2)   # (B, T, D)

        # Zero-out encodings for missing observations
        h_s1 = h_s1 * mask_s1.unsqueeze(-1)   # (B, T, D)
        h_s2 = h_s2 * mask_s2.unsqueeze(-1)   # (B, T, D)

        # Availability-guided gating
        gate_inp = torch.cat([
            h_s1, h_s2,
            mask_s1.unsqueeze(-1),   # (B, T, 1)
            mask_s2.unsqueeze(-1),   # (B, T, 1)
        ], dim=-1)                   # (B, T, 2D+2)
        gates = self.gate_net(gate_inp)   # (B, T, 2)
        g1, g2 = gates[..., 0:1], gates[..., 1:2]

        fused = g1 * h_s1 + g2 * h_s2    # (B, T, D)
        fused = self.fusion_proj(fused)   # (B, T, D)

        # Build padding mask (fully-missing months)
        pad_mask = build_temporal_mask(mask_s1, mask_s2)   # (B, T) bool

        # Temporal Transformer (safe mask handled inside TemporalTransformerEncoder)
        enc_out = self.temporal_enc(fused, src_key_padding_mask=pad_mask)  # (B, T, D)

        # Mean pooling: use original pad_mask but ensure at least 1 token per sample
        safe_valid_mask = ~pad_mask
        all_masked = pad_mask.all(dim=1, keepdim=True)   # (B, 1)
        # For fully-masked samples, pool uniformly over all T tokens
        fallback = all_masked.expand(-1, pad_mask.size(1))  # (B, T)
        pooling_mask = safe_valid_mask | fallback           # (B, T)
        valid = pooling_mask.float().unsqueeze(-1)          # (B, T, 1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (enc_out * valid).sum(dim=1) / denom      # (B, D)

        return self.head(pooled)


# ── Baseline: S1-only ─────────────────────────────────────────────────────────

class S1OnlyModel(nn.Module):
    """Baseline using only Sentinel-1 features."""

    def __init__(self, n_s1=2, d_model=128, n_heads=4, n_layers=3,
                 dim_ff=256, n_classes=10, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_s1, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = TemporalTransformerEncoder(d_model, n_heads, n_layers, dim_ff, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, s1, s2, mask_s1, mask_s2):
        h = self.enc(s1) * mask_s1.unsqueeze(-1)
        pad_mask = (mask_s1 < 0.5)
        enc = self.temporal(h, src_key_padding_mask=pad_mask)
        all_m = pad_mask.all(dim=1, keepdim=True)
        pool_mask = (~pad_mask | all_m.expand_as(pad_mask)).float().unsqueeze(-1)
        pooled = (enc * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


# ── Baseline: S2-only ─────────────────────────────────────────────────────────

class S2OnlyModel(nn.Module):
    """Baseline using only Sentinel-2 features."""

    def __init__(self, n_s2=11, d_model=128, n_heads=4, n_layers=3,
                 dim_ff=256, n_classes=10, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_s2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = TemporalTransformerEncoder(d_model, n_heads, n_layers, dim_ff, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, s1, s2, mask_s1, mask_s2):
        h = self.enc(s2) * mask_s2.unsqueeze(-1)
        pad_mask = (mask_s2 < 0.5)
        enc = self.temporal(h, src_key_padding_mask=pad_mask)
        all_m = pad_mask.all(dim=1, keepdim=True)
        pool_mask = (~pad_mask | all_m.expand_as(pad_mask)).float().unsqueeze(-1)
        pooled = (enc * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


# ── Baseline: Concat Fusion ───────────────────────────────────────────────────

class ConcatFusionModel(nn.Module):
    """Baseline: concatenate S1+S2 features, no missing-awareness."""

    def __init__(self, n_s1=2, n_s2=11, d_model=128, n_heads=4, n_layers=3,
                 dim_ff=256, n_classes=10, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_s1 + n_s2, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal = TemporalTransformerEncoder(d_model, n_heads, n_layers, dim_ff, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, s1, s2, mask_s1, mask_s2):
        # Zero-out missing (but no gating)
        s1_m = s1 * mask_s1.unsqueeze(-1)
        s2_m = s2 * mask_s2.unsqueeze(-1)
        combined = torch.cat([s1_m, s2_m], dim=-1)
        h = self.enc(combined)
        # Padding mask: both missing
        pad_mask = build_temporal_mask(mask_s1, mask_s2)
        enc = self.temporal(h, src_key_padding_mask=pad_mask)
        all_m = pad_mask.all(dim=1, keepdim=True)
        pool_mask = (~pad_mask | all_m.expand_as(pad_mask)).float().unsqueeze(-1)
        pooled = (enc * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


# ── Factory ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "macmt":  MACMT,
    "s1only": S1OnlyModel,
    "s2only": S2OnlyModel,
    "concat": ConcatFusionModel,
}


def build_model(name: str, n_classes: int = 10, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_classes=n_classes, **kwargs)
