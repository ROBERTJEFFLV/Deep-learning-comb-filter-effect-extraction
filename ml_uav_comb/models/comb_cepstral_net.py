"""Minimal cepstral-bin classification network.

Architecture (from first principles):
  Input:  per-frame normalized cepstral patches  X ∈ R^(T × Q)
  Module 1: Quefrency Conv — local peak-shape matching along quefrency axis
  Module 2: Projection — flatten conv features to per-frame vector
  Module 3: Causal GRU — temporal smoothing (suppresses burst errors)
  Module 4: K+1 classifier — no-pattern (class 0) + K distance bins

The network answers ONE question per window:
  "Is there a reliable comb peak, and if so, which distance bin?"

Distance is decoded from the predicted bin.  Pattern confidence = 1 - P(no-pattern).
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


C_SPEED = 343.0  # m/s


def compute_cepstral_geometry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute quefrency bin range and distance-bin mapping from config."""
    sr = int(cfg["audio"]["target_sr"])
    n_fft = int(cfg["audio"]["n_fft"])
    freq_min = float(cfg["audio"]["freq_min"])
    freq_max = float(cfg["audio"]["freq_max"])

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    n_band = int(band_mask.sum())

    cep_cfg = cfg.get("cepstral", {})
    tau_min_s = float(cep_cfg.get("tau_min_s", 0.00025))
    tau_max_s = float(cep_cfg.get("tau_max_s", 0.004))

    df = float(sr) / float(n_fft)
    quef_tau_factor = 1.0 / (n_band * df)
    cep_min_bin = max(2, round(tau_min_s / quef_tau_factor))
    cep_max_bin = min(n_band // 2, round(tau_max_s / quef_tau_factor) + 1)
    Q = cep_max_bin - cep_min_bin  # number of quefrency bins

    # Distance center for each bin (cm)
    bin_centers_cm = np.array([
        C_SPEED * (cep_min_bin + k) * quef_tau_factor / 2.0 * 100.0
        for k in range(Q)
    ], dtype=np.float32)

    return {
        "n_band": n_band,
        "cep_min_bin": cep_min_bin,
        "cep_max_bin": cep_max_bin,
        "Q": Q,
        "K": Q,  # K distance bins = Q quefrency bins
        "quef_tau_factor": quef_tau_factor,
        "bin_centers_cm": bin_centers_cm,
    }


class CombCepstralNet(nn.Module):
    """Minimal correct network for cepstral-bin distance classification.

    Parameters (from cfg["model"]):
        q_conv_channels: int = 16    — quefrency conv feature maps
        q_proj_dim: int = 32         — per-frame projection dimension
        gru_hidden: int = 32         — causal GRU hidden dimension
        gru_layers: int = 1          — GRU depth
        gru_dropout: float = 0.1     — dropout between GRU layers
        use_temporal: bool = True     — if False, skip GRU (ablation A1)

    Input:  batch["x"] shape [B, T, Q]
    Output: {"logits": [B, K+1], "bin_centers_cm": [K]}
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        geom = compute_cepstral_geometry(cfg)
        self.Q = geom["Q"]
        self.K = geom["K"]
        self.num_classes = self.K + 1  # class 0 = no-pattern

        # Register bin centers for distance decoding
        self.register_buffer(
            "bin_centers_cm",
            torch.from_numpy(geom["bin_centers_cm"]).float(),
        )

        model_cfg = cfg.get("model", {})
        C_q = int(model_cfg.get("q_conv_channels", 16))
        D = int(model_cfg.get("q_proj_dim", 32))
        H = int(model_cfg.get("gru_hidden", 32))
        n_layers = int(model_cfg.get("gru_layers", 1))
        dropout = float(model_cfg.get("gru_dropout", 0.1))
        self.use_temporal = bool(model_cfg.get("use_temporal", True))

        # Module 1: Quefrency conv — learns local peak templates
        self.q_conv = nn.Sequential(
            nn.Conv1d(1, C_q, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(C_q, C_q, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Module 2: Flatten + project to per-frame vector
        self.q_proj = nn.Sequential(
            nn.Linear(C_q * self.Q, D),
            nn.ReLU(inplace=True),
        )

        # Module 3: Causal temporal smoother (optional for ablation)
        if self.use_temporal:
            self.gru = nn.GRU(
                input_size=D,
                hidden_size=H,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            classifier_in = H
        else:
            self.gru = None
            classifier_in = D

        # Module 4: K+1 classifier
        self.classifier = nn.Linear(classifier_in, self.num_classes)

        self._init_weights(cfg)

    def _init_weights(self, cfg: Dict[str, Any] | None = None) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Prior-based bias initialization for the classifier.
        # Prevents the "predict everything as pattern" collapse at step 0:
        # set bias[0] (no-pattern) so that softmax p[0] ≈ no_pattern_prior.
        no_pat_prior = float(
            (cfg or {}).get("model", {}).get("no_pattern_prior", 0.9)
        )
        no_pat_prior = max(1e-4, min(1.0 - 1e-4, no_pat_prior))
        n_pat = self.num_classes - 1
        if n_pat > 0:
            # softmax(b0, 0, ..., 0)[0] = exp(b0)/(exp(b0)+n_pat) = no_pat_prior
            b0 = math.log(no_pat_prior * n_pat / (1.0 - no_pat_prior))
            with torch.no_grad():
                self.classifier.bias.data[0] = b0

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"]  # [B, T, Q]
        B, T, Q = x.shape
        assert Q == self.Q, f"Expected Q={self.Q}, got {Q}"

        # Per-frame quefrency encoding
        x_flat = x.reshape(B * T, 1, Q)            # [B*T, 1, Q]
        x_conv = self.q_conv(x_flat)                # [B*T, C_q, Q]
        x_proj = self.q_proj(x_conv.reshape(B * T, -1))  # [B*T, D]
        x_seq = x_proj.reshape(B, T, -1)            # [B, T, D]

        # Temporal smoothing
        if self.use_temporal and self.gru is not None:
            gru_out, _ = self.gru(x_seq)             # [B, T, H]
            h_last = gru_out[:, -1, :]               # [B, H]
        else:
            h_last = x_seq[:, -1, :]                 # [B, D]

        # Classify
        logits = self.classifier(h_last)              # [B, K+1]

        return {
            "logits": logits,
            "bin_centers_cm": self.bin_centers_cm,
        }

    def decode_distance(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode logits to distance and pattern confidence.

        Returns:
            pred_class: [B] — predicted class (0=no-pattern, 1..K=bins)
            pred_distance_cm: [B] — distance (0 for no-pattern)
            pattern_prob: [B] — P(pattern present) = 1 - P(class 0)
        """
        probs = F.softmax(logits, dim=-1)
        pred_class = logits.argmax(dim=-1)

        # Expected distance (soft decoding from pattern classes only)
        pattern_probs = probs[:, 1:]  # [B, K]
        pattern_probs_norm = pattern_probs / (pattern_probs.sum(dim=-1, keepdim=True) + 1e-8)
        soft_distance = (pattern_probs_norm * self.bin_centers_cm.unsqueeze(0)).sum(dim=-1)

        # Hard distance from argmax
        hard_distance = torch.zeros_like(pred_class, dtype=torch.float32)
        has_pattern = pred_class > 0
        if has_pattern.any():
            bin_idx = pred_class[has_pattern] - 1  # 0-indexed
            hard_distance[has_pattern] = self.bin_centers_cm[bin_idx]

        pattern_prob = 1.0 - probs[:, 0]

        return {
            "pred_class": pred_class,
            "hard_distance_cm": hard_distance,
            "soft_distance_cm": soft_distance,
            "pattern_prob": pattern_prob,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
