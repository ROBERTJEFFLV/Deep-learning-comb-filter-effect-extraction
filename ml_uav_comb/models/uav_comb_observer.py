"""Distance-grid causal observer with TCN temporal modeling."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from ml_uav_comb.models.branches import ScalarSequenceBranch, SpectrogramBranch
from ml_uav_comb.models.tcn import CausalTCN


def build_distance_grid_cm(cfg: Dict[str, Any]) -> torch.Tensor:
    model_cfg = cfg["model"]
    num_candidates = int(model_cfg.get("num_candidates", 64))
    min_cm = float(model_cfg.get("distance_grid_cm_min", 20.0))
    max_cm = float(model_cfg.get("distance_grid_cm_max", 300.0))
    mode = str(model_cfg.get("distance_grid_mode", "uniform")).strip().lower()
    min_cm = max(min_cm, 1e-3)
    max_cm = max(max_cm, min_cm + 1e-3)
    if num_candidates < 2:
        raise ValueError("model.num_candidates must be >= 2")
    if mode == "uniform":
        return torch.linspace(min_cm, max_cm, num_candidates, dtype=torch.float32)
    if mode == "log":
        return torch.logspace(
            np.log10(min_cm),
            np.log10(max_cm),
            num_candidates,
            dtype=torch.float32,
        )
    raise ValueError(f"unsupported distance_grid_mode: {mode}")


def summarize_distance_logits(
    distance_logits: torch.Tensor,
    validity_logit: torch.Tensor,
    distance_grid_cm: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(distance_logits, dim=-1)
    mean_cm = torch.sum(probs * distance_grid_cm.unsqueeze(0), dim=-1)
    var_cm2 = torch.sum(probs * torch.square(distance_grid_cm.unsqueeze(0) - mean_cm.unsqueeze(-1)), dim=-1)
    var_cm2 = torch.clamp(var_cm2, min=float(eps))
    logvar = torch.log(var_cm2 + float(eps))
    entropy = -torch.sum(probs * torch.log(probs + float(eps)), dim=-1)
    top_probs, top_idx = torch.topk(probs, k=2, dim=-1)
    top1_cm = distance_grid_cm[top_idx[:, 0]]
    top2_cm = distance_grid_cm[top_idx[:, 1]]
    margin = top_probs[:, 0] - top_probs[:, 1]
    validity_prob = torch.sigmoid(validity_logit)
    return {
        "distance_probs": probs,
        "measurement_distance_cm": mean_cm,
        "measurement_var_from_logits": var_cm2,
        "measurement_logvar": logvar,
        "measurement_entropy": entropy,
        "measurement_top1_cm": top1_cm,
        "measurement_top2_cm": top2_cm,
        "measurement_margin": margin,
        "measurement_validity_prob": validity_prob,
    }


class UAVCombObserver(nn.Module):
    def __init__(self, cfg: Dict[str, Any], scalar_dim: int, use_stpacc: bool) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        dropout = float(model_cfg["dropout"])
        self.use_stpacc = bool(use_stpacc)

        self.phase_branch = SpectrogramBranch(
            in_channels=3,
            channels=list(model_cfg["phase_branch_channels"]),
            out_dim=int(model_cfg["phase_out_dim"]),
            dropout=dropout,
        )
        self.comb_branch = SpectrogramBranch(
            in_channels=3,
            channels=list(model_cfg["comb_branch_channels"]),
            out_dim=int(model_cfg["comb_out_dim"]),
            dropout=dropout,
        )
        self.scalar_branch = ScalarSequenceBranch(
            in_dim=scalar_dim * 3,
            hidden_dim=int(model_cfg["scalar_hidden"]),
            out_dim=int(model_cfg["scalar_out_dim"]),
            dropout=dropout,
        )
        fused_dim = int(model_cfg["phase_out_dim"] + model_cfg["comb_out_dim"] + model_cfg["scalar_out_dim"])
        if self.use_stpacc:
            self.stpacc_branch = SpectrogramBranch(
                in_channels=1,
                channels=list(model_cfg["stp_branch_channels"]),
                out_dim=int(model_cfg["stp_out_dim"]),
                dropout=dropout,
            )
            fused_dim += int(model_cfg["stp_out_dim"])
        else:
            self.stpacc_branch = None

        distance_grid_cm = build_distance_grid_cm(cfg)
        self.register_buffer("distance_grid_cm", distance_grid_cm, persistent=False)
        num_candidates = int(distance_grid_cm.numel())
        temporal_in = fused_dim
        self.temporal = CausalTCN(
            in_channels=temporal_in,
            channels=list(model_cfg["tcn_channels"]),
            kernel_size=int(model_cfg["tcn_kernel_size"]),
            dilations=list(model_cfg["tcn_dilations"]),
            dropout=float(model_cfg["tcn_dropout"]),
        )
        last_dim = int(self.temporal.out_channels)
        self.distance_logits_head = nn.Linear(last_dim, num_candidates)
        self.validity_head = nn.Linear(last_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        phase_feat = self.phase_branch(batch["phase"])
        comb_feat = self.comb_branch(batch["comb"])
        scalar_in = torch.cat(
            [
                batch["scalar"] * batch["scalar_observed_mask"],
                batch["scalar_observed_mask"],
                batch["scalar_reliable_mask"],
            ],
            dim=-1,
        )
        scalar_feat = self.scalar_branch(scalar_in)
        learned = [phase_feat, comb_feat, scalar_feat]
        if self.stpacc_branch is not None:
            learned.append(self.stpacc_branch(batch["stpacc"]))
        learned_seq = torch.cat(learned, dim=-1)

        temporal_repr = self.temporal(learned_seq)
        last_repr = temporal_repr[:, -1, :]
        distance_logits = self.distance_logits_head(last_repr)
        validity_logit = self.validity_head(last_repr).squeeze(-1)
        distance_grid_cm = self.distance_grid_cm.to(device=distance_logits.device, dtype=distance_logits.dtype)

        derived = summarize_distance_logits(
            distance_logits=distance_logits,
            validity_logit=validity_logit,
            distance_grid_cm=distance_grid_cm,
            eps=1e-6,
        )
        return {
            "distance_logits": distance_logits,
            "measurement_validity_logit": validity_logit,
            **derived,
            "distance_grid_cm": distance_grid_cm,
            "temporal_repr": temporal_repr,
            "last_repr": last_repr,
            # Compatibility aliases, not used as primary supervision.
            "distance": derived["measurement_distance_cm"],
            "confidence": validity_logit,
        }
