"""Multi-branch CRNN for UAV comb-motion distance estimation."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from ml_uav_comb.models.branches import ScalarSequenceBranch, SpectrogramBranch
from ml_uav_comb.models.heads import MLPHead


class UAVCombCRNN(nn.Module):
    def __init__(self, cfg: Dict[str, Any], scalar_dim: int, use_stpacc: bool) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        dropout = float(model_cfg["dropout"])
        self.use_stpacc = bool(use_stpacc)
        self.scalar_dim = int(scalar_dim)

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

        self.temporal = nn.GRU(
            input_size=fused_dim,
            hidden_size=int(model_cfg["rnn_hidden"]),
            num_layers=int(model_cfg["rnn_layers"]),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if int(model_cfg["rnn_layers"]) > 1 else 0.0,
        )
        center_dim = int(model_cfg["rnn_hidden"]) * 2
        self.distance_head = MLPHead(center_dim, 128, 1)
        self.sign_head = MLPHead(center_dim, 128, int(model_cfg["sign_classes"]))
        self.confidence_head = MLPHead(center_dim, 128, 1)

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
        fused = [phase_feat, comb_feat, scalar_feat]
        if self.stpacc_branch is not None:
            fused.append(self.stpacc_branch(batch["stpacc"]))

        fused_feat = torch.cat(fused, dim=-1)
        temporal, _ = self.temporal(fused_feat)
        center_idx = temporal.shape[1] // 2
        center_repr = temporal[:, center_idx, :]
        return {
            "distance": self.distance_head(center_repr).squeeze(-1),
            "sign_logits": self.sign_head(center_repr),
            "confidence": self.confidence_head(center_repr).squeeze(-1),
            "center_repr": center_repr,
        }
