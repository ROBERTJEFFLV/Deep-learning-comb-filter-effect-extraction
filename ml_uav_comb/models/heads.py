"""Output heads for distance, sign, and confidence."""

from __future__ import annotations

from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, final_activation: nn.Module | None = None) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

