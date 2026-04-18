"""Causal temporal convolution network blocks."""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CausalTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(float(dropout))
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.activation(self.conv1(x))
        out = self.dropout(out)
        out = self.activation(self.conv2(out))
        out = self.dropout(out)
        return self.activation(out + residual)


class CausalTCN(nn.Module):
    """Causal TCN over sequences of shape [B, T, C]."""

    def __init__(
        self,
        in_channels: int,
        channels: Iterable[int],
        kernel_size: int,
        dilations: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        channels_list = [int(v) for v in channels]
        dilation_list = [int(v) for v in dilations]
        if not channels_list:
            raise ValueError("channels must not be empty")
        if not dilation_list:
            raise ValueError("dilations must not be empty")

        blocks: List[nn.Module] = []
        prev = int(in_channels)
        for idx, out in enumerate(channels_list):
            dilation = dilation_list[idx] if idx < len(dilation_list) else dilation_list[-1]
            blocks.append(
                CausalTCNBlock(
                    in_channels=prev,
                    out_channels=int(out),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
            prev = int(out)
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] -> [B, C, T]
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        return out.transpose(1, 2)
