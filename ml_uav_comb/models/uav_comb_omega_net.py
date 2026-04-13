"""Small CNN + end-of-window omega regressor with frequency-aware pooling."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride) if (in_ch != out_ch or stride != (1, 1)) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(y + self.skip(x), inplace=True)


class EndOfWindowOmegaHead(nn.Module):
    """Use the full 68-frame context, but regress only the aligned window-end omega."""

    def __init__(self, channels: int, hidden: int, omega_min: float, omega_max: float) -> None:
        super().__init__()
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.omega_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.pattern_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, time_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        refined = self.temporal(time_feat)
        last_feat = refined[:, :, -1]
        omega_raw = self.omega_head(last_feat).squeeze(-1)
        pattern_logit = self.pattern_head(last_feat).squeeze(-1)
        omega_pred = self.omega_min + (self.omega_max - self.omega_min) * torch.sigmoid(omega_raw)
        return {
            "omega_pred": omega_pred,
            "pattern_logit": pattern_logit,
            "pattern_prob": torch.sigmoid(pattern_logit),
        }


class FrequencyAwareAggregator(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        hidden = max(8, int(hidden))
        self.attn = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(
        self,
        feat: torch.Tensor,
        freq_coord_map: torch.Tensor,
        freq_hz_map: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        target_size = feat.shape[-2:]
        if freq_coord_map.shape[-2:] != target_size:
            freq_coord_map = F.interpolate(freq_coord_map, size=target_size, mode="bilinear", align_corners=False)
        if freq_hz_map.shape[-2:] != target_size:
            freq_hz_map = F.interpolate(freq_hz_map, size=target_size, mode="bilinear", align_corners=False)

        attn_logits = self.attn(torch.cat([feat, freq_coord_map], dim=1)).squeeze(1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(feat * attn_weights.unsqueeze(1), dim=-1)
        pooled_freq_hz = torch.sum(freq_hz_map.squeeze(1) * attn_weights, dim=-1)
        out = {
            "time_feat": pooled,
        }
        if return_debug:
            out["frequency_pool_weights"] = attn_weights
            out["pooled_frequency_hz"] = pooled_freq_hz
        return out


class UAVCombOmegaNet(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        channels: List[int] = list(model_cfg.get("encoder_channels", [16, 32, 64]))
        if not channels:
            raise ValueError("model.encoder_channels must not be empty")
        omega_min = float(model_cfg["distance_cm_min"]) * (4.0 * math.pi / (343.0 * 100.0))
        omega_max = float(model_cfg["distance_cm_max"]) * (4.0 * math.pi / (343.0 * 100.0))
        self.frequency_coord_mode = str(model_cfg.get("frequency_coord_mode", "standardized")).lower()
        audio_cfg = cfg["audio"]
        fft_freqs = torch.fft.rfftfreq(int(audio_cfg["n_fft"]), d=1.0 / float(audio_cfg["target_sr"]))
        freq_mask = (fft_freqs >= float(audio_cfg["freq_min"])) & (fft_freqs <= float(audio_cfg["freq_max"]))
        self.register_buffer("frequency_bins_hz", fft_freqs[freq_mask].to(dtype=torch.float32), persistent=True)
        self.stem = nn.Conv2d(2, channels[0], kernel_size=3, padding=1)
        blocks = []
        in_ch = channels[0]
        for idx, out_ch in enumerate(channels):
            # Preserve the full 68-step time axis and only compress along frequency.
            stride = (1, 1) if idx == 0 else (1, 2)
            blocks.append(ResidualBlock2D(in_ch, out_ch, stride=stride))
            blocks.append(ResidualBlock2D(out_ch, out_ch, stride=(1, 1)))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.frequency_pool = FrequencyAwareAggregator(
            in_ch,
            int(model_cfg.get("frequency_pool_hidden", max(32, in_ch // 2))),
        )
        self.regressor = EndOfWindowOmegaHead(
            in_ch,
            int(model_cfg.get("regression_hidden", max(32, in_ch))),
            omega_min,
            omega_max,
        )

    def _expand_frequency_grid(self, frequencies_hz: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if frequencies_hz.ndim == 1:
            if int(frequencies_hz.shape[0]) != int(x.shape[-1]):
                raise ValueError(f"frequencies_hz length mismatch: expected {x.shape[-1]}, got {frequencies_hz.shape[0]}")
            frequencies_hz = frequencies_hz.unsqueeze(0).expand(x.shape[0], -1)
        elif frequencies_hz.ndim == 2:
            if frequencies_hz.shape[0] == 1 and x.shape[0] > 1:
                frequencies_hz = frequencies_hz.expand(x.shape[0], -1)
            elif frequencies_hz.shape[0] != x.shape[0]:
                raise ValueError(
                    f"frequencies_hz batch mismatch: expected {x.shape[0]}, got {frequencies_hz.shape[0]}"
                )
            if int(frequencies_hz.shape[1]) != int(x.shape[-1]):
                raise ValueError(
                    f"frequencies_hz length mismatch: expected {x.shape[-1]}, got {frequencies_hz.shape[1]}"
                )
        else:
            raise ValueError(f"expected frequencies_hz shape [F] or [B,F], got {tuple(frequencies_hz.shape)}")
        freq = frequencies_hz.to(device=x.device, dtype=x.dtype)
        return freq[:, None, None, :].expand(-1, 1, x.shape[-2], -1)

    def _encode_frequency_coordinates(self, freq_hz_map: torch.Tensor) -> torch.Tensor:
        if self.frequency_coord_mode == "raw":
            return freq_hz_map
        if self.frequency_coord_mode == "minmax":
            min_val = freq_hz_map.amin(dim=-1, keepdim=True)
            max_val = freq_hz_map.amax(dim=-1, keepdim=True)
            denom = (max_val - min_val).clamp_min(1e-6)
            return 2.0 * (freq_hz_map - min_val) / denom - 1.0
        mean = freq_hz_map.mean(dim=-1, keepdim=True)
        std = freq_hz_map.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (freq_hz_map - mean) / std

    def _prepare_inputs(
        self,
        batch_or_x: Dict[str, torch.Tensor] | torch.Tensor,
        frequencies_hz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_or_x, dict):
            x = batch_or_x["x"]
        else:
            x = batch_or_x

        if x.ndim != 4:
            raise ValueError(f"expected x shape [B,C,T,F], got {tuple(x.shape)}")

        resolved_frequencies = self.frequency_bins_hz if frequencies_hz is None else frequencies_hz

        if x.shape[1] == 1:
            freq_hz_map = self._expand_frequency_grid(resolved_frequencies, x)
            freq_coord_map = self._encode_frequency_coordinates(freq_hz_map)
            model_input = torch.cat([x, freq_coord_map], dim=1)
            return model_input, freq_coord_map, freq_hz_map

        if x.shape[1] == 2:
            freq_coord_map = x[:, 1:2]
            freq_hz_map = self._expand_frequency_grid(resolved_frequencies, x[:, :1])
            return x, freq_coord_map, freq_hz_map

        raise ValueError(f"expected channel count 1 or 2, got {x.shape[1]}")

    def forward(
        self,
        batch_or_x: Dict[str, torch.Tensor] | torch.Tensor,
        frequencies_hz: torch.Tensor | None = None,
        *,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x, freq_coord_map, freq_hz_map = self._prepare_inputs(batch_or_x, frequencies_hz=frequencies_hz)
        feat = F.relu(self.stem(x), inplace=True)
        feat = self.encoder(feat)
        pooled = self.frequency_pool(feat, freq_coord_map=freq_coord_map, freq_hz_map=freq_hz_map, return_debug=return_debug)
        out = self.regressor(pooled["time_feat"])
        if return_debug:
            out.update(
                {
                    "frequency_pool_weights": pooled["frequency_pool_weights"],
                    "pooled_frequency_hz": pooled["pooled_frequency_hz"],
                    "encoded_feature_map": feat,
                }
            )
        return out
