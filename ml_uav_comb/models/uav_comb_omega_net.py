"""Per-bin temporal encoder → cross-frequency fusion → end-of-window omega regressor.

Architecture v2: GRU-based per-bin encoder with full temporal receptive field,
multi-head cross-frequency fusion with pairwise interaction, and increased capacity.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Temporal residual block operating on [B, C, T]. (Kept for legacy compat.)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(y + self.skip(x), inplace=True)


class PerBinGRUEncoder(nn.Module):
    """Bidirectional GRU per-bin temporal encoder.

    Unlike Conv1D (receptive field=9 for 2 layers), a bidirectional GRU sees the
    entire temporal window (68 frames) in one pass. This is critical for capturing
    the direction and rate of moving-pattern progression.

    Input:  [B, C_in, T, F]
    Output: [B, D, T, F]  where D = hidden_size (bidirectional outputs are projected)

    NOTE: Very memory-hungry with large batch sizes (B*F sequences).
    Prefer DilatedTCNEncoder for GPU-limited settings.
    """

    def __init__(self, in_channels: int, hidden_size: int, num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_channels = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B * F, T, C)
        gru_out, _ = self.gru(x)
        out = self.norm(self.proj(gru_out))
        out = out.reshape(B, F, T, self.hidden_size).permute(0, 3, 2, 1)
        return out


class DilatedCausalConv1D(nn.Module):
    """Single dilated causal conv layer with residual connection (legacy)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                              dilation=dilation, padding=0)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.pad, 0))
        y = self.conv(padded)
        y = F.gelu(self.norm(y))
        y = self.drop(y)
        return x + y


class DilatedCausalConv2DPerBin(nn.Module):
    """Dilated causal conv operating per-bin via Conv2d(k,1).

    Avoids the B*F memory explosion of the Conv1d approach by using 2D
    convolutions with kernel (k_time, 1_freq), keeping batch dim at B.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                              dilation=(dilation, 1), padding=0)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        padded = F.pad(x, (0, 0, self.pad, 0))  # pad T only (left)
        y = self.conv(padded)
        y = F.gelu(self.norm(y))
        y = self.drop(y)
        return x + y


class DilatedTCNEncoder(nn.Module):
    """Dilated TCN per-bin encoder using Conv2d(k,1) for memory efficiency.

    Processes each frequency bin independently along the time axis using
    Conv2d with (k_time, 1) kernels. Mathematically equivalent to Conv1d
    per-bin but avoids the B*F memory explosion.

    Input:  [B, C_in, T, F]
    Output: [B, D, T, F]
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 dilations: List[int] = None, kernel_size: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            DilatedCausalConv2DPerBin(hidden_channels, kernel_size, d, dropout)
            for d in dilations
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.out_channels = hidden_channels

        rf = 1
        for d in dilations:
            rf += (kernel_size - 1) * d
        self._receptive_field = rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, F] — process directly as 2D, no B*F reshape
        x = self.input_proj(x)
        for layer in self.layers:
            if self.training and x.requires_grad:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.output_proj(x)
        return x  # [B, D, T, F]


class PerBinTemporalEncoder(nn.Module):
    """Conv1D per-bin encoder (legacy). Use PerBinGRUEncoder for new training."""

    def __init__(self, in_channels: int, temporal_channels: List[int]) -> None:
        super().__init__()
        layers = []
        ch_in = in_channels
        for ch_out in temporal_channels:
            layers.append(ResidualBlock1D(ch_in, ch_out))
            ch_in = ch_out
        self.net = nn.Sequential(*layers)
        self.out_channels = temporal_channels[-1] if temporal_channels else in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * F, C, T)
        x = self.net(x)
        D = x.shape[1]
        x = x.reshape(B, F, D, T).permute(0, 2, 3, 1)
        return x


class CrossFrequencyFusion(nn.Module):
    """Fuse per-bin temporal features across frequency dimension.

    Input:  [B, D, T, F]
    Output: {"time_feat": [B, D_out, T]}

    v3: Gradient-friendly design. Uses mean pool + learned linear projection
    instead of softmax attention (which killed gradients in v2: 6x attenuation).
    Frequency mixing via 1×1 conv is retained but simplified.
    """

    def __init__(self, in_channels: int, hidden: int, num_freq_bins: int,
                 n_heads: int = 4) -> None:
        super().__init__()
        self.num_freq_bins = num_freq_bins
        # Learnable frequency position embedding
        self.freq_embed = nn.Parameter(torch.randn(1, 1, 1, num_freq_bins) * 0.02)
        # Pairwise frequency interaction via 1×1 convs + residual
        self.freq_mix = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
        )
        self.freq_norm = nn.GroupNorm(min(8, in_channels), in_channels)
        # Gradient-friendly pooling: learned linear projection F→1 (no softmax)
        # This preserves gradient magnitude 1:1 instead of dividing by F
        self.freq_proj = nn.Linear(num_freq_bins, 1, bias=False)
        # Initialize to mean pooling so starting point is reasonable
        nn.init.constant_(self.freq_proj.weight, 1.0 / num_freq_bins)
        self.post_fusion = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
        )
        self.out_channels = in_channels

    def forward(
        self,
        feat: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, D, T, F = feat.shape
        freq_pos = self.freq_embed.expand(B, 1, T, F)

        # Frequency mixing: cross-frequency interaction via 1×1 conv + residual
        mix_input = torch.cat([feat, freq_pos], dim=1)  # [B, D+1, T, F]
        feat = feat + self.freq_norm(self.freq_mix(mix_input))  # residual

        # Gradient-friendly pooling: learned linear projection across F
        # feat: [B, D, T, F] → permute to [B, D, T, F] then linear on last dim
        pooled = self.freq_proj(feat).squeeze(-1)  # [B, D, T]
        pooled = self.post_fusion(pooled)

        out: Dict[str, torch.Tensor] = {"time_feat": pooled}
        if return_debug:
            # Report effective weights for visualization
            out["cross_freq_attention_weights"] = self.freq_proj.weight.detach().expand(B, T, F)
        return out


class TemporalStateAggregator(nn.Module):
    """Causal temporal self-attention for explicit latent state modeling.

    Each temporal position attends to itself and all earlier positions,
    producing a state representation that integrates information from the
    full observable history within the window.

    Input:  [B, D, T]
    Output: [B, D, T]
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, time_feat: torch.Tensor) -> torch.Tensor:
        x = time_feat.permute(0, 2, 1)  # [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)
        attn_out = self.proj(attn_out)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x.permute(0, 2, 1)  # [B, D, T]


class EndOfWindowOmegaHead(nn.Module):
    """Multi-output head: omega regression + pattern + observability + uncertainty."""

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
        # Phase 2: observability prediction (continuous score, gating for inference)
        self.observability_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Phase 3: uncertainty estimation (log-variance for heteroscedastic loss)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, time_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        refined = self.temporal(time_feat)
        last_feat = refined[:, :, -1]
        omega_raw = self.omega_head(last_feat).squeeze(-1)
        pattern_logit = self.pattern_head(last_feat).squeeze(-1)
        observability_logit = self.observability_head(last_feat).squeeze(-1)
        log_variance = self.uncertainty_head(last_feat).squeeze(-1).clamp(-5.0, 5.0)
        # tanh-based output: better gradient flow than sigmoid near boundaries
        mid = 0.5 * (self.omega_min + self.omega_max)
        half_range = 0.5 * (self.omega_max - self.omega_min)
        omega_pred = mid + half_range * torch.tanh(omega_raw)
        return {
            "omega_pred": omega_pred,
            "pattern_logit": pattern_logit,
            "pattern_prob": torch.sigmoid(pattern_logit),
            "observability_logit": observability_logit,
            "observability_pred": F.softplus(observability_logit),
            "log_variance": log_variance,
        }


# Keep old classes available for loading legacy checkpoints
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
        omega_min = float(model_cfg["distance_cm_min"]) * (4.0 * math.pi / (343.0 * 100.0))
        omega_max = float(model_cfg["distance_cm_max"]) * (4.0 * math.pi / (343.0 * 100.0))

        # Input channels: 4 dynamic channels (log_mag_band, dt1, dt_long, abs_dt1)
        self.input_channels = int(model_cfg.get("input_channels", 4))

        # Cross-frequency fusion config
        audio_cfg = cfg["audio"]
        fft_freqs = torch.fft.rfftfreq(int(audio_cfg["n_fft"]), d=1.0 / float(audio_cfg["target_sr"]))
        freq_mask = (fft_freqs >= float(audio_cfg["freq_min"])) & (fft_freqs <= float(audio_cfg["freq_max"]))
        self.register_buffer("frequency_bins_hz", fft_freqs[freq_mask].to(dtype=torch.float32), persistent=True)
        num_freq_bins = int(freq_mask.sum().item())

        # Per-bin encoder: TCN (default), GRU, or legacy Conv1D
        encoder_type = str(model_cfg.get("per_bin_encoder_type", "tcn"))
        if encoder_type == "gru":
            gru_hidden = int(model_cfg.get("per_bin_gru_hidden", 128))
            gru_layers = int(model_cfg.get("per_bin_gru_layers", 2))
            gru_dropout = float(model_cfg.get("per_bin_gru_dropout", 0.1))
            self.per_bin_encoder = PerBinGRUEncoder(
                in_channels=self.input_channels,
                hidden_size=gru_hidden,
                num_layers=gru_layers,
                dropout=gru_dropout,
            )
        elif encoder_type == "tcn":
            tcn_hidden = int(model_cfg.get("per_bin_tcn_hidden", 128))
            tcn_dilations = list(model_cfg.get("per_bin_tcn_dilations", [1, 2, 4, 8, 16]))
            tcn_kernel = int(model_cfg.get("per_bin_tcn_kernel", 3))
            tcn_dropout = float(model_cfg.get("per_bin_tcn_dropout", 0.1))
            self.per_bin_encoder = DilatedTCNEncoder(
                in_channels=self.input_channels,
                hidden_channels=tcn_hidden,
                dilations=tcn_dilations,
                kernel_size=tcn_kernel,
                dropout=tcn_dropout,
            )
        else:
            temporal_channels: List[int] = list(model_cfg.get("per_bin_temporal_channels", [32, 64]))
            if not temporal_channels:
                temporal_channels = [32, 64]
            self.per_bin_encoder = PerBinTemporalEncoder(
                in_channels=self.input_channels,
                temporal_channels=temporal_channels,
            )

        feat_dim = self.per_bin_encoder.out_channels
        cross_freq_hidden = int(model_cfg.get("cross_freq_hidden", max(32, feat_dim // 2)))
        cross_freq_heads = int(model_cfg.get("cross_freq_heads", 4))

        self.cross_freq_fusion = CrossFrequencyFusion(
            in_channels=feat_dim,
            hidden=cross_freq_hidden,
            num_freq_bins=num_freq_bins,
            n_heads=cross_freq_heads,
        )

        # Phase 3: Temporal state aggregation (causal self-attention)
        # DISABLED by default: causes 150x gradient attenuation (vanishing gradients)
        # TCN with RF=55 already covers temporal context
        use_temporal_state = bool(model_cfg.get("use_temporal_state", False))
        if use_temporal_state:
            n_heads = int(model_cfg.get("temporal_state_heads", 4))
            state_dropout = float(model_cfg.get("temporal_state_dropout", 0.1))
            self.temporal_state = TemporalStateAggregator(
                d_model=feat_dim, n_heads=n_heads, dropout=state_dropout,
            )
        else:
            self.temporal_state = None

        regression_hidden = int(model_cfg.get("regression_hidden", max(64, feat_dim)))
        self.regressor = EndOfWindowOmegaHead(
            feat_dim,
            regression_hidden,
            omega_min,
            omega_max,
        )

    def _prepare_inputs(
        self,
        batch_or_x: Dict[str, torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(batch_or_x, dict):
            x = batch_or_x["x"]
        else:
            x = batch_or_x

        if x.ndim != 4:
            raise ValueError(f"expected x shape [B,C,T,F], got {tuple(x.shape)}")

        # Handle legacy single-channel input [B,1,T,F]
        if x.shape[1] == 1 and self.input_channels > 1:
            x = x.expand(-1, self.input_channels, -1, -1)

        return x

    def forward(
        self,
        batch_or_x: Dict[str, torch.Tensor] | torch.Tensor,
        frequencies_hz: torch.Tensor | None = None,
        *,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = self._prepare_inputs(batch_or_x)

        # Per-bin temporal encoding: [B, C, T, F] → [B, D, T, F]
        per_bin_feat = self.per_bin_encoder(x)

        # Phase 2: Cross-frequency consistency loss (cosine-based)
        if self.training:
            norm_feat = F.normalize(per_bin_feat, dim=1, eps=1e-5)
            consensus = F.normalize(norm_feat.mean(dim=-1, keepdim=True), dim=1, eps=1e-5)
            cos_sim = (norm_feat * consensus).sum(dim=1)  # [B, T, F]
            cross_freq_consistency = (1.0 - cos_sim).mean()
        else:
            cross_freq_consistency = per_bin_feat.new_tensor(0.0)

        # Cross-frequency fusion: [B, D, T, F] → {"time_feat": [B, D, T]}
        fused = self.cross_freq_fusion(per_bin_feat, return_debug=return_debug)
        time_feat = fused["time_feat"]

        # Phase 3: Temporal state aggregation
        if self.temporal_state is not None:
            time_feat = self.temporal_state(time_feat)

        # End-of-window regression
        out = self.regressor(time_feat)
        out["cross_freq_consistency"] = cross_freq_consistency

        if return_debug:
            if "cross_freq_attention_weights" in fused:
                out["cross_freq_attention_weights"] = fused["cross_freq_attention_weights"]
            out["per_bin_feature_map"] = per_bin_feat
        return out
