"""Deterministic physical likelihood builder for raw physical inputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_spacing_grid_hz(cfg: Dict[str, Any]) -> np.ndarray:
    phys_cfg = cfg["physical_likelihood"] if "physical_likelihood" in cfg else cfg
    spacing_min = float(phys_cfg["spacing_hz_min"])
    spacing_max = float(phys_cfg["spacing_hz_max"])
    num_candidates = int(phys_cfg["num_candidates"])
    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2")
    return np.linspace(spacing_min, spacing_max, num_candidates, dtype=np.float32)


def spacing_to_distance_cm(spacing_hz: np.ndarray, sound_speed_m_s: float) -> np.ndarray:
    spacing_hz = np.asarray(spacing_hz, dtype=np.float32)
    return (100.0 * float(sound_speed_m_s) / (2.0 * np.maximum(spacing_hz, 1e-6))).astype(np.float32)


def distance_to_spacing_hz(distance_cm: float, sound_speed_m_s: float) -> float:
    distance_cm = float(distance_cm)
    if not np.isfinite(distance_cm) or distance_cm <= 0.0:
        return float("nan")
    return float(100.0 * float(sound_speed_m_s) / (2.0 * distance_cm))


def gaussian_soft_target_numpy(
    spacing_grid_hz: np.ndarray,
    target_spacing_hz: float,
    sigma_hz: float,
    eps: float = 1e-6,
) -> np.ndarray:
    spacing_grid_hz = np.asarray(spacing_grid_hz, dtype=np.float32)
    if not np.isfinite(target_spacing_hz):
        return np.zeros_like(spacing_grid_hz, dtype=np.float32)
    sigma_hz = max(float(sigma_hz), 1e-6)
    diff = (spacing_grid_hz - float(target_spacing_hz)) / sigma_hz
    weights = np.exp(-0.5 * np.square(diff)).astype(np.float32) + float(eps)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(spacing_grid_hz, dtype=np.float32)
    return (weights / total).astype(np.float32)


def gaussian_soft_target_torch(
    spacing_grid_hz: torch.Tensor,
    target_spacing_hz: torch.Tensor,
    sigma_hz: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    sigma_hz = max(float(sigma_hz), 1e-6)
    diff = (spacing_grid_hz.unsqueeze(0) - target_spacing_hz.unsqueeze(-1)) / sigma_hz
    weights = torch.exp(-0.5 * diff * diff) + float(eps)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    denom = torch.clamp(weights.sum(dim=-1, keepdim=True), min=float(eps))
    return weights / denom


def _scalar_index(cfg: Dict[str, Any], field_name: str) -> int:
    fields = list(cfg["features"]["acoustic_scalar_fields"])
    for idx, name in enumerate(fields):
        if str(name) == str(field_name):
            return int(idx)
    return -1


class DeterministicPhysicalLikelihoodBuilder(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.phys_cfg = cfg["physical_likelihood"]
        self.sound_speed_m_s = float(self.phys_cfg["sound_speed_m_s"])
        self.spacing_grid_np = build_spacing_grid_hz(cfg)
        self.distance_grid_np = spacing_to_distance_cm(self.spacing_grid_np, self.sound_speed_m_s)
        self.register_buffer("spacing_grid_hz", torch.from_numpy(self.spacing_grid_np), persistent=False)
        self.register_buffer("distance_grid_cm", torch.from_numpy(self.distance_grid_np), persistent=False)
        self.likelihood_eps = float(self.phys_cfg.get("likelihood_eps", 1e-6))
        self.use_subbands = bool(self.phys_cfg.get("use_subbands", True))
        self.subband_edges_hz = [float(v) for v in self.phys_cfg.get("subband_edges_hz", [])]
        self.use_stpacc_projection = bool(self.phys_cfg.get("use_stpacc_projection", True))
        self.is_sound_idx = _scalar_index(cfg, "is_sound_present")
        self.rho_idx = _scalar_index(cfg, "comb_shift_rho")
        self._basis_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
        self._subband_cache: Dict[Tuple[str, str, int], List[torch.Tensor]] = {}

    def _periodicity_basis(self, frequencies_hz: torch.Tensor) -> torch.Tensor:
        device = frequencies_hz.device
        dtype = frequencies_hz.dtype
        key = (str(device), str(dtype), int(frequencies_hz.numel()))
        if key in self._basis_cache:
            return self._basis_cache[key]

        spacing = self.spacing_grid_hz.to(device=device, dtype=dtype)
        freq = frequencies_hz.reshape(1, -1).to(device=device, dtype=dtype)
        pattern = 0.5 * (1.0 + torch.cos(2.0 * np.pi * (freq / torch.clamp(spacing[:, None], min=1e-6))))
        pattern = pattern / torch.clamp(pattern.mean(dim=-1, keepdim=True), min=self.likelihood_eps)
        self._basis_cache[key] = pattern
        return pattern

    def _subband_masks(self, frequencies_hz: torch.Tensor) -> List[torch.Tensor]:
        device = frequencies_hz.device
        dtype = frequencies_hz.dtype
        key = (str(device), str(dtype), int(frequencies_hz.numel()))
        if key in self._subband_cache:
            return self._subband_cache[key]

        freq = frequencies_hz.reshape(-1)
        if not self.use_subbands:
            self._subband_cache[key] = []
            return []

        masks: List[torch.Tensor] = []
        boundaries = [float(torch.min(freq).item())] + self.subband_edges_hz + [float(torch.max(freq).item()) + 1.0]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            mask = ((freq >= float(lo)) & (freq < float(hi))).to(dtype=dtype)
            if torch.sum(mask) > 0:
                masks.append(mask)
        self._subband_cache[key] = masks
        return masks

    def forward(
        self,
        *,
        phase: torch.Tensor,
        comb: torch.Tensor,
        scalar: torch.Tensor,
        scalar_observed_mask: torch.Tensor,
        scalar_reliable_mask: torch.Tensor,
        stpacc: torch.Tensor | None,
        frequencies_hz: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # phase/comb: [B, C, T, F], scalar/masks: [B, T, S], stpacc: [B, 1, T, L]
        # Inputs are expected to be raw physical quantities (not normalized).
        if frequencies_hz.ndim == 2:
            frequencies = frequencies_hz[0]
        else:
            frequencies = frequencies_hz
        frequencies = frequencies.to(device=phase.device, dtype=phase.dtype)

        log_mag = phase[:, 0, :, :]
        smooth_d1 = comb[:, 0, :, :]
        abs_d1 = torch.abs(comb[:, min(1, int(comb.shape[1]) - 1), :, :])
        log_mag_centered = log_mag - torch.mean(log_mag, dim=-1, keepdim=True)
        trough = torch.relu(-log_mag_centered)
        d1_strength = torch.relu(abs_d1) + 0.25 * torch.relu(torch.abs(smooth_d1))
        trough = trough * (1.0 + d1_strength)
        trough = trough / torch.clamp(torch.mean(trough, dim=-1, keepdim=True), min=self.likelihood_eps)

        basis = self._periodicity_basis(frequencies).to(device=phase.device, dtype=phase.dtype)
        periodic_score = torch.einsum("btf,df->btd", trough, basis) / float(max(int(trough.shape[-1]), 1))

        channels: List[torch.Tensor] = [periodic_score]
        for band_mask in self._subband_masks(frequencies):
            band = band_mask.to(device=phase.device, dtype=phase.dtype)
            denom = torch.clamp(torch.sum(band), min=1.0)
            band_score = torch.einsum("btf,df,f->btd", trough, basis, band) / denom
            channels.append(band_score)

        if self.use_stpacc_projection and stpacc is not None:
            stp = torch.relu(stpacc[:, 0, :, :])
            if stp.shape[-1] != basis.shape[0]:
                b, t, l = stp.shape
                stp = F.interpolate(
                    stp.reshape(b * t, 1, l),
                    size=int(basis.shape[0]),
                    mode="linear",
                    align_corners=False,
                ).reshape(b, t, int(basis.shape[0]))
            stp = stp / torch.clamp(torch.mean(stp, dim=-1, keepdim=True), min=self.likelihood_eps)
            channels.append(stp)

        obs_quality = torch.mean(scalar_observed_mask, dim=-1, keepdim=True).expand(-1, -1, basis.shape[0])
        rel_quality = torch.mean(scalar_reliable_mask, dim=-1, keepdim=True).expand(-1, -1, basis.shape[0])
        channels.append(obs_quality)
        channels.append(rel_quality)

        if self.is_sound_idx >= 0:
            is_sound = torch.clamp(
                scalar[:, :, self.is_sound_idx : self.is_sound_idx + 1],
                min=0.0,
                max=1.0,
            ).expand(-1, -1, basis.shape[0])
            channels.append(is_sound)
        else:
            is_sound = rel_quality

        if self.rho_idx >= 0:
            rho = torch.abs(
                torch.clamp(
                    scalar[:, :, self.rho_idx : self.rho_idx + 1],
                    min=-1.0,
                    max=1.0,
                )
            ).expand(-1, -1, basis.shape[0])
            channels.append(rho)

        raw_likelihood_seq = torch.stack(channels, dim=-1)
        raw_distance_logits_seq = torch.mean(raw_likelihood_seq, dim=-1)
        raw_distance_logits_seq = raw_distance_logits_seq * (0.5 + 0.5 * is_sound)

        return {
            "raw_likelihood_seq": raw_likelihood_seq,
            "raw_distance_logits_seq": raw_distance_logits_seq,
            "spacing_grid_hz": self.spacing_grid_hz.to(device=phase.device, dtype=phase.dtype),
            "distance_grid_cm": self.distance_grid_cm.to(device=phase.device, dtype=phase.dtype),
        }
