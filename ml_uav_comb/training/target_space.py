"""Distance target-space transforms shared across export, train, and eval."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is available in runtime/tests
    torch = None


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def encode_distance_target(value: Any, target_space: str, eps: float = 1e-3) -> Any:
    if _is_torch_tensor(value):
        clipped = torch.clamp(value, min=eps)
        if target_space == "raw":
            return value
        if target_space == "log":
            return torch.log(clipped)
        if target_space == "inverse":
            return 1.0 / clipped
        raise ValueError(f"unsupported target_space: {target_space}")

    array = np.asarray(value)
    clipped = np.clip(array, eps, None)
    if target_space == "raw":
        return array
    if target_space == "log":
        return np.log(clipped)
    if target_space == "inverse":
        return 1.0 / clipped
    raise ValueError(f"unsupported target_space: {target_space}")


def decode_distance_target(value: Any, target_space: str, eps: float = 1e-3) -> Any:
    if _is_torch_tensor(value):
        if target_space == "raw":
            return value
        if target_space == "log":
            return torch.exp(value)
        if target_space == "inverse":
            denom = torch.clamp(value, min=eps)
            return 1.0 / denom
        raise ValueError(f"unsupported target_space: {target_space}")

    array = np.asarray(value)
    if target_space == "raw":
        return array
    if target_space == "log":
        return np.exp(array)
    if target_space == "inverse":
        denom = np.clip(array, eps, None)
        return 1.0 / denom
    raise ValueError(f"unsupported target_space: {target_space}")

