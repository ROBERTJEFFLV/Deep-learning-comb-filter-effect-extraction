"""Loss functions for K+1 cepstral-bin classification."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute inverse-frequency class weights with smoothing.

    Args:
        targets: [N] integer class labels
        num_classes: K+1
        smoothing: Laplace smoothing count per class

    Returns:
        weights: [num_classes] normalized inverse-frequency weights
    """
    counts = torch.zeros(num_classes, dtype=torch.float32, device=targets.device)
    for c in range(num_classes):
        counts[c] = float((targets == c).sum()) + smoothing
    weights = counts.sum() / (num_classes * counts)
    return weights


def cepstral_bin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    soft_targets: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute classification loss for K+1 bin prediction.

    Args:
        logits: [B, K+1] raw logits
        targets: [B] integer class labels (hard, used when soft_targets is None)
        class_weights: [K+1] per-class weights (or None)
        label_smoothing: label smoothing factor (ignored when soft_targets given)
        focal_gamma: focal loss gamma (0 = standard CE)
        soft_targets: [B, K+1] pre-computed soft target distributions from dataset;
            when provided they replace label smoothing (pattern windows use Gaussian
            bin distributions, no-pattern windows fall back to hard one-hot).

    Returns:
        dict with "loss_total", "loss_ce", and auxiliary terms
    """
    n_classes = logits.shape[-1]

    if soft_targets is not None:
        # Use dataset-provided soft distributions directly
        target_one_hot = soft_targets.to(logits.device)
        # For no-pattern windows (target==0) soft_targets is None in dataset,
        # so they're collated as zeros — replace with hard one-hot
        no_pattern_mask = (targets == 0)
        if no_pattern_mask.any():
            hard_one_hot = F.one_hot(
                targets[no_pattern_mask], num_classes=n_classes
            ).float()
            target_one_hot = target_one_hot.clone()
            target_one_hot[no_pattern_mask] = hard_one_hot

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if focal_gamma > 0:
            focal_weight = (1 - probs) ** focal_gamma
            ce_per_class = -target_one_hot * log_probs
            weighted = focal_weight * ce_per_class
        else:
            weighted = -target_one_hot * log_probs

        if class_weights is not None:
            weighted = weighted * class_weights.unsqueeze(0)

        loss = weighted.sum(dim=-1).mean()

    elif focal_gamma > 0:
        # Focal loss with hard targets (+ optional label smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_one_hot = F.one_hot(targets, num_classes=n_classes).float()

        if label_smoothing > 0:
            target_one_hot = (
                target_one_hot * (1 - label_smoothing)
                + label_smoothing / n_classes
            )

        focal_weight = (1 - probs) ** focal_gamma
        ce_per_class = -target_one_hot * log_probs
        focal_per_class = focal_weight * ce_per_class

        if class_weights is not None:
            focal_per_class = focal_per_class * class_weights.unsqueeze(0)

        loss = focal_per_class.sum(dim=-1).mean()
    else:
        # Standard cross-entropy with optional label smoothing
        loss = F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    return {
        "loss_total": loss,
        "loss_ce": loss.detach(),
    }
