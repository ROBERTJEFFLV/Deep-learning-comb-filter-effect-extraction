"""Loss functions for likelihood-first observer training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def likelihood_ce_loss(
    distance_logits: torch.Tensor,
    target_grid: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    valid = train_mask > 0.5
    if not torch.any(valid):
        return distance_logits.new_tensor(0.0)
    logits = distance_logits[valid]
    target = target_grid[valid]
    target = target / torch.clamp(target.sum(dim=-1, keepdim=True), min=1e-6)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target * log_probs).sum(dim=-1)
    return torch.mean(loss)


def measurement_mean_loss(
    measurement_distance_cm: torch.Tensor,
    target_distance_cm: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    valid = train_mask > 0.5
    if not torch.any(valid):
        return measurement_distance_cm.new_tensor(0.0)
    return F.smooth_l1_loss(measurement_distance_cm[valid], target_distance_cm[valid])


def measurement_validity_loss(
    validity_logit: torch.Tensor,
    target_validity: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    valid = train_mask > 0.5
    if not torch.any(valid):
        return validity_logit.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(validity_logit[valid], target_validity[valid])


def combined_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_likelihood_ce: float,
    lambda_measurement_mean: float,
    lambda_measurement_validity: float,
    teacher_consistency_weight: float = 0.0,
    teacher_conf_threshold: float = 0.8,
) -> Dict[str, torch.Tensor]:
    del teacher_consistency_weight, teacher_conf_threshold
    loss_likelihood = likelihood_ce_loss(
        distance_logits=pred["distance_logits"],
        target_grid=batch["distance_target_grid"],
        train_mask=batch["measurement_distance_train_mask"],
    )
    loss_mean = measurement_mean_loss(
        measurement_distance_cm=pred["measurement_distance_cm"],
        target_distance_cm=batch["measurement_distance_target_cm"],
        train_mask=batch["measurement_distance_train_mask"],
    )
    loss_validity = measurement_validity_loss(
        validity_logit=pred["measurement_validity_logit"],
        target_validity=batch["measurement_validity_target"],
        train_mask=batch["measurement_validity_train_mask"],
    )
    total = (
        float(lambda_likelihood_ce) * loss_likelihood
        + float(lambda_measurement_mean) * loss_mean
        + float(lambda_measurement_validity) * loss_validity
    )
    return {
        "total": total,
        "likelihood_ce": loss_likelihood,
        "measurement_mean": loss_mean,
        "measurement_validity": loss_validity,
    }
