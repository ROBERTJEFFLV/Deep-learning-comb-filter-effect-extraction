"""RangeKF wrapper for distance-grid observer measurement tracking."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from processing.range_kf import RangeKF


class DistanceGridRangeTracker:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        filter_cfg = cfg.get("filter", {})
        kf_cfg = cfg.get("kf", {})
        self.enabled = bool(filter_cfg.get("enabled", True))
        self.validity_threshold = float(filter_cfg.get("validity_threshold", 0.5))
        self.invalid_r_scale = float(filter_cfg.get("invalid_R_scale", 25.0))
        self.entropy_r_scale = float(filter_cfg.get("entropy_R_scale", 4.0))
        self.margin_r_scale = float(filter_cfg.get("margin_R_scale", 2.0))
        self.min_r = float(filter_cfg.get("min_R", 1.0))
        self.max_r = float(filter_cfg.get("max_R", 10000.0))
        self.skip_update_when_invalid = bool(filter_cfg.get("skip_update_when_invalid", True))
        self.kf = RangeKF(
            R=float(kf_cfg.get("R", 4.0)),
            sigma_a=float(kf_cfg.get("sigma_a", 8.0)),
            v_max=float(kf_cfg.get("v_max", 8.0)),
            a_max=float(kf_cfg.get("a_max", 25.0)),
            d_min=0.0,
        )
        self.last_timestamp_sec: Optional[float] = None

    def _effective_r(
        self,
        measurement_logvar: float,
        validity_prob: float,
        entropy: float,
        margin: float,
    ) -> float:
        base_var = float(np.exp(np.clip(measurement_logvar, -20.0, 20.0)))
        if not np.isfinite(base_var) or base_var <= 0.0:
            base_var = self.max_r

        v = float(np.clip(validity_prob, 0.0, 1.0))
        validity_factor = 1.0
        if v < self.validity_threshold:
            denom = max(self.validity_threshold, 1e-6)
            validity_factor += self.invalid_r_scale * (self.validity_threshold - v) / denom

        entropy_factor = 1.0 + self.entropy_r_scale * max(float(entropy), 0.0)
        margin_factor = 1.0 + self.margin_r_scale * max(0.0, 1.0 - float(np.clip(margin, 0.0, 1.0)))
        r_eff = base_var * validity_factor * entropy_factor * margin_factor
        return float(np.clip(r_eff, self.min_r, self.max_r))

    def step(
        self,
        *,
        measurement_distance_cm: float,
        measurement_logvar: float,
        measurement_validity_prob: float,
        measurement_entropy: float,
        measurement_margin: float,
        timestamp_sec: float,
    ) -> Dict[str, float]:
        if self.last_timestamp_sec is not None and np.isfinite(timestamp_sec):
            dt = max(float(timestamp_sec - self.last_timestamp_sec), 1e-3)
            self.kf.predict(dt)
        self.last_timestamp_sec = float(timestamp_sec)

        if not self.enabled:
            measurement_used = 1.0
            r_eff = float(np.exp(np.clip(measurement_logvar, -20.0, 20.0)))
            self.kf.update(float(measurement_distance_cm), R_override=r_eff)
        else:
            r_eff = self._effective_r(
                measurement_logvar=float(measurement_logvar),
                validity_prob=float(measurement_validity_prob),
                entropy=float(measurement_entropy),
                margin=float(measurement_margin),
            )
            is_invalid = float(measurement_validity_prob) < self.validity_threshold
            if self.skip_update_when_invalid and is_invalid:
                measurement_used = 0.0
            else:
                self.kf.update(float(measurement_distance_cm), R_override=r_eff)
                measurement_used = 1.0

        covariance = self.kf.covariance
        return {
            "posterior_distance_cm": float("nan") if self.kf.distance is None else float(self.kf.distance),
            "posterior_velocity_cm_s": float("nan") if self.kf.velocity is None else float(self.kf.velocity),
            "posterior_covariance": float("nan")
            if covariance is None
            else float(np.asarray(covariance, dtype=np.float64)[0, 0]),
            "measurement_used_flag": float(measurement_used),
            "R_eff": float(r_eff),
        }


# Backward-compatible alias for older imports.
LikelihoodRangeTracker = DistanceGridRangeTracker
