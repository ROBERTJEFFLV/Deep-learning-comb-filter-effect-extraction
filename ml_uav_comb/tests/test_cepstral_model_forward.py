"""Unit tests for CombCepstralNet (K+1 cepstral-bin classification).

Tests:
  - Forward pass shape and dtype for A2 (full model) and A1 (no GRU)
  - Loss backward pass (gradients flow through all modules)
  - Prior-based bias initialisation keeps p[no-pattern] ≈ 0.5
  - Cepstral geometry is self-consistent (bin centres are monotone)
  - MAD normalisation in the dataset is zero-mean on the median
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.models.comb_cepstral_net import CombCepstralNet, compute_cepstral_geometry
from ml_uav_comb.data_pipeline.cepstral_dataset import _compute_cepstral_patches


_CFG_PATH = "ml_uav_comb/configs/cepstral_tiny_debug.yaml"


def _make_batch(cfg, B: int = 2):
    geom = compute_cepstral_geometry(cfg)
    # window_frames is in the omega extractor config, not in 'data' section.
    # Default to 68 which matches schema v7 caches; tests only need a valid T.
    T = int(cfg.get("omega_extractor", {}).get("window_frames",
            cfg.get("cepstral", {}).get("window_frames", 68)))
    Q = geom["Q"]
    x = torch.randn(B, T, Q)
    return {"x": x}, geom


class TestCombCepstralNetForward(unittest.TestCase):
    def setUp(self):
        self.cfg = load_yaml_config(_CFG_PATH)

    def test_a2_full_model_output_shape(self):
        model = CombCepstralNet(self.cfg)
        batch, geom = _make_batch(self.cfg)
        out = model(batch)
        self.assertIn("logits", out)
        K1 = geom["K"] + 1
        self.assertEqual(out["logits"].shape, (2, K1))

    def test_a1_no_gru_output_shape(self):
        cfg = dict(self.cfg)
        cfg["model"] = dict(self.cfg["model"])
        cfg["model"]["use_temporal"] = False
        model = CombCepstralNet(cfg)
        batch, geom = _make_batch(cfg)
        out = model(batch)
        K1 = geom["K"] + 1
        self.assertEqual(out["logits"].shape, (2, K1))

    def test_output_dtype_is_float32(self):
        model = CombCepstralNet(self.cfg)
        batch, _ = _make_batch(self.cfg)
        out = model(batch)
        self.assertEqual(out["logits"].dtype, torch.float32)

    def test_bin_centers_registered_buffer(self):
        model = CombCepstralNet(self.cfg)
        geom = compute_cepstral_geometry(self.cfg)
        self.assertEqual(model.bin_centers_cm.shape[0], geom["K"])

    def test_gradients_flow_through_all_modules(self):
        model = CombCepstralNet(self.cfg)
        batch, _ = _make_batch(self.cfg)
        out = model(batch)
        loss = out["logits"].sum()
        loss.backward()
        grads_found = {
            name: param.grad is not None
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        missing = [n for n, ok in grads_found.items() if not ok]
        self.assertEqual(missing, [], f"No grad for: {missing}")

    def test_a1_gradients_flow(self):
        cfg = dict(self.cfg)
        cfg["model"] = dict(self.cfg["model"])
        cfg["model"]["use_temporal"] = False
        model = CombCepstralNet(cfg)
        batch, _ = _make_batch(cfg)
        out = model(batch)
        out["logits"].sum().backward()
        grads_found = [p.grad is not None for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(grads_found))


class TestPriorBiasInit(unittest.TestCase):
    def setUp(self):
        self.cfg = load_yaml_config(_CFG_PATH)

    def test_prior_05_gives_balanced_p0(self):
        cfg = dict(self.cfg)
        cfg["model"] = dict(self.cfg["model"])
        cfg["model"]["no_pattern_prior"] = 0.5
        model = CombCepstralNet(cfg)
        b0 = model.classifier.bias[0].item()
        K = model.K
        # softmax([b0, 0, ..., 0])[0] should be ≈ 0.5
        logits = torch.zeros(K + 1)
        logits[0] = b0
        p0 = F.softmax(logits, dim=0)[0].item()
        self.assertAlmostEqual(p0, 0.5, places=3)

    def test_prior_09_gives_high_p0(self):
        cfg = dict(self.cfg)
        cfg["model"] = dict(self.cfg["model"])
        cfg["model"]["no_pattern_prior"] = 0.9
        model = CombCepstralNet(cfg)
        b0 = model.classifier.bias[0].item()
        K = model.K
        logits = torch.zeros(K + 1)
        logits[0] = b0
        p0 = F.softmax(logits, dim=0)[0].item()
        self.assertAlmostEqual(p0, 0.9, places=3)


class TestCepstralGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = load_yaml_config(_CFG_PATH)

    def test_bin_centers_are_monotone_increasing(self):
        geom = compute_cepstral_geometry(self.cfg)
        c = geom["bin_centers_cm"]
        self.assertTrue(np.all(np.diff(c) > 0), "bin centres not monotone")

    def test_q_equals_cep_max_minus_min(self):
        geom = compute_cepstral_geometry(self.cfg)
        self.assertEqual(geom["Q"], geom["cep_max_bin"] - geom["cep_min_bin"])

    def test_k_equals_q(self):
        geom = compute_cepstral_geometry(self.cfg)
        self.assertEqual(geom["K"], geom["Q"])


class TestCepstralPatchNormalisation(unittest.TestCase):
    def setUp(self):
        self.cfg = load_yaml_config(_CFG_PATH)

    def test_normalised_patches_have_near_zero_median(self):
        geom = compute_cepstral_geometry(self.cfg)
        T, Q = 32, geom["Q"]
        rng = np.random.default_rng(42)
        log_mag = rng.standard_normal((T, geom["n_band"])).astype(np.float32)
        patches = _compute_cepstral_patches(
            log_mag, geom["cep_min_bin"], geom["cep_max_bin"], use_normalization=True
        )
        self.assertEqual(patches.shape, (T, Q))
        per_frame_median = np.median(patches, axis=1)
        self.assertTrue(np.all(np.abs(per_frame_median) < 1e-4),
                        f"MAD normalisation: median not near 0, got {per_frame_median[:3]}")

    def test_unnormalised_patches_raw_cepstrum(self):
        geom = compute_cepstral_geometry(self.cfg)
        T = 8
        rng = np.random.default_rng(7)
        log_mag = rng.standard_normal((T, geom["n_band"])).astype(np.float32)
        patches_norm = _compute_cepstral_patches(
            log_mag, geom["cep_min_bin"], geom["cep_max_bin"], use_normalization=True
        )
        patches_raw = _compute_cepstral_patches(
            log_mag, geom["cep_min_bin"], geom["cep_max_bin"], use_normalization=False
        )
        # Raw and normalised should differ
        self.assertFalse(np.allclose(patches_norm, patches_raw))


class TestSoftBinTargets(unittest.TestCase):
    """Tests for the Gaussian soft-bin target generation."""

    def setUp(self):
        self.cfg = load_yaml_config(_CFG_PATH)
        from ml_uav_comb.data_pipeline.cepstral_dataset import (
            distance_cm_to_soft_target,
            distance_cm_to_bin,
        )
        self.distance_cm_to_soft_target = distance_cm_to_soft_target
        self.distance_cm_to_bin = distance_cm_to_bin
        geom = compute_cepstral_geometry(self.cfg)
        self.geom = geom

    def test_soft_target_sums_to_one(self):
        geom = self.geom
        soft = self.distance_cm_to_soft_target(
            20.0, geom["quef_tau_factor"], geom["cep_min_bin"], geom["Q"],
            soft_bin_sigma=1.5,
        )
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=5)
        self.assertEqual(soft.shape[0], geom["K"] + 1)

    def test_soft_target_peak_at_correct_bin(self):
        geom = self.geom
        distance_cm = 20.0
        hard_bin = self.distance_cm_to_bin(
            distance_cm, geom["quef_tau_factor"], geom["cep_min_bin"], geom["Q"]
        )
        soft = self.distance_cm_to_soft_target(
            distance_cm, geom["quef_tau_factor"], geom["cep_min_bin"], geom["Q"],
            soft_bin_sigma=1.5,
        )
        # Peak class (1-indexed) should be hard_bin + 1
        self.assertEqual(int(soft[1:].argmax()), hard_bin)

    def test_soft_target_no_pattern_mass(self):
        geom = self.geom
        soft = self.distance_cm_to_soft_target(
            20.0, geom["quef_tau_factor"], geom["cep_min_bin"], geom["Q"],
            soft_bin_sigma=1.5, no_pattern_mass=0.1,
        )
        self.assertAlmostEqual(float(soft[0]), 0.1, places=5)
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=5)

    def test_focal_loss_with_soft_targets(self):
        from ml_uav_comb.training.cepstral_losses import cepstral_bin_loss
        B, K1 = 4, 29
        logits = torch.randn(B, K1, requires_grad=True)
        hard_targets = torch.tensor([0, 1, 5, 0], dtype=torch.long)

        geom = self.geom
        soft_list = []
        for i, (tgt, is_pat) in enumerate(zip([0, 1, 5, 0], [False, True, True, False])):
            if not is_pat:
                s = np.zeros(K1, dtype=np.float32)
                s[0] = 1.0
            else:
                s = self.distance_cm_to_soft_target(
                    20.0, geom["quef_tau_factor"], geom["cep_min_bin"], geom["Q"],
                    soft_bin_sigma=1.5,
                )
            soft_list.append(torch.from_numpy(s).float())
        soft_targets = torch.stack(soft_list, dim=0)  # [B, K+1]

        out = cepstral_bin_loss(logits, hard_targets, focal_gamma=2.0, soft_targets=soft_targets)
        self.assertIn("loss_total", out)
        self.assertTrue(torch.isfinite(out["loss_total"]))
        out["loss_total"].backward()


if __name__ == "__main__":
    unittest.main()
