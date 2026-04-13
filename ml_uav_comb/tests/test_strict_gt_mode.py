from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_uav_comb.data_pipeline.dataset_index import infer_label_bundle
from ml_uav_comb.features.feature_utils import load_optional_labels, load_yaml_config


SCALAR_FIELD_NAMES = np.asarray(
    ["sum_abs_d1_smooth", "comb_shift_lag", "comb_shift_rho", "is_sound_present"],
    dtype=object,
)
TEACHER_FIELD_NAMES = np.asarray(
    [
        "heuristic_distance_raw_cm",
        "heuristic_distance_kf_cm",
        "heuristic_distance_raw_available",
        "heuristic_distance_kf_available",
        "velocity_kf_cm_s",
        "acceleration_kf_cm_s2",
        "heuristic_measure_available",
        "shift_direction_raw",
        "shift_direction_hysteresis",
    ],
    dtype=object,
)


def _base_cfg() -> dict:
    cfg = load_yaml_config("ml_uav_comb/configs/tiny_debug.yaml")
    cfg["dataset"]["supervision_mode"] = "strict_gt"
    cfg["dataset"]["use_physics_gate_as_distance_mask"] = False
    cfg["dataset"]["use_physics_gate_as_distance_weight"] = True
    cfg["dataset"]["distance_weight_floor"] = 0.25
    cfg["dataset"]["distance_weight_reliable"] = 1.0
    return cfg


def _target_scalar(is_sound: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    return np.asarray([amplitude, 0.0, 0.9, is_sound], dtype=np.float32)


def _target_reliable_mask(reliable: bool) -> np.ndarray:
    if reliable:
        return np.ones(4, dtype=np.float32)
    return np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _target_teacher(distance_cm: float = 30.0) -> np.ndarray:
    return np.asarray([distance_cm, distance_cm, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


class TestStrictGTMode(unittest.TestCase):
    def _write_csv(self, tmpdir: str, rows: list[dict], fieldnames: list[str]) -> Path:
        path = Path(tmpdir) / "labels.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_all_zero_valid_mask_is_explicit_confidence_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_csv(
                tmpdir,
                rows=[
                    {"time_sec": 0.0, "distance_cm": 30.0, "valid_mask": 0},
                    {"time_sec": 1.0, "distance_cm": 20.0, "valid_mask": 0},
                ],
                fieldnames=["time_sec", "distance_cm", "valid_mask"],
            )
            label_dict = load_optional_labels(csv_path, "rec_1")
            assert label_dict is not None
            bundle = infer_label_bundle(
                target_time_sec=0.5,
                target_scalar=_target_scalar(),
                target_reliable_mask=_target_reliable_mask(True),
                target_teacher=_target_teacher(),
                scalar_field_names=SCALAR_FIELD_NAMES,
                teacher_field_names=TEACHER_FIELD_NAMES,
                label_dict=label_dict,
                cfg=_base_cfg(),
            )
            self.assertEqual(bundle["measurement_validity_target"], 0.0)
            self.assertEqual(bundle["validity_target_source"], "explicit_valid_mask_gt")
            self.assertEqual(bundle["measurement_validity_train_mask"], 1.0)

    def test_row_without_distance_keeps_validity_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_csv(
                tmpdir,
                rows=[{"time_sec": 0.5, "valid_mask": 1}],
                fieldnames=["time_sec", "valid_mask"],
            )
            label_dict = load_optional_labels(csv_path, "rec_1")
            assert label_dict is not None
            bundle = infer_label_bundle(
                target_time_sec=0.5,
                target_scalar=_target_scalar(),
                target_reliable_mask=_target_reliable_mask(True),
                target_teacher=_target_teacher(),
                scalar_field_names=SCALAR_FIELD_NAMES,
                teacher_field_names=TEACHER_FIELD_NAMES,
                label_dict=label_dict,
                cfg=_base_cfg(),
            )
            self.assertEqual(bundle["measurement_distance_train_mask"], 0.0)
            self.assertEqual(bundle["measurement_validity_target"], 1.0)
            self.assertEqual(bundle["measurement_validity_train_mask"], 1.0)

    def test_no_explicit_validity_falls_back_to_reliable_mask(self) -> None:
        label_dict = {
            "time_sec": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "distance_cm": np.asarray([40.0, 30.0, 20.0], dtype=np.float32),
            "distance_valid": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "row_has_distance_value": np.ones(3, dtype=np.float32),
            "row_has_distance_valid": np.ones(3, dtype=np.float32),
            "motion_sign": np.asarray(["unknown", "unknown", "unknown"], dtype=object),
            "explicit_motion_sign_mask": np.zeros(3, dtype=np.float32),
            "row_has_motion_sign_value": np.zeros(3, dtype=np.float32),
            "row_has_sign_annotation": np.zeros(3, dtype=np.float32),
            "confidence_valid": np.full(3, np.nan, dtype=np.float32),
            "row_has_conf_value": np.zeros(3, dtype=np.float32),
            "has_conf_gt": np.asarray([0.0], dtype=np.float32),
            "has_distance_valid": np.asarray([1.0], dtype=np.float32),
            "has_sign_annotation": np.asarray([0.0], dtype=np.float32),
        }
        bundle = infer_label_bundle(
            target_time_sec=0.5,
            target_scalar=_target_scalar(),
            target_reliable_mask=_target_reliable_mask(False),
            target_teacher=_target_teacher(),
            scalar_field_names=SCALAR_FIELD_NAMES,
            teacher_field_names=TEACHER_FIELD_NAMES,
            label_dict=label_dict,
            cfg=_base_cfg(),
        )
        self.assertEqual(bundle["validity_target_source"], "dist_reliable_fallback")
        self.assertEqual(bundle["measurement_validity_train_mask"], 1.0)

    def test_shared_label_file_filters_recording_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_csv(
                tmpdir,
                rows=[
                    {"recording_id": "rec_a", "time_sec": 0.0, "distance_cm": 10.0},
                    {"recording_id": "rec_b", "time_sec": 0.5, "distance_cm": 20.0},
                    {"recording_id": "rec_a", "time_sec": 1.0, "distance_cm": 30.0},
                ],
                fieldnames=["recording_id", "time_sec", "distance_cm"],
            )
            label_dict = load_optional_labels(csv_path, "rec_a")
            assert label_dict is not None
            self.assertTrue(np.allclose(label_dict["time_sec"], np.asarray([0.0, 1.0], dtype=np.float32)))
            self.assertTrue(np.allclose(label_dict["distance_cm"], np.asarray([10.0, 30.0], dtype=np.float32)))

    def test_distance_loss_weight_keeps_unreliable_gt_samples_trainable(self) -> None:
        label_dict = {
            "time_sec": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "distance_cm": np.asarray([40.0, 30.0, 20.0], dtype=np.float32),
            "distance_valid": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "row_has_distance_value": np.ones(3, dtype=np.float32),
            "row_has_distance_valid": np.ones(3, dtype=np.float32),
            "motion_sign": np.asarray(["unknown", "unknown", "unknown"], dtype=object),
            "explicit_motion_sign_mask": np.zeros(3, dtype=np.float32),
            "row_has_motion_sign_value": np.zeros(3, dtype=np.float32),
            "row_has_sign_annotation": np.zeros(3, dtype=np.float32),
            "confidence_valid": np.full(3, np.nan, dtype=np.float32),
            "row_has_conf_value": np.zeros(3, dtype=np.float32),
            "has_conf_gt": np.asarray([0.0], dtype=np.float32),
            "has_distance_valid": np.asarray([1.0], dtype=np.float32),
            "has_sign_annotation": np.asarray([0.0], dtype=np.float32),
        }
        reliable_bundle = infer_label_bundle(
            target_time_sec=0.5,
            target_scalar=_target_scalar(),
            target_reliable_mask=_target_reliable_mask(True),
            target_teacher=_target_teacher(),
            scalar_field_names=SCALAR_FIELD_NAMES,
            teacher_field_names=TEACHER_FIELD_NAMES,
            label_dict=label_dict,
            cfg=_base_cfg(),
        )
        unreliable_bundle = infer_label_bundle(
            target_time_sec=0.5,
            target_scalar=_target_scalar(),
            target_reliable_mask=_target_reliable_mask(False),
            target_teacher=_target_teacher(),
            scalar_field_names=SCALAR_FIELD_NAMES,
            teacher_field_names=TEACHER_FIELD_NAMES,
            label_dict=label_dict,
            cfg=_base_cfg(),
        )
        self.assertEqual(reliable_bundle["measurement_distance_train_mask"], 1.0)
        self.assertEqual(unreliable_bundle["measurement_distance_train_mask"], 1.0)
        self.assertGreater(reliable_bundle["dist_loss_weight"], unreliable_bundle["dist_loss_weight"])
        self.assertGreaterEqual(unreliable_bundle["dist_loss_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
