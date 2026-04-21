"""Trainer for CombCepstralNet K+1 classification.

Reuses infrastructure from omega_trainer (ContiguousSequenceBatchSampler,
_move_batch, GPU/memory logging utilities) and the CepstralBinDataset.

Key differences from omega_trainer:
  - Input: normalized cepstral patches [B, T, Q]
  - Loss: weighted cross-entropy (K+1 classes)
  - Metrics: classification accuracy, distance MAE from bin decode
  - Class weights computed via quick scan at startup
"""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_uav_comb.data_pipeline.cepstral_dataset import CepstralBinDataset
from ml_uav_comb.models.comb_cepstral_net import CombCepstralNet
from ml_uav_comb.training.cepstral_losses import cepstral_bin_loss
from ml_uav_comb.training.cepstral_metrics import compute_cepstral_metrics, reduce_cepstral_metrics
from ml_uav_comb.training.omega_trainer import (
    ContiguousSequenceBatchSampler,
    _create_dataloader_from_dataset,
    _move_batch,
    _set_torch_runtime,
    _resolve_amp_dtype,
    _query_gpu_stats,
    _query_host_memory_stats,
    _format_gpu_stats,
    _format_host_memory_stats,
    _dynamic_epoch_split_enabled,
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _autocast_context(device: torch.device, cfg: Dict[str, Any]):
    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"
    if not use_amp:
        return nullcontext()
    dtype = _resolve_amp_dtype(cfg)
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_cepstral_model(cfg: Dict[str, Any]) -> CombCepstralNet:
    return CombCepstralNet(cfg)


def create_cepstral_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    max_cache_files = int(cfg["training"].get("max_cache_files_per_worker", 2))
    dataset = CepstralBinDataset(
        cfg["dataset"]["index_path"],
        split=split,
        cfg=cfg,
        max_cache_files=max_cache_files,
    )
    return _create_dataloader_from_dataset(cfg, dataset, split)


def create_cepstral_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return tuple(create_cepstral_dataloader(cfg, s) for s in ("train", "val", "test"))


def _create_cepstral_dynamic_epoch_dataloaders(
    cfg: Dict[str, Any],
    pooled_dataset: CepstralBinDataset,
    epoch: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict[str, int]]]:
    """Dynamic epoch split: rotate recordings across train/val/test each epoch."""
    split_names = ("train", "val", "test")
    ratio = np.asarray(
        cfg["dataset"].get("dynamic_epoch_split_ratio", [5, 1, 1]),
        dtype=np.float64,
    ).ravel()
    all_codes = np.asarray(
        sorted({int(c) for c in pooled_dataset.recording_code.tolist()}),
        dtype=np.int64,
    )
    rng = np.random.default_rng(int(cfg.get("experiment", {}).get("seed", 0)) + epoch)
    shuffled = rng.permutation(all_codes)
    total = int(shuffled.shape[0])
    scaled = (ratio / float(ratio.sum())) * float(total)
    counts = np.floor(scaled).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        fracs = scaled - counts
        for idx in np.argsort(-fracs)[:remainder]:
            counts[int(idx)] += 1

    code_map: Dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(split_names, counts.tolist()):
        code_map[name] = shuffled[offset: offset + int(count)]
        offset += int(count)

    loaders: Dict[str, DataLoader] = {}
    summary: Dict[str, Dict[str, int]] = {}
    for name in split_names:
        sub = pooled_dataset.subset_by_recording_codes(code_map[name], split=name)
        loaders[name] = _create_dataloader_from_dataset(cfg, sub, name)
        summary[name] = {
            "num_recordings": int(code_map[name].shape[0]),
            "num_windows": int(len(sub)),
        }
    return loaders, summary


def _compute_class_weights_from_dataset(
    dataset: CepstralBinDataset,
    num_classes: int,
    max_weight_ratio: float = 50.0,
) -> torch.Tensor:
    """Estimate inverse-frequency class weights from training split.

    Only active classes (those with ≥1 real sample) participate in weighting.
    Empty classes get weight=1.0.  All weights are capped at max_weight_ratio
    times the minimum active weight to prevent explosion from near-empty bins.
    """
    from ml_uav_comb.data_pipeline.cepstral_dataset import distance_cm_to_bin

    raw_counts = np.zeros(num_classes, dtype=np.float64)
    for entry in dataset.recording_entries.values():
        try:
            cache = dataset._get_cache(str(entry["cache_path"]))
            pattern_binary = np.asarray(cache["frame_pattern_binary_target"], dtype=np.float32)
            distance_cm = np.asarray(cache["frame_distance_cm"], dtype=np.float32)
            valid_mask = np.isfinite(pattern_binary) & (pattern_binary >= 0.5) & np.isfinite(distance_cm)
            raw_counts[0] += float(np.sum(~valid_mask))
            for dc in distance_cm[valid_mask]:
                bin_idx = distance_cm_to_bin(
                    float(dc), dataset.quef_tau_factor, dataset.cep_min_bin, dataset.Q
                )
                raw_counts[min(bin_idx + 1, num_classes - 1)] += 1.0
        except Exception:
            pass

    active = raw_counts > 0
    if not np.any(active):
        return torch.ones(num_classes, dtype=torch.float32)

    # Inverse-frequency weights for active classes
    active_counts = raw_counts[active]
    active_freq = active_counts / active_counts.sum()
    inv_freq = 1.0 / active_freq

    # Cap at max_weight_ratio × minimum
    min_w = inv_freq.min()
    inv_freq = np.clip(inv_freq, min_w, max_weight_ratio * min_w)
    inv_freq /= inv_freq.mean()  # normalize active classes to mean=1

    weights = np.ones(num_classes, dtype=np.float32)
    weights[active] = inv_freq.astype(np.float32)
    return torch.from_numpy(weights)


def _evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()
    if len(loader.dataset) == 0:
        return {}
    metric_list: List[Dict[str, float]] = []
    num_classes = model.num_classes
    bin_centers = model.bin_centers_cm.to(device)
    max_eval_batches = max(0, int(cfg["training"].get("max_eval_batches", 0)))
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device, bool(cfg["training"].get("non_blocking_to_device", True)))
            with _autocast_context(device, cfg):
                out = model(batch)
            metric_list.append(compute_cepstral_metrics(
                out["logits"],
                batch["target"],
                bin_centers,
                batch["distance_cm"],
            ))
            if max_eval_batches > 0 and (batch_idx + 1) >= max_eval_batches:
                break
    return reduce_cepstral_metrics(metric_list)


def train_model(cfg: Dict[str, Any], resume_from: Optional[str] = None) -> Dict[str, Any]:
    _set_torch_runtime()
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dynamic_epoch_split = _dynamic_epoch_split_enabled(cfg)
    max_cache_files = int(cfg["training"].get("max_cache_files_per_worker", 2))

    pooled_dataset = (
        CepstralBinDataset(cfg["dataset"]["index_path"], split="all", cfg=cfg, max_cache_files=max_cache_files)
        if dynamic_epoch_split
        else None
    )
    train_loader = create_cepstral_dataloader(cfg, "train") if not dynamic_epoch_split else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = build_cepstral_model(cfg).to(device)
    _log(f"model | params={base_model.count_parameters():,} num_classes={base_model.num_classes} Q={base_model.Q}")

    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg)
    epochs = int(cfg["training"].get("epochs", 30))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 5.0))
    non_blocking = bool(cfg["training"].get("non_blocking_to_device", True))

    # Use a higher learning rate for the classifier head.
    # Adam normalizes gradient magnitude, so biases only change by ±LR/step.
    # With the prior init (bias[0] ≈ 3+), the classifier needs many steps to
    # reach the no-pattern/pattern balance point at the backbone LR.
    # A dedicated classifier_lr (default 10x backbone) fixes this.
    backbone_lr = float(cfg["training"].get("learning_rate", 5e-4))
    classifier_lr = float(cfg["training"].get("classifier_lr", backbone_lr * 10.0))
    classifier_params = list(base_model.classifier.parameters())
    classifier_ids = {id(p) for p in classifier_params}
    backbone_params = [p for p in base_model.parameters() if id(p) not in classifier_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": classifier_lr},
        ],
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    _log(f"optimizer | backbone_lr={backbone_lr:.2e} classifier_lr={classifier_lr:.2e}")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    scheduler = None
    if cfg["training"].get("lr_scheduler", "none") == "cosine_warmup":
        warmup_epochs = int(cfg["training"].get("lr_warmup_epochs", 3))
        lr_min = float(cfg["training"].get("lr_min", 1e-5))
        lr_base = backbone_lr

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return max(lr_min / lr_base, (epoch + 1) / warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return max(lr_min / lr_base, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # Pre-compute class weights from a training data sample
    class_weights: Optional[torch.Tensor] = None
    if bool(cfg["training"].get("use_class_weights", True)):
        try:
            probe_dataset = (
                pooled_dataset if dynamic_epoch_split else train_loader.dataset
            )
            max_wr = float(cfg["training"].get("class_weight_max_ratio", 15.0))
            class_weights = _compute_class_weights_from_dataset(
                probe_dataset, base_model.num_classes, max_weight_ratio=max_wr
            ).to(device)
            _log(f"class_weights | no-pattern={class_weights[0]:.3f} bin_mean={class_weights[1:].mean():.3f} ratio={class_weights[1:].mean()/class_weights[0]:.1f}x")
        except Exception as e:
            _log(f"class_weights | failed to compute: {e}, using uniform")

    focal_gamma = float(cfg["training"].get("focal_gamma", 0.0))
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    history_path = checkpoint_dir / "train_history.json"

    start_epoch = 0
    best_val = float("-inf")  # maximise val_pat_recall - val_fp_rate
    history: List[Dict[str, Any]] = []

    if resume_from is not None:
        state = torch.load(str(resume_from), map_location=device)
        base_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if "scaler_state" in state and scaler.is_enabled():
            scaler.load_state_dict(state["scaler_state"])
        if scheduler is not None and "scheduler_state" in state and state["scheduler_state"] is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val_combined_score", float("-inf")))
        _log(f"resumed | epoch={start_epoch} best_val_combined={best_val:.4f}")

    # No torch.compile for small models (overhead not worth it)
    model = base_model

    log_interval = max(1, int(cfg["training"].get("log_interval_steps", 20)))

    _log(
        f"train setup | device={device} amp={use_amp} "
        f"lr={cfg['training'].get('learning_rate', 5e-4)} epochs={epochs} "
        f"focal_gamma={focal_gamma} label_smoothing={label_smoothing} "
        f"dynamic_split={dynamic_epoch_split}"
    )

    best_epoch = -1
    max_train_steps = max(0, int(cfg["training"].get("max_train_steps_per_epoch", 0)))

    for epoch in range(start_epoch, epochs):
        if dynamic_epoch_split:
            epoch_loaders, split_summary = _create_cepstral_dynamic_epoch_dataloaders(
                cfg, pooled_dataset, epoch
            )
            train_loader = epoch_loaders["train"]
            val_loader = epoch_loaders["val"]
            _log(
                f"epoch split | epoch={epoch:03d} "
                f"train_rec={split_summary['train']['num_recordings']} train_win={split_summary['train']['num_windows']} "
                f"val_rec={split_summary['val']['num_recordings']} val_win={split_summary['val']['num_windows']}"
            )
        else:
            val_loader = create_cepstral_dataloader(cfg, "val")

        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        metric_list: List[Dict[str, float]] = []
        num_batches = len(train_loader)
        prev_step_end = time.perf_counter()

        for step, batch in enumerate(train_loader):
            batch_ready = time.perf_counter()
            data_wait_sec = batch_ready - prev_step_end
            step_start = batch_ready

            batch = _move_batch(batch, device, non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                out = model(batch)
                losses = cepstral_bin_loss(
                    out["logits"],
                    batch["target"],
                    class_weights=class_weights,
                    label_smoothing=label_smoothing,
                    focal_gamma=focal_gamma,
                    soft_targets=batch.get("soft_target"),
                )

            if scaler.is_enabled():
                scaler.scale(losses["loss_total"]).backward()
            else:
                losses["loss_total"].backward()

            if grad_clip > 0.0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            bin_centers = model.bin_centers_cm.to(device)
            metric = compute_cepstral_metrics(
                out["logits"].detach(),
                batch["target"],
                bin_centers,
                batch["distance_cm"],
            )
            metric["loss_total"] = float(losses["loss_total"].item())
            metric_list.append(metric)

            step_sec = time.perf_counter() - step_start
            prev_step_end = time.perf_counter()

            if step % log_interval == 0:
                gpu_stats = _query_gpu_stats(device)
                host_stats = _query_host_memory_stats()
                eta_min = (max(0, num_batches - step - 1) * step_sec) / 60.0
                _log(
                    f"epoch {epoch:03d} step {step:04d}/{num_batches:04d} "
                    f"loss={metric['loss_total']:.4f} "
                    f"acc={metric['accuracy']:.3f} "
                    f"pat_recall={metric['pattern_recall']:.3f} "
                    f"fp_rate={metric['fp_rate']:.3f} "
                    f"dist_mae={metric['distance_mae_cm']:.2f}cm "
                    f"data_wait={data_wait_sec:.2f}s step={step_sec:.3f}s eta={eta_min:.1f}m "
                    f"{_format_gpu_stats(gpu_stats)} {_format_host_memory_stats(host_stats)}"
                )

            if max_train_steps > 0 and (step + 1) >= max_train_steps:
                break

        train_metrics = reduce_cepstral_metrics(metric_list)
        val_metrics = _evaluate_epoch(model, val_loader, device, cfg)

        if scheduler is not None:
            scheduler.step()

        val_mae = float(val_metrics.get("distance_mae_cm", float("inf")))
        val_acc = float(val_metrics.get("accuracy", 0.0))
        # Use combined recall–fp score for checkpoint selection.
        # This ensures we keep the model that best detects patterns (not just
        # the one that maximises accuracy by predicting all-no-pattern).
        val_recall = float(val_metrics.get("pattern_recall", 0.0))
        val_fp = float(val_metrics.get("fp_rate", 0.0))
        val_combined = val_recall - val_fp
        is_best = val_combined > best_val
        if is_best:
            best_val = val_combined
            best_epoch = epoch

        epoch_record = {
            "epoch": epoch,
            "train": {k: v for k, v in train_metrics.items() if not k.startswith("_")},
            "val": {k: v for k, v in val_metrics.items() if not k.startswith("_")},
        }
        history.append(epoch_record)

        state = {
            "epoch": epoch,
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_combined_score": best_val,
            "cfg": cfg,
        }
        torch.save(state, str(latest_path))
        if is_best:
            torch.save(state, str(best_path))

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        _log(
            f"epoch {epoch:03d} done | "
            f"train_loss={train_metrics.get('loss_total', 0.0):.4f} "
            f"train_acc={train_metrics.get('accuracy', 0.0):.3f} "
            f"val_acc={val_acc:.3f} "
            f"val_pat_recall={val_recall:.3f} "
            f"val_fp={val_fp:.3f} "
            f"val_combined={val_combined:.4f} "
            f"val_dist_mae={val_mae:.2f}cm "
            f"{'[BEST]' if is_best else ''}"
        )

    _log(f"training complete | best_epoch={best_epoch} best_val_combined={best_val:.4f}")

    # Final test evaluation
    test_loader = (
        _create_cepstral_dynamic_epoch_dataloaders(cfg, pooled_dataset, epochs)[0]["test"]
        if dynamic_epoch_split
        else create_cepstral_dataloader(cfg, "test")
    )
    best_state = torch.load(str(best_path), map_location=device)
    base_model.load_state_dict(best_state["model_state"])
    test_metrics = _evaluate_epoch(base_model, test_loader, device, cfg)
    _log(
        f"test eval | "
        f"acc={test_metrics.get('accuracy', 0.0):.4f} "
        f"dist_mae={test_metrics.get('distance_mae_cm', 0.0):.2f}cm "
        f"pat_recall={test_metrics.get('pattern_recall', 0.0):.4f} "
        f"fp_rate={test_metrics.get('fp_rate', 0.0):.4f}"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_distance_mae_cm": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }
