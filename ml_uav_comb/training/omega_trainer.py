"""Trainer for omega-regression LS replacement model."""
from __future__ import annotations

import csv
import json
import random
import resource
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate

from ml_uav_comb.data_pipeline.offline_omega_feature_extractor import omega_to_distance_cm
from ml_uav_comb.data_pipeline.omega_dataset import OmegaWindowDataset
from ml_uav_comb.models.uav_comb_omega_net import UAVCombOmegaNet
from ml_uav_comb.training.omega_losses import combined_omega_loss, compute_omega_regression_weights, compute_pattern_weights
from ml_uav_comb.training.omega_metrics import compute_omega_metrics, reduce_metric_list


class ContiguousSequenceBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: OmegaWindowDataset,
        *,
        sequence_length: int,
        chunk_step: int,
        chunks_per_batch: int,
        drop_last: bool,
        shuffle_chunks: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.sequence_length = max(1, int(sequence_length))
        self.chunk_step = max(1, int(chunk_step))
        self.chunks_per_batch = max(1, int(chunks_per_batch))
        self.drop_last = bool(drop_last)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.seed = int(seed)
        self._epoch = 0
        self._full_chunk_starts, self._short_chunk_starts, self._short_chunk_lengths = self._build_chunk_descriptors()

    def _build_chunk_descriptors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        full_chunk_starts: List[int] = []
        short_chunk_starts: List[int] = []
        short_chunk_lengths: List[int] = []
        for offset, length, _recording_code in self.dataset.recording_segments:
            if length < self.sequence_length:
                if length > 0 and not self.drop_last:
                    short_chunk_starts.append(int(offset))
                    short_chunk_lengths.append(int(length))
                continue
            for local_start in range(0, length - self.sequence_length + 1, self.chunk_step):
                full_chunk_starts.append(int(offset + local_start))
        return (
            np.asarray(full_chunk_starts, dtype=np.int64),
            np.asarray(short_chunk_starts, dtype=np.int64),
            np.asarray(short_chunk_lengths, dtype=np.int32),
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _expand_batch(self, chunk_starts: List[int], chunk_length: int) -> List[int]:
        indices: List[int] = []
        for start in chunk_starts:
            indices.extend(range(int(start), int(start) + int(chunk_length)))
        return indices

    def __iter__(self) -> Iterator[List[int]]:
        current: List[int] = []
        if self._full_chunk_starts.size > 0:
            if self.shuffle_chunks:
                order = np.arange(self._full_chunk_starts.shape[0], dtype=np.int64)
                rng = np.random.default_rng(self.seed + self._epoch)
                rng.shuffle(order)
                full_iter: Iterator[int] = (int(self._full_chunk_starts[idx]) for idx in order)
            else:
                full_iter = (int(start) for start in self._full_chunk_starts)

            for chunk_start in full_iter:
                current.append(chunk_start)
                if len(current) == self.chunks_per_batch:
                    yield self._expand_batch(current, self.sequence_length)
                    current = []

        if current and not self.drop_last:
            yield self._expand_batch(current, self.sequence_length)

        if not self.drop_last:
            for chunk_start, chunk_length in zip(self._short_chunk_starts, self._short_chunk_lengths):
                yield list(range(int(chunk_start), int(chunk_start) + int(chunk_length)))

    def __len__(self) -> int:
        batches = int(self._full_chunk_starts.shape[0]) // self.chunks_per_batch
        if not self.drop_last and (int(self._full_chunk_starts.shape[0]) % self.chunks_per_batch):
            batches += 1
        if not self.drop_last:
            batches += int(self._short_chunk_starts.shape[0])
        return batches


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


def _set_torch_runtime() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def _resolve_amp_dtype(cfg: Dict[str, Any]) -> torch.dtype:
    amp_dtype = str(cfg["training"].get("amp_dtype", "auto")).lower()
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, cfg: Dict[str, Any]):
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_resolve_amp_dtype(cfg))


def build_model(cfg: Dict[str, Any]) -> UAVCombOmegaNet:
    return UAVCombOmegaNet(cfg)


def _maybe_compile_model(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.nn.Module:
    if bool(cfg["training"].get("use_compile", True)) and hasattr(torch, "compile"):
        return torch.compile(model, mode=str(cfg["training"].get("compile_mode", "reduce-overhead")))
    return model


def _make_chunk_collate_fn(chunk_length: int):
    chunk_length = max(1, int(chunk_length))

    def collate(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_collate(batch_items)
        n = len(batch_items)
        if n == 0:
            batch["chunk_id"] = torch.empty((0,), dtype=torch.long)
            return batch
        if n % chunk_length == 0:
            num_chunks = n // chunk_length
            chunk_ids = np.repeat(np.arange(num_chunks, dtype=np.int64), chunk_length)
        else:
            chunk_ids = np.zeros((n,), dtype=np.int64)
        batch["chunk_id"] = torch.from_numpy(chunk_ids)
        return batch

    return collate


def _create_dataloader_from_dataset(cfg: Dict[str, Any], dataset: Any, split: str) -> DataLoader:
    num_workers = int(cfg["training"].get("num_workers", 0))
    pin_memory = bool(cfg["training"].get("pin_memory", torch.cuda.is_available()))
    dynamic_epoch_split = _dynamic_epoch_split_enabled(cfg)
    # Keep persistent workers only for train. Eval loaders are created on demand and should release memory promptly.
    persistent_workers = (
        bool(cfg["training"].get("persistent_workers", num_workers > 0))
        if (num_workers > 0 and split == "train" and not dynamic_epoch_split)
        else False
    )
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2)) if num_workers > 0 else None

    is_train = split == "train"
    seq_len = int(cfg["training"].get("sequence_length", 8)) if is_train else int(cfg["training"].get("eval_sequence_length", cfg["training"].get("sequence_length", 8)))
    chunk_step = int(cfg["training"].get("chunk_step", seq_len)) if is_train else int(cfg["training"].get("eval_chunk_step", seq_len))
    chunks_per_batch = int(cfg["training"].get("batch_size", 1)) if is_train else int(cfg["training"].get("eval_batch_size", cfg["training"].get("batch_size", 1)))

    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        common["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_sampler=ContiguousSequenceBatchSampler(
            dataset,
            sequence_length=seq_len,
            chunk_step=chunk_step,
            chunks_per_batch=chunks_per_batch,
            drop_last=bool(cfg["training"].get("drop_last_train_batch", True)) if is_train else False,
            shuffle_chunks=is_train,
            seed=int(cfg.get("experiment", {}).get("seed", 0)),
        ),
        collate_fn=_make_chunk_collate_fn(seq_len),
        **common,
    )


def create_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    max_cache_files = int(cfg["training"].get("max_cache_files_per_worker", 2))
    dataset = OmegaWindowDataset(cfg["dataset"]["index_path"], split=split, max_cache_files=max_cache_files)
    return _create_dataloader_from_dataset(cfg, dataset, split)


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return tuple(create_dataloader(cfg, split_name) for split_name in ("train", "val", "test"))


def _dynamic_epoch_split_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("dataset", {}).get("dynamic_epoch_split", False))


def _resolve_dynamic_epoch_ratio(cfg: Dict[str, Any]) -> np.ndarray:
    ratio = np.asarray(cfg.get("dataset", {}).get("dynamic_epoch_split_ratio", [5, 1, 1]), dtype=np.float64).reshape(-1)
    if int(ratio.shape[0]) != 3:
        raise ValueError(f"dataset.dynamic_epoch_split_ratio must have length 3, got {ratio.tolist()}")
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0.0):
        raise ValueError(f"dataset.dynamic_epoch_split_ratio must be positive finite values, got {ratio.tolist()}")
    return ratio


def _allocate_dynamic_split_counts(num_items: int, ratio: np.ndarray) -> np.ndarray:
    if num_items <= 0:
        return np.zeros((ratio.shape[0],), dtype=np.int64)
    scaled = (ratio / float(np.sum(ratio))) * float(num_items)
    counts = np.floor(scaled).astype(np.int64)
    remainder = int(num_items - int(counts.sum()))
    if remainder > 0:
        frac = scaled - counts
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return counts


def _dynamic_epoch_recording_codes(
    pooled_dataset: OmegaWindowDataset,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, np.ndarray]:
    split_names = ("train", "val", "test")
    ratio = _resolve_dynamic_epoch_ratio(cfg)
    recording_codes = np.asarray(sorted({int(code) for code in np.asarray(pooled_dataset.recording_code).tolist()}), dtype=np.int64)
    rng = np.random.default_rng(int(cfg.get("experiment", {}).get("seed", 0)) + int(epoch))
    shuffled_codes = rng.permutation(recording_codes)
    counts = _allocate_dynamic_split_counts(int(shuffled_codes.shape[0]), ratio)
    code_map: Dict[str, np.ndarray] = {}
    offset = 0
    for split_name, count in zip(split_names, counts.tolist()):
        next_offset = offset + int(count)
        code_map[split_name] = np.asarray(shuffled_codes[offset:next_offset], dtype=np.int64)
        offset = next_offset
    return code_map


def _create_dynamic_epoch_dataloaders(
    cfg: Dict[str, Any],
    pooled_dataset: OmegaWindowDataset,
    epoch: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict[str, int]]]:
    recording_codes_by_split = _dynamic_epoch_recording_codes(pooled_dataset, cfg, epoch)
    loaders: Dict[str, DataLoader] = {}
    split_summary: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_dataset = pooled_dataset.subset_by_recording_codes(recording_codes_by_split[split_name], split=split_name)
        loaders[split_name] = _create_dataloader_from_dataset(cfg, split_dataset, split_name)
        split_summary[split_name] = {
            "num_recordings": int(recording_codes_by_split[split_name].shape[0]),
            "num_windows": int(len(split_dataset)),
        }
    return loaders, split_summary


def _move_batch(batch: Dict[str, Any], device: torch.device, non_blocking: bool) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def _query_gpu_stats(device: torch.device) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if device.type != "cuda":
        return stats

    try:
        stats["torch_mem_allocated_mb"] = float(torch.cuda.memory_allocated(device) / (1024**2))
        stats["torch_mem_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024**2))
    except Exception:
        pass

    if shutil.which("nvidia-smi") is None:
        return stats

    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2.0)
        parts = [part.strip() for part in result.stdout.strip().split(",")]
        if len(parts) >= 5:
            stats["gpu_util_pct"] = float(parts[0])
            stats["gpu_mem_used_mb"] = float(parts[1])
            stats["gpu_mem_total_mb"] = float(parts[2])
            stats["gpu_temp_c"] = float(parts[3])
            stats["gpu_power_w"] = float(parts[4])
    except Exception:
        return stats
    return stats


def _format_gpu_stats(stats: Dict[str, float]) -> str:
    if not stats:
        return "gpu=na"
    parts = []
    if "gpu_util_pct" in stats:
        parts.append(f"gpu_util={stats['gpu_util_pct']:.0f}%")
    if "gpu_mem_used_mb" in stats and "gpu_mem_total_mb" in stats:
        parts.append(f"gpu_mem={stats['gpu_mem_used_mb']:.0f}/{stats['gpu_mem_total_mb']:.0f}MB")
    if "torch_mem_allocated_mb" in stats and "torch_mem_reserved_mb" in stats:
        parts.append(f"torch_mem={stats['torch_mem_allocated_mb']:.0f}/{stats['torch_mem_reserved_mb']:.0f}MB")
    if "gpu_temp_c" in stats:
        parts.append(f"gpu_temp={stats['gpu_temp_c']:.0f}C")
    return " ".join(parts) if parts else "gpu=na"


def _query_host_memory_stats() -> Dict[str, float]:
    stats: Dict[str, float] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        stats["proc_rss_mb"] = float(parts[1]) / 1024.0
                        break
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            meminfo = {}
            for line in f:
                key, value = line.split(":", 1)
                meminfo[key.strip()] = value.strip()
        if "MemAvailable" in meminfo:
            stats["host_mem_available_mb"] = float(meminfo["MemAvailable"].split()[0]) / 1024.0
        if "MemTotal" in meminfo:
            stats["host_mem_total_mb"] = float(meminfo["MemTotal"].split()[0]) / 1024.0
    except Exception:
        pass

    try:
        maxrss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        stats["proc_maxrss_mb"] = maxrss_kb / 1024.0
    except Exception:
        pass

    return stats


def _format_host_memory_stats(stats: Dict[str, float]) -> str:
    parts = []
    if "proc_rss_mb" in stats:
        parts.append(f"rss={stats['proc_rss_mb']:.0f}MB")
    if "proc_maxrss_mb" in stats:
        parts.append(f"maxrss={stats['proc_maxrss_mb']:.0f}MB")
    if "host_mem_available_mb" in stats and "host_mem_total_mb" in stats:
        parts.append(f"host_mem_avail={stats['host_mem_available_mb']:.0f}/{stats['host_mem_total_mb']:.0f}MB")
    return " ".join(parts) if parts else "host_mem=na"


def _evaluate_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    if len(loader.dataset) == 0:
        return {}
    metric_list: List[Dict[str, float]] = []
    max_eval_batches = max(0, int(cfg["training"].get("max_eval_batches", 0)))
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device, bool(cfg["training"].get("non_blocking_to_device", True)))
            with _autocast_context(device, cfg):
                out = model(batch)
            regression_weights = compute_omega_regression_weights(batch)
            metric_list.append(
                compute_omega_metrics(
                    out["omega_pred"],
                    batch["omega_target"],
                    regression_weights,
                    pattern_logit=out["pattern_logit"],
                    pattern_target=batch["pattern_target"],
                    pattern_weights=compute_pattern_weights(batch),
                )
            )
            if max_eval_batches > 0 and (batch_idx + 1) >= max_eval_batches:
                break
    return reduce_metric_list(metric_list)


def train_model(cfg: Dict[str, Any], resume_from: Optional[str] = None) -> Dict[str, Any]:
    _set_torch_runtime()
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dynamic_epoch_split = _dynamic_epoch_split_enabled(cfg)
    max_cache_files = int(cfg["training"].get("max_cache_files_per_worker", 2))
    pooled_dataset = OmegaWindowDataset(cfg["dataset"]["index_path"], split="all", max_cache_files=max_cache_files) if dynamic_epoch_split else None
    train_loader = create_dataloader(cfg, "train") if not dynamic_epoch_split else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = build_model(cfg).to(device)
    grad_clip = float(cfg["training"].get("grad_clip_norm", 5.0))
    epochs = int(cfg["training"].get("epochs", 20))
    non_blocking = bool(cfg["training"].get("non_blocking_to_device", True))
    amp_dtype = _resolve_amp_dtype(cfg)
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    latest_path = checkpoint_dir / "latest.pt"
    history_path = checkpoint_dir / "train_history.json"
    start_epoch = 0
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    if resume_from is not None:
        state = torch.load(str(resume_from), map_location=device)
        base_model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if "scaler_state" in state and scaler.is_enabled():
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val_distance_mae_cm", float("inf")))

    model = _maybe_compile_model(base_model, cfg)

    _log(
        "train setup | "
        f"device={device} seq_len={int(cfg['training'].get('sequence_length', 8))} "
        f"chunks_per_batch={int(cfg['training'].get('batch_size', 1))} "
        f"effective_windows_per_step={int(cfg['training'].get('sequence_length', 8)) * int(cfg['training'].get('batch_size', 1))} "
        f"eval_chunks_per_batch={int(cfg['training'].get('eval_batch_size', int(cfg['training'].get('batch_size', 1))))} "
        f"num_workers={int(cfg['training'].get('num_workers', 0))} "
        f"amp={use_amp} amp_dtype={str(amp_dtype).replace('torch.', '')} "
        f"compile={bool(cfg['training'].get('use_compile', True)) and hasattr(torch, 'compile')} "
        f"dynamic_epoch_split={dynamic_epoch_split}"
    )

    best_epoch = -1
    log_interval = max(1, int(cfg["training"].get("log_interval_steps", 20)))
    max_train_steps_per_epoch = max(0, int(cfg["training"].get("max_train_steps_per_epoch", 0)))
    for epoch in range(start_epoch, epochs):
        if dynamic_epoch_split:
            if pooled_dataset is None:
                raise RuntimeError("dynamic epoch split enabled but pooled dataset is missing")
            epoch_loaders, split_summary = _create_dynamic_epoch_dataloaders(cfg, pooled_dataset, epoch)
            train_loader = epoch_loaders["train"]
            val_loader = epoch_loaders["val"]
            test_loader = epoch_loaders["test"]
            _log(
                "epoch split | "
                f"epoch={epoch:03d} "
                f"train_rec={split_summary['train']['num_recordings']} train_win={split_summary['train']['num_windows']} "
                f"val_rec={split_summary['val']['num_recordings']} val_win={split_summary['val']['num_windows']} "
                f"test_rec={split_summary['test']['num_recordings']} test_win={split_summary['test']['num_windows']}"
            )
        else:
            if train_loader is None:
                raise RuntimeError("static train loader is missing")
            val_loader = create_dataloader(cfg, "val")
            test_loader = None
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
                losses = combined_omega_loss(out, batch, cfg)
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

            metric = compute_omega_metrics(
                out["omega_pred"].detach(),
                batch["omega_target"],
                losses["sample_weights"],
                pattern_logit=out["pattern_logit"].detach(),
                pattern_target=batch["pattern_target"],
                pattern_weights=losses["pattern_weights"],
            )
            metric.update(
                {
                    "loss_total": float(losses["loss_total"].item()),
                    "loss_distance": float(losses["loss_distance"].item()),
                    "loss_omega": float(losses["loss_omega"].item()),
                    "loss_pattern": float(losses["loss_pattern"].item()),
                    "loss_velocity": float(losses["loss_velocity"].item()),
                    "loss_acceleration": float(losses["loss_acceleration"].item()),
                    "loss_dynamic": float(losses["loss_dynamic"].item()),
                }
            )
            metric_list.append(metric)

            step_sec = time.perf_counter() - step_start
            prev_step_end = time.perf_counter()
            if step % log_interval == 0:
                gpu_stats = _query_gpu_stats(device)
                host_stats = _query_host_memory_stats()
                remaining_steps = max(0, num_batches - step - 1)
                eta_min = (remaining_steps * step_sec) / 60.0
                _log(
                    f"epoch {epoch:03d} step {step:04d}/{num_batches:04d} "
                    f"loss={metric['loss_total']:.6f} "
                    f"dist_mae={metric['distance_mae_cm']:.3f}cm "
                    f"pattern_acc={metric.get('pattern_acc', 0.0):.3f} "
                    f"omega_loss={metric['loss_omega']:.6f} "
                    f"pattern_loss={metric.get('loss_pattern', 0.0):.6f} "
                    f"delta_omega_loss={metric['loss_velocity']:.3e} "
                    f"acc_omega_loss={metric['loss_acceleration']:.3e} "
                    f"data_wait={data_wait_sec:.3f}s step={step_sec:.3f}s eta={eta_min:.1f}m "
                    f"{_format_gpu_stats(gpu_stats)} {_format_host_memory_stats(host_stats)}"
                )
            if max_train_steps_per_epoch > 0 and (step + 1) >= max_train_steps_per_epoch:
                break

        train_metrics = reduce_metric_list(metric_list)
        val_metrics = _evaluate_epoch(model, val_loader, device, cfg)
        summary = {
            "epoch": int(epoch),
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        history.append(summary)
        state = {
            "epoch": int(epoch),
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_distance_mae_cm": best_val,
            "history": history,
            "config": cfg,
        }
        torch.save(state, latest_path)
        current_val = float(val_metrics.get("distance_mae_cm", train_metrics.get("distance_mae_cm", float("inf"))))
        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            state["best_val_distance_mae_cm"] = best_val
            torch.save(state, best_path)
        _log(
            f"epoch {epoch:03d} done | "
            f"train_dist_mae={train_metrics.get('distance_mae_cm', 0.0):.3f}cm "
            f"val_dist_mae={val_metrics.get('distance_mae_cm', 0.0):.3f}cm"
        )

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    final_test_metrics: Dict[str, float] = {}
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        base_model.load_state_dict(best_state["model_state"])
        if dynamic_epoch_split:
            if pooled_dataset is None:
                raise RuntimeError("dynamic epoch split enabled but pooled dataset is missing")
            epoch_loaders, _ = _create_dynamic_epoch_dataloaders(cfg, pooled_dataset, best_epoch if best_epoch >= 0 else 0)
            test_loader = epoch_loaders["test"]
        else:
            test_loader = create_dataloader(cfg, "test")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        final_test_metrics = _evaluate_epoch(model, test_loader, device, cfg)
        if final_test_metrics:
            _log(
                f"final test | best_epoch={best_epoch:03d} "
                f"test_dist_mae={final_test_metrics.get('distance_mae_cm', 0.0):.3f}cm"
            )
    return {
        "device": str(device),
        "epochs": epochs,
        "best_epoch": int(best_epoch),
        "best_val_distance_mae_cm": float(best_val),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "history_path": str(history_path),
        "final_test_metrics": final_test_metrics,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    output_csv: Optional[str] = None,
) -> Dict[str, float]:
    _set_torch_runtime()
    model.eval()
    if len(loader.dataset) == 0:
        return {}
    rows: List[Dict[str, Any]] = []
    metrics: List[Dict[str, float]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, bool(cfg["training"].get("non_blocking_to_device", True)))
            with _autocast_context(device, cfg):
                out = model(batch)
            regression_weights = compute_omega_regression_weights(batch)
            pattern_weights = compute_pattern_weights(batch)
            metrics.append(
                compute_omega_metrics(
                    out["omega_pred"],
                    batch["omega_target"],
                    regression_weights,
                    pattern_logit=out["pattern_logit"],
                    pattern_target=batch["pattern_target"],
                    pattern_weights=pattern_weights,
                )
            )
            pred_omega_tensor = out["omega_pred"].detach().float()
            pred_omega = pred_omega_tensor.cpu().numpy()
            pred_distance = omega_to_distance_cm(pred_omega_tensor).cpu().numpy()
            pattern_prob = out["pattern_prob"].detach().float().cpu().numpy()
            pattern_pred = (pattern_prob >= float(cfg.get("inference", {}).get("pattern_threshold", 0.5))).astype(np.float32)
            target_omega_tensor = batch["omega_target"].detach().float()
            target_omega = target_omega_tensor.cpu().numpy()
            target_distance = omega_to_distance_cm(target_omega_tensor).cpu().numpy()
            target_pattern = batch["pattern_target"].detach().cpu().numpy()
            target_time = batch["target_time_sec"].detach().cpu().numpy()
            valid = np.isfinite(target_omega).astype(np.float32)
            weight_np = regression_weights.detach().cpu().numpy()
            pattern_weight_np = pattern_weights.detach().cpu().numpy()
            seq_index = batch["sequence_index"].detach().cpu().numpy()
            for i in range(len(pred_omega)):
                rows.append(
                    {
                        "recording_id": batch["recording_id"][i],
                        "sequence_index": int(seq_index[i]),
                        "target_time_sec": float(target_time[i]),
                        "omega_pred": float(pred_omega[i]),
                        "omega_target": float(target_omega[i]),
                        "distance_pred_cm": float(pred_distance[i]),
                        "distance_pred_cm_gated": float(pred_distance[i]) if float(pattern_pred[i]) > 0.5 else float("nan"),
                        "distance_target_cm": float(target_distance[i]),
                        "distance_valid": float(valid[i]),
                        "pattern_prob": float(pattern_prob[i]),
                        "pattern_pred": float(pattern_pred[i]),
                        "pattern_target": float(target_pattern[i]),
                        "sample_weight": float(weight_np[i]),
                        "pattern_sample_weight": float(pattern_weight_np[i]),
                    }
                )
    if output_csv is not None and rows:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return reduce_metric_list(metrics)
