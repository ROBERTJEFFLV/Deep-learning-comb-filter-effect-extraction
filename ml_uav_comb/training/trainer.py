"""Native PyTorch trainer for the UAV comb-motion subsystem."""

from __future__ import annotations

import json
import os
import random
import resource
import shutil
import subprocess
import time
from pathlib import Path
from collections import deque
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Sampler

from ml_uav_comb.data_pipeline.dataset import CachedWindowDataset
from ml_uav_comb.models.uav_comb_crnn import UAVCombCRNN
from ml_uav_comb.models.uav_comb_observer import UAVCombObserver
from ml_uav_comb.training.evaluate import evaluate_model
from ml_uav_comb.training.losses import combined_loss
from ml_uav_comb.training.metrics import compute_batch_metrics, reduce_metric_list


class RecordingGroupedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: CachedWindowDataset,
        *,
        shuffle_groups: bool,
        shuffle_within_group: bool,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.shuffle_groups = bool(shuffle_groups)
        self.shuffle_within_group = bool(shuffle_within_group)
        self.seed = int(seed)
        self._epoch = 0
        self._groups = [list(indices) for _, indices in sorted(dataset.indices_by_cache_path.items())]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self._epoch)
        groups = [list(group) for group in self._groups]
        if self.shuffle_groups:
            rng.shuffle(groups)
        for group in groups:
            if self.shuffle_within_group:
                rng.shuffle(group)
            for idx in group:
                yield idx

    def __len__(self) -> int:
        return len(self.dataset)


class RecordingInterleaveBatchSampler(Sampler):
    def __init__(
        self,
        dataset: CachedWindowDataset,
        *,
        batch_size: int,
        drop_last: bool,
        max_samples_per_recording: int = 2,
        shuffle_recordings: bool = True,
        shuffle_within_recording: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.max_samples_per_recording = max(1, int(max_samples_per_recording))
        self.shuffle_recordings = bool(shuffle_recordings)
        self.shuffle_within_recording = bool(shuffle_within_recording)
        self.seed = int(seed)
        self._epoch = 0
        self._groups = [
            (str(recording_id), list(indices))
            for recording_id, indices in sorted(dataset.indices_by_recording_id.items())
        ]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        groups = [(recording_id, list(indices)) for recording_id, indices in self._groups]
        if self.shuffle_recordings:
            rng.shuffle(groups)
        if self.shuffle_within_recording:
            groups = [(recording_id, rng.sample(indices, len(indices))) for recording_id, indices in groups]

        queue = deque((recording_id, deque(indices)) for recording_id, indices in groups if indices)
        batch: list[int] = []
        batch_counts: Dict[str, int] = {}
        stalled = 0

        while queue:
            recording_id, indices = queue.popleft()
            if batch_counts.get(recording_id, 0) >= self.max_samples_per_recording:
                queue.append((recording_id, indices))
                stalled += 1
                if stalled >= len(queue):
                    if batch and (not self.drop_last or len(batch) == self.batch_size):
                        yield batch
                    batch = []
                    batch_counts = {}
                    stalled = 0
                continue

            batch.append(indices.popleft())
            batch_counts[recording_id] = batch_counts.get(recording_id, 0) + 1
            stalled = 0
            if indices:
                queue.append((recording_id, indices))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                batch_counts = {}

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


class ContiguousRecordingChunkBatchSampler(Sampler):
    def __init__(
        self,
        dataset: CachedWindowDataset,
        *,
        batch_size: int,
        drop_last: bool,
        chunk_step: int = 0,
        min_chunk_span_cm: float = 0.0,
        min_valid_fraction: float = 0.75,
        shuffle_chunks: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.chunk_step = max(1, int(chunk_step) if int(chunk_step) > 0 else self.batch_size // 2)
        self.min_chunk_span_cm = float(min_chunk_span_cm)
        self.min_valid_fraction = float(max(0.0, min(1.0, min_valid_fraction)))
        self.shuffle_chunks = bool(shuffle_chunks)
        self.seed = int(seed)
        self._epoch = 0
        self._chunks = self._build_chunks()

    def _row_distance_target(self, row: Dict[str, Any]) -> float:
        if "measurement_distance_target_cm" in row:
            return self.dataset._row_float(row, "measurement_distance_target_cm", default=float("nan"))
        return self.dataset._row_float(row, "distance_cm", default=float("nan"))

    def _row_distance_mask(self, row: Dict[str, Any]) -> float:
        if "measurement_distance_train_mask" in row:
            return self.dataset._row_float(row, "measurement_distance_train_mask", default=0.0)
        if "valid_dist_gt_mask" in row:
            return self.dataset._row_float(row, "valid_dist_gt_mask", default=0.0)
        return 1.0

    def _row_time(self, row: Dict[str, Any]) -> float:
        if "target_time_sec" in row:
            return self.dataset._row_float(row, "target_time_sec", default=0.0)
        if "center_time_sec" in row:
            return self.dataset._row_float(row, "center_time_sec", default=0.0)
        return 0.0

    def _build_chunks(self) -> List[List[int]]:
        grouped: Dict[str, List[Tuple[float, int, float, float]]] = {}
        for idx, row in enumerate(self.dataset.rows):
            recording_id = str(row.get("recording_id", ""))
            grouped.setdefault(recording_id, []).append(
                (
                    self._row_time(row),
                    idx,
                    self._row_distance_target(row),
                    self._row_distance_mask(row),
                )
            )

        chunks: List[List[int]] = []
        min_valid = max(1, int(round(self.batch_size * self.min_valid_fraction)))
        for _, entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            if len(entries) < self.batch_size:
                continue
            for start in range(0, len(entries) - self.batch_size + 1, self.chunk_step):
                chunk = entries[start : start + self.batch_size]
                valid_targets = [
                    target_cm for _, _, target_cm, mask in chunk if mask > 0.5 and float(target_cm) == float(target_cm)
                ]
                if len(valid_targets) < min_valid:
                    continue
                span_cm = max(valid_targets) - min(valid_targets)
                if span_cm < self.min_chunk_span_cm:
                    continue
                chunks.append([idx for _, idx, _, _ in chunk])
        return chunks

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        chunks = [list(chunk) for chunk in self._chunks]
        if self.shuffle_chunks:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(chunks)
        for chunk in chunks:
            if len(chunk) == self.batch_size:
                yield chunk
            elif chunk and not self.drop_last:
                yield chunk

    def __len__(self) -> int:
        return len(self._chunks)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


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


def _should_log_step(step_idx: int, num_batches: int, warmup_steps: int, interval_steps: int) -> bool:
    if step_idx <= warmup_steps:
        return True
    if interval_steps > 0 and step_idx % interval_steps == 0:
        return True
    return step_idx == num_batches


def _format_gpu_stats(stats: Dict[str, float]) -> str:
    if not stats:
        return "gpu=na"
    parts = []
    if "gpu_util_pct" in stats:
        parts.append(f"gpu_util={stats['gpu_util_pct']:.0f}%")
    if "gpu_mem_used_mb" in stats and "gpu_mem_total_mb" in stats:
        parts.append(f"gpu_mem={stats['gpu_mem_used_mb']:.0f}/{stats['gpu_mem_total_mb']:.0f}MB")
    if "torch_mem_allocated_mb" in stats and "torch_mem_reserved_mb" in stats:
        parts.append(
            f"torch_mem={stats['torch_mem_allocated_mb']:.0f}/{stats['torch_mem_reserved_mb']:.0f}MB"
        )
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
        parts.append(
            f"host_mem_avail={stats['host_mem_available_mb']:.0f}/{stats['host_mem_total_mb']:.0f}MB"
        )
    return " ".join(parts) if parts else "host_mem=na"


def _maybe_mark_cudagraph_step_begin() -> None:
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None and hasattr(compiler_mod, "cudagraph_mark_step_begin"):
        try:
            compiler_mod.cudagraph_mark_step_begin()
        except Exception:
            pass


def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    max_cache_files = int(cfg["training"].get("max_cache_files_per_worker", 2))
    train_ds = CachedWindowDataset(cfg["dataset"]["index_path"], split="train", max_cache_files=max_cache_files)
    val_ds = CachedWindowDataset(cfg["dataset"]["index_path"], split="val", max_cache_files=max_cache_files)
    test_ds = CachedWindowDataset(cfg["dataset"]["index_path"], split="test", max_cache_files=max_cache_files)
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    pin_memory = bool(cfg["training"].get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(cfg["training"].get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg["training"].get("prefetch_factor", 2))
    group_by_recording = bool(cfg["training"].get("group_batches_by_recording", True))
    shuffle_within_group = bool(cfg["training"].get("shuffle_within_recording", True))
    batch_sampler_mode = str(cfg["training"].get("batch_sampler_mode", "plain")).strip().lower()
    max_samples_per_recording = int(cfg["training"].get("max_samples_per_recording_per_batch", 2))
    chunk_step = int(cfg["training"].get("chunk_step", 0))
    min_chunk_span_cm = float(cfg["training"].get("min_chunk_span_cm", 0.0))
    min_valid_fraction = float(cfg["training"].get("min_chunk_valid_fraction", 0.75))
    sampler_seed = int(cfg.get("experiment", {}).get("seed", 0))
    drop_last_train_batch = bool(cfg["training"].get("drop_last_train_batch", True))

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = persistent_workers
        common_kwargs["prefetch_factor"] = max(1, prefetch_factor)

    if batch_sampler_mode == "recording_interleave":
        train_batch_sampler = RecordingInterleaveBatchSampler(
            train_ds,
            batch_size=batch_size,
            drop_last=drop_last_train_batch,
            max_samples_per_recording=max_samples_per_recording,
            shuffle_recordings=True,
            shuffle_within_recording=shuffle_within_group,
            seed=sampler_seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, **{
            key: value for key, value in common_kwargs.items() if key != "batch_size"
        })
    elif batch_sampler_mode == "contiguous_recording_chunks":
        train_batch_sampler = ContiguousRecordingChunkBatchSampler(
            train_ds,
            batch_size=batch_size,
            drop_last=drop_last_train_batch,
            chunk_step=chunk_step,
            min_chunk_span_cm=min_chunk_span_cm,
            min_valid_fraction=min_valid_fraction,
            shuffle_chunks=True,
            seed=sampler_seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            **{key: value for key, value in common_kwargs.items() if key != "batch_size"},
        )
    else:
        train_sampler: Optional[Sampler[int]] = None
        if group_by_recording:
            train_sampler = RecordingGroupedSampler(
                train_ds,
                shuffle_groups=True,
                shuffle_within_group=shuffle_within_group,
                seed=sampler_seed,
            )
        train_loader = DataLoader(
            train_ds,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=drop_last_train_batch,
            **common_kwargs,
        )

    return (
        train_loader,
        DataLoader(val_ds, shuffle=False, **common_kwargs),
        DataLoader(test_ds, shuffle=False, **common_kwargs),
    )


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    scalar_dim = len(cfg["features"]["acoustic_scalar_fields"])
    model_type = str(cfg["model"].get("model_type", "distance_grid_tcn_observer")).strip().lower()
    if model_type in ("distance_grid_tcn_observer", "likelihood_tcn_observer"):
        return UAVCombObserver(cfg=cfg, scalar_dim=scalar_dim, use_stpacc=bool(cfg["features"]["use_stpacc"]))
    if model_type == "crnn":
        return UAVCombCRNN(cfg=cfg, scalar_dim=scalar_dim, use_stpacc=bool(cfg["features"]["use_stpacc"]))
    raise ValueError(f"unsupported model_type: {model_type}")


def _batch_to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool = False) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def materialize_lazy_modules(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> None:
    has_lazy = any(isinstance(module, torch.nn.modules.lazy.LazyModuleMixin) for module in model.modules())
    if not has_lazy:
        return

    iterator = iter(dataloader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("cannot materialize lazy modules from an empty dataloader") from exc
    batch = _batch_to_device(batch, device, non_blocking=non_blocking)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(batch)
    model.train(was_training)


def warmup_compiled_training_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    loss_cfg: Dict[str, float],
    *,
    amp_enabled: bool,
    autocast_dtype: torch.dtype,
    non_blocking: bool = False,
) -> None:
    iterator = iter(dataloader)
    try:
        batch = next(iterator)
    except StopIteration:
        return
    batch = _batch_to_device(batch, device, non_blocking=non_blocking)
    was_training = model.training
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
        pred = model(batch)
        losses = combined_loss(
            pred=pred,
            batch=batch,
            lambda_likelihood_ce=loss_cfg["lambda_likelihood_ce"],
            lambda_measurement_mean=loss_cfg["lambda_measurement_mean"],
            lambda_measurement_validity=loss_cfg["lambda_measurement_validity"],
            teacher_consistency_weight=loss_cfg.get("teacher_consistency_weight", 0.0),
            teacher_conf_threshold=loss_cfg.get("teacher_conf_threshold", 0.8),
        )
    if amp_enabled and scaler is not None:
        scaler.scale(losses["total"]).backward()
    else:
        losses["total"].backward()
    optimizer.zero_grad(set_to_none=True)
    model.train(was_training)


def warmup_compiled_eval_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    autocast_dtype: torch.dtype,
    non_blocking: bool = False,
) -> None:
    iterator = iter(dataloader)
    try:
        batch = next(iterator)
    except StopIteration:
        return
    batch = _batch_to_device(batch, device, non_blocking=non_blocking)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
            model(batch)
    model.train(was_training)


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    loss_cfg: Dict[str, float],
    amp_enabled: bool,
    autocast_dtype: torch.dtype,
    non_blocking: bool,
    epoch_index: int = 0,
    stage_name: str = "",
    phase_name: str = "",
    log_interval_steps: int = 100,
    log_warmup_steps: int = 5,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch_index)
    loss_records = []
    metric_records = []
    num_batches = len(dataloader)
    epoch_start = time.perf_counter()
    wait_start = epoch_start
    for step_idx, batch in enumerate(dataloader, start=1):
        batch_ready = time.perf_counter()
        data_wait_sec = batch_ready - wait_start
        batch = _batch_to_device(batch, device, non_blocking=non_blocking)
        step_start = time.perf_counter()
        _maybe_mark_cudagraph_step_begin()
        if is_train:
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                pred = model(batch)
                losses = combined_loss(
                    pred=pred,
                    batch=batch,
                    lambda_likelihood_ce=loss_cfg["lambda_likelihood_ce"],
                    lambda_measurement_mean=loss_cfg["lambda_measurement_mean"],
                    lambda_measurement_validity=loss_cfg["lambda_measurement_validity"],
                    teacher_consistency_weight=loss_cfg.get("teacher_consistency_weight", 0.0),
                    teacher_conf_threshold=loss_cfg.get("teacher_conf_threshold", 0.8),
                )
        else:
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                    pred = model(batch)
                    losses = combined_loss(
                        pred=pred,
                        batch=batch,
                        lambda_likelihood_ce=loss_cfg["lambda_likelihood_ce"],
                        lambda_measurement_mean=loss_cfg["lambda_measurement_mean"],
                        lambda_measurement_validity=loss_cfg["lambda_measurement_validity"],
                        teacher_consistency_weight=loss_cfg.get("teacher_consistency_weight", 0.0),
                        teacher_conf_threshold=loss_cfg.get("teacher_conf_threshold", 0.8),
                    )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(loss_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(loss_cfg["grad_clip_norm"]))
                optimizer.step()
        loss_records.append({key: float(value.detach().cpu().item()) for key, value in losses.items()})
        metric_records.append(compute_batch_metrics(pred, batch))
        step_end = time.perf_counter()
        step_sec = step_end - step_start
        total_elapsed = step_end - epoch_start
        steps_left = max(num_batches - step_idx, 0)
        avg_step_sec = total_elapsed / max(step_idx, 1)
        eta_sec = avg_step_sec * steps_left
        wait_start = time.perf_counter()

        if _should_log_step(
            step_idx=step_idx,
            num_batches=num_batches,
            warmup_steps=max(0, int(log_warmup_steps)),
            interval_steps=max(0, int(log_interval_steps)),
        ):
            gpu_stats = _query_gpu_stats(device)
            host_stats = _query_host_memory_stats()
            _log(
                f"{stage_name}/{phase_name} epoch={epoch_index + 1} "
                f"step={step_idx}/{num_batches} "
                f"loss={losses['total'].detach().cpu().item():.4f} "
                f"data_wait={data_wait_sec:.3f}s "
                f"step={step_sec:.3f}s "
                f"eta={eta_sec/60.0:.1f}m "
                f"{_format_gpu_stats(gpu_stats)} "
                f"{_format_host_memory_stats(host_stats)}"
            )

    reduced_metrics = reduce_metric_list(metric_records)
    if loss_records:
        loss_keys = loss_records[0].keys()
        reduced_losses = {
            f"loss_{key}": float(sum(item[key] for item in loss_records) / len(loss_records))
            for key in loss_keys
        }
    else:
        reduced_losses = {}
    return {**reduced_losses, **reduced_metrics}


def _save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Dict[str, Any],
    epoch: int,
    stage_name: str,
    metrics: Dict[str, float],
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "scaler_state": None if scaler is None else scaler.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "stage_name": stage_name,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def _stage_a_score(metrics: Dict[str, float]) -> float:
    return -float(metrics.get("loss_total", 0.0))


def _stage_b_score(metrics: Dict[str, float]) -> float:
    return float(metrics.get("measurement_mae_cm_gt", float("inf")))


def train_model(cfg: Dict[str, Any], resume_from: str | Path | None = None) -> Dict[str, Any]:
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and bool(cfg["training"].get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda" and bool(cfg["training"].get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    model.to(device)
    non_blocking = bool(cfg["training"].get("non_blocking_to_device", True)) and device.type == "cuda"
    materialize_lazy_modules(model, train_loader, device, non_blocking=non_blocking)
    compile_enabled = bool(cfg["training"].get("use_torch_compile", False))
    if compile_enabled and hasattr(torch, "compile"):
        compile_mode = str(cfg["training"].get("torch_compile_mode", "reduce-overhead"))
        _log(f"torch.compile enabled mode={compile_mode}")
        model = torch.compile(model, mode=compile_mode)
    _log(
        f"training start device={device} "
        f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)} "
        f"test_windows={len(test_loader.dataset)} batch_size={int(cfg['training']['batch_size'])} "
        f"train_batches={len(train_loader)} num_workers={int(cfg['training']['num_workers'])} "
        f"amp={bool(cfg['training'].get('use_amp', True)) and device.type == 'cuda'} "
        f"pin_memory={bool(cfg['training'].get('pin_memory', torch.cuda.is_available()))} "
        f"tf32={bool(cfg['training'].get('allow_tf32', True)) and device.type == 'cuda'} "
        f"compile={compile_enabled and hasattr(torch, 'compile')}"
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    history: Dict[str, Any] = {"stage_a": [], "stage_b": []}
    best_stage_a = None
    best_stage_b = None
    best_stage_a_score = float("-inf")
    best_stage_b_score = float("inf")
    skip_stage_a = False

    base_loss_cfg = {
        "lambda_likelihood_ce": float(cfg["training"]["lambda_likelihood_ce"]),
        "lambda_measurement_mean": float(cfg["training"]["lambda_measurement_mean"]),
        "lambda_measurement_validity": float(cfg["training"]["lambda_measurement_validity"]),
        "grad_clip_norm": float(cfg["training"]["grad_clip_norm"]),
        "teacher_consistency_weight": float(cfg["training"].get("teacher_consistency_weight", 0.0)),
        "teacher_conf_threshold": float(cfg["training"].get("teacher_conf_threshold", 0.8)),
    }
    amp_enabled = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    amp_dtype_name = str(cfg["training"].get("amp_dtype", "bfloat16")).strip().lower()
    if amp_dtype_name in ("bf16", "bfloat16"):
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)
    _log(f"autocast_dtype={amp_dtype_name}")
    log_interval_steps = int(cfg["training"].get("log_interval_steps", 100))
    log_warmup_steps = int(cfg["training"].get("log_warmup_steps", 5))
    compile_warmup_train = bool(cfg["training"].get("compile_warmup_train", True))
    compile_warmup_eval = bool(cfg["training"].get("compile_warmup_eval", True))

    if compile_enabled and compile_warmup_train:
        _log("compile warmup train begin")
        warmup_compiled_training_step(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_cfg=base_loss_cfg,
            amp_enabled=amp_enabled,
            autocast_dtype=autocast_dtype,
            non_blocking=non_blocking,
        )
        _log("compile warmup train done")
    if compile_enabled and compile_warmup_eval:
        _log("compile warmup eval begin")
        warmup_compiled_eval_step(
            model=model,
            dataloader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            autocast_dtype=autocast_dtype,
            non_blocking=non_blocking,
        )
        _log("compile warmup eval done")

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        resume_stage = str(state.get("stage_name", "")).strip().lower()
        if resume_stage != "stage_a":
            raise ValueError(
                f"resume_from currently supports stage_a checkpoints only; got stage_name={resume_stage!r}"
            )
        model.load_state_dict(state["model_state"])
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = state.get("scaler_state")
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        best_stage_a = resume_path
        best_stage_a_score = _stage_a_score(state.get("metrics", {}))
        history["stage_a"].append(
            {
                "epoch": int(state.get("epoch", 0)),
                "resumed_from": str(resume_path),
                "val": state.get("metrics", {}),
            }
        )
        skip_stage_a = True
        _log(f"resumed from stage_a checkpoint: {resume_path}")
        _log("skipping stage_a and continuing with stage_b")

    if not skip_stage_a:
        for epoch in range(int(cfg["training"]["stage_a_epochs"])):
            _log(f"stage_a epoch {epoch + 1}/{int(cfg['training']['stage_a_epochs'])} begin")
            train_stats = _run_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                loss_cfg=base_loss_cfg,
                amp_enabled=amp_enabled,
                autocast_dtype=autocast_dtype,
                non_blocking=non_blocking,
                epoch_index=epoch,
                stage_name="stage_a",
                phase_name="train",
                log_interval_steps=log_interval_steps,
                log_warmup_steps=log_warmup_steps,
            )
            val_stats = _run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                scaler=None,
                device=device,
                loss_cfg=base_loss_cfg,
                amp_enabled=amp_enabled,
                autocast_dtype=autocast_dtype,
                non_blocking=non_blocking,
                epoch_index=epoch,
                stage_name="stage_a",
                phase_name="val",
                log_interval_steps=log_interval_steps,
                log_warmup_steps=1,
            )
            history["stage_a"].append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
            _log(
                f"stage_a epoch {epoch + 1} done "
                f"train_loss={train_stats.get('loss_total', 0.0):.4f} "
                f"val_loss={val_stats.get('loss_total', 0.0):.4f} "
                f"val_measurement_mae={val_stats.get('measurement_mae_cm_gt', float('nan')):.4f}"
            )
            current_score = _stage_a_score(val_stats)
            if current_score > best_stage_a_score:
                best_stage_a_score = current_score
                best_stage_a = checkpoint_dir / "best_stage_a.pt"
                _save_checkpoint(best_stage_a, model, optimizer, scaler, cfg, epoch + 1, "stage_a", val_stats)
                _log(f"stage_a new best checkpoint: {best_stage_a}")

    if best_stage_a is not None:
        state = torch.load(best_stage_a, map_location=device)
        model.load_state_dict(state["model_state"])
        _log(f"loaded best stage_a checkpoint: {best_stage_a}")

    for epoch in range(int(cfg["training"]["stage_b_epochs"])):
        _log(f"stage_b epoch {epoch + 1}/{int(cfg['training']['stage_b_epochs'])} begin")
        train_stats = _run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_cfg=base_loss_cfg,
            amp_enabled=amp_enabled,
            autocast_dtype=autocast_dtype,
            non_blocking=non_blocking,
            epoch_index=epoch + int(cfg["training"]["stage_a_epochs"]),
            stage_name="stage_b",
            phase_name="train",
            log_interval_steps=log_interval_steps,
            log_warmup_steps=log_warmup_steps,
        )
        val_stats = _run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            loss_cfg=base_loss_cfg,
            amp_enabled=amp_enabled,
            autocast_dtype=autocast_dtype,
            non_blocking=non_blocking,
            epoch_index=epoch + int(cfg["training"]["stage_a_epochs"]),
            stage_name="stage_b",
            phase_name="val",
            log_interval_steps=log_interval_steps,
            log_warmup_steps=1,
        )
        history["stage_b"].append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        _log(
            f"stage_b epoch {epoch + 1} done "
            f"train_loss={train_stats.get('loss_total', 0.0):.4f} "
            f"val_loss={val_stats.get('loss_total', 0.0):.4f} "
            f"val_measurement_mae={val_stats.get('measurement_mae_cm_gt', float('nan')):.4f}"
        )
        current_score = _stage_b_score(val_stats)
        if current_score < best_stage_b_score:
            best_stage_b_score = current_score
            best_stage_b = checkpoint_dir / "best_stage_b.pt"
            _save_checkpoint(best_stage_b, model, optimizer, scaler, cfg, epoch + 1, "stage_b", val_stats)
            _log(f"stage_b new best checkpoint: {best_stage_b}")

    if best_stage_b is not None:
        state = torch.load(best_stage_b, map_location=device)
        model.load_state_dict(state["model_state"])
        _log(f"loaded best stage_b checkpoint: {best_stage_b}")

    _log("starting final test evaluation")
    eval_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
        output_csv=cfg["evaluation"]["prediction_csv"],
    )
    history_path = checkpoint_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _log(f"training finished history_path={history_path}")
    return {
        "device": str(device),
        "resumed_from": None if resume_from is None else str(resume_from),
        "best_stage_a": None if best_stage_a is None else str(best_stage_a),
        "best_stage_b": None if best_stage_b is None else str(best_stage_b),
        "history_path": str(history_path),
        "test_metrics": eval_metrics,
    }
