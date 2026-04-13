#!/usr/bin/env python3
"""Benchmark batch-size / dataloader-worker sweet spots for training throughput."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_uav_comb.data_pipeline.export_dataset import build_dataset, dataset_artifacts_ready
from ml_uav_comb.features.feature_utils import load_yaml_config
from ml_uav_comb.training.losses import combined_loss
from ml_uav_comb.training.trainer import (
    _batch_to_device,
    _format_gpu_stats,
    _query_gpu_stats,
    build_model,
    create_dataloaders,
    materialize_lazy_modules,
)


def _parse_int_list(value: str) -> List[int]:
    parsed = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed.append(int(chunk))
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _autocast_dtype_from_cfg(cfg: Dict[str, Any]) -> torch.dtype:
    amp_dtype_name = str(cfg["training"].get("amp_dtype", "bfloat16")).strip().lower()
    return torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16


def _configure_runtime(cfg: Dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    if bool(cfg["training"].get("cudnn_benchmark", True)):
        torch.backends.cudnn.benchmark = True
    if bool(cfg["training"].get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def _sample_batch_size(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("unable to infer batch size from batch tensors")


def _cleanup_trial(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    dynamo_mod = getattr(torch, "_dynamo", None)
    if dynamo_mod is not None and hasattr(dynamo_mod, "reset"):
        dynamo_mod.reset()


def benchmark_trial(
    base_cfg: Dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    measure_steps: int,
    gpu_sample_interval: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["num_workers"] = int(num_workers)
    cfg["training"]["persistent_workers"] = int(num_workers) > 0

    train_loader = None
    val_loader = None
    test_loader = None
    model = None
    optimizer = None
    scaler = None
    try:
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _configure_runtime(cfg, device)
        model = build_model(cfg)
        model.to(device)
        non_blocking = bool(cfg["training"].get("non_blocking_to_device", True)) and device.type == "cuda"
        materialize_lazy_modules(model, train_loader, device, non_blocking=non_blocking)

        compile_enabled = bool(cfg["training"].get("use_torch_compile", False)) and hasattr(torch, "compile")
        if compile_enabled:
            compile_mode = str(cfg["training"].get("torch_compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=compile_mode)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        amp_enabled = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
        autocast_dtype = _autocast_dtype_from_cfg(cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)

        loss_cfg = {
            "lambda_likelihood_ce": float(cfg["training"]["lambda_likelihood_ce"]),
            "lambda_measurement_mean": float(cfg["training"]["lambda_measurement_mean"]),
            "lambda_measurement_validity": float(cfg["training"]["lambda_measurement_validity"]),
            "grad_clip_norm": float(cfg["training"]["grad_clip_norm"]),
            "teacher_consistency_weight": float(cfg["training"].get("teacher_consistency_weight", 0.0)),
            "teacher_conf_threshold": float(cfg["training"].get("teacher_conf_threshold", 0.8)),
        }

        model.train(True)
        total_steps = int(warmup_steps) + int(measure_steps)
        iterator = iter(train_loader)
        step_times: List[float] = []
        data_wait_times: List[float] = []
        losses: List[float] = []
        gpu_utils: List[float] = []
        gpu_mem_used: List[float] = []
        samples_seen = 0
        wall_total_sec = 0.0
        max_torch_reserved_mb = 0.0
        max_torch_allocated_mb = 0.0

        wait_start = time.perf_counter()
        for step_idx in range(1, total_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            batch_ready = time.perf_counter()
            data_wait_sec = batch_ready - wait_start
            batch = _batch_to_device(batch, device, non_blocking=non_blocking)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=autocast_dtype):
                pred = model(batch)
                loss_map = combined_loss(
                    pred=pred,
                    batch=batch,
                    lambda_likelihood_ce=loss_cfg["lambda_likelihood_ce"],
                    lambda_measurement_mean=loss_cfg["lambda_measurement_mean"],
                    lambda_measurement_validity=loss_cfg["lambda_measurement_validity"],
                    teacher_consistency_weight=loss_cfg["teacher_consistency_weight"],
                    teacher_conf_threshold=loss_cfg["teacher_conf_threshold"],
                )
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and scaler is not None:
                scaler.scale(loss_map["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=loss_cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_map["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=loss_cfg["grad_clip_norm"])
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                max_torch_allocated_mb = max(
                    max_torch_allocated_mb,
                    float(torch.cuda.max_memory_allocated(device) / (1024**2)),
                )
                max_torch_reserved_mb = max(
                    max_torch_reserved_mb,
                    float(torch.cuda.max_memory_reserved(device) / (1024**2)),
                )
            step_end = time.perf_counter()
            step_sec = step_end - step_start
            wait_start = time.perf_counter()

            if step_idx <= warmup_steps:
                continue

            measured_idx = step_idx - warmup_steps
            data_wait_times.append(data_wait_sec)
            step_times.append(step_sec)
            losses.append(float(loss_map["total"].detach().cpu().item()))
            wall_total_sec += data_wait_sec + step_sec
            samples_seen += _sample_batch_size(batch)

            if device.type == "cuda" and (
                measured_idx == 1
                or measured_idx == measure_steps
                or measured_idx % max(1, gpu_sample_interval) == 0
            ):
                gpu_stats = _query_gpu_stats(device)
                if "gpu_util_pct" in gpu_stats:
                    gpu_utils.append(float(gpu_stats["gpu_util_pct"]))
                if "gpu_mem_used_mb" in gpu_stats:
                    gpu_mem_used.append(float(gpu_stats["gpu_mem_used_mb"]))

        avg_step_sec = statistics.mean(step_times) if step_times else math.nan
        avg_wait_sec = statistics.mean(data_wait_times) if data_wait_times else math.nan
        avg_loss = statistics.mean(losses) if losses else math.nan
        samples_per_sec = (float(samples_seen) / wall_total_sec) if wall_total_sec > 0.0 else math.nan
        steps_per_sec = (float(measure_steps) / wall_total_sec) if wall_total_sec > 0.0 else math.nan
        data_wait_ratio = (avg_wait_sec / max(avg_step_sec + avg_wait_sec, 1.0e-9)) if step_times else math.nan
        avg_gpu_util = statistics.mean(gpu_utils) if gpu_utils else math.nan
        peak_gpu_mem_mb = max(gpu_mem_used) if gpu_mem_used else math.nan
        end_gpu_stats = _query_gpu_stats(device)
        return {
            "status": "ok",
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "warmup_steps": int(warmup_steps),
            "measure_steps": int(measure_steps),
            "avg_loss": avg_loss,
            "avg_step_sec": avg_step_sec,
            "avg_data_wait_sec": avg_wait_sec,
            "data_wait_ratio": data_wait_ratio,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "avg_gpu_util_pct": avg_gpu_util,
            "peak_gpu_mem_used_mb": peak_gpu_mem_mb,
            "peak_torch_mem_allocated_mb": max_torch_allocated_mb,
            "peak_torch_mem_reserved_mb": max_torch_reserved_mb,
            "compile_enabled": bool(compile_enabled),
            "gpu_summary": _format_gpu_stats(end_gpu_stats),
        }
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "status": "oom",
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "error": message,
            }
        raise
    finally:
        _cleanup_trial(train_loader, val_loader, test_loader, model, optimizer, scaler)


def _recommend(results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    ok_rows = [row for row in results if row.get("status") == "ok" and math.isfinite(row.get("samples_per_sec", math.nan))]
    if not ok_rows:
        return None
    ranked = sorted(
        ok_rows,
        key=lambda row: (
            float(row["samples_per_sec"]),
            -float(row.get("avg_data_wait_sec", 0.0)),
            -float(row.get("peak_torch_mem_reserved_mb", 0.0)),
        ),
        reverse=True,
    )
    return ranked[0]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml_uav_comb/configs/rir_runs_global.yaml")
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--batch-sizes", default="16,24,32,48,64,80,96")
    parser.add_argument("--num-workers-list", default="2,4,6")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=40)
    parser.add_argument("--gpu-sample-interval", type=int, default=10)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    ready, reason = dataset_artifacts_ready(cfg)
    if args.force_build:
        print("[dataset] force rebuild requested", flush=True)
        build_dataset(cfg, build_jobs=args.build_jobs)
    elif not ready:
        print(f"[dataset] building artifacts because: {reason}", flush=True)
        build_dataset(cfg, build_jobs=args.build_jobs)
    else:
        print("[dataset] reusing existing artifacts", flush=True)

    batch_sizes = _parse_int_list(args.batch_sizes)
    num_workers_list = _parse_int_list(args.num_workers_list)
    output_csv = Path(args.output_csv) if args.output_csv else Path(cfg["training"]["checkpoint_dir"]) / "sweet_spot_benchmark.csv"
    output_json = Path(args.output_json) if args.output_json else Path(cfg["training"]["checkpoint_dir"]) / "sweet_spot_benchmark.json"

    print(
        f"[benchmark] config={args.config} batch_sizes={batch_sizes} "
        f"num_workers={num_workers_list} warmup={args.warmup_steps} measure={args.measure_steps}",
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    total_trials = len(batch_sizes) * len(num_workers_list)
    trial_idx = 0
    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            trial_idx += 1
            print(
                f"[trial {trial_idx}/{total_trials}] batch_size={batch_size} num_workers={num_workers} begin",
                flush=True,
            )
            trial_start = time.perf_counter()
            row = benchmark_trial(
                cfg,
                batch_size=batch_size,
                num_workers=num_workers,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                gpu_sample_interval=max(1, int(args.gpu_sample_interval)),
            )
            row["trial_wall_sec"] = time.perf_counter() - trial_start
            results.append(row)
            if row["status"] == "ok":
                print(
                    f"[trial {trial_idx}/{total_trials}] ok "
                    f"bs={batch_size} nw={num_workers} "
                    f"samples_per_sec={row['samples_per_sec']:.1f} "
                    f"step={row['avg_step_sec']:.4f}s "
                    f"wait={row['avg_data_wait_sec']:.4f}s "
                    f"gpu_util={row['avg_gpu_util_pct']:.0f}% "
                    f"torch_reserved={row['peak_torch_mem_reserved_mb']:.0f}MB",
                    flush=True,
                )
            else:
                print(
                    f"[trial {trial_idx}/{total_trials}] {row['status']} "
                    f"bs={batch_size} nw={num_workers} error={row.get('error', '')[:160]}",
                    flush=True,
                )

    recommendation = _recommend(results)
    payload = {"results": results, "recommended": recommendation}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_csv, results)

    print(f"[output] csv={output_csv}", flush=True)
    print(f"[output] json={output_json}", flush=True)
    if recommendation is None:
        print("[recommendation] none (no successful trials)", flush=True)
    else:
        print(
            "[recommendation] "
            f"batch_size={recommendation['batch_size']} "
            f"num_workers={recommendation['num_workers']} "
            f"samples_per_sec={recommendation['samples_per_sec']:.1f} "
            f"avg_step={recommendation['avg_step_sec']:.4f}s "
            f"avg_wait={recommendation['avg_data_wait_sec']:.4f}s "
            f"gpu_util={recommendation['avg_gpu_util_pct']:.0f}% "
            f"torch_reserved={recommendation['peak_torch_mem_reserved_mb']:.0f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
