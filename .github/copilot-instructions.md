# Copilot Instructions

## Project Overview

Real-time acoustic ranging system that estimates reflector distance using comb filter effects. Two main subsystems:

1. **Real-time pipeline** (root-level modules): WebSocket audio → STFT feature extraction → cosine fitting → Kalman filtering → Pygame visualization
2. **ML subsystem** (`ml_uav_comb/`): Offline deep learning pipeline that replaces the classical cosine-fitting step with a neural network (omega regression)

## Architecture

### Real-time Pipeline (root modules)

Data flows through a multi-threaded pipeline coordinated in `main.py`:

```
WSClient (network/) → deque → AudioPlayer (playback/) → Queue → AudioProcessor (processing/) → Queue → Visualization (ui/)
```

- **WSClient**: Receives PCM int16 frames over WebSocket with a 16-byte binary header (magic `0xA55A1234`, frame_id, ts_ms, payload_len). Runs in a daemon thread with asyncio.
- **AudioPlayer**: sounddevice callback-driven playback. Converts int16→float32, feeds both the processing queue and AudioRecorder.
- **AudioProcessor**: Core DSP — STFT, d1 differential features, direction detection via cross-correlation (`comb_shift.py`), windowed cosine fitting (`sine_fit_22.py`), and Kalman + RTS smoothing (`range_kf.py`).
- **Visualization**: Pygame main-loop rendering + CSV logging (`sine_fit_log.csv`).

Inter-thread communication uses `collections.deque` (WSClient→AudioPlayer) and `queue.Queue` (AudioPlayer→AudioProcessor→Visualization). A shared `running_fn` callable with `stop=True` sentinel coordinates graceful shutdown.

### ML Subsystem (`ml_uav_comb/`)

Two parallel tracks exist; the **omega regression** track is the active default:

```
audio → offline omega feature extractor → cached .npz windows → UAVCombOmegaNet (PyTorch) → omega_pred → distance_cm
```

Key components:
- **Data pipeline**: `data_pipeline/export_omega_dataset.py` builds cached features; `omega_dataset.py` reads them
- **Model**: `models/uav_comb_omega_net.py` — Per-bin GRU encoder → cross-frequency fusion → omega regressor
- **Training**: `training/omega_trainer.py` with `ContiguousSequenceBatchSampler` for temporal ordering
- **Config**: YAML files in `ml_uav_comb/configs/` (e.g., `omega_default.yaml`)

The legacy **observer/logits** track (`models/uav_comb_observer.py`, `training/trainer.py`) is retained for comparison but is not the active pipeline.

### Core Physics

Distance from comb filter spectral spacing: `d = c × ω / (4π) × 100 cm` where `c = 343 m/s`.

The cosine fitting model is `amplitude(f) = A · cos(ω · f)` with φ=0 constraint. The fitter searches a grid of ω values, solves for A in closed form, and selects by weighted SSE.

## Build & Run Commands

### Dependencies

```bash
pip install -r requirements.txt
# ML subsystem additionally requires: torch (PyTorch)
```

### Real-time System

```bash
python main.py
```

Requires a running WebSocket audio source at the URI configured in `config.py`.

### ML Training Pipeline

```bash
# Build dataset cache
python -m ml_uav_comb.scripts.build_omega_dataset --config ml_uav_comb/configs/omega_default.yaml --build-jobs 4

# Train
python -m ml_uav_comb.scripts.train_omega --config ml_uav_comb/configs/omega_default.yaml

# Evaluate checkpoint
python -m ml_uav_comb.scripts.eval_omega_checkpoint --config ml_uav_comb/configs/omega_default.yaml --checkpoint <path> --split test

# Inference on a WAV file
python -m ml_uav_comb.scripts.infer_omega_wav --config ml_uav_comb/configs/omega_default.yaml --checkpoint <path> --wav rec_1.wav
```

### Tests

```bash
# Run all tests
python -m pytest ml_uav_comb/tests/

# Run a single test file
python -m pytest ml_uav_comb/tests/test_omega_model_forward.py

# Run a single test method
python -m pytest ml_uav_comb/tests/test_omega_model_forward.py::TestOmegaModelForward::test_forward_returns_minimal_dual_head_outputs
```

### Offline Analysis

```bash
# Offline processing with Kalman filter (configure input WAV inside the script)
python offline_process_kalman.py

# Compare two CSV measurement results
python comparison.py
```

## Key Conventions

### Configuration

- **Real-time pipeline**: All parameters live in `config.py` as module-level constants. Each constant has a comment noting which module consumes it. Parameter dicts (`LS_PARAMS`, `KF_PARAMS`, `RTS_PARAMS`, `OUTPUT_FILTER`) group related settings.
- **ML pipeline**: YAML config files in `ml_uav_comb/configs/`. The config dict is passed through the entire pipeline. Use `omega_tiny_debug.yaml` for fast iteration and tests.

### Threading Model

All real-time threads are daemon threads. The shared `running_fn(stop=False)` callable returns `True`/`False` for the running state and accepts `stop=True` to trigger shutdown. Signal handlers (SIGINT) and Pygame events (ESC, QUIT) both call `running_fn(stop=True)`.

### Audio Format

PCM int16, mono, 48kHz throughout. The WebSocket protocol uses little-endian encoding. Float32 normalization (`/ 32768.0`) happens at the AudioPlayer→AudioProcessor boundary.

### ML Cache & Artifacts

- Dataset caches: `ml_uav_comb/cache/<config_name>/` (`.npz` features, `dataset_index.json`, `normalization_stats.npz`)
- Training outputs: `ml_uav_comb/artifacts/<config_name>/` (checkpoints, prediction CSVs)
- Both directories are gitignored. The training script reuses existing caches if present.

### Test Patterns

Tests use `unittest.TestCase` and are run with pytest. Test support utilities in `ml_uav_comb/tests/support.py` provide helpers for generating synthetic WAV files and label CSVs. Tests use `omega_tiny_debug.yaml` config for fast execution.

### Language

Code comments and log messages are predominantly in Chinese (Mandarin). README and documentation are bilingual (Chinese primary, English in ML subsystem docs). Variable names and API interfaces are in English.
