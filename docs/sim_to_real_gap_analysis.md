# Sim-to-Real Gap Analysis & RIR Correction Equation

## 1. Overview

This document reports the quantitative analysis of the gap between simulated (RIR-convolved) training data and real UAV flight recordings, measured through the 43-bin `smooth_d1` feature pipeline. The goal is to derive a correction equation for the RIR bank that reduces the domain gap before model training.

**Key pipeline parameters:**
- `n_fft=512`, `target_sr=48000`, `freq_min=1000`, `freq_max=5000`
- 43 frequency bins: 1031.25 Hz to 4968.75 Hz (93.75 Hz spacing)
- Primary feature: `smooth_d1` (lagged difference of smoothed spectrogram)
- Aggregate metric: `sum_abs_d1` = sum of |smooth_d1| across 43 bins

**Data:**
- Real: 68 flight recordings (~281K subsampled frames)
- Simulated: 14,130 runs (800 sampled for this analysis, ~1M frames)
- RIR bank: 101 distances (5–55 cm, 0.5 cm step), COMSOL-simulated impulse responses at 100 kHz

---

## 2. Observability Boundary

### 2.1 Hard Distance Cutoff

In real data, `pattern=1` (observable comb pattern) drops to **exactly 0%** beyond 25 cm:

| Distance Threshold | pattern=1 Rate | N frames |
|---|---|---|
| d > 15 cm | 5.06% | 263,872 |
| d > 20 cm | 2.88% | 252,102 |
| d > 24 cm | 0.62% | 241,762 |
| **d > 25 cm** | **0.00%** | **239,264** |
| d > 30 cm | 0.00% | 227,649 |

**Finding:** 25 cm is a hard observability wall in real data. No combination of speed makes features visible beyond this distance.

### 2.2 Speed-Distance Relationship (d < 25 cm)

Within 0–25 cm, observability depends on both distance and speed:

| Distance | Observable / Total | Speed of Observable (median) |
|---|---|---|
| 0–5 cm | 70.1% | 0.062 m/s |
| 5–10 cm | 78.0% | 0.120 m/s |
| 10–15 cm | 67.8% | 0.186 m/s |
| 15–20 cm | 51.8% | 0.218 m/s |
| 20–25 cm | 56.5% | 0.259 m/s |
| 25–30 cm | **0.0%** | — |

### 2.3 Fitted Boundary

**Best affine boundary** (F1 = 0.768):
```
Observable ⟺ v_perp > 3.30 × d_meters − 0.50
```

**Best linear boundary** (F1 = 0.530):
```
Observable ⟺ v_perp > 1.25 × d_meters
```

The affine fit is significantly better because the intercept captures the fact that very close distances (< 15 cm) are observable even at near-zero speed.

**Note:** The user's initial hypothesis of `v > 0.65 × d_meter` is too lenient — the actual threshold is steeper (3.30× vs 0.65×), and has a hard wall at 25 cm that no speed can overcome.

### 2.4 Sim vs Real Observability

| Condition | Simulated | Real |
|---|---|---|
| pattern=1 at 25–30 cm, v=0.2–0.5 | **100%** | **0%** |
| pattern=1 at 40–50 cm, v=0.2–0.5 | **94.1%** | **0%** |
| pattern=1 at 50–75 cm, v=0.5–1.0 | **100%** | **0%** |
| Overall pattern=1 rate | **96.4%** | **9.2%** |

**Critical gap:** Simulations predict observable patterns far beyond the real 25 cm cutoff, inflating pattern=1 rate by ~10×.

---

## 3. Intensity Gap (sum_abs_d1)

### 3.1 2D Grid: Real Data

Mean `sum_abs_d1` by (distance, speed):

| Distance | v < 0.05 | v = 0.1–0.2 | v = 0.2–0.5 | v = 0.5–1.0 | v = 2–5 |
|---|---|---|---|---|---|
| 0–5 cm | 0.555 | 0.557 | 0.465 | 0.491 | 0.600 |
| 5–10 cm | 0.582 | 0.525 | 0.522 | 0.557 | 0.560 |
| 15–20 cm | 0.469 | 0.451 | 0.473 | 0.470 | 0.399 |
| 25–30 cm | 0.438 | 0.434 | 0.456 | 0.478 | 0.416 |
| 40–50 cm | 0.412 | 0.431 | 0.417 | 0.434 | 0.439 |
| 100–150 cm | 0.378 | 0.360 | 0.351 | 0.387 | 0.354 |
| 150–250 cm | 0.362 | 0.382 | 0.362 | 0.343 | 0.363 |

**Observations:**
- `sum_abs_d1` ranges from ~0.55 (near) to ~0.33 (noise floor at d >> 100 cm)
- Speed has minimal effect on intensity — the signal strength is primarily distance-dependent
- Decay is gradual, not step-like

### 3.2 2D Grid: Simulated Data

| Distance | v < 0.05 | v = 0.1–0.2 | v = 0.2–0.5 | v = 0.5–1.0 |
|---|---|---|---|---|
| 5–10 cm | 0.375 | 0.397 | 0.454 | 0.550 |
| 15–20 cm | 0.350 | 0.344 | 0.349 | 0.371 |
| 25–30 cm | 0.331 | 0.324 | 0.355 | 0.346 |
| 40–50 cm | 0.310 | 0.308 | 0.309 | 0.341 |

**Note:** Sim has NO data at v > 1.0 m/s (speed range too narrow).

### 3.3 Intensity Ratio (Sim / Real)

| Distance | Sim/Real Ratio |
|---|---|
| 5–10 cm | 0.791 |
| 10–15 cm | 0.767 |
| 15–20 cm | 0.758 |
| 20–25 cm | 0.721 |
| 25–30 cm | 0.753 |
| 30–40 cm | 0.761 |
| 40–50 cm | 0.750 |

**Mean ratio: 0.757 ± 0.019**

→ Sim features are consistently ~24% weaker than real. Apply intensity correction factor **1.32** to RIR output.

### 3.4 Distance-Dependent Attenuation Model

Fitting `sum_abs_d1(d) = noise_floor + A × exp(−d / λ)`:

```
noise_floor = 0.330    (asymptotic level at d >> 100 cm)
A           = 0.180    (signal amplitude at d = 0)
λ           = 75.2 cm  (decay constant)

Signal halves every 52.1 cm
```

At d = 25 cm (hard cutoff): signal_above_floor ≈ 0.108, SNR ≈ 0.33  
At d = 50 cm: signal_above_floor ≈ 0.053, SNR ≈ 0.16  
At d = 100 cm: signal_above_floor ≈ 0.048, SNR ≈ 0.15  

The signal above noise floor at the 25 cm cutoff is only ~33% of the noise level — this explains why the pattern disappears.

---

## 4. Spectral Shape Gap (Bin Profile)

### 4.1 Real Bin Profile

Real data consistently shows a **downward-tilting spectral shape**: low-frequency bins (1031 Hz) are 1.3–1.4× the mean, while high-frequency bins (4969 Hz) are 1.1–1.2× the mean. This tilt is present at all distances.

Representative profile (10–15 cm, normalized to mean=1):

| Bin | Freq (Hz) | Normalized Intensity |
|---|---|---|
| 0 | 1031 | **1.340** |
| 10 | 1969 | 1.109 |
| 20 | 2906 | 0.831 |
| 30 | 3844 | 0.977 |
| 42 | 4969 | **1.175** |

Note: slight uptick at highest bins (4781–4969 Hz) creates a "U-shape" rather than monotone decline.

### 4.2 RIR Spectral Response

The RIR bank shows extreme frequency variation that does **not** match real data:

| Distance | hi/lo ratio (5kHz / 1kHz) |
|---|---|
| 5 cm | 0.109 |
| 15 cm | 0.081 |
| 25 cm | 0.383 |
| 35 cm | 0.230 |
| 45 cm | 0.067 |
| 55 cm | 0.200 |

**RIR has 5–15× stronger low-frequency response than high-frequency**, far more extreme than the 1.3–1.4× tilt in real data. This means the RIR over-attenuates high-frequency bins.

### 4.3 Shape Correlation (Real vs Sim)

After processing through the full pipeline (RIR → ego noise convolution → FFT → smooth_d1):

| Distance | Shape Correlation |
|---|---|
| 5–10 cm | 0.890 |
| 10–15 cm | 0.611 |
| 15–20 cm | 0.627 |
| 20–25 cm | 0.637 |
| 30–40 cm | 0.622 |
| 40–50 cm | 0.607 |
| 50–75 cm | 0.501 |

**Shape correlation is only 0.50–0.64 at most distances** (drops from 0.89 at very close range). The pipeline partially corrects the RIR's extreme tilt but not enough.

### 4.4 Spectral Correction Equation

The correction needed for each frequency bin (averaged across distances):

```
C(f) = real_shape(f) / rir_shape(f)
```

Fitting a cubic polynomial:

```
C(f_norm) = 14.87 × f³ + 6.09 × f² + 0.82 × f + 1.40

where f_norm = (f − 3000) / 3937.5
```

R² = 0.62 (moderate fit due to bin-to-bin variability in the RIR's comb structure).

**Per-bin correction values (key bins):**

| Bin | Freq (Hz) | Mean Correction | Description |
|---|---|---|---|
| 0 | 1031 | 1.35 | Slight boost needed |
| 10 | 1969 | 0.70 | Attenuate (RIR too strong here) |
| 20 | 2906 | 0.82 | Slight attenuation |
| 30 | 3844 | 2.06 | Strong boost (RIR too weak) |
| 35 | 4312 | 1.82 | Strong boost |
| 40 | 4781 | 2.92 | Very strong boost |
| 42 | 4969 | 8.40 | Extreme boost needed |

**Key insight:** The highest bins (4300–5000 Hz) need 2–8× amplification in the RIR to match real spectral shape. This is the dominant source of shape mismatch.

---

## 5. Noise Floor Analysis

Using real data at d > 100 cm as the noise floor reference:

| Distance | Signal Excess | Noise Floor | SNR |
|---|---|---|---|
| 5–10 cm | 0.00366 | 0.00860 | 0.424 |
| 10–15 cm | 0.00328 | 0.00860 | 0.377 |
| 15–20 cm | 0.00225 | 0.00860 | 0.259 |
| 25–30 cm | 0.00212 | 0.00860 | 0.243 |
| 50–75 cm | 0.00067 | 0.00860 | 0.079 |

At 50–75 cm, only **8 of 43 bins** (19%) have SNR > 0.1. The signal is almost entirely buried in noise.

---

## 6. Complete RIR Correction Equation

### 6.1 Three-Component Correction

The RIR bank should be corrected with a multiplicative function `G(f, d)` applied in the frequency domain:

```
RIR_corrected(f, d) = RIR_original(f, d) × G(f, d)
```

where:

```
G(f, d) = G_intensity × G_spectral(f) × G_attenuation(d)
```

**Component 1 — Intensity correction** (uniform scaling):
```
G_intensity = 1.32
```

**Component 2 — Spectral shape correction** (frequency-dependent):
```
G_spectral(f) = 14.87·x³ + 6.09·x² + 0.82·x + 1.40
where x = (f − 3000) / 3937.5
```

This boosts high frequencies (4–5 kHz) by 2–5× and slightly adjusts low frequencies.

**Component 3 — Distance-dependent attenuation** (to add realistic fade-out):
```
G_attenuation(d) = clip(exp(−d_cm / 75.2), 0, 1)
```

This applies a smooth exponential decay with distance, matching the observed signal attenuation in real data.

### 6.2 Observability Gate

After correction, add an observability gate to the training pipeline:

```python
observable = (d_cm < 25.0) and (v_perp > 3.30 * d_meters - 0.50)
```

When `observable = False`, the frame's `pattern_target` should be set to 0, and distance regression loss should be down-weighted or masked.

### 6.3 Where to Apply

The correction should be applied in `rir_bank_augment.py`, inside the `augment_rir_bank()` function, **after** the reflection gain augmentation:

```python
def augment_rir_bank(...):
    # 1. Existing: split direct + reflection, apply alpha
    rir_aug = h_direct + alpha * h_reflection
    
    # 2. NEW: Apply frequency-domain correction
    for i in range(rir_aug.shape[0]):
        H = np.fft.rfft(rir_aug[i])
        freqs = np.fft.rfftfreq(len(rir_aug[i]), d=1.0/fs)
        x = (freqs - 3000.0) / 3937.5
        G_spectral = 14.87*x**3 + 6.09*x**2 + 0.82*x + 1.40
        G_spectral = np.clip(G_spectral, 0.1, 20.0)  # safety clip
        H *= G_spectral * 1.32  # intensity + spectral
        rir_aug[i] = np.fft.irfft(H, n=len(rir_aug[i]))
```

The distance attenuation and observability gate should be applied at the **label/loss level** in the training pipeline, not in the RIR itself (since the RIR bank is indexed by distance already).

---

## 7. Summary of Gaps

| Dimension | Sim | Real | Gap Factor | Fix |
|---|---|---|---|---|
| Overall intensity | 0.76× real | 1.0× | 1.32× | G_intensity |
| High-freq (>4kHz) bins | 0.15–0.50× | 1.0× | 2–8× | G_spectral |
| Pattern=1 rate | 96.4% | 9.2% | 10× | Observability gate |
| Observable distance | up to 75 cm | up to 25 cm | 3× | Hard cutoff + affine boundary |
| Distance range | 5–55 cm | 0–338 cm | 6× | Extend sim range or clamp labels |
| Speed range (v_perp) | 0–1.0 m/s | 0–30+ m/s | 30× | Extend trajectory sampler |
| Shape correlation | — | — | 0.50–0.64 | G_spectral should improve to >0.8 |

## 8. Recommended Implementation Order

1. **Apply spectral + intensity correction to RIR bank** → regenerate simulated data
2. **Add observability gate** with 25 cm hard cutoff + affine boundary → fix pattern imbalance
3. **Extend trajectory sampler** speed range → better speed coverage
4. **Retrain** with corrected data → measure sim-to-real gap reduction
5. **Validate** by comparing corrected sim bin profiles with real profiles
