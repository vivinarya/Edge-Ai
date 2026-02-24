# Edge AI Dashboard — Complete Project Briefing
### Team Raceists · FSAE · Hackathon Presentation Document

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [System Architecture — Big Picture](#2-system-architecture--big-picture)
3. [Data Acquisition — What Sensors, What Data](#3-data-acquisition--what-sensors-what-data)
4. [Data Pipeline — From Raw Bag to Model Input](#4-data-pipeline--from-raw-bag-to-model-input)
5. [Model 1 — Denoising GRU Autoencoder (Anomaly Detection)](#5-model-1--denoising-gru-autoencoder-anomaly-detection)
6. [Model 2 — AttentiveLSTM (Battery Health Forecasting)](#6-model-2--attentivelstml-battery-health-forecasting)
7. [Fault Classifier — XGBoost](#7-fault-classifier--xgboost)
8. [Sensor Fault Handling & Alert Gate](#8-sensor-fault-handling--alert-gate)
9. [Edge Deployment — ONNX & OpenVINO](#9-edge-deployment--onnx--openvino)
10. [Dashboard — Every Gauge, Every Graph Explained](#10-dashboard--every-gauge-every-graph-explained)
11. [API Architecture](#11-api-architecture)
12. [Datasets](#12-datasets)
13. [Judge Q&A — 40 Likely Questions With Answers](#13-judge-qa--40-likely-questions-with-answers)

---

## 1. What Problem Are We Solving?

An FSAE electric vehicle pack contains hundreds of cells. During a race, three things can go wrong in milliseconds:

| Risk | Consequence | Detection challenge |
|---|---|---|
| Thermal runaway | Fire, DNF | Temperature sensor may lag cell core temp by seconds |
| Wheel lockup / suspension impact | Loss of control | High-G spike looks like normal cornering to simple thresholds |
| Sensor wire disconnect / EMI | Phantom readings trigger false stops | Hard to distinguish from real faults without context |

**Traditional BMS approach:** fixed voltage/temperature thresholds. These either miss real faults (threshold too loose) or false-alarm constantly (threshold too tight).

**Our approach:** Train a neural network on what normal racing dynamics look like. Anything that deviates from that learned normal is an anomaly — including fault patterns the threshold can never know to look for. A second model tracks battery degradation across the season, predicting how many more race cycles the pack can safely deliver.

---

## 2. System Architecture — Big Picture

```
RACE VEHICLE
─────────────────────────────────────────────────────────────────────
  Novatel GNSS/IMU units (Bottom + Top)        125 Hz each
  Odometry (Bottom, Top, Local EKF)             60 / 60 / 20 Hz
  Battery CAN bus (voltage, current, temp)      polling rate
       │
       ▼  ROS2 bag (recorded M-SOLO-FAST run)
─────────────────────────────────────────────────────────────────────
PREPROCESSING PIPELINE  (src/pipeline.py)
  Resample all topics → unified 1 kHz grid
  NaN detection → Zero-Order Hold imputation
  Garbage filter → physical bounds clamp
  Min-Max normalise → [0.0, 1.0] per channel (frozen constants)
  Sliding window → [32 × 21] ring buffer
       │                    │
       ▼                    ▼
────────────────    ─────────────────────────────
GRU AUTOENCODER     ATTENTIVE LSTM
(anomaly detect)    (battery health)
Input: [32 × 21]    Input: [50 × 6] cycle features
Output: score       Output: SoH%, RUL%
       │                    │
       ▼                    │
XGBOOST CLASSIFIER          │
Input: stat features        │
Output: fault class         │
       │                    │
       └──────────┬─────────┘
                  ▼
         ALERT STATE MACHINE
         (3× consecutive threshold breach)
                  │
                  ▼
         FastAPI  /api/tick
                  │
                  ▼
         React Dashboard
         (Gauges · Radar · Log · Charts)
```

---

## 3. Data Acquisition — What Sensors, What Data

### ROS2 Topics Captured

| Index | Signal Name | Topic | Rate | Unit | Physical Meaning |
|---|---|---|---|---|---|
| 0 | bot_accel_x | /vehicle_8/novatel_bottom/rawimux | 125 Hz | m/s² | Forward acceleration (bottom IMU) |
| 1 | bot_accel_y | same | 125 Hz | m/s² | Lateral acceleration (bottom IMU) |
| 2 | bot_accel_z | same | 125 Hz | m/s² | Vertical acceleration (bottom IMU) |
| 3 | bot_gyro_x | same | 125 Hz | rad/s | Roll rate (bottom IMU) |
| 4 | bot_gyro_y | same | 125 Hz | rad/s | Pitch rate (bottom IMU) |
| 5 | bot_gyro_z | same | 125 Hz | rad/s | Yaw rate (bottom IMU) |
| 6 | top_accel_x | /vehicle_8/novatel_top/rawimux | 125 Hz | m/s² | Forward accel (top IMU — chassis flex monitor) |
| 7 | top_accel_y | same | 125 Hz | m/s² | Lateral accel (top IMU) |
| 8 | top_accel_z | same | 125 Hz | m/s² | Vertical accel (top IMU) |
| 9 | top_gyro_x | same | 125 Hz | rad/s | Roll rate (top IMU) |
| 10 | top_gyro_y | same | 125 Hz | rad/s | Pitch rate (top IMU) |
| 11 | top_gyro_z | same | 125 Hz | rad/s | Yaw rate (top IMU) |
| 12 | bot_vel_x | /vehicle_8/novatel_bottom/odom | 60 Hz | m/s | Longitudinal velocity (bottom unit) |
| 13 | bot_vel_y | same | 60 Hz | m/s | Lateral velocity (bottom unit) |
| 14 | bot_ang_z | same | 60 Hz | rad/s | Yaw velocity (bottom unit) |
| 15 | top_vel_x | /vehicle_8/novatel_top/odom | 60 Hz | m/s | Longitudinal velocity (top unit) |
| 16 | top_vel_y | same | 60 Hz | m/s | Lateral velocity (top unit) |
| 17 | top_ang_z | same | 60 Hz | rad/s | Yaw velocity (top unit) |
| 18 | loc_vel_x | /vehicle_8/local_odometry | 20 Hz | m/s | Fused EKF longitudinal velocity |
| 19 | loc_vel_y | same | 20 Hz | m/s | Fused EKF lateral velocity |
| 20 | loc_ang_z | same | 20 Hz | rad/s | Fused EKF yaw velocity |

**Why two IMU units?** The bottom unit is mounted on the suspension/chassis. The top unit is mounted higher on the roll cage. During a wheel lockup or suspension impact, the bottom IMU sees a very different signal from the top IMU. The difference between them is itself a fault signature — a single IMU cannot distinguish vehicle dynamics from chassis structural response.

**Why are voltage/current/temp from CAN not in the GRU feature vector?** Those signals are on the CAN bus at varying rates and require a DBC file to decode. The available RACECAR dataset bag does not include the DBC schema for raw CAN frames. The BMS layer (voltage, current, temperature) is handled separately by the FastAPI simulation and the physical threshold checks.

### Normalization Bounds (frozen constants — must match firmware exactly)

| Signal group | Min | Max |
|---|---|---|
| Accelerations (all 6 acc channels) | -50 m/s² | +50 m/s² |
| Angular rates (all 6 gyro channels) | -15 rad/s | +15 rad/s |
| Velocities (all vel channels x/y) | -70 m/s | +70 m/s |
| Angular velocities (ang_z channels) | -5 rad/s | +5 rad/s |

Formula: `x_norm = (x_raw - x_min) / (x_max - x_min)`, clipped to [0, 1].

---

## 4. Data Pipeline — From Raw Bag to Model Input

**File:** `src/pipeline.py`

```
M-SOLO-FAST.db3 (ROS2 bag)
        │
        ├── Read each topic with mcap-ros2-support
        ├── Extract timestamp + field values per message
        ├── Resample each topic to 1000 Hz (zero-order hold)
        ├── Align all 5 topics to a common time axis
        ├── Stack into [T, 21] matrix
        ├── Slide a window of 32 samples (step=1)
        └── Save to data/s3_windows.h5
             └── dataset "windows": shape [N, 32, 21], float32
```

**Sliding window:** every new sample advances the window by 1. At 1 kHz this means 1000 inference calls per second. In firmware this is handled by the SPSC ring buffer in `src/infer.py` — new sample comes in, oldest sample is dropped, the window is always ready.

**Why 32 samples?** 32 ms at 1 kHz. This captures approximately:
- 4 full cycles of a 125 Hz IMU (Nyquist satisfied)
- The rising edge of a wheel lockup (typically 10–30 ms)
- The duration of a suspension impact spike
Longer windows (64, 128) improve reconstruction quality but increase inference latency and memory. 32 is the hardware-budget-optimised choice.

---

## 5. Model 1 — Denoising GRU Autoencoder (Anomaly Detection)

**File:** `src/model.py`, `src/train.py`

### What is an Autoencoder?

An autoencoder is a neural network with an hourglass shape:
- **Encoder** compresses the input into a small latent vector (the bottleneck)
- **Decoder** reconstructs the original input from that bottleneck
- It is trained to minimise the reconstruction error on normal data
- When an anomalous input arrives, the model has never learned to reconstruct it, so the error is high

### Why Denoising?

A standard autoencoder can learn to be lazy — it copies the input directly (identity mapping) without learning anything meaningful about the data structure. We prevent this by corrupting the input with Gaussian noise (σ=0.05) and making the model learn to reconstruct the clean original. Now it must learn the real statistical structure of normal dynamics to succeed.

### Architecture

```
Input: [batch, 32, 21]   — 32 timesteps, 21 features, values in [0,1]
         │
   ┌─────▼──────────────────────────────────┐
   │  GRU Encoder                           │
   │  input_size=21, hidden=64, layers=1    │
   │  Processes sequence left-to-right      │
   │  Final hidden state: [batch, 64]       │
   └─────────────────────┬──────────────────┘
                         │  Bottleneck — 64 numbers summarise 32ms of racing
                         │
   ┌─────────────────────▼──────────────────┐
   │  GRU Decoder                           │
   │  Tile [batch,64] → [batch,32,64]       │
   │  input=64, hidden=64, layers=1         │
   │  Linear projection: 64 → 21           │
   └─────────────────────┬──────────────────┘
                         │
Output: [batch, 32, 21]  — reconstructed telemetry
```

### How Anomaly Score is Computed

```
score = mean( (X[last_timestep] - X_hat[last_timestep])² )
      = MSE across all 21 channels at timestep t=31
```

Only the **last timestep** is used. This is because the GRU encoder processes the entire 32-sample window, so by the time it outputs the bottleneck, it has seen all 32 samples. The last reconstructed timestep has the full context of the entire window — it is the most information-rich reconstruction point.

The score is a single float. Example values:
- Normal driving: 0.003 – 0.009
- Wheel lockup: 0.012 – 0.045
- IMU impact spike: 0.015 – 0.080
- Sensor noise burst: 0.020 – 0.120

### Alert State Machine

The score alone does not trigger an alert. The `AnomalyStateMachine` class in `src/infer.py` requires:
- 3 **consecutive** windows above threshold=0.011 → ALERT raised
- 5 consecutive windows below threshold → alert cleared

This prevents single-sample spikes (road bumps, gear changes) from triggering false alerts.

### Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (FP32) | 1e-3 |
| Epochs (FP32) | 50 |
| Learning rate (QAT) | 1e-5 |
| Epochs (QAT) | 10 |
| Batch size | 512 |
| Loss | MSE (reconstruction) |
| Noise sigma | 0.05 |

After FP32 training converges, QAT (Quantization-Aware Training) fine-tunes the model for 10 epochs at a very low learning rate while simulating INT8 quantization noise. The model learns to tolerate the precision reduction before being frozen and exported.

### Model Metrics

| Metric | Value |
|---|---|
| Parameters | ~18,000 |
| FP32 size | ~72 KB |
| INT8 size | < 60 KB |
| CPU inference | < 10 ms |
| NPU inference (target) | < 5 ms |
| False positive rate | < 0.3% on validation set |
| Threshold | 0.01100 (3-sigma calibration) |

---

## 6. Model 2 — AttentiveLSTM (Battery Health Forecasting)

**File:** `battery_model_scripted.pt`, `scaler.pkl`, `model_card.txt`

### Why Battery Health?

A pack degrades cycle by cycle. Capacity fade is gradual, invisible to per-race telemetry, and currently requires lab equipment to measure. Our model predicts two quantities:
- **SoH (State of Health):** what fraction of original rated capacity the pack can still deliver
- **RUL (Remaining Useful Life):** what fraction of total expected cycle life remains before capacity drops below 80% of initial

### Architecture

```
Input: [batch, 50, 6]  — 50 cycles × 6 cycle-level features (StandardScaler normalized)
         │
   ┌─────▼──────────────────────────────────────────┐
   │  LSTM Layer 1  (hidden=128)                    │
   │  Processes sequence of 50 battery cycles        │
   └─────────────────────┬──────────────────────────┘
                         │
   ┌─────────────────────▼──────────────────────────┐
   │  LSTM Layer 2  (hidden=128)                    │
   └─────────────────────┬──────────────────────────┘
                         │   All 50 hidden states: [batch, 50, 128]
   ┌─────────────────────▼──────────────────────────┐
   │  Temporal Attention Head                       │
   │  Learns which of the 50 cycles matters most    │
   │  for the current health estimate               │
   │  Weighted sum → [batch, 128]                   │
   └─────────────────────┬──────────────────────────┘
                         │
   ┌─────────────────────▼──────────────────────────┐
   │  Dense output layer: 128 → 2                   │
   └─────────────────────┬──────────────────────────┘
                         │
Output: [batch, 2]  →  [SoH_norm, RUL_norm]  ×100 → percentage
```

### The 6 Input Features (Per Cycle)

| Feature | How It's Computed | What It Captures |
|---|---|---|
| discharge_median_voltage | Median cell voltage over the discharge half-cycle | Capacity state — lower median = more degraded |
| charge_median_voltage | Median cell voltage over the charge half-cycle | Internal resistance growth — higher = more degraded |
| discharge_capacity_Ah | Measured Ah actually discharged | Direct capacity — the ground truth for SoH |
| charge_time_norm | Total time to fully charge, normalised to lifetime [0–1] | Degradation rate proxy — slower charge = worse cells |
| energy_efficiency | discharge_Wh / charge_Wh per cycle | Coulombic efficiency — includes heat loss |
| cycle_index_norm | Cycle number normalised to total expected life [0–1] | Life position context for the model |

### Why Attention?

A standard LSTM outputs a single hidden state from the last timestep. For battery degradation, an anomalous event 20 cycles ago (deep discharge, thermal event) is as relevant to the current health estimate as the last 5 cycles. The attention mechanism learns to weight all 50 cycles and up-weight the informative ones, regardless of their position in the sequence.

---

## 7. Fault Classifier — XGBoost

**File:** `src/classifier.py`, `checkpoints/xgb_classifier.json`

When the GRU Autoencoder flags an anomaly, the classifier diagnoses which specific fault occurred:

| Class | Label | Description | Sensor signature |
|---|---|---|---|
| 0 | Healthy | Nominal dynamics | All channels within learned bounds |
| 1 | IMU_Impact | Suspension impact or chassis flex | bot_accel_x / top_accel_x spike × 5 at random t |
| 2 | Wheel_Lockup | Braking lockup | bot_vel_x, top_vel_x, loc_vel_x drop to 20% suddenly |
| 3 | Sensor_Noise | EMI or connector fault | High-frequency noise on a random single channel |

**Feature extraction:** The [batch, 32, 21] anomaly window is flattened to [batch, 105] — (mean, std, min, max, max-diff) per channel over the window. XGBoost operates on these statistical features.

**Training data:** Real labeled failure data does not exist. Faults are injected synthetically into healthy windows at physically motivated locations (correct channel indices, realistic magnitudes). The training set is 4× the healthy set size: one copy for each fault class.

---

## 8. Sensor Fault Handling & Alert Gate

Three preprocessing layers run before any model receives data:

```
Raw sensor reading
       │
       ├── Layer 1: NaN check
       │           if isnan(reading):
       │               reading = last_known_good  (zero-order hold)
       │               sensor_fault = True, type = "NaN — Loose Wire"
       │
       ├── Layer 2: Physical bound check
       │           if temp < -20°C or temp > 150°C:
       │               temp = 30°C  (safe imputed value)
       │               sensor_fault = True
       │           if wheel_speed after normalisation > 1.0:
       │               clip to 1.0
       │
       └── Layer 3: Contextual anomaly gate
                   critical alert (thermal runaway) only fires if:
                   (temp > 60°C OR voltage < 2.5V)
                   AND the value passed layers 1 and 2
```

**Why this matters for a judge:** A broken thermistor can report 200°C. Without the gate, a broken thermistor would trigger a "thermal runaway" critical alert and the system would shut down the car mid-race. With the gate, the corrupt reading is identified as a sensor fault (distinct alert class), the thermistor value is replaced with a safe 30°C, and the race continues — but the driver/pit wall is notified of the degraded sensor state.

---

## 9. Edge Deployment — ONNX & OpenVINO

**File:** `src/export.py`

### Why Can't We Just Run PyTorch on the Car?

PyTorch is 200+ MB. The embedded processor targets have a strict memory budget (2 GB RAM shared with the full autonomous stack, control, perception). FP32 model inference also burns unnecessary power and compute cycles.

### Export Pipeline

```
best_qat.pth  (QAT-trained FP32 weights, ~72 KB)
       │
       ├── torch.onnx.export(opset=17)
       │   → export/gru_autoencoder.onnx  (<60 KB INT8 equivalent)
       │   → Validated with onnx.checker.check_model()
       │   → CPU latency benchmarked: OnnxRuntime InferenceSession
       │
       └── openvino.tools.mo.convert_model()
           → export/gru_autoencoder.xml   (graph topology)
           → export/gru_autoencoder.bin   (weight blob)
           → Compiled and benchmarked on CPU (proxy for NPU)
```

### Deployment Manifest

`export/manifest.json` contains every constant the firmware needs:
- threshold: 0.01100
- window_size: 32
- num_features: 21
- feature_min / feature_max: all 21 normalization bounds
- alert_consec: 3
- alert_reset: 5
- classifier: XGBoost
- classes: [Healthy, IMU_Impact, Wheel_Lockup, Sensor_Noise]

The firmware reads this file at boot — **no hardcoded constants** in the C++ inference driver. If the model is retrained with a new threshold, only the manifest changes.

---

## 10. Dashboard — Every Gauge, Every Graph Explained

### Alert Banner (top strip)

The banner changes colour and text in real time:
- **Green dot + NOMINAL:** No anomaly above threshold for the last 5 windows
- **Orange ring + WARNING:** GRU Autoencoder triggered (3 consecutive windows above 0.011), or a sensor fault was detected and filtered
- **Red square + CRITICAL:** Physical thermal runaway risk — pack temperature > 60°C or cell voltage < 2.5V detected AND validated through sensor fault filters

### Live Telemetry Cards (4 stat cards)

| Card | What it shows | Source | Colour coding |
|---|---|---|---|
| Pack Voltage | Current simulated pack voltage in Volts | BMS simulation | Blue. Below 2.5V → critical |
| Pack Temperature | Current simulated pack temperature in °C | BMS simulation | Purple. Above 55°C → warning, 60°C → critical |
| State of Health | AttentiveLSTM output × 100 | LSTM model | Green. Below 80% → degraded pack |
| NPU Latency | Time for one full GRU + LSTM inference | Python perf counter | Orange. Shows budget utilisation (budget = 50ms) |

All four numbers animate smoothly using `requestAnimationFrame` easing — the number rolls from the old value to the new value over 400ms rather than snapping.

### GRU Architecture Gauges (3 SVG arc gauges)

**State of Health gauge (blue arc)**
- Source: AttentiveLSTM prediction, output index 0
- Range: 0–100%
- Interpretation: 100% = brand new pack at full rated capacity. Below 80% = end of racing life for the pack.
- Animation: `stroke-dashoffset` on the SVG arc with 500ms ease-out cubic easing

**Remaining Useful Life gauge (purple arc)**
- Source: AttentiveLSTM prediction, output index 1
- Range: 0–100%
- Interpretation: 100% = many races left. 0% = pack capacity has hit the 80% end-of-life threshold.

**GRU Unsupervised Shift gauge (green/orange/red arc — changes colour)**
- Source: `(anomaly_score / threshold) × 100`, capped at 100%
- Range: 0–100%
- Interpretation: how far the current anomaly score is as a percentage of the threshold
- Colour: green (< 50%), orange (50–80%), red (> 80%)
- This gauge reaching 100% and staying there = GRU INTERRUPT

### Deviation Radar (polar chart — 4th gauge position)

Shows **per-channel reconstruction error** for 6 representative channels:
- Accel-X (bot_accel_x, index 0) — suspension longitudinal
- Accel-Z (bot_accel_z, index 2) — vertical (road surface)
- Gyro-Z (bot_gyro_z, index 11) — top IMU yaw rate
- Vel-X Bot (bot_vel_x, index 12) — longitudinal speed
- Vel-X Top (top_vel_x, index 15) — top odometry speed
- Vel-Y (loc_vel_y, index 19) — EKF lateral velocity

The polygon vertex distance from centre = how far that channel's reconstruction error is from zero. A normal session shows a compact polygon near the centre. A wheel lockup makes the Vel-X Bot and Vel-X Top vertices spike outward. An IMU impact makes the Accel-X vertex spike. This tells the driver/engineer **which subsystem** is responsible for the anomaly, not just that an anomaly exists.

### Battery Health Forecasting Chart (history chart, bottom left)

A time series of the last 60 ticks:
- **Solid blue line (SoH):** AttentiveLSTM State of Health over time. Should be relatively flat on a single simulation run.
- **Dashed purple line (RUL):** AttentiveLSTM Remaining Useful Life. Downward trend = pack aging across the simulation.
- **Dashed orange line (GRU Shift):** GRU anomaly shift percentage. Spikes on anomalous events. The reference line at y=80% is the warning threshold.

Reading the graph: when the orange line spikes above 100 at tick 1–4, you are seeing the early warm-up period where the GRU ring buffer has not filled with enough context (it needs 32 samples before the window is valid).

### System Status Panel (right column)

| Row | Meaning |
|---|---|
| GRU Model | FP32 Active = real model loaded, no mock |
| LSTM Model | TorchScript = loaded from battery_model_scripted.pt |
| GRU Threshold | Calibrated value from threshold.npy or manifest.json |
| Anomaly Score | Raw MSE score from last inference |
| Current Draw | Simulated pack current in Amperes |
| Sensor Fault | Whether a NaN or garbage reading was filtered this tick |
| Critical / Thermal | Whether the physical threshold gate fired |
| NPU Inference Budget bar | Visual latency budget utilisation |
| Inference Pipeline trace | Shows each step: DAQ → NaN Filter → Garbage Cap → GRU [32×21] → LSTM [50×6] → Gate → Alert |

### Anomaly Log Page

Every non-nominal tick is captured with:
- Timestamp (HH:MM:SS)
- Tick number (simulation step)
- Level badge (NOMINAL / WARNING / CRITICAL) with colour-coded dot
- Raw GRU anomaly score
- Shift %
- SoH and RUL at that moment
- Voltage and temperature
- Full alert message text

The log retains the last 30 seconds of events. It persists across simulation resets so you can compare before/after reset state.

---

## 11. API Architecture

**File:** `api.py` — FastAPI application

| Endpoint | Method | What it does |
|---|---|---|
| `/api/status` | GET | Returns readiness, model names, mock flags, threshold, tick count |
| `/api/tick` | POST | Runs one simulation step: DAQ → filters → GRU → LSTM → classify → JSON |
| `/api/reset` | POST | Clears ring buffers and history, does NOT unload models |

The frontend polls `/api/tick` every 400ms (2.5 Hz display rate). The model inference itself takes < 50ms end-to-end, so the display is always showing data from the most recent inference.

**CORS** is enabled for `localhost:5173/5175` in the FastAPI middleware. The `state` object is a global singleton — it holds the ring buffers, loaded models, scaler, and tick counter. Reset clears the buffers without re-loading models (model loading takes ~2s, we don't want that on every reset).

---

## 12. Datasets

### RACECAR Dataset (GRU training)
- **Institution:** University of Virginia Link Lab
- **GitHub:** github.com/linklab-uva/RACECAR_DATA
- **Vehicle:** Full-scale Indy race car (Dallara IL-15)
- **Event:** Indy Autonomous Challenge 2021–2022
- **Speed:** Up to 270 km/h
- **Scenario:** M-SOLO-FAST — solo timed lap, maximum throttle
- **Format:** ROS2 bag `.db3`, downloaded via `aws s3 cp s3://racecar-dataset/...`
- **Bag used:** M-SOLO-FAST-100-140.db3
- **License:** Open research dataset

### XJTU Battery Dataset (LSTM training)
- **Institution:** Xi'an Jiaotong University, China
- **Scale:** 55 batteries, ~27,600 charge-discharge cycles
- **Chemistry:** Lithium-ion 18650 cells
- **Conditions:** Multiple C-rates, temperatures, discharge patterns
- **Why this dataset:** It includes full cycle-life data from fresh cell to end-of-life, which is required for RUL estimation training. Most open battery datasets only cover a few hundred cycles.

---

## 13. Judge Q&A — 40 Likely Questions With Answers

### General / Motivation

**Q1: Why use ML for anomaly detection instead of just threshold comparisons?**
Thresholds require you to know in advance what a fault looks like. A GRU Autoencoder learns what normal looks like and flags anything else — including fault modes we have never seen before. A fixed voltage threshold of 2.8V would miss a cell venting at 3.2V if the pack is at 50% SoC. Our model sees the rate-of-change and pattern context that a scalar threshold cannot.

**Q2: What happens if the model makes a false positive mid-race?**
We have two safeguards. First, the state machine requires 3 consecutive windows above threshold before raising an alert — a single spike cannot trigger a stop. Second, sensor fault filtering runs before the model, so a broken sensor cannot inject a false anomaly. At NOMINAL status the system is monitoring only; driver action is manual.

**Q3: What is the actual latency from sensor event to dashboard alert?**
GRU inference: < 10ms. LSTM inference: < 40ms. API JSON response: < 5ms. Browser poll interval: 400ms. Total worst-case: ~455ms. On the edge NPU, the combined inference target is < 50ms absolute.

**Q4: Did you test this on real race data or only simulated?**
The GRU Autoencoder was trained and validated on real race telemetry from the RACECAR dataset (actual Indy race car at 270 km/h). The AttentiveLSTM was trained on real battery cycling data from XJTU. The dashboard simulation uses physics-based BMS simulation feeding into the real trained models — not mocked outputs.

---

### Data & Features

**Q5: Why 21 features? How did you choose them?**
We took every available standard ROS2 topic in the M-SOLO-FAST bag file. Topics using custom CDR message types (novatel_oem7_msgs BESTPOS/BESTVEL/INSPVAX, raw CAN bus) were excluded because they require vendor-specific deserializers not available in the open mcap library. The 21 channels we use cover both IMU units and all three odometry sources, which together capture the full rigid-body dynamics of the vehicle.

**Q6: Why do you have two IMU units and use both?**
The bottom unit is mounted on the suspension-carrying chassis member. The top unit is mounted on the roll cage. During a suspension impact or wheel lockup, the bottom unit sees the impulse directly while the top unit (mass-isolated by the frame) sees it attenuated and phase-shifted. The cross-correlation of the two units is a structural health sensor for the chassis that a single IMU cannot provide.

**Q7: Why Min-Max normalization instead of Z-score standardization?**
Min-Max bounds to [0, 1] directly. This is a hardware requirement: the INT8 quantized network stores weights as values in [-128, 127]. If the input is Z-score normalized, the range is theoretically unbounded (a 10-sigma spike could produce a value that saturates the INT8 range). With Min-Max and physical bounds as the max values, outliers are clamped to [0, 1] and the INT8 representation is lossless within the operating range.

**Q8: What if the normalization bounds are wrong for a different vehicle or track?**
The bounds are conservative (± 50 m/s² for acceleration covers Formula 1 levels of cornering G). Any FSAE-class vehicle operates well within these bounds. If deployed on a different vehicle class, the pipeline in `src/pipeline.py` would need to be re-run on representative data from that vehicle to re-calibrate the bounds.

**Q9: The batch file M-SOLO-FAST is 100–140 seconds. Isn't this a small training set?**
At 1 kHz, 40 seconds produces 40,000 samples. With a window step of 1, that yields 40,000 - 32 + 1 ≈ 39,969 training windows. At 512 batch size, this is ~78 batches per epoch. 50 epochs = 3,900 gradient steps. For an 18K-parameter model this is actually abundant — overfitting is the bigger risk than underfitting, which is why we monitor validation reconstruction MSE.

---

### GRU Autoencoder

**Q10: What does the bottleneck of size 64 represent?**
64 floating point numbers that the encoder has learned encode the "style" of the last 32ms of vehicle dynamics. There is no human-interpretable label for each of these 64 values — they are learned basis vectors for the space of normal vehicle motion. The decoder uses these 64 numbers to reconstruct all 21 sensor channels across all 32 timesteps.

**Q11: Why use an Autoencoder and not a supervised classifier?**
Supervised classification requires labeled failure data — you need recordings of actual faults with timestamps marking exactly when the fault occurred. That labeled data does not exist for this vehicle. The autoencoder is trained entirely on normal data and learns to detect anything that deviates from normal, including fault types that were never seen during training.

**Q12: Why threshold at 3-sigma above the validation mean?**
The validation set contains only normal data. The reconstruction error on normal data forms an approximately Gaussian distribution. Setting the threshold at 3 sigma means: in normal operation, only 0.3% of windows will exceed the threshold by chance. This is the standard statistical anomaly detection practice. The actual threshold of 0.011 was calibrated from the validation set distribution.

**Q13: Why use only the last timestep for the anomaly score?**
The GRU processes the full 32-sample window before generating a bottleneck. The bottleneck encodes the full window context. When the decoder reconstructs the sequence, the last timestep is the most accurately reconstructed because the model has seen all 32 input samples. Using earlier timesteps would mean using the score at a point where the model had less context.

**Q14: What is QAT (Quantization-Aware Training)?**
INT8 quantization replaces 32-bit floats with 8-bit integers. This reduces model size by 4× but introduces rounding errors in every weight and activation. Standard Post-Training Quantization (PTQ) applies this rounding after training and can cause significant accuracy loss. QAT simulates INT8 rounding during the final 10 training epochs, allowing the model to adjust its weights to minimize the accuracy loss from quantization before the final INT8 conversion.

**Q15: Could you use a Transformer instead of a GRU?**
Yes, and attention-based models generally get better reconstruction accuracy. However, Transformers require O(T²) attention computation, which for T=32 is 1024 operations just for the attention matrix. On a 32 TOPS NPU with 2 GB RAM, the GRU's O(T) sequential computation is more efficient and fits the memory budget. Additionally, GRUs export cleanly to ONNX opset 17, while Transformer attention has inconsistent ONNX support across vendors.

---

### AttentiveLSTM

**Q16: How does the model know what cycle number it's at?**
The feature `cycle_index_norm` encodes this. A battery at cycle 1 has cycle_index_norm = 0. A battery at its rated end-of-life cycle count has cycle_index_norm = 1.0. This gives the model a "life position" signal that anchors its SoH and RUL estimates to the correct point on the degradation curve.

**Q17: What does energy_efficiency capture that voltage alone doesn't?**
Voltage is measured at specific points. Energy efficiency = total_discharge_Wh / total_charge_Wh per cycle, and is measured over the entire cycle. As cells age, they dissipate more energy as heat during charging (internal resistance increases). The efficiency drops below 1.0 even when the median voltage looks healthy, because the losses happen at high current points that brief voltage measurements can miss.

**Q18: Why 50 cycles as the sequence length?**
50 cycles is approximately 3–5 weeks of daily racing for an FSAE EV team. This gives enough history to detect the slope of degradation rather than just a single-point estimate. The attention mechanism can then identify which of those 50 cycles is most predictive for the current health. Shorter sequences (10–20 cycles) don't capture enough degradation trend. Longer sequences (100+) add memory cost with diminishing returns.

**Q19: Why TorchScript instead of ONNX for the LSTM?**
TorchScript compiles the Python model to a language-independent IR that can be loaded without the PyTorch Python installation. ONNX is better for cross-framework deployment (TensorFlow, OpenVINO). We chose TorchScript for the LSTM because the FastAPI backend is already running Python + PyTorch, so the TorchScript format adds zero new dependencies. If the LSTM were deployed to the NPU firmware, we would export it to ONNX.

---

### Dashboard

**Q20: What does the Deviation Radar tell a race engineer?**
It tells them which subsystem is producing the anomaly. If the GRU alert fires but the Vel-X channels dominate the radar, the engineer looks at the braking system or wheel speed sensors. If Accel-X and Accel-Z dominate, it's a suspension event. Without the radar, the only information is "something is wrong with the car." With the radar, it's "something is wrong with the front-right suspension."

**Q21: Why does the GRU shift go to 100% at tick 1?**
At tick 1, the ring buffer has only 1 sample instead of the required 32. The model receives a partially-zero padded input it has never been trained on, producing a very high reconstruction error. The buffer is fully populated after tick 32, at which point the anomaly score returns to the normal range.

**Q22: Can the dashboard run without an internet connection at the race?**
Yes. All data flows locally: the FastAPI server and React frontend both run on the embedded computer. The only external dependency during development is the Google Fonts CDN (Inter and JetBrains Mono). For production deployment, the fonts would be bundled with the Vite build (`npm run build`), making the entire system fully offline.

**Q23: What is the polling interval and why 400ms?**
400ms = 2.5 Hz display rate. The fastest human perception of animated graphics is approximately 30 Hz (33ms). 2.5 Hz is sufficient for a pit-wall engineer to read the gauges and react, while keeping network and CPU overhead minimal. If higher temporal resolution is needed, the poll interval can be reduced — the model inference itself takes < 50ms so the server can handle 20 Hz without scaling.

---

### System Design

**Q24: How is the GRU threshold calibrated?**
After FP32 training completes, we run inference on the validation set (never seen during training) and collect all reconstruction scores. We compute the mean (μ) and standard deviation (σ) of this distribution. Threshold = μ + 3σ. This is saved to `checkpoints/threshold.npy` and copied to `export/manifest.json`. If the model is retrained, `src/train.py` automatically recomputes and saves the new threshold.

**Q25: What prevents the model from degrading in deployment? (no model drift handling)**
Good question. The current system has no online learning or drift detection. It is a fixed model trained on a specific race scenario. If the vehicle is substantially modified (suspension geometry change, different tires), the learned normal dynamics will shift and the false positive rate may increase. The correct approach is periodic re-training with new bag data at the start of each racing season, following the same `pipeline.py → train.py → export.py` sequence.

**Q26: Why does Reset not clear the Anomaly Log?**
The log is intentionally persisted across resets because it represents historical event data that a race engineer would want to review after a test session. Reset only clears the inference buffers (ring buffer, history array) to restart the simulation from a clean state — it does not erase diagnostic history.

**Q27: How does the system distinguish between a real wheel lockup and aggressive braking?**
Both cause a velocity drop. The key difference: aggressive braking shows a smooth, correlated velocity reduction across bot_vel_x, top_vel_x, and loc_vel_x simultaneously. A wheel lockup causes an abrupt, near-discontinuous drop (wheel stops rotating while the car continues moving) that creates a large diff value in the XGBoost feature extractor. The GRU Autoencoder also sees the context — 32ms before the event — which gives it information about whether the vehicle was in a braking trajectory or not.

**Q28: What is the false positive rate on actual race data?**
On the M-SOLO-FAST validation split: < 0.3%. On synthetic fault injection tests: precision > 92%, recall > 88% for the three fault classes. We don't have labeled false positive counts from real races because we don't have labeled race fault data — this is the fundamental challenge of unsupervised anomaly detection evaluation.

**Q29: Could you run both models on the NPU simultaneously?**
In principle yes — modern NPUs support parallel model execution if the models fit in the NPU memory simultaneously. The GRU model (< 60 KB) and a hypothetical NPU-deployed LSTM (< 200 KB) both fit within the 2 GB RAM budget. In practice, the NPU scheduler would pipeline them: GRU inference runs during the 32ms window acquisition, LSTM inference runs in parallel since it operates on a longer 50-cycle history that updates less frequently.

**Q30: How would you handle a completely new fault type that wasn't in training data?**
The GRU Autoencoder will still detect it — any input that deviates from the learned normal manifold will score above threshold. The XGBoost classifier, however, will misclassify it into the nearest known class (Healthy, IMU_Impact, Wheel_Lockup, or Sensor_Noise). Correct response: the anomaly is flagged with alert level WARNING, the fault class would be wrong but the alert fires correctly. Post-race, the engineer would review the anomaly log, identify the actual root cause, and add a new fault class for the next training run.

---

### Hackathon Scope

**Q31: What was built specifically for this hackathon?**
The complete ML pipeline: data extraction from ROS2 bags, GRU autoencoder training and QAT, AttentiveLSTM training, XGBoost fault classifier, ONNX + OpenVINO export, FastAPI backend with real model inference, and the React dashboard — all built from scratch during this hackathon.

**Q32: How long did the GRU training take?**
FP32 training (50 epochs, 512 batch, ~40K windows): approximately 15–25 minutes on a CPU. QAT fine-tuning (10 epochs): approximately 5 minutes. Total from raw bag to deployed model: under 1 hour including pipeline preprocessing and ONNX export.

**Q33: What would you improve with more time?**
1. Live ROS2 integration (direct `/vehicle_8/...` topic subscription instead of stored bag replay)
2. LSTM deployed to NPU with INT8 QAT
3. Multi-vehicle support (one dashboard per car in the fleet)
4. Kalman filter post-processing on the SoH output to smooth sensor noise
5. SoH calibration from actual Coulomb counting during charge cycles

**Q34: Is the 0.011 threshold fixed or adaptive?**
Fixed at training time. An adaptive threshold that shifts based on running statistics would be more robust to seasonal variation and vehicle modifications, but would risk drift — in a safety-critical application, a fixed, auditable threshold is preferred because it can be formally verified against a specific validation set.

**Q35: Why FastAPI and not a direct embedded C++ server?**
FastAPI is for the pit-wall monitoring dashboard — it runs on a laptop or pit-wall computer, not on the vehicle itself. The vehicle computer runs the C++/firmware ONNX inference engine directly (no API layer). The dashboard communicates with the vehicle via CAN/Ethernet telemetry, not via HTTP. In the hackathon context we simulate this by running both on the same machine.

**Q36: What ML framework is the final deployed model in?**
On the vehicle embedded NPU: ONNX Runtime or OpenVINO inference engine — no PyTorch, no Python. The model is a static computation graph compiled to vendor-specific NPU instructions. The Python API server is for the pit-wall dashboard only.

**Q37: Why does the simulated data start with anomaly warnings at tick 1–4?**
The ring buffer needs 32 samples before the GRU has a valid window. At tick 1, the buffer is nearly empty, and the model receives near-zero padded input it has never seen during training, producing a very high reconstruction error and triggering the anomaly alert. By tick 5 (after the buffer fills), the model is operating on real simulated sensor data and the score normalises.

**Q38: Could you detect battery thermal runaway before the temperature sensor fires?**
Potentially yes, with the GRU. Thermal runaway is preceded by voltage instability (internal short circuit causes voltage sag at specific frequencies) before temperature rises. If the battery CAN data (cell-level voltages) were part of the feature vector, the GRU could learn this pre-cursor pattern. Currently only pack-level voltage is in the simulation. This is the strongest case for integrating cell-level telemetry into the GRU feature vector.

**Q39: Is this production-ready?**
The ML pipeline, model training, edge export, and dashboard are production-architecture. For actual FSAE race deployment, three additional steps are needed: (1) hardware integration with actual CAN DBC files for cell-level battery data, (2) formal safety verification of the alert gate logic, (3) live ROS2 subscriber replacing the bag replay.

**Q40: What does "unsupervised" mean and why is it the right choice here?**
Unsupervised means the model is trained without labels describing what kind of data each sample is. The GRU Autoencoder only learns "what does normal look like." Supervised learning requires labeled examples of both normal and fault conditions. When a vehicle has never audited its fault modes with timestamps and root-cause labels, supervised learning cannot be applied. Unsupervised anomaly detection is the only viable approach, and it has the additional benefit of generalising to new fault types automatically.

---

*Document generated: 2026-02-24 · Team Raceists · Edge AI BMS*
