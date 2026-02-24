# Edge AI Dashboard — Team Raceists

Real-time battery management and chassis anomaly detection system for an FSAE electric vehicle. Two independent ML pipelines run on a 32 TOPS edge NPU: a **Denoising GRU Autoencoder** for unsupervised mechanical anomaly detection and an **AttentiveLSTM** for battery State-of-Health and Remaining Useful Life forecasting.

Live inference is served through a **FastAPI** backend and visualised in a **React** dashboard with sub-50ms end-to-end latency.

---

## System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              SENSOR LAYER                   │
                        │  5 ROS2 topics · 125/60/20 Hz · 21 channels │
                        └───────────────┬─────────────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  Preprocessing Pipeline     │
                         │  NaN → ZOH  │  Garbage cap  │
                         │  Min-Max normalise [0,1]    │
                         │  Ring buffer  [32 × 21]     │
                         └──────┬───────────────┬──────┘
                                │               │
               ┌────────────────▼───┐   ┌───────▼─────────────────┐
               │  GRU Autoencoder   │   │  AttentiveLSTM          │
               │  Input  [32 × 21]  │   │  Input  [50 × 6]        │
               │  Hidden  64        │   │  Output [SoH, RUL]      │
               │  Output [32 × 21]  │   │  Dataset: XJTU Battery  │
               │  Score = MSE[-1]   │   └─────────────────────────┘
               │  Threshold 0.011   │
               │  → XGBoost classify│
               └────────────────────┘
                                │
               ┌────────────────▼──────────────────────────────────┐
               │  FastAPI  /api/tick  →  React Dashboard           │
               │  Alert gate · Anomaly log · Live gauges · Charts  │
               └───────────────────────────────────────────────────┘
```

---

## Datasets

### GRU Autoencoder — RACECAR Dataset
- **Source:** [linklab-uva/RACECAR_DATA](https://github.com/linklab-uva/RACECAR_DATA) (University of Virginia Link Lab)
- **Format:** ROS2 bag (`.db3`) — multi-modal sensor recordings from full-scale autonomous Indy race cars at speeds up to 270 km/h
- **Scenario used:** `M-SOLO-FAST` — solo timed lap, maximum speed
- **Topics extracted:**

| Topic | Rate | Channels | Description |
|---|---|---|---|
| `/vehicle_8/novatel_bottom/rawimux` | 125 Hz | 6 | Bottom IMU — accel xyz + gyro xyz |
| `/vehicle_8/novatel_top/rawimux` | 125 Hz | 6 | Top IMU — accel xyz + gyro xyz |
| `/vehicle_8/novatel_bottom/odom` | 60 Hz | 3 | Bottom odometry — vel xy + angular z |
| `/vehicle_8/novatel_top/odom` | 60 Hz | 3 | Top odometry — vel xy + angular z |
| `/vehicle_8/local_odometry` | 20 Hz | 3 | Fused EKF odometry |

**Total feature vector: 21 channels.** All resampled to 1 kHz, windowed into `[32 × 21]` tensors.

### AttentiveLSTM — XJTU Battery Dataset
- **Source:** [XJTU Sigma Battery Dataset](https://github.com/Wang-ML-Lab/XJTU-Battery-Dataset) — Xi'an Jiaotong University
- **Scale:** 55 batteries, ~27,600 charge-discharge cycles
- **Features (6 per macro-step cycle):**

| Feature | Description |
|---|---|
| `discharge_median_voltage` | Median cell voltage during discharge — capacity proxy |
| `charge_median_voltage` | Median cell voltage during charge — internal resistance proxy |
| `discharge_capacity_Ah` | Ah actually delivered per cycle |
| `charge_time_norm` | Normalized charge duration across battery lifetime [0–1] |
| `energy_efficiency` | discharge_Wh / charge_Wh — Coulombic efficiency |
| `cycle_index_norm` | Normalized cycle count across total life [0–1] |

**Preprocessing:** StandardScaler fitted on training split → `scaler.pkl`.  
**Sequence:** 50 cycles of history per inference → `[50 × 6]` input.

---

## Model Results

### GRU Autoencoder — Anomaly Detection

| Metric | Value |
|---|---|
| Architecture | GRU Encoder (hidden=64) + GRU Decoder |
| Parameters | ~18,000 |
| Training epochs | 50 (FP32) + 10 (QAT) |
| Loss | MSE reconstruction error |
| Anomaly threshold | 0.01100 (3σ above val-set mean) |
| False positive rate | < 0.3% on validation set |
| INT8 model size | < 60 KB |
| CPU inference | < 10 ms |
| NPU inference (target) | < 5 ms |
| ONNX opset | 17 |

**Input:** `[batch, 32, 21]` normalized to [0, 1]  
**Output:** `[batch, 32, 21]` reconstructed telemetry + scalar MSE anomaly score  
**Alert classes (XGBoost):** Healthy · IMU\_Impact · Wheel\_Lockup · Sensor\_Noise

### AttentiveLSTM — Battery Health Forecasting

| Metric | Value |
|---|---|
| Architecture | 2-layer LSTM + temporal attention head |
| Parameters | ~55,982 |
| Dataset | XJTU Battery (55 batteries, ~27,600 cycles) |
| Sequence length | 50 cycles |
| Loss | MSE (SoH + RUL jointly) |
| SoH output | [0, 1] → percentage of rated capacity remaining |
| RUL output | [0, 1] → fraction of cycle life remaining |
| Format | TorchScript (`battery_model_scripted.pt`) |

**Input:** `[batch, 50, 6]` StandardScaler-normalized features  
**Output:** `[batch, 2]` → `[SoH, RUL_normalised]`

---

## Project Structure

```
c:\gru\
├── src/
│   ├── config.py          Frozen constants — normalization bounds, window size, paths
│   ├── pipeline.py        ROS2 bag → 1 kHz resampling → HDF5 dataset builder
│   ├── model.py           Denoising GRU Autoencoder (FP32 + QAT compatible)
│   ├── train.py           FP32 baseline + QAT + 3σ threshold calibration
│   ├── infer.py           Ring buffer + AnomalyStateMachine + ONNX runtime
│   ├── classifier.py      XGBoost fault classifier (4 classes)
│   ├── export.py          ONNX (opset 17) → OpenVINO IR → manifest.json
│   ├── battery_loader.py  XJTU dataset loader
│   └── fusion.py          Sensor fusion utilities
├── frontend/              React + Vite dashboard
│   └── src/
│       ├── App.jsx
│       └── components/    Gauge, Radar, Chart, Log, Models, Docs, Status
├── checkpoints/
│   ├── threshold.npy      Calibrated anomaly threshold (0.011)
│   ├── xgb_classifier.json XGBoost fault classifier
│   └── lbl_encoder.pkl    Class label encoder
├── export/
│   ├── manifest.json      Deployment manifest (threshold + normalization bounds)
│   └── xgb_classifier.json
├── api.py                 FastAPI backend — /api/status, /api/tick, /api/reset
├── app.py                 Streamlit dashboard (legacy)
├── scaler.pkl             StandardScaler for LSTM input features
├── battery_model_scripted.pt  AttentiveLSTM TorchScript
├── model_card.txt         LSTM model card
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn[standard] streamlit joblib xgboost
cd frontend && npm install
```

### 2. Download RACECAR dataset (GRU training data)

```bash
# List available scenarios
aws s3 ls s3://racecar-dataset/RACECAR-ROS2/ --no-sign-request

# Download M-SOLO-FAST bag (~few GB)
aws s3 cp s3://racecar-dataset/RACECAR-ROS2/S3/M-SOLO-FAST/ data/M_SOLO_FAST/ \
    --recursive --no-sign-request
```

### 3. Build the GRU training dataset

```bash
python src/pipeline.py
# → data/s3_windows.h5  (32-sample windows, 21 features)
```

### 4. Train models

```bash
# GRU Autoencoder (FP32 + QAT)
python src/train.py
# → checkpoints/best_fp32.pth
# → checkpoints/best_qat.pth
# → checkpoints/threshold.npy

# XGBoost fault classifier
python src/classifier.py
# → checkpoints/xgb_classifier.json
```

### 5. Export to ONNX + OpenVINO

```bash
python src/export.py
# → export/gru_autoencoder.onnx   (< 60 KB)
# → export/gru_autoencoder.xml/.bin (OpenVINO IR)
# → export/manifest.json
```

### 6. Run the dashboard

```bash
# Terminal 1 — FastAPI backend
uvicorn api:app --host 0.0.0.0 --port 8004

# Terminal 2 — React frontend
cd frontend && npm run dev
# → http://localhost:5173
```

---

## Hardware Budget

| Resource | Used | Budget | Margin |
|---|---|---|---|
| GRU INT8 model | < 60 KB | 60 KB | — |
| GRU inference | < 10 ms CPU · < 5 ms NPU | 50 ms | 5–10× |
| LSTM inference | < 40 ms CPU | 50 ms | 1.2× |
| Ring buffer | 672 bytes (32×21) | 2 GB RAM | 99.9% |
| False positive rate | < 0.3% | < 0.3% | — |

---

## Sensor Fault Handling

The preprocessing pipeline distinguishes three kinds of signal degradation before any model receives data:

1. **NaN imputation** — disconnected wire → zero-order hold on last valid reading, flagged as `sensor_fault`
2. **Garbage filter** — physically impossible values (temp < -20°C or > 150°C, wheel speed > 300 km/h) → clamped to safe defaults
3. **Contextual gate** — critical thermal/voltage alerts only fire if both filters passed, preventing a broken sensor from triggering a false emergency stop

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML training | PyTorch 2.2, scikit-learn, XGBoost |
| Edge export | ONNX (opset 17), OpenVINO IR |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, Vite, Recharts |
| Legacy UI | Streamlit |
| Data pipeline | ROS2 bag, MCAP, h5py |

---

*Team Raceists — FSAE Edge AI BMS*
