import React, { useState } from 'react';

const SECTIONS = [
    {
        title: 'Project Overview',
        content: `Team Raceists' Edge AI Dashboard is a real-time battery and chassis anomaly monitoring system built for FSAE electric vehicles. Two independent ML pipelines run simultaneously on a 32 TOPS edge NPU: an unsupervised GRU Autoencoder detects mechanical and electrical anomalies from 21-channel sensor data, and an AttentiveLSTM forecasts battery State of Health and Remaining Useful Life from charge-discharge cycle history.

The dashboard provides sub-50ms inference latency, contextual sensor fault filtering, and live visualisation of all model outputs through a FastAPI backend and React frontend. The entire model graph — from training on the M-SOLO-FAST race dataset to ONNX export and OpenVINO IR conversion — is implemented and version-controlled in this repository.`,
    },
    {
        title: 'Data Acquisition & Feature Engineering',
        content: `Telemetry is extracted from five ROS2 topics logged in the M-SOLO-FAST race bag:

  /vehicle_8/novatel_bottom/rawimux   125 Hz   IMU: accel xyz + gyro xyz (Bottom unit)
  /vehicle_8/novatel_top/rawimux      125 Hz   IMU: accel xyz + gyro xyz (Top unit)
  /vehicle_8/novatel_bottom/odom       60 Hz   Odometry: vel xyz + angular (Bottom)
  /vehicle_8/novatel_top/odom          60 Hz   Odometry: vel xyz + angular (Top)
  /vehicle_8/local_odometry            20 Hz   Fused EKF odometry

All 21 channels are resampled to a common 1 kHz grid using zero-order hold and passed through the preprocessing pipeline in src/pipeline.py. A 32-sample sliding window (WINDOW_SIZE=32) is stored in a ring buffer, giving the model a 32ms view of vehicle dynamics.

Normalization uses frozen min/max bounds per feature (FEATURE_MIN / FEATURE_MAX in src/config.py) calibrated from the training split. These bounds are duplicated in the deployment manifest (export/manifest.json) to guarantee firmware/model consistency.

The six wheel-speed channels (bot_vel_x, bot_vel_y, bot_ang_z, top_vel_x, top_vel_y, loc_vel_x) are the most sensitive fault indicators. A 300 km/h wheel-speed spike from a corrupted CAN frame is clipped by the normalization bounds before it reaches the model.`,
    },
    {
        title: 'Anomaly Detection — Denoising GRU Autoencoder',
        content: `Implemented in: src/model.py, src/train.py, src/infer.py

Architecture:
  Input    [batch, 32, 21]  →  GRU Encoder (hidden=64, 1 layer)
                            →  Bottleneck hidden state [batch, 64]
                            →  GRU Decoder (mirror)
  Output   [batch, 32, 21]  reconstructed telemetry

Training uses a denoising objective: Gaussian noise (sigma=0.05) is added to the input, and the model learns to recover the clean signal. This prevents the identity-mapping shortcut and forces the model to learn the statistical manifold of nominal driving dynamics. Loss: mean squared error. Optimizer: Adam, lr=1e-3. Trained for 50 epochs (FP32, src/train.py).

Inference (src/infer.py):
  A lock-free ring buffer (RingBuffer class) holds the most recent 32 samples. On each new sample arriving from the CAN stream, the buffer slides forward and the model runs in O(1) time. Per-channel squared error at the final timestep is the anomaly score. The AnomalyStateMachine requires three consecutive exceedances above threshold=0.011 (calibrated at 3-sigma above validation mean) before raising a GRU INTERRUPT, reducing false positives from transient noise.

The Deviation Radar on the Dashboard tab maps six representative error channels onto a polar chart, identifying which sensor subsystem is deviating from learned nominal physics.`,
    },
    {
        title: 'Fault Classification — XGBoost Classifier',
        content: `Implemented in: src/classifier.py

After the GRU Autoencoder flags an anomaly window, a second model classifies the specific fault type. Classes:

  0  Healthy         — Nominal track dynamics
  1  IMU_Impact      — Suspension impact or chassis flex (spike on bot/top accel channels)
  2  Wheel_Lockup    — Sudden velocity drop on wheel speed channels (indices 12, 15, 18)
  3  Sensor_Noise    — EMI or connector fault (high-frequency noise on random channel)

Since labeled failure data does not exist for this vehicle, the training set is generated synthetically by injecting the above fault patterns into healthy windows from the M-SOLO-FAST dataset. The injected faults are physically motivated: IMU impacts multiply accel values by 5× at a random onset time; wheel lockup linearly ramps velocity channels to 20% of nominal; sensor noise adds Gaussian noise (sigma=0.3) to a randomly chosen channel.

Feature extraction flattens [B, 32, 21] into [B, 105] (mean, std, min, max, max-diff per channel over the window). XGBoost (100 trees, max_depth=6, multi:softprob objective) trains stratified 80/20 split. The trained model is saved to checkpoints/xgb_classifier.json and loaded at API startup. The fault class is included in the manifest.json.`,
    },
    {
        title: 'Battery Health Forecasting — AttentiveLSTM',
        content: `Implemented in: battery_model_scripted.pt, scaler.pkl, model_card.txt

Architecture: 2-layer LSTM (hidden=128) with a temporal attention head. Parameters: 55,982. Input shape: [batch, 50, 6]. Output: [batch, 2] → [SoH, RUL_norm].

The six input features per timestep (standardized by scaler.pkl, a scikit-learn StandardScaler):
  discharge_median_voltage   — Median cell voltage during discharge, proxy for reversible capacity
  charge_median_voltage      — Median cell voltage during charge, proxy for internal resistance growth
  discharge_capacity_Ah      — Ah actually discharged per cycle
  charge_time_norm           — Normalized charge duration [0–1 across battery lifetime]
  energy_efficiency          — discharge_Wh / charge_Wh, implicit internal-resistance proxy
  cycle_index_norm           — Normalized cycle count [0–1], gives the model its life-position context

Dataset: XJTU Battery Dataset, 55 batteries, ~27,600 charge-discharge cycles. All 6 features are computed per cycle (macro-steps), so the 50-step sequence spans 50 cycles of history.

The temporal attention mechanism up-weights recent cycles during a voltage-sag event, allowing the model to distinguish a transient 8C discharge pulse from genuine capacity fade without misclassifying either. Model is exported as a TorchScript file (torch.jit.script) for zero-dependency loading on the edge backend.`,
    },
    {
        title: 'Sensor Fault Handling & Contextual Gate',
        content: `Implemented in: api.py (simulate_daq), src/infer.py (normalize)

Three-layer preprocessing runs before any model receives data:

Layer 1 — NaN detection
  Disconnected voltage wires produce float('nan') on the CAN bus. The pipeline detects this and applies zero-order hold (last known good value is held in state.last_v). The event is tagged sensor_fault=True with type "NaN — Loose Voltage Wire."

Layer 2 — Garbage filtering
  Broken thermistors report physically impossible temperatures (-40°C or 200°C). Boundary checks (< -20°C or > 150°C for temperature) replace the reading with 30°C. Similarly, wheel-speed CAN spikes >300 km/h are clipped to [0,1] by the gru_normalize function.

Layer 3 — Contextual anomaly gate
  A critical physical anomaly (thermal runaway risk: temperature > 60°C OR voltage < 2.5V) is only raised if the triggering reading passed layers 1 and 2. This prevents a broken thermistor from triggering an emergency stop. The Anomaly Log records the full fault classification at every event with the three-layer outcome.`,
    },
    {
        title: 'Edge Deployment — ONNX Export & OpenVINO IR',
        content: `Implemented in: src/export.py, export/gru_autoencoder.onnx, export/manifest.json

The full export pipeline (python src/export.py) runs four steps:

Step 1 — Load QAT checkpoint
  checkpoints/best_qat.pth contains the Quantization-Aware Training variant of the GRU autoencoder. QAT simulates INT8 quantization during the final 10 training epochs (LR=1e-5) so the model learns to tolerate quantization noise before being frozen.

Step 2 — ONNX export
  torch.onnx.export traces the model with a dummy [1, 32, 21] input and produces a static computation graph. Opset version: 17. Dynamic batch axis is retained so the NPU compiler can batch inferences. The exported file is export/gru_autoencoder.onnx. ONNX checker validates the graph and OnnxRuntime benchmarks CPU latency.

Step 3 — OpenVINO IR conversion (optional)
  export_openvino() calls openvino.tools.mo.convert_model to produce gru_autoencoder.xml + .bin. compress_to_fp16=False preserves the INT8 precision set by QAT. The IR graph is compiled and benchmarked on CPU as a proxy for the target NPU.

Step 4 — Deployment manifest
  export/manifest.json bundles the threshold (0.011), normalization bounds (FEATURE_MIN / FEATURE_MAX), window size, hidden size, alert parameters, and XGBoost classifier path. Firmware reads this manifest at boot — no hardcoded constants in the C++ inference driver.

  File          Size       Budget
  best_fp32.pth  ~220 KB    FP32 development
  best_qat.pth   ~220 KB    QAT source
  gru_autoencoder.onnx  < 60 KB  INT8 edge target`,
    },
    {
        title: 'API & Frontend Architecture',
        content: `Backend: FastAPI (api.py), served by Uvicorn on port 8004.

Endpoints:
  GET  /api/status   Returns model load state, mock flags, threshold, tick count
  POST /api/tick     Runs one simulation step: DAQ → filter → GRU → LSTM → classify → JSON
  POST /api/reset    Clears simulation buffers without unloading models

The /api/tick response includes all telemetry (voltage, current, temperature), both model outputs (soh, rul, anomaly_score, anomaly_shift_pct, feat_err[21]), alert level and message, latency, and a rolling 60-tick history buffer.

Frontend: React 18 + Vite (frontend/), served on port 5175.

Pages:
  Dashboard    Live telemetry cards, GRU + LSTM gauges, deviation radar, history chart, system status
  Anomaly Log  Rolling 30-second event table — every non-nominal tick is captured with timestamp, level, all telemetry values, and the model scores at that moment
  ML Models    Architecture cards with live runtime diagnostics for both models
  Docs         This document

State management: all data flows from the /api/tick poll (400ms interval) into React state. The anomaly log persists across simulation resets. Model status labels (FP32 Active / TorchScript) are fetched from /api/status on load and after every reset, so they never fall back to "MOCK" incorrectly.`,
    },
];

function Section({ title, content }) {
    const [open, setOpen] = useState(false);
    return (
        <div className="card" style={{ overflow: 'hidden' }}>
            <button onClick={() => setOpen(o => !o)} style={{
                width: '100%', padding: '15px 20px',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: 'transparent', border: 'none', cursor: 'pointer',
                textAlign: 'left', fontFamily: 'inherit',
                transition: 'background 0.15s ease',
            }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--surface2)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{title}</span>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                    stroke="var(--text3)" strokeWidth="2" strokeLinecap="round"
                    style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease', flexShrink: 0 }}>
                    <polyline points="6 9 12 15 18 9" />
                </svg>
            </button>
            {open && (
                <div style={{
                    padding: '2px 20px 20px', borderTop: '1px solid var(--border)',
                    fontSize: 12, color: 'var(--text2)', lineHeight: 1.9,
                    whiteSpace: 'pre-line', animation: 'fadeIn 0.2s ease',
                }}>
                    <div style={{ height: 14 }} />
                    {content}
                </div>
            )}
        </div>
    );
}

export default function Docs() {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>Documentation</div>
                <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>
                    Technical reference — Raceists Edge AI BMS · All sections reflect implemented code
                </div>
            </div>
            {SECTIONS.map(s => <Section key={s.title} {...s} />)}
        </div>
    );
}
