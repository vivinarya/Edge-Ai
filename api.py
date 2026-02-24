"""
api.py — FastAPI Backend for Raciests BMS React Dashboard
Real models only: GRU (checkpoints/best_fp32.pth) + LSTM (battery_model_scripted.pt)

Run from the c:\\gru directory:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch
import pickle
import joblib
import json
import time
import sys
from pathlib import Path
from collections import deque

# ── Resolve project root (same dir as this file) ──────────────────────────────
ROOT = Path(__file__).parent.resolve()

sys.path.insert(0, str(ROOT / "src"))
from model  import DenoisingGRUAutoencoder
from config import (
    WINDOW_SIZE, NUM_FEATURES,
    FEATURE_MIN, FEATURE_MAX,
    ALERT_CONSECUTIVE, ALERT_RESET,
)
from infer import MockCANStream, normalize as gru_normalize

# ── Absolute paths ─────────────────────────────────────────────────────────────
GRU_CHECKPOINT  = ROOT / "checkpoints" / "best_fp32.pth"
GRU_THRESHOLD_F = ROOT / "checkpoints" / "threshold.npy"
EXPORT_MANIFEST = ROOT / "export"      / "manifest.json"
LSTM_MODEL      = ROOT / "battery_model_scripted.pt"
SCALER_FILE     = ROOT / "scaler.pkl"

print(f"\n[BMS API] Project root: {ROOT}")
print(f"  GRU  checkpoint : {GRU_CHECKPOINT}  exists={GRU_CHECKPOINT.exists()}")
print(f"  GRU  threshold  : {GRU_THRESHOLD_F} exists={GRU_THRESHOLD_F.exists()}")
print(f"  Manifest        : {EXPORT_MANIFEST} exists={EXPORT_MANIFEST.exists()}")
print(f"  LSTM model      : {LSTM_MODEL}      exists={LSTM_MODEL.exists()}")
print(f"  Scaler          : {SCALER_FILE}     exists={SCALER_FILE.exists()}\n")

app = FastAPI(title="Raciests BMS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Threshold from manifest (most accurate source) ─────────────────────────────
def _load_threshold() -> float:
    if EXPORT_MANIFEST.exists():
        with open(EXPORT_MANIFEST) as f:
            return float(json.load(f)["threshold"])
    if GRU_THRESHOLD_F.exists():
        return float(np.load(str(GRU_THRESHOLD_F))[0])
    raise RuntimeError("No threshold file found (export/manifest.json or checkpoints/threshold.npy)")

# ── Global BMS State ───────────────────────────────────────────────────────────
class BMSState:
    def __init__(self):
        self.tick            = 0
        self.seq_buf_lstm    = np.zeros((50, 6),              dtype=np.float32)
        self.seq_buf_gru     = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
        self.can_stream      = MockCANStream(fault_at_sec=999999)
        self.last_v          = 3.7
        self.history         = deque(maxlen=60)

        # ── Load scaler (required — raises if missing) ─────────────────────────
        if not SCALER_FILE.exists():
            raise FileNotFoundError(f"scaler.pkl not found at {SCALER_FILE}")
        self.scaler = joblib.load(str(SCALER_FILE))
        print(f"[BMS API] ✓ scaler loaded from {SCALER_FILE}")

        # ── Load GRU (required — raises if missing) ────────────────────────────
        if not GRU_CHECKPOINT.exists():
            raise FileNotFoundError(f"GRU checkpoint not found at {GRU_CHECKPOINT}")
        self.gru_model = DenoisingGRUAutoencoder()
        self.gru_model.load_state_dict(
            torch.load(str(GRU_CHECKPOINT), map_location="cpu", weights_only=False)
        )
        self.gru_model.eval()
        self.gru_threshold = _load_threshold()
        print(f"[BMS API] ✓ GRU loaded (threshold={self.gru_threshold:.6f})")

        # ── Load LSTM (required — raises if missing) ───────────────────────────
        if not LSTM_MODEL.exists():
            raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL}")
        self.lstm_model = torch.jit.load(str(LSTM_MODEL), map_location="cpu")
        self.lstm_model.eval()
        print(f"[BMS API] ✓ LSTM loaded from {LSTM_MODEL}")

        print("[BMS API] All models loaded — no mocks active.\n")

# Instantiate at startup — crashes loudly if any file is missing
state = BMSState()


# ── Signal simulation helpers ───────────────────────────────────────────────────

def simulate_daq(t: int, can_stream: MockCANStream):
    """Simulate one DAQ tick with realistic FSAE profiles + occasional faults."""
    base_v = 3.7 - (t % 100) * 0.01          # discharge curve
    base_i = 40.0 * np.sin(t * 0.1)           # regen / throttle oscillation
    base_t = 30.0 + (t % 100) * 0.3           # thermal rise to ~60 °C

    # Rare dangerous events
    if np.random.rand() > 0.98: base_t += 35.0  # thermal spike above 60 °C
    if np.random.rand() > 0.98: base_v -= 1.5   # voltage sag below 2.5 V

    noisy_v = base_v + np.random.normal(0, 0.05)
    noisy_i = base_i + np.random.normal(0, 0.25)
    noisy_t = base_t + np.random.normal(0, 0.10)

    # Sensor fault injection (1 % probability)
    if   np.random.rand() > 0.99: noisy_t = -40.0        # broken thermistor
    elif np.random.rand() > 0.99: noisy_v = float('nan') # disconnected wire

    # GRU feature vector from MockCANStream (correct physics distribution)
    raw_sample = can_stream.read_sample()
    if np.random.rand() > 0.98: raw_sample[12] = 85.0    # wheel speed spike 300 km/h
    gru_features = gru_normalize(raw_sample)              # → [0, 1] via FEATURE_MIN/MAX

    # IMU anomaly when voltage sags (physical correlation)
    if not np.isnan(noisy_v) and noisy_v < 2.5:
        gru_features[0] = 1.0
        gru_features[6] = 1.0

    return noisy_v, noisy_i, noisy_t, gru_features


def run_inference(s: BMSState):
    """Run GRU (anomaly) + LSTM (SoH/RUL) on current buffers. No mocks."""
    t0 = time.perf_counter()

    # ── 1. GRU Autoencoder (anomaly detection) ─────────────────────────────────
    # Input: [1, WINDOW_SIZE=32, NUM_FEATURES=21]  normalized to [0,1]
    x_gru  = torch.tensor(s.seq_buf_gru, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        x_hat  = s.gru_model(x_gru, add_noise=False)
        diff   = (x_gru - x_hat).numpy()                # [1, 32, 21]
        diff_sq = (diff ** 2)[0, -1, :]                 # per-feature error at last timestep [21]
        score   = float(diff_sq.mean())                 # scalar MSE
    alert = score >= s.gru_threshold

    # ── 2. AttentiveLSTM (SoH + normalized RUL) ────────────────────────────────
    # Input: [1, 50, 6]  scaled with scaler.pkl (StandardScaler)
    x_scaled = s.scaler.transform(s.seq_buf_lstm)       # (50, 6) → rescaled
    x_lstm   = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = s.lstm_model(x_lstm)                     # [1, 2]
        soh  = float(np.clip(pred[0, 0].item() * 100.0, 0, 100))
        rul  = float(np.clip(pred[0, 1].item() * 100.0, 0, 100))

    latency = (time.perf_counter() - t0) * 1000
    return score, alert, soh, rul, diff_sq.tolist(), latency


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return {
        "ready":         True,
        "mock_gru":      False,
        "mock_lstm":     False,
        "gru_threshold": state.gru_threshold,
        "tick":          state.tick,
        "gru_checkpoint": str(GRU_CHECKPOINT),
        "lstm_model":     str(LSTM_MODEL),
        "scaler":         str(SCALER_FILE),
    }


@app.post("/api/tick")
def api_tick():
    """Advance simulation one step and return all telemetry data."""
    s = state
    s.tick += 1
    t = s.tick

    v_raw, i_raw, temp_raw, gru_features = simulate_daq(t, s.can_stream)

    sensor_fault_type = None

    # NaN imputation: zero-order hold (last valid voltage)
    if np.isnan(v_raw):
        v = s.last_v
        sensor_fault_type = "NaN — Loose Voltage Wire"
    else:
        v = float(v_raw)
        s.last_v = v

    # Garbage filter: impossible physical temperature readings
    if temp_raw < -20.0 or temp_raw > 150.0:
        temp = 30.0
        sensor_fault_type = f"Garbage — Thermistor {temp_raw:.1f}°C"
    else:
        temp = float(temp_raw)

    sensor_fault     = sensor_fault_type is not None
    critical_anomaly = (temp > 60.0 or v < 2.5) and not sensor_fault

    # ── Update sequence buffers ──────────────────────────────────────────────
    # LSTM buffer: 6 features per the model card:
    # [discharge_median_voltage, charge_median_voltage, discharge_capacity_Ah,
    #  charge_time_norm, energy_efficiency, cycle_index_norm]
    pseudo_lstm = np.array([
        v,                          # 0: discharge_median_voltage (using live pack V as proxy)
        4.05,                       # 1: charge_median_voltage    (FSAE pack fully charged)
        2.5,                        # 2: discharge_capacity_Ah    (nominal cell capacity)
        (t % 100) / 100.0,          # 3: charge_time_norm
        0.95 - (t / 10000.0),       # 4: energy_efficiency degrading slowly over cycles
        min(t / 2000.0, 1.0),       # 5: cycle_index_norm
    ], dtype=np.float32)

    s.seq_buf_lstm[:-1] = s.seq_buf_lstm[1:]
    s.seq_buf_lstm[-1]  = pseudo_lstm

    # GRU buffer: [WINDOW_SIZE=32, NUM_FEATURES=21] normalized [0,1]
    s.seq_buf_gru[:-1]  = s.seq_buf_gru[1:]
    s.seq_buf_gru[-1]   = gru_features

    # ── Run real inference ───────────────────────────────────────────────────
    score, gru_alert, soh, rul, feat_err, lat = run_inference(s)

    shift_pct = float(min((score / s.gru_threshold) * 100.0, 100.0))

    # ── Alert logic (sensor fault gate prevents false physical anomalies) ────
    if critical_anomaly:
        alert_level = "critical"
        alert_msg   = "Thermal Runaway Risk — Temp >60°C or Voltage <2.5V"
    elif sensor_fault:
        alert_level = "warn"
        alert_msg   = f"Sensor Fault Filtered — {sensor_fault_type}"
    elif gru_alert:
        alert_level = "warn"
        alert_msg   = f"GRU INTERRUPT — Unsupervised Shift Score {score:.5f} >= {s.gru_threshold:.5f}"
    else:
        alert_level = "nominal"
        alert_msg   = "Nominal Track / Race Conditions"

    point = {
        "tick": t, "soh": round(soh, 2), "rul": round(rul, 2),
        "score": round(score, 6), "latency": round(lat, 2)
    }
    s.history.append(point)

    # ── Live terminal monitoring output ─────────────────────────────────────
    ALERT_ICONS = {"nominal": "✅", "warn": "⚠️ ", "critical": "🚨"}
    icon = ALERT_ICONS.get(alert_level, "  ")
    print(
        f"[Tick {t:>5}] "
        f"V={v:.3f}V  I={float(i_raw):+6.1f}A  T={temp:.1f}°C  │  "
        f"SoH={soh:5.2f}%  RUL={rul:5.2f}%  │  "
        f"GRU={score:.5f} (Shift={shift_pct:5.1f}%)  │  "
        f"Lat={lat:5.2f}ms  {icon} {alert_level.upper()}"
    )

    return JSONResponse({
        "tick":              t,
        "voltage":           round(v, 3),
        "current":           round(float(i_raw), 2),
        "temperature":       round(temp, 2),
        "soh":               round(soh, 2),
        "rul":               round(rul, 2),
        "anomaly_score":     round(score, 6),
        "anomaly_shift_pct": round(shift_pct, 2),
        "gru_alert":         bool(gru_alert),
        "sensor_fault":      bool(sensor_fault),
        "critical_anomaly":  bool(critical_anomaly),
        "alert_level":       alert_level,
        "alert_msg":         alert_msg,
        "latency_ms":        round(lat, 2),
        "gru_threshold":     round(s.gru_threshold, 6),
        "mock_gru":          False,
        "mock_lstm":         False,
        "feat_err":          feat_err[:21],
        "history":           list(s.history),
    })


@app.post("/api/reset")
def api_reset():
    """Reset simulation history and buffers without reloading models."""
    state.tick           = 0
    state.history.clear()
    state.seq_buf_lstm[:]= 0
    state.seq_buf_gru[:] = 0
    state.last_v         = 3.7
    state.can_stream     = MockCANStream(fault_at_sec=999999)
    return {"ok": True, "message": "Simulation reset. Models remain loaded."}
