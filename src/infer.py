"""
infer.py — Real-Time Anomaly Detection Engine
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Simulates the on-device inference loop:
  Ring Buffer (32-sample SPSC) → Normalize → Model → Anomaly Score → State Machine

Usage:
    # Against live CAN stream (replace MockCANStream with real driver):
    python src/infer.py --mode live

    # Against a recorded HDF5 window file (for validation):
    python src/infer.py --mode replay --data data/s3_windows.h5
"""

import argparse
import sys
import time
import json
import numpy as np
import torch
from collections import deque
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from model  import DenoisingGRUAutoencoder
from config import (
    WINDOW_SIZE, NUM_FEATURES, FEATURE_MIN, FEATURE_MAX,
    CHECKPOINT_DIR, EXPORT_DIR,
    ALERT_CONSECUTIVE, ALERT_RESET,
)

X_MIN = np.array(FEATURE_MIN, dtype=np.float32)
X_MAX = np.array(FEATURE_MAX, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ring Buffer (lock-free SPSC simulation in Python)
# ─────────────────────────────────────────────────────────────────────────────

class RingBuffer:
    """
    Circular buffer: 32 samples × 14 features = 448 bytes at INT8.
    In C firmware: SPSC lock-free with atomic write pointer.
    """
    def __init__(self, size: int = WINDOW_SIZE, features: int = NUM_FEATURES):
        self.buf = deque(maxlen=size)
        self.size = size

    def push(self, sample: np.ndarray) -> None:
        """Add one sample [F,]. Oldest sample auto-dropped when full."""
        self.buf.append(sample)

    def ready(self) -> bool:
        """True once buffer holds exactly WINDOW_SIZE samples."""
        return len(self.buf) == self.size

    def get_window(self) -> Optional[np.ndarray]:
        """Return current window [T, F] without clearing buffer."""
        if not self.ready():
            return None
        return np.array(self.buf, dtype=np.float32)   # [32, 14]


# ─────────────────────────────────────────────────────────────────────────────
# Normalization (inline — must match firmware constants exactly)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x: np.ndarray) -> np.ndarray:
    denom = X_MAX - X_MIN
    return ((x - X_MIN) / denom).clip(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly State Machine
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyStateMachine:
    """
    ALERT counter logic:
      3 consecutive scores ≥ threshold → INTERRUPT (CAN frame 0x6FF)
      5 consecutive NOMINAL readings → clear ALERT counter
    """
    STATE_NOMINAL    = "NOMINAL"
    STATE_ALERT      = "ALERT"
    STATE_INTERRUPT  = "INTERRUPT"

    def __init__(self, threshold: float):
        self.threshold    = threshold
        self.alert_count  = 0
        self.nominal_run  = 0
        self.state        = self.STATE_NOMINAL
        self.trigger_count = 0

    def update(self, score: float) -> str:
        if score >= self.threshold:
            self.alert_count += 1
            self.nominal_run  = 0
        else:
            self.nominal_run  += 1
            if self.nominal_run >= ALERT_RESET:
                self.alert_count = 0   # Clear alert after 5 nominal

        if self.alert_count >= ALERT_CONSECUTIVE:
            self.state          = self.STATE_INTERRUPT
            self.trigger_count += 1
            self.alert_count    = 0    # Reset after interrupt emitted
        else:
            self.state = (self.STATE_ALERT if self.alert_count > 0
                          else self.STATE_NOMINAL)

        return self.state


# ─────────────────────────────────────────────────────────────────────────────
# Mock CAN Stream (replaces real driver for testing)
# ─────────────────────────────────────────────────────────────────────────────

class MockCANStream:
    """
    Generates synthetic 21-feature data at 1 kHz.
    Feature order (must match config.py):
      Bot IMU(6): accel xyz, gyro xyz
      Top IMU(6): accel xyz, gyro xyz  (slightly different mount angle)
      Bot odom(3): vel_x, vel_y, ang_z
      Top odom(3): vel_x, vel_y, ang_z
      Local odom(3): vel_x, vel_y, ang_z

    Fault at t=2.0s: accel spike on both IMUs + velocity drop (simulated suspension failure)
    """
    def __init__(self, fault_at_sec: float = 2.0, fault_duration: int = 50):
        self.t        = 0
        self.dt       = 1.0 / 1000
        self.fault_at  = int(fault_at_sec / self.dt)
        self.fault_end = self.fault_at + fault_duration

    def read_sample(self) -> np.ndarray:
        """Returns one sample [21] in physical units matching FEATURE_MIN/MAX."""
        t  = self.t * self.dt
        rn = np.random.default_rng(self.t)   # deterministic seed per tick
        rn.shuffle  # warm up

        v = 40.0 + 10 * np.sin(2 * np.pi * 0.1 * t)   # vehicle speed m/s

        x = np.array([
            # Bot IMU — accel xyz (m/s^2)
            5 * np.sin(2 * np.pi * 2 * t) + rn.normal(0, 0.3),
            2 * np.sin(2 * np.pi * 3 * t) + rn.normal(0, 0.2),
            9.8 + rn.normal(0, 0.05),
            # Bot IMU — gyro xyz (rad/s)
            0.1 * np.sin(2 * np.pi * t) + rn.normal(0, 0.01),
            0.05 * np.cos(2 * np.pi * 2 * t) + rn.normal(0, 0.01),
            0.2 * np.sin(2 * np.pi * 0.5 * t) + rn.normal(0, 0.02),
            # Top IMU — accel xyz (slightly different due to mount position)
            5 * np.sin(2 * np.pi * 2 * t) + rn.normal(0, 0.5),
            2 * np.sin(2 * np.pi * 3 * t) + rn.normal(0, 0.3),
            9.8 + rn.normal(0, 0.08),
            # Top IMU — gyro xyz
            0.1 * np.sin(2 * np.pi * t) + rn.normal(0, 0.015),
            0.05 * np.cos(2 * np.pi * 2 * t) + rn.normal(0, 0.012),
            0.2 * np.sin(2 * np.pi * 0.5 * t) + rn.normal(0, 0.025),
            # Bot odom — vel_x, vel_y, ang_z
            v + rn.normal(0, 0.1),
            rn.normal(0, 0.05),
            0.01 * np.sin(2 * np.pi * 0.2 * t),
            # Top odom — vel_x, vel_y, ang_z
            v + rn.normal(0, 0.1),
            rn.normal(0, 0.05),
            0.01 * np.sin(2 * np.pi * 0.2 * t),
            # Local odom — vel_x, vel_y, ang_z
            v + rn.normal(0, 0.15),
            rn.normal(0, 0.08),
            0.01 * np.sin(2 * np.pi * 0.2 * t),
        ], dtype=np.float32)

        # Inject fault: IMU spike + sudden velocity drop (suspension/wheel fault)
        if self.fault_at <= self.t < self.fault_end:
            x[0]  *= 8.0   # bot accel_x spike (hard impact)
            x[6]  *= 8.0   # top accel_x spike
            x[12]  = 5.0   # bot_vel_x drops (lockup)
            x[15]  = 5.0   # top_vel_x drops
            x[18]  = 5.0   # local_vel_x drops

        self.t += 1
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, model_path: str, threshold: float,
                 use_onnx: bool = False):
        self.threshold = threshold
        self.use_onnx  = use_onnx

        if use_onnx:
            import onnxruntime as ort
            onnx_path = str(Path(EXPORT_DIR) / "gru_autoencoder.onnx")
            self.sess = ort.InferenceSession(onnx_path,
                                             providers=["CPUExecutionProvider"])
        # 1. Load Autoencoder
        self.model = DenoisingGRUAutoencoder()
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=False))
            print(f"  Loaded GRU checkpoint: {model_path}")
        else:
            print("  [INFO] No checkpoint — using untrained model (demo/shape check only)")
        self.model.eval()

        # 2. Load XGBoost Classifier (if available)
        self.xgb_model = None
        self.classes = ["Healthy", "IMU_Impact", "Wheel_Lockup", "Sensor_Noise"]
        
        xgb_path = Path(EXPORT_DIR) / "xgb_classifier.json"
        if not xgb_path.exists():
            xgb_path = Path(CHECKPOINT_DIR) / "xgb_classifier.json"
            
        if xgb_path.exists():
            import xgboost as xgb
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))
            print(f"  Loaded XGBoost classifier: {xgb_path}")

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract statistical features from a single [32, 21] window."""
        f_mean = np.mean(window, axis=0)
        f_std  = np.std(window, axis=0)
        f_min  = np.min(window, axis=0)
        f_max  = np.max(window, axis=0)
        f_diff = np.max(np.abs(np.diff(window, axis=0)), axis=0)
        
        # [1, 105]
        return np.concatenate([f_mean, f_std, f_min, f_max, f_diff])[np.newaxis, :]

    def classify_fault(self, window: np.ndarray) -> str:
        """Run the trained XGBoost model to classify a suspected anomalous window."""
        if self.xgb_model is None:
            return "UNKNOWN_FAULT"
            
        x_feat = self.extract_features(window)
        pred_idx = self.xgb_model.predict(x_feat)[0]
        return self.classes[pred_idx]

    def infer(self, window: np.ndarray) -> tuple[float, str]:
        """window: [32, 21] normalized float32 → (anomaly_score, fault_class)"""
        x = window[np.newaxis]   # [1, 32, 21]

        if self.use_onnx:
            out  = self.sess.run(None, {"x": x})[0]   # [1, 32, 21]
            diff = x - out                             # [1, 32, 21]
        else:
            xt = torch.from_numpy(x)
            with torch.no_grad():
                xh = self.model(xt, add_noise=False)
            diff = (xt - xh).numpy()                  # [1, 32, 21]

        score = float(((diff ** 2).mean(axis=-1)).mean())
        
        # Classify only if score is above threshold
        fault_class = "Healthy"
        if score >= self.threshold:
            fault_class = self.classify_fault(window)
            
        return score, fault_class


# ─────────────────────────────────────────────────────────────────────────────
# Live Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_live(engine: InferenceEngine, duration_sec: int = 4):
    """Simulate real-time 1 kHz inference loop."""
    print(f"\nRunning live simulation for {duration_sec}s at 1 kHz...")
    print(f"Threshold = {engine.threshold:.6f}  |  "
          f"Alert after {ALERT_CONSECUTIVE} consecutive hits\n")
    print(f"{'Tick':>6} | {'Score':>10} | {'State':<12} | {'Latency ms':>10}")
    print("-" * 50)

    buf    = RingBuffer()
    stream = MockCANStream(fault_at_sec=2.0)
    fsm    = AnomalyStateMachine(engine.threshold)

    total_ticks   = duration_sec * 1000
    interrupts    = 0
    latencies_ms  = []

    for tick in range(total_ticks):
        t0     = time.perf_counter()
        sample = stream.read_sample()
        buf.push(normalize(sample))

        state = AnomalyStateMachine.STATE_NOMINAL
        score = 0.0

        if buf.ready():
            w     = buf.get_window()        # [32, 14]
            score, fault_class = engine.infer(w)
            state = fsm.update(score)
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(lat_ms)

            if state == AnomalyStateMachine.STATE_INTERRUPT:
                interrupts += 1

            # Print every 100 ticks (10 Hz display rate)
            if tick % 100 == 0 or state != AnomalyStateMachine.STATE_NOMINAL:
                marker = f" ⚠ INTERRUPT! [{fault_class}]" if state == AnomalyStateMachine.STATE_INTERRUPT else ("" if fault_class == "Healthy" else f" ({fault_class})")
                print(f"{tick:>6} | {score:>10.5f} | {state:<12} | "
                      f"{lat_ms:>10.3f}{marker}")

        # 1 kHz timing
        elapsed = time.perf_counter() - t0
        sleep   = max(0.0, 1.0 / 1000 - elapsed)
        time.sleep(sleep)

    # Summary
    if latencies_ms:
        arr = np.array(latencies_ms)
        print(f"\n{'='*50}")
        print(f"Latency  — mean: {arr.mean():.2f} ms | "
              f"p99: {np.percentile(arr, 99):.2f} ms | "
              f"max: {arr.max():.2f} ms")
        print(f"Interrupts triggered: {interrupts}")
        ok = np.percentile(arr, 99) < 50
        print(f"Latency budget (50 ms): {'✓ PASS' if ok else '✗ FAIL'}")


# ─────────────────────────────────────────────────────────────────────────────
# Replay Mode (HDF5 validation)
# ─────────────────────────────────────────────────────────────────────────────

def run_replay(engine: InferenceEngine, h5_path: str, max_windows: int = 5000):
    import h5py
    print(f"\nReplay validation on {h5_path} (max {max_windows} windows)...")
    fsm    = AnomalyStateMachine(engine.threshold)
    scores = []

    with h5py.File(h5_path, "r") as f:
        ds = f["windows"]
        n  = min(len(ds), max_windows)
        for i in range(n):
            w     = ds[i].astype(np.float32)
            score, fault_class = engine.infer(w)
            fsm.update(score)
            scores.append(score)

    arr   = np.array(scores)
    above = (arr >= engine.threshold).sum()
    print(f"Windows: {n:,} | Above threshold: {above} "
          f"({above/n*100:.2f}%) | Interrupts: {fsm.trigger_count}")
    print(f"Score — μ={arr.mean():.5f} | σ={arr.std():.5f} | "
          f"max={arr.max():.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SUPRA SAEINDIA 2025 — Anomaly Inference")
    parser.add_argument("--mode",   default="live",   choices=["live", "replay"])
    parser.add_argument("--data",   default="data/s3_windows.h5")
    parser.add_argument("--onnx",   action="store_true", help="Use ONNX runtime")
    parser.add_argument("--duration", type=int, default=4, help="Live sim duration (sec)")
    args = parser.parse_args()

    # Load threshold
    thresh_path = Path(CHECKPOINT_DIR) / "threshold.npy"
    manifest_path = Path(EXPORT_DIR) / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            threshold = json.load(f)["threshold"]
    elif thresh_path.exists():
        threshold = float(np.load(str(thresh_path))[0])
    else:
        print("[WARNING] No threshold found. Using default 0.05.")
        threshold = 0.05

    # Load model — always use FP32 checkpoint for inference
    # (dynamic quantized model uses different state_dict format not compatible
    #  with standard load_state_dict; use export.py + ONNX for INT8 path)
    fp32_ckpt = str(Path(CHECKPOINT_DIR) / "best_fp32.pth")
    qat_ckpt  = str(Path(CHECKPOINT_DIR) / "best_qat.pth")
    ckpt = fp32_ckpt if Path(fp32_ckpt).exists() else qat_ckpt
    if not Path(ckpt).exists() and not args.onnx:
        print("[ERROR] No checkpoint found. Run: python src/train.py first.")

    engine = InferenceEngine(ckpt if Path(ckpt).exists() else "", threshold, args.onnx)

    if args.mode == "live":
        run_live(engine, duration_sec=args.duration)
    else:
        run_replay(engine, args.data)


if __name__ == "__main__":
    main()
