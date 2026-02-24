"""
pipeline.py — Data Engineering & Cleaning Pipeline
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Reads: RACECAR M-SOLO-FAST-100-140.db3 (ROS2 SQLite bag)
Uses:  rosbags library (pure Python, no ROS2 install needed)

Feature vector (21D):
  • Bottom IMU  (6): accel xyz, gyro xyz   @ 125 Hz
  • Top IMU     (6): accel xyz, gyro xyz   @ 125 Hz
  • Bottom odom (3): vel_x, vel_y, ang_z   @  60 Hz
  • Top odom    (3): vel_x, vel_y, ang_z   @  60 Hz
  • Local odom  (3): vel_x, vel_y, ang_z   @  20 Hz

Pipeline steps:
  1. Read .db3 via rosbags → deserialize standard Imu + Odometry messages
  2. Upsample all signals to 1 kHz uniform grid (linear interp)
  3. EMI smoothing: 5-sample causal moving average (mechanical channels only)
  4. NaN linear interpolation
  5. Min-Max normalization (frozen constants)
  6. Segment into overlapping 32-sample windows
  7. Save to compressed HDF5
"""

import sys
import numpy as np
import h5py
from pathlib import Path
from scipy import interpolate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SAMPLE_RATE_HZ, WINDOW_SIZE, NUM_FEATURES,
    FEATURE_MIN, FEATURE_MAX,
    DB3_BAG_PATH, DATA_H5_PATH,
    TOPICS,
)

X_MIN = np.array(FEATURE_MIN, dtype=np.float32)
X_MAX = np.array(FEATURE_MAX, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DB3 Extraction via rosbags
# ─────────────────────────────────────────────────────────────────────────────

def _extract_imu_vals(msg) -> list:
    return [
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        msg.angular_velocity.x,    msg.angular_velocity.y,    msg.angular_velocity.z,
    ]

def _extract_odom_vals(msg) -> list:
    return [
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z,
    ]

def extract_db3(bag_path: str) -> dict:
    """
    Extract timestamp + signal arrays from a ROS2 .db3 bag using rosbags.
    Returns dict: topic → {"ts": float64[N], "vals": float32[N, D]}
    """
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print("[ERROR] pip install rosbags")
        sys.exit(1)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    raw = {t: {"ts": [], "vals": []} for t in TOPICS}

    bag_dir = str(Path(bag_path).parent)
    print(f"  Opening bag directory: {bag_dir}")

    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic in TOPICS]
        if not connections:
            print("[ERROR] None of the required topics found.")
            sys.exit(1)
        found = {c.topic for c in connections}
        print(f"  Found {len(found)}/{len(TOPICS)} topics: {[t.split('/')[-1] for t in found]}")

        total_msgs = sum(c.msgcount for c in connections)
        for conn, ts_ns, data in tqdm(
            reader.messages(connections=connections),
            desc="  Reading messages", unit="msg", total=total_msgs
        ):
            try:
                msg  = typestore.deserialize_cdr(data, conn.msgtype)
                ts_s = ts_ns * 1e-9

                if "rawimux" in conn.topic:
                    vals = _extract_imu_vals(msg)
                elif "odom" in conn.topic or "odometry" in conn.topic:
                    vals = _extract_odom_vals(msg)
                else:
                    continue

                raw[conn.topic]["ts"].append(ts_s)
                raw[conn.topic]["vals"].append(vals)
            except Exception:
                continue

    for t in raw:
        if raw[t]["ts"]:
            raw[t]["ts"]   = np.array(raw[t]["ts"],   dtype=np.float64)
            raw[t]["vals"] = np.array(raw[t]["vals"],  dtype=np.float32)
            print(f"  {t.split('/')[-1]}: {len(raw[t]['ts']):,} msgs")
        else:
            print(f"  [WARN] No data for {t}")
            raw[t]["ts"]   = np.zeros(2, dtype=np.float64)
            raw[t]["vals"] = np.zeros((2, 3), dtype=np.float32)

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build 1 kHz uniform time grid
# ─────────────────────────────────────────────────────────────────────────────

def build_time_grid(raw: dict) -> np.ndarray:
    valid = [v for v in raw.values() if len(v["ts"]) > 10]
    t_start = max(v["ts"][0]  for v in valid)
    t_end   = min(v["ts"][-1] for v in valid)
    n       = int((t_end - t_start) * SAMPLE_RATE_HZ)
    print(f"  Grid: {t_start:.1f}s to {t_end:.1f}s ({(t_end-t_start):.1f}s, {n:,} samples)")
    return np.linspace(t_start, t_end, n, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Upsample to 1 kHz
# ─────────────────────────────────────────────────────────────────────────────

def upsample(ts: np.ndarray, vals: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    if vals.ndim == 1:
        vals = vals[:, None]
    n_cols = vals.shape[1]
    out = np.empty((len(t_grid), n_cols), dtype=np.float32)
    for j in range(n_cols):
        col = vals[:, j]
        f   = interpolate.interp1d(ts, col, kind="linear",
                                   bounds_error=False,
                                   fill_value=(col[0], col[-1]))
        out[:, j] = f(t_grid)
    return out


def build_feature_matrix(raw: dict, t_grid: np.ndarray) -> np.ndarray:
    """
    Assemble 21-feature matrix at 1 kHz.
    Order: bot_imu(6), top_imu(6), bot_odom(3), top_odom(3), local_odom(3)
    """
    bot_imu  = upsample(raw["/vehicle_8/novatel_bottom/rawimux"]["ts"],
                        raw["/vehicle_8/novatel_bottom/rawimux"]["vals"], t_grid)  # [N,6]
    top_imu  = upsample(raw["/vehicle_8/novatel_top/rawimux"]["ts"],
                        raw["/vehicle_8/novatel_top/rawimux"]["vals"],    t_grid)  # [N,6]
    bot_odom = upsample(raw["/vehicle_8/novatel_bottom/odom"]["ts"],
                        raw["/vehicle_8/novatel_bottom/odom"]["vals"],    t_grid)  # [N,3]
    top_odom = upsample(raw["/vehicle_8/novatel_top/odom"]["ts"],
                        raw["/vehicle_8/novatel_top/odom"]["vals"],       t_grid)  # [N,3]
    loc_odom = upsample(raw["/vehicle_8/local_odometry"]["ts"],
                        raw["/vehicle_8/local_odometry"]["vals"],         t_grid)  # [N,3]

    return np.concatenate([bot_imu, top_imu, bot_odom, top_odom, loc_odom], axis=1)  # [N,21]


# ─────────────────────────────────────────────────────────────────────────────
# 4. EMI Smoothing (mechanical channels only)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_causal_ma(X: np.ndarray, window: int = 5,
                     cols: slice = slice(0, 12)) -> np.ndarray:
    """
    5-sample causal MA on IMU channels (cols 0–11) only.
    Odom channels (12–20) left unsmoothed — velocity transients are anomaly signal.
    """
    kernel = np.ones(window, dtype=np.float32) / window
    out    = X.copy()
    for j in range(*cols.indices(X.shape[1])):
        out[:, j] = np.convolve(X[:, j], kernel, mode="full")[:len(X)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. NaN Imputation
# ─────────────────────────────────────────────────────────────────────────────

def impute_nans(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    idx = np.arange(len(X))
    for j in range(X.shape[1]):
        col  = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            ok = ~mask
            if ok.sum() < 2:
                out[:, j] = 0.0
            else:
                out[:, j] = np.interp(idx, idx[ok], col[ok])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize(X: np.ndarray) -> np.ndarray:
    denom = X_MAX - X_MIN
    denom[denom == 0] = 1.0
    return ((X - X_MIN) / denom).clip(0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Window Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_windows(X_norm: np.ndarray,
                    window: int = WINDOW_SIZE,
                    stride: int = 4) -> np.ndarray:
    N   = len(X_norm)
    num = (N - window) // stride + 1
    idx = np.arange(window)[None, :] + np.arange(num)[:, None] * stride
    return X_norm[idx]   # [num_windows, 32, 21]


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(bag_path: str = DB3_BAG_PATH,
                 out_path: str = DATA_H5_PATH,
                 stride:   int = 4) -> None:

    if not Path(bag_path).exists():
        print(f"[ERROR] Bag not found: {bag_path}")
        sys.exit(1)

    print("=" * 60)
    print("  SUPRA SAEINDIA 2025 — Preprocessing Pipeline (21D)")
    print("=" * 60)

    print("\n[1/6] Extracting topics from .db3 bag...")
    raw = extract_db3(bag_path)

    print("\n[2/6] Building 1 kHz time grid...")
    t_grid = build_time_grid(raw)

    print(f"\n[3/6] Upsampling to 1 kHz & assembling {NUM_FEATURES}-feature matrix...")
    X_raw = build_feature_matrix(raw, t_grid)
    print(f"  Shape: {X_raw.shape}  (expect [N, {NUM_FEATURES}])")

    print("\n[4/6] EMI smoothing (IMU channels only)...")
    X_sm = smooth_causal_ma(X_raw)

    print("\n[5/6] NaN imputation...")
    X_clean = impute_nans(X_sm)
    nans = np.isnan(X_clean).sum()
    print(f"  NaN check: {nans} remaining {'✓' if nans == 0 else '← WARN'}")

    X_norm = normalize(X_clean)
    print(f"  Normalize range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")

    print(f"\n[6/6] Segmenting windows (size={WINDOW_SIZE}, stride={stride})...")
    windows = segment_windows(X_norm, stride=stride)
    print(f"  Windows: {len(windows):,}  shape: {windows.shape}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "windows", data=windows,
            compression="gzip", compression_opts=4,
            chunks=(min(1024, len(windows)), WINDOW_SIZE, NUM_FEATURES),
        )
        f.attrs.update({
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_size":    WINDOW_SIZE,
            "num_features":   NUM_FEATURES,
            "stride":         stride,
            "source_bag":     str(bag_path),
        })

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"\n✓ Saved → {out_path}  ({size_mb:.1f} MB)  |  {len(windows):,} windows")
    print("  Next step: python src/train.py")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bag",    default=DB3_BAG_PATH)
    p.add_argument("--out",    default=DATA_H5_PATH)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()
    run_pipeline(args.bag, args.out, args.stride)
