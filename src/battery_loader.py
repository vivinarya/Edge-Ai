"""
battery_loader.py — XJTU-Battery / VED / NASA Battery CSV Parser
SUPRA SAEINDIA 2025 | Task 1.1: Multimodal Anomaly Detection

Supports three public battery datasets:
  1. XJTU-SY   — Xi'an Jiaotong University (IEEE DataPort)
  2. VED        — Vehicle Energy Dataset, Univ. of Michigan (GitHub)
  3. NASA       — Battery Data Set (Kaggle mirror or NASA NTRS)

All are converted to a unified 72-column electrical feature array at 1 Hz,
which is then upsampled to 1 kHz in fusion.py via ZOH.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Unified output schema (72 features) ───────────────────────────────────────
# Indices 0–47:  cell voltages V[0..47]  (padded with last valid if < 48 cells)
# Indices 48–63: cell temps T[0..15]     (padded)
# Index 64:      pack current (A)
# Index 65:      pack voltage (V)
# Index 66:      SoC estimate (%)
# Index 67:      delta_V = max(V) - min(V)
# Index 68:      thermal gradient = max(T) - min(T)
# Index 69:      dV_dt (V/s)
# Index 70:      dI_dt (A/s)
# Index 71:      internal resistance = ΔV/ΔI (mΩ)

ELEC_FEATURES = 72
NUM_CELLS      = 48
NUM_TEMPS      = 16

ELEC_MIN = np.array([
    *([2.5] * NUM_CELLS),    # cell voltages min
    *([-20.0] * NUM_TEMPS),  # cell temps min (°C)
    -400.0,                  # pack current
     0.0,                    # pack voltage
     0.0,                    # SoC
     0.0,                    # delta_V
     0.0,                    # thermal gradient
    -2.0,                    # dV_dt
    -500.0,                  # dI_dt
     0.0,                    # internal resistance
], dtype=np.float32)

ELEC_MAX = np.array([
    *([4.25] * NUM_CELLS),
    *([85.0]  * NUM_TEMPS),
     400.0,
     200.0,
     100.0,
     0.5,
     40.0,
     2.0,
     500.0,
     500.0,
], dtype=np.float32)


def _pad_or_trim(arr: np.ndarray, target: int,
                 pad_value: Optional[float] = None) -> np.ndarray:
    """Pad (with last value or given value) or trim array to target length."""
    n = len(arr)
    if n >= target:
        return arr[:target]
    fill = arr[-1] if (pad_value is None and n > 0) else (pad_value or 0.0)
    return np.concatenate([arr, np.full(target - n, fill)])


def _derive_features(cell_v: np.ndarray, cell_t: np.ndarray,
                     pack_i: np.ndarray, pack_v: np.ndarray,
                     soc: np.ndarray) -> pd.DataFrame:
    """Compute delta_V, thermal_grad, dV_dt, dI_dt, Rint per timestep."""
    delta_v   = cell_v.max(axis=1) - cell_v.min(axis=1)
    therm_g   = cell_t.max(axis=1) - cell_t.min(axis=1)
    dv_dt     = np.gradient(pack_v)
    di_dt     = np.gradient(pack_i)
    # R_int = ΔV/ΔI  (exclude near-zero dI to avoid division artifacts)
    rint      = np.where(np.abs(di_dt) > 0.01, np.abs(dv_dt / di_dt) * 1000, 0.0)
    return np.stack([delta_v, therm_g, dv_dt, di_dt, rint], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# XJTU-SY Parser
# ─────────────────────────────────────────────────────────────────────────────

def load_xjtu(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load XJTU-SY battery CSV files.
    Expected CSV columns: Cycle_Index, Current(A), Voltage(V), Capacity(Ah),
                           Temperature (°C), ...
    Returns: (timestamps_s, features [N, 72])
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    all_rows = []
    for csv in csvs:
        df = pd.read_csv(csv)
        # Normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)

    # Try to extract key columns with flexible naming
    def col(df, *candidates):
        for c in candidates:
            for col in df.columns:
                if c.lower() in col:
                    return df[col].values.astype(np.float32)
        return np.zeros(len(df), dtype=np.float32)

    pack_v  = col(df, "voltage")
    pack_i  = col(df, "current")
    temp    = col(df, "temp", "temperature")
    soc     = col(df, "soc", "capacity")    # use capacity as SoC proxy

    N = len(df)
    # Replicate single channel measurements across cell array
    cell_v = np.tile(pack_v[:, np.newaxis] / 16, (1, NUM_CELLS))  # 16S3P approx
    cell_t = np.tile(temp[:, np.newaxis], (1, NUM_TEMPS))

    packed_v = _pad_or_trim(pack_v, N)
    packed_i = _pad_or_trim(pack_i, N)

    derived = _derive_features(cell_v, cell_t, packed_i, packed_v, soc)
    features = np.concatenate([
        cell_v[:N],
        cell_t[:N],
        packed_i[:, np.newaxis],
        packed_v[:, np.newaxis],
        soc[:N, np.newaxis],
        derived,
    ], axis=1)   # [N, 72]

    timestamps = np.arange(N, dtype=np.float64)   # 1 Hz assumed
    return timestamps, features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# VED Parser (Vehicle Energy Dataset)
# ─────────────────────────────────────────────────────────────────────────────

def load_ved(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Vehicle Energy Dataset CSVs.
    VED columns include: time(s), speed(km/h), soc(%), voltage(V), current(A),
                          hvac_power, aux_power, ...
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.rglob("*.csv"))[:10]   # cap at 10 files
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    all_rows = []
    t_offset = 0.0
    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            if "time" in df.columns:
                df["time"] = df["time"].astype(float) + t_offset
                t_offset = df["time"].max() + 1.0
            all_rows.append(df)
        except Exception:
            continue

    df = pd.concat(all_rows, ignore_index=True)

    def col(df, *names, default=0.0):
        for n in names:
            if n in df.columns:
                return df[n].fillna(default).values.astype(np.float32)
        return np.full(len(df), default, dtype=np.float32)

    pack_v  = col(df, "voltage", "hv_voltage")
    pack_i  = col(df, "current", "hv_current")
    soc     = col(df, "soc", "state_of_charge", default=50.0)
    temp    = col(df, "temp", "battery_temp", default=25.0)
    ts      = col(df, "time")

    N       = len(df)
    cell_v  = np.tile((pack_v / 96)[:, np.newaxis], (1, NUM_CELLS))  # 96S pack
    cell_t  = np.tile(temp[:, np.newaxis], (1, NUM_TEMPS))
    derived = _derive_features(cell_v, cell_t, pack_i, pack_v, soc)

    features = np.concatenate([
        cell_v, cell_t,
        pack_i[:, np.newaxis], pack_v[:, np.newaxis],
        soc[:, np.newaxis], derived,
    ], axis=1)

    return ts.astype(np.float64), features.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect and load
# ─────────────────────────────────────────────────────────────────────────────

def load_battery(data_dir: str,
                 dataset: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """
    Auto-detect battery dataset type and load.
    Returns: (timestamps [N], features [N, 72])
    """
    data_dir = Path(data_dir)
    if dataset == "auto":
        # Heuristic: VED has 'trip' in folder names, XJTU has 'batch'
        if any(data_dir.rglob("*trip*")) or any(data_dir.rglob("*VED*")):
            dataset = "ved"
        else:
            dataset = "xjtu"
        print(f"  Auto-detected dataset type: {dataset}")

    if dataset == "xjtu":
        return load_xjtu(str(data_dir))
    elif dataset == "ved":
        return load_ved(str(data_dir))
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_elec(X: np.ndarray) -> np.ndarray:
    """Normalize electrical features to [0, 1] using frozen bounds."""
    denom = ELEC_MAX - ELEC_MIN
    denom[denom == 0] = 1.0
    return ((X - ELEC_MIN) / denom).clip(0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     default="data/battery")
    p.add_argument("--dataset", default="auto", choices=["auto", "xjtu", "ved"])
    args = p.parse_args()

    ts, feats = load_battery(args.dir, args.dataset)
    print(f"Battery data: {len(ts):,} samples | Shape: {feats.shape}")
    print(f"  Time range: {ts[0]:.1f}s → {ts[-1]:.1f}s")
    print(f"  Value range: [{feats.min():.3f}, {feats.max():.3f}]")
    normed = normalize_elec(feats)
    print(f"  Normalized: [{normed.min():.3f}, {normed.max():.3f}]")
