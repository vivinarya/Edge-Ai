"""
fusion.py — Multimodal Mechanical × Electrical Feature Fusion
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Joins:
  Source A: RACECAR mechanical features [N_mech, 10] @ 1 kHz
  Source B: Battery electrical features [N_elec, 72] @ 1 Hz → 1 kHz ZOH

Outputs: Synthetic feature vector [N, 85] at 1 kHz (10 + 72 + 3 cross-modal)
"""

import numpy as np
from pathlib import Path
from scipy import interpolate


MECH_DIM   = 10
ELEC_DIM   = 72
CROSS_DIM  = 3
TOTAL_DIM  = MECH_DIM + ELEC_DIM + CROSS_DIM   # 85


# ─────────────────────────────────────────────────────────────────────────────
# Z-Score Outlier Rejection
# ─────────────────────────────────────────────────────────────────────────────

def reject_outliers(X: np.ndarray, z_thresh: float = 5.0) -> np.ndarray:
    """
    Causal Z-score outlier rejection: replaces outlier samples with
    the previous valid value (carry-forward). Applied per feature column.

    Catches: velocity > 300 km/h, cell_v < 1.5 V, temp > 85°C, etc.
    """
    X = X.copy()
    mu    = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma[sigma == 0] = 1.0

    for t in range(1, len(X)):
        z    = np.abs((X[t] - mu) / sigma)
        mask = z > z_thresh
        if mask.any():
            X[t, mask] = X[t - 1, mask]   # carry-forward

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Electrical upsampling: 1 Hz → 1 kHz via ZOH
# ─────────────────────────────────────────────────────────────────────────────

def upsample_zoh(ts_elec: np.ndarray, X_elec: np.ndarray,
                 t_grid: np.ndarray) -> np.ndarray:
    """
    Zero-Order Hold upsample from 1 Hz → 1 kHz.
    Each 1s frame is repeated 1000 times with the same value.
    MANDATORY for voltage channels — preserves sudden sag events.
    """
    idx = np.searchsorted(ts_elec, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(X_elec) - 1)
    return X_elec[idx].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal derived features
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_modal(X_mech: np.ndarray,
                        X_elec_1khz: np.ndarray) -> np.ndarray:
    """
    Compute 3 cross-modal coupling features per timestep.

    Feature 0: Mechanical power proxy   = ‖accel_xyz‖ × |vel_x|
    Feature 1: Electrical power         = pack_V × pack_I
    Feature 2: Mech/Elec ratio          = feat0 / (feat1 + ε)

    Anomalies disturb the mech-elec coupling ratio:
    - High G-force with no power response → mechanical fault (loss of traction)
    - Voltage sag without mechanical cause → electrical fault (cell failure)
    """
    # accel = cols 0,1,2 of mechanical; vel_x = col 6
    accel_norm  = np.linalg.norm(X_mech[:, 0:3], axis=1)   # [N]
    vel_x       = np.abs(X_mech[:, 6])                       # [N]
    mech_power  = accel_norm * vel_x                          # [N]

    # pack_V = elec col 65, pack_I = col 64
    pack_v      = X_elec_1khz[:, 65]                          # [N]
    pack_i      = np.abs(X_elec_1khz[:, 64])                  # [N]
    elec_power  = pack_v * pack_i                              # [N]

    ratio = mech_power / (elec_power + 1e-6)

    return np.stack([mech_power, elec_power, ratio], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main fusion function
# ─────────────────────────────────────────────────────────────────────────────

def fuse(X_mech: np.ndarray,
         ts_elec: np.ndarray,
         X_elec: np.ndarray,
         t_grid: np.ndarray) -> np.ndarray:
    """
    Fuse mechanical (already @ 1 kHz) with electrical (@ 1 Hz).

    Args:
        X_mech      : [N, 10] mechanical features at 1 kHz (normalized)
        ts_elec     : [M]     electrical timestamps in seconds (1 Hz, M≈N/1000)
        X_elec      : [M, 72] electrical features at 1 Hz (normalized)
        t_grid      : [N]     shared 1 kHz time grid

    Returns:
        X_fused: [N, 85] (10 mech + 72 elec + 3 cross-modal)
    """
    N = len(X_mech)

    # Step 1: Z-score reject outliers in BOTH modalities
    X_mech  = reject_outliers(X_mech,  z_thresh=5.0)
    X_elec  = reject_outliers(X_elec,  z_thresh=5.0)

    # Step 2: Upsample electrical to 1 kHz via ZOH
    X_elec_1k = upsample_zoh(ts_elec, X_elec, t_grid)   # [N, 72]

    # Step 3: Trim to same length
    N = min(len(X_mech), len(X_elec_1k))
    X_mech    = X_mech[:N]
    X_elec_1k = X_elec_1k[:N]
    t_grid    = t_grid[:N]

    # Step 4: Cross-modal features
    X_cross = compute_cross_modal(X_mech, X_elec_1k)   # [N, 3]

    # Step 5: Concatenate → [N, 85]
    X_fused = np.concatenate([X_mech, X_elec_1k, X_cross], axis=1)

    print(f"  Fusion complete: {X_fused.shape}  "
          f"[{MECH_DIM} mech + {ELEC_DIM} elec + {CROSS_DIM} cross]")
    return X_fused.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization of cross-modal features (frozen bounds)
# ─────────────────────────────────────────────────────────────────────────────

CROSS_MIN = np.array([0.0,    0.0,  0.0], dtype=np.float32)  # mech_power, elec_power, ratio
CROSS_MAX = np.array([3000.0, 200000.0, 100.0], dtype=np.float32)

def normalize_cross(X_cross: np.ndarray) -> np.ndarray:
    denom = CROSS_MAX - CROSS_MIN
    denom[denom == 0] = 1.0
    return ((X_cross - CROSS_MIN) / denom).clip(0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    # Smoke test with synthetic data
    N_mech  = 900_000   # 900 s @ 1 kHz
    N_elec  = 900       # 900 s @ 1 Hz

    X_mech  = np.random.rand(N_mech, MECH_DIM).astype(np.float32)
    t_grid  = np.linspace(0, 900, N_mech)
    ts_elec = np.linspace(0, 900, N_elec)
    X_elec  = np.random.rand(N_elec, ELEC_DIM).astype(np.float32)

    X_fused = fuse(X_mech, ts_elec, X_elec, t_grid)
    print(f"Output shape: {X_fused.shape}")               # [900000, 85]
    print(f"Value range:  [{X_fused.min():.3f}, {X_fused.max():.3f}]")
