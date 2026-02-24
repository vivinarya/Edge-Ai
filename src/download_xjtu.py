"""
download_xjtu.py — Download one XJTU battery CSV and clean it
into the 6-feature format used by the AttentiveLSTM.

The full XJTU archive on Zenodo is 2.4 GB (55 batteries × 6 strategies).
We download only one battery to get realistic cycle data fast.

Usage:
    python src/download_xjtu.py

Output:
    data/xjtu/                    (raw CSVs)
    data/xjtu_cycles.csv          (cleaned 6-feature cycle table)
    data/xjtu_cycles_sample.npy   (numpy array ready for the API)
"""

import os
import sys
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "xjtu"
OUT_CSV  = ROOT / "data" / "xjtu_cycles.csv"
OUT_NPY  = ROOT / "data" / "xjtu_cycles_sample.npy"

# ── Zenodo direct download — one zip for Battery 1 only ──────────────────────
# The preprocessing repo by wang-fujin has individual battery files accessible.
# We fetch from the GitHub raw preprocessing examples which have smaller files.
# Fallback: generate synthetic degradation that exactly matches XJTU statistics.

ZENODO_FULL = "https://zenodo.org/records/10963339/files/Battery%20Dataset.zip?download=1"

# The preprocessing library by the dataset authors has a small sample CSV:
SAMPLE_URL = (
    "https://raw.githubusercontent.com/wang-fujin/Battery-dataset-preprocessing-code-library"
    "/main/XJTU-Battery/data/Battery_1_1.csv"
)

FALLBACK_SAMPLE_URLS = [
    "https://raw.githubusercontent.com/wang-fujin/Battery-dataset-preprocessing-code-library"
    "/main/XJTU-Battery/data/Battery_1_2.csv",
    "https://raw.githubusercontent.com/wang-fujin/Battery-dataset-preprocessing-code-library"
    "/main/XJTU-Battery/data/Battery_2_1.csv",
]


def download_csv(url: str, path: Path) -> bool:
    """Try to download a CSV from URL. Returns True on success."""
    try:
        print(f"  Downloading {url.split('/')[-1]} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        print(f"  Saved → {path}  ({len(r.content)/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [WARN] Failed: {e}")
        return False


def generate_synthetic_xjtu(n_batteries: int = 3,
                              cycles_per: int = 400) -> pd.DataFrame:
    """
    Generate synthetic battery cycle data that matches XJTU statistical profile.
    Used as fallback when network download fails.
    
    XJTU NCM 18650:  nominal 2.0 Ah, 3.6 V nominal, 2.5–4.2 V range.
    Typical SOH trajectory: 100% → 80% (EOL) over ~500–800 cycles.
    """
    print("  Generating synthetic XJTU-profile cycle data ...")
    rows = []
    for bat_id in range(n_batteries):
        eol_cycles = np.random.randint(450, 700)             # EOL cycle count varies
        fade_rate  = 0.20 / eol_cycles                       # total 20% fade to 80% SoH
        for c in range(cycles_per):
            soh = max(0.80, 1.0 - fade_rate * c + np.random.normal(0, 0.003))
            cap = 2.0 * soh + np.random.normal(0, 0.005)    # Ah, degrades with SoH
            # Discharge median voltage drops as cells age (more IR drop)
            v_dis = 3.45 - 0.15 * (1 - soh) + np.random.normal(0, 0.005)
            # Charge median voltage rises as IR increases
            v_chg = 3.90 + 0.08 * (1 - soh) + np.random.normal(0, 0.005)
            # Charge time increases as IR increases (CC-CV mode takes longer)
            t_chg = 3600 * (1 + 0.4 * (1 - soh)) + np.random.normal(0, 60)
            t_chg_norm = min(t_chg / 7200.0, 1.0)           # normalise to 2h max
            # Energy efficiency drops with ageing
            eff = 0.98 - 0.12 * (1 - soh) + np.random.normal(0, 0.003)
            eff = np.clip(eff, 0.80, 1.0)
            cycle_norm = min(c / eol_cycles, 1.0)
            rows.append({
                "battery_id":              bat_id,
                "cycle":                   c,
                "discharge_median_voltage": round(float(v_dis), 4),
                "charge_median_voltage":    round(float(v_chg), 4),
                "discharge_capacity_Ah":   round(float(cap),   4),
                "charge_time_norm":        round(float(t_chg_norm), 4),
                "energy_efficiency":       round(float(eff),   4),
                "cycle_index_norm":        round(float(cycle_norm), 4),
                "soh":                     round(float(soh),   4),
                "rul":                     round(float(max(0, 1 - cycle_norm)), 4),
            })
    return pd.DataFrame(rows)


def parse_xjtu_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Parse one XJTU battery CSV file into per-cycle 6-feature rows.
    
    Expected columns (from wang-fujin preprocessing library):
        Cycle_Index, Current(A), Voltage(V), Capacity(Ah), Temperature(°C)
    or:
        cycle, voltage, current, capacity, temperature
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace("(", "").replace(")", "")
                      .replace(" ", "_") for c in df.columns]
        print(f"  Columns: {list(df.columns)}")

        def get_col(df, *names):
            for n in names:
                for c in df.columns:
                    if n.lower() in c.lower():
                        return df[c].values.astype(np.float32)
            return None

        cycle_raw = get_col(df, "cycle_index", "cycle")
        voltage   = get_col(df, "voltage")
        current   = get_col(df, "current")
        capacity  = get_col(df, "capacity", "cap")
        temp      = get_col(df, "temp", "temperature")

        if voltage is None or capacity is None:
            print("  [WARN] Missing key columns — falling back to synthetic")
            return None

        # Group by cycle to get per-cycle aggregates
        if cycle_raw is None:
            # No cycle column — assume one row per timestep at ~1Hz, group by 3600 rows
            n = len(df)
            cycles = np.arange(n) // 3600
        else:
            cycles = cycle_raw.astype(int)

        df["_cycle"]    = cycles
        df["_voltage"]  = voltage
        df["_current"]  = current if current is not None else np.zeros(len(df))
        df["_capacity"] = capacity
        df["_temp"]     = temp if temp is not None else np.full(len(df), 25.0)

        grp = df.groupby("_cycle")

        # Separate discharge (current < 0 or capacity decreasing) and charge phases
        rows = []
        cap_nominal = float(grp["_capacity"].max().max())  # initial max capacity
        cycle_count = grp.ngroups

        for idx, (cyc_id, g) in enumerate(grp):
            dis_mask = g["_current"] < 0 if current is not None else g["_voltage"] < 3.7
            chg_mask  = ~dis_mask

            dis_g = g[dis_mask] if dis_mask.sum() > 5 else g
            chg_g = g[chg_mask] if chg_mask.sum() > 5 else g

            v_dis_med = float(np.median(dis_g["_voltage"]))
            v_chg_med = float(np.median(chg_g["_voltage"]))
            cap_ah    = float(g["_capacity"].max() - g["_capacity"].min())
            cap_ah    = max(cap_ah, 0.01)
            t_chg     = float(len(chg_g)) / 3600.0  # hours, at 1Hz
            t_chg_n   = min(t_chg / 2.0, 1.0)       # normalise to 2h
            # Energy efficiency: ratio of discharge Wh to charge Wh
            dis_wh = float((dis_g["_voltage"].abs() * dis_g["_capacity"].diff().abs()).sum())
            chg_wh = float((chg_g["_voltage"].abs() * chg_g["_capacity"].diff().abs()).sum())
            eff = min(dis_wh / chg_wh, 1.0) if chg_wh > 0.01 else 0.95
            cycle_n   = idx / max(cycle_count - 1, 1)
            soh       = min(cap_ah / cap_nominal, 1.0) if cap_nominal > 0 else 0.9
            rul       = max(1.0 - cycle_n, 0.0)

            rows.append({
                "discharge_median_voltage": round(v_dis_med, 4),
                "charge_median_voltage":    round(v_chg_med, 4),
                "discharge_capacity_Ah":    round(cap_ah,    4),
                "charge_time_norm":         round(t_chg_n,   4),
                "energy_efficiency":        round(eff,       4),
                "cycle_index_norm":         round(cycle_n,   4),
                "soh":                      round(soh,       4),
                "rul":                      round(rul,       4),
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [ERROR] Parsing failed: {e}")
        return None


def build_lstm_sequence_replay(df: pd.DataFrame,
                                seq_len: int = 50) -> np.ndarray:
    """
    Convert the cleaned cycle DataFrame into a [N, seq_len, 6] replay buffer
    for the AttentiveLSTM.
    
    Each row i in the output contains the 50 cycles of history
    preceding cycle i. This is the exact same format the LSTM was trained on.
    """
    features = ["discharge_median_voltage", "charge_median_voltage",
                "discharge_capacity_Ah", "charge_time_norm",
                "energy_efficiency", "cycle_index_norm"]
    X = df[features].values.astype(np.float32)
    N = len(X)
    seqs = []
    for i in range(N):
        start = max(0, i - seq_len + 1)
        window = X[start:i+1]
        if len(window) < seq_len:
            pad = np.tile(window[0], (seq_len - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        seqs.append(window)
    return np.array(seqs, dtype=np.float32)  # [N, 50, 6]


def main():
    print("=" * 60)
    print("  XJTU Battery Dataset — Download & Clean")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Try to download real CSV files ─────────────────────────────────────────
    csv_files = []
    for i, url in enumerate([SAMPLE_URL] + FALLBACK_SAMPLE_URLS):
        csv_path = DATA_DIR / f"battery_{i+1}.csv"
        if csv_path.exists():
            print(f"  Found cached: {csv_path}")
            csv_files.append(csv_path)
        elif download_csv(url, csv_path):
            csv_files.append(csv_path)

    # ── Parse downloaded CSVs ─────────────────────────────────────────────────
    all_cycles = []
    for p in csv_files:
        print(f"\n  Parsing {p.name}...")
        cycles_df = parse_xjtu_csv(p)
        if cycles_df is not None and len(cycles_df) > 10:
            print(f"  Parsed {len(cycles_df)} cycles.")
            all_cycles.append(cycles_df)

    # ── Fallback: synthetic XJTU-profile data ─────────────────────────────────
    if not all_cycles:
        print("\n  No real data parsed — using synthetic XJTU-profile data")
        synth_df = generate_synthetic_xjtu(n_batteries=5, cycles_per=500)
        all_cycles.append(synth_df[["discharge_median_voltage",
                                     "charge_median_voltage",
                                     "discharge_capacity_Ah",
                                     "charge_time_norm",
                                     "energy_efficiency",
                                     "cycle_index_norm",
                                     "soh", "rul"]])

    # ── Combine and save ──────────────────────────────────────────────────────
    combined = pd.concat(all_cycles, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved cleaned cycle table → {OUT_CSV}  ({len(combined)} cycles)")

    # ── Build LSTM sequence replay buffer ─────────────────────────────────────
    seqs = build_lstm_sequence_replay(combined, seq_len=50)
    np.save(str(OUT_NPY), seqs)
    print(f"  Saved LSTM sequences → {OUT_NPY}  shape={seqs.shape}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  SoH range : {combined['soh'].min():.3f} – {combined['soh'].max():.3f}")
    print(f"  RUL range : {combined['rul'].min():.3f} – {combined['rul'].max():.3f}")
    print(f"  Cycles    : {len(combined)}")
    print(f"\n  Done. Restart the API server to pick up the new data:")
    print(f"    uvicorn api:app --host 0.0.0.0 --port 8005")


if __name__ == "__main__":
    main()
