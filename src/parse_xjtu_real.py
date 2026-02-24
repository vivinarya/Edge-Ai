import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("c:/gru")
DATA_DIR = ROOT / "data" / "xjtu_real"
OUT_CSV  = ROOT / "data" / "xjtu_cycles.csv"
OUT_NPY  = ROOT / "data" / "xjtu_cycles_sample.npy"

def parse_xjtu_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        # Clean column names
        df.columns = [c.strip().lower().replace("(", "").replace(")", "").replace(" ", "_").replace("°c", "c") for c in df.columns]

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

        if voltage is None or capacity is None or current is None or cycle_raw is None:
            print(f"Skipping {csv_path.name} — missing columns: {list(df.columns)}")
            return None

        df["_cycle"]    = cycle_raw.astype(int)
        df["_voltage"]  = voltage
        df["_current"]  = current
        df["_capacity"] = capacity

        grp = df.groupby("_cycle")
        rows = []
        cap_nominal = float(grp["_capacity"].max().max())
        cycle_count = grp.ngroups
        
        # XJTU batteries are ~2Ah nominal, but use actual max from file to be safe
        cap_nominal = max(cap_nominal, 1.8)

        for idx, (cyc_id, g) in enumerate(grp):
            # XJTU dataset: Discharge is usually negative current or decreasing voltage/capacity
            # but actually in XJTU, discharging is negative current
            dis_mask = g["_current"] < 0
            chg_mask = g["_current"] > 0
            
            dis_g = g[dis_mask] if dis_mask.sum() > 5 else g
            chg_g = g[chg_mask] if chg_mask.sum() > 5 else g

            v_dis_med = float(np.median(dis_g["_voltage"]))
            v_chg_med = float(np.median(chg_g["_voltage"]))
            
            cap_start = float(g["_capacity"].iloc[0])
            cap_end   = float(g["_capacity"].iloc[-1])
            # In XJTU, Capacity(Ah) column often resets. Let's find max - min in discharge.
            cap_ah    = float(dis_g["_capacity"].max() - dis_g["_capacity"].min()) if not dis_g.empty else 0.0
            
            # if cap_ah is broken, just use overall max-min for the cycle
            if cap_ah < 0.1:
                cap_ah = float(g["_capacity"].max() - g["_capacity"].min())

            t_chg     = float(len(chg_g)) / 3600.0  # seconds to hours at 1Hz
            t_chg_n   = min(t_chg / 2.0, 1.0)       # norm to 2h max
            
            # Energy efficency Wh ratio
            dis_wh = float((dis_g["_voltage"].abs() * (dis_g["_current"].abs() / 3600.0)).sum())
            chg_wh = float((chg_g["_voltage"].abs() * (chg_g["_current"].abs() / 3600.0)).sum())
            eff = min(dis_wh / chg_wh, 1.0) if chg_wh > 0.1 else 0.95
            
            cycle_n = cyc_id / 1000.0  # Normalize out of 1000 roughly, model saw this
            soh     = min(cap_ah / cap_nominal, 1.0)
            rul     = max(1.0 - (cyc_id / 1000.0), 0.0)

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
        print(f"Error parsing {csv_path}: {e}")
        return None

def build_lstm_sequence_replay(df: pd.DataFrame, seq_len: int = 50) -> np.ndarray:
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
    return np.array(seqs, dtype=np.float32)

def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found in data/xjtu_real")
        return
        
    print(f"Found {len(csv_files)} CSV files. Processing first file to be fast...")
    df = parse_xjtu_csv(csv_files[0])
    
    if df is not None:
        df.to_csv(OUT_CSV, index=False)
        seqs = build_lstm_sequence_replay(df)
        np.save(OUT_NPY, seqs)
        
        print(f"Summary for {csv_files[0].name}:")
        print(f"Cycles: {len(df)}")
        print(df[["soh", "rul", "discharge_capacity_Ah", "energy_efficiency"]].describe().round(3).to_string())
        print("\nSuccessfully updated data/xjtu_cycles_sample.npy with REAL Zenodo data!")

if __name__ == "__main__":
    main()
