"""
probe_can.py — CAN ID Discovery & Dynamic Topic Audit
Reads SQLite db3 directly (fast) to identify:
  1. All available dynamic /vehicle_8/ topics with message counts
  2. CAN IDs from from_can_bus with frequency + byte variance analysis
  3. Identifies wheel speed candidates (4 correlated periodic signals)
"""
import sqlite3
import struct
import numpy as np
from collections import defaultdict
from pathlib import Path
import sys

DB = "data/M_SOLO_FAST/M-SOLO-FAST-100-140.db3"

def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # ── Step 1: Audit all topics ──────────────────────────────────────────────
    cur.execute("SELECT id, name, type FROM topics")
    topics = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    print("=" * 75)
    print("DYNAMIC /vehicle_8/ TOPICS")
    print("=" * 75)
    DYNAMIC_KEYS = ["rawimux", "odom", "bestvel", "bestpos", "fix",
                    "can_bus", "gps", "corrimu", "inspvax", "imu"]
    for tid, (name, typ) in topics.items():
        if any(k in name for k in DYNAMIC_KEYS):
            cur.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (tid,))
            cnt = cur.fetchone()[0]
            short = name.replace("/vehicle_8/", "")
            print(f"  {cnt:>8,}  {short:<45} [{typ.split('/')[-1]}]")

    # ── Step 2: CAN frame analysis ────────────────────────────────────────────
    can_tid = [tid for tid, (n, t) in topics.items() if "from_can_bus" in n]
    if not can_tid:
        print("\n[WARN] No from_can_bus topic found")
        con.close()
        return

    print(f"\n{'=' * 75}")
    print("CAN FRAME ANALYSIS (sampling 10,000 frames via SQL LIMIT)")
    print("=" * 75)

    # Fast query — only CAN topic, first 10K messages
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? LIMIT 10000",
        (can_tid[0],)
    )
    rows = cur.fetchall()
    print(f"  Loaded {len(rows)} CAN frames from SQLite")

    frame_data = defaultdict(list)   # can_id → [(ts_ns, payload_bytes)]

    for ts_ns, raw_bytes in rows:
        try:
            b = bytes(raw_bytes)
            # CDR deserialize can_msgs/Frame manually:
            # [4 CDR header][4 seq][4 sec][4 nanosec][4 frame_id_len][frame_id][4-align]
            # [4 can_id][1 dlc][1 is_error][1 is_rtr][1 is_extended][8 data]
            off = 4  # skip CDR header
            off += 4  # seq
            off += 4  # stamp.sec
            off += 4  # stamp.nanosec
            fid_len = struct.unpack_from("<I", b, off)[0]
            off += 4 + fid_len
            # 4-byte align
            if off % 4:
                off += 4 - (off % 4)
            can_id = struct.unpack_from("<I", b, off)[0]
            off += 4
            dlc = b[off]; off += 1
            off += 3  # flags
            payload = b[off:off + 8]

            if 0 < can_id < 0x7FF and 1 <= dlc <= 8:
                frame_data[can_id].append((ts_ns, list(payload[:dlc]) + [0] * (8 - dlc)))
        except Exception:
            continue

    total_decoded = sum(len(v) for v in frame_data.values())
    print(f"  Decoded {total_decoded} frames across {len(frame_data)} unique CAN IDs\n")

    # ── Step 3: Identify wheel speed candidates ───────────────────────────────
    print(f"{'ID (hex)':>10} {'ID (dec)':>8} {'N':>6} {'Hz':>7}  "
          f"{'High-Var Bytes':20}  Notes")
    print("-" * 75)

    sorted_ids = sorted(frame_data.items(), key=lambda x: -len(x[1]))

    # Collect summary for threshold decision
    high_var_ids = []
    for cid, frames in sorted_ids[:30]:
        if len(frames) < 20:
            continue
        arr = np.array([f[1] for f in frames], dtype=np.uint8)  # [N, 8]
        var = arr.var(axis=0)
        mean = arr.mean(axis=0)
        # Update rate
        ts_vals = [f[0] for f in frames]
        dt_s = (ts_vals[-1] - ts_vals[0]) * 1e-9
        hz = len(frames) / max(dt_s, 0.001)

        # Wheel speed heuristic: 4 bytes with similar variance (4 symmetrical signals)
        top_b = np.argsort(var)[::-1][:4]
        top_var = var[top_b]
        var_symmetry = top_var.std() / (top_var.mean() + 1e-6)  # low = symmetric

        note = ""
        if 10 < hz < 200 and top_var.mean() > 5:
            note = " << DYNAMIC"
            if var_symmetry < 0.5 and len(top_b) == 4:
                note = " <<< WHEEL SPEED CANDIDATE"
                high_var_ids.append((cid, top_b.tolist(), hz))

        hv = " ".join([f"b{b}({var[b]:.0f})" for b in top_b])
        print(f"  0x{cid:03X}  {cid:>8d}  {len(frames):>6}  {hz:>6.1f}Hz  {hv:<20} {note}")

    print()
    print("=" * 75)
    print("WHEEL SPEED CANDIDATES SUMMARY")
    print("=" * 75)
    if high_var_ids:
        for cid, active_bytes, hz in high_var_ids:
            print(f"  CAN ID 0x{cid:03X} @ {hz:.1f} Hz — bytes {active_bytes}")
    else:
        print("  No clear 4-signal symmetric CAN IDs found in sample.")
        print("  Will use top-N dynamic CAN byte signals as candidate features.")

    con.close()


if __name__ == "__main__":
    main()
