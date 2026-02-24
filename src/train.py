"""
train.py — Training Loop for Denoising GRU Autoencoder
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Phase 1: FP32 baseline training (50 epochs)
Phase 2: QAT fine-tuning (10 epochs, lr=1e-5)
"""

import sys
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model  import DenoisingGRUAutoencoder
from config import (
    BATCH_SIZE, EPOCHS_FP32, EPOCHS_QAT, LR_FP32, LR_QAT,
    WINDOW_SIZE, NUM_FEATURES, DATA_H5_PATH, CHECKPOINT_DIR,
    ANOMALY_THRESHOLD_SIGMA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """Lazy HDF5 dataset — memory efficient for 942 GB source."""

    def __init__(self, h5_path: str):
        self.f  = h5py.File(h5_path, "r")
        self.ds = self.f["windows"]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.ds[idx])

    def __del__(self):
        if hasattr(self, "f"):
            self.f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction Loss
# ─────────────────────────────────────────────────────────────────────────────

class ReconLoss(nn.Module):
    """MSE loss over full [B, T, F] tensor."""
    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(x_hat, x)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: FP32 Training
# ─────────────────────────────────────────────────────────────────────────────

def train_fp32(model: DenoisingGRUAutoencoder,
               train_loader: DataLoader,
               val_loader: DataLoader,
               device: torch.device) -> str:
    """Train FP32 model. Returns path to best checkpoint."""

    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / "best_fp32.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FP32)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FP32, eta_min=1e-6)
    criterion = ReconLoss()
    best_val  = float("inf")

    model.to(device)
    model.train()

    for epoch in range(1, EPOCHS_FP32 + 1):
        # ── Train ──
        train_loss = 0.0
        model.train()
        for x in tqdm(train_loader, desc=f"[FP32 {epoch:02d}/{EPOCHS_FP32}] Train",
                       leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x, add_noise=True)
            loss  = criterion(x_hat, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ──
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x in val_loader:
                x     = x.to(device)
                x_hat = model(x, add_noise=False)
                val_loss += criterion(x_hat, x).item()
        val_loss /= len(val_loader)

        print(f"  Epoch {epoch:02d} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best FP32 checkpoint (val={best_val:.5f})")

    print(f"\nFP32 training complete. Best val loss: {best_val:.5f}")
    return best_path


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Quantization-Aware Training (QAT)
# ─────────────────────────────────────────────────────────────────────────────

def train_qat(model: DenoisingGRUAutoencoder,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: torch.device) -> str:
    """
    Post-training dynamic quantization for GRU.

    PyTorch's QAT (prepare_qat) does NOT support GRU/LSTM natively.
    The correct path for RNNs is quantize_dynamic, which quantizes
    GRU weights to INT8 while keeping activations FP32.
    This achieves ~4× weight compression and faster CPU inference.

    For NPU INT8 deployment, use export.py → ONNX → OpenVINO IR.
    """
    ckpt_dir = Path(CHECKPOINT_DIR)
    qat_path = str(ckpt_dir / "best_qat.pth")

    print("\nApplying dynamic quantization (INT8 weights) to GRU modules...")
    model_cpu = model.cpu().eval()

    q_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.GRU, torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Measure sizes
    import io, os
    def model_size_kb(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return buf.tell() / 1024

    fp32_kb = model_size_kb(model_cpu)
    int8_kb = model_size_kb(q_model)
    print(f"  FP32 size : {fp32_kb:.1f} KB")
    print(f"  INT8 size : {int8_kb:.1f} KB  ({100*int8_kb/fp32_kb:.0f}% of FP32)")

    # Validate accuracy on val set
    criterion = ReconLoss()
    val_loss  = 0.0
    q_model.eval()
    with torch.no_grad():
        for x in tqdm(val_loader, desc="INT8 validation", leave=False):
            x = x.cpu()
            val_loss += criterion(q_model(x, add_noise=False), x).item()
    val_loss /= len(val_loader)
    print(f"  INT8 val loss: {val_loss:.5f}")

    torch.save(q_model.state_dict(), qat_path)
    print(f"QAT checkpoint saved: {qat_path}")
    return qat_path



# ─────────────────────────────────────────────────────────────────────────────
# Threshold Calibration (3σ)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_threshold(model: DenoisingGRUAutoencoder,
                         val_loader: DataLoader,
                         device: torch.device) -> float:
    """
    Compute the 99.7th percentile (3σ) anomaly score on the validation set.
    This threshold is frozen into firmware at deployment time.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for x in tqdm(val_loader, desc="Calibrating threshold"):
            s = model.anomaly_score(x.to(device))
            scores.extend(s.cpu().numpy().tolist())

    scores = np.array(scores)
    mu     = scores.mean()
    sigma  = scores.std()
    thresh = mu + ANOMALY_THRESHOLD_SIGMA * sigma

    print(f"\nThreshold Calibration:")
    print(f"  μ = {mu:.6f} | σ = {sigma:.6f}")
    print(f"  3σ threshold = {thresh:.6f}")
    print(f"  Covers {(scores <= thresh).mean()*100:.2f}% of healthy windows")
    return float(thresh)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs-fp32",  type=int, default=15,
                   help="FP32 epochs (default 15 — use 50 on GPU)")
    p.add_argument("--epochs-qat",   type=int, default=5,
                   help="QAT epochs (default 5)")
    p.add_argument("--patience",     type=int, default=5,
                   help="Early-stopping patience (default 5)")
    p.add_argument("--max-samples",  type=int, default=0,
                   help="Cap training samples for fast runs (0=all)")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--data",         default=DATA_H5_PATH)
    args = p.parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] Dataset not found: {args.data}")
        print("Run: python src/pipeline.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Epochs : FP32={args.epochs_fp32}  QAT={args.epochs_qat}  "
          f"patience={args.patience}  batch={args.batch_size}")

    # ── Load dataset ──
    dataset = WindowDataset(args.data)
    total   = len(dataset)

    if args.max_samples > 0 and args.max_samples < total:
        from torch.utils.data import Subset
        indices = torch.randperm(total)[:args.max_samples]
        dataset = Subset(dataset, indices.tolist())
        print(f"Subset : {args.max_samples:,} / {total:,} windows")

    n_val   = max(int(0.1 * len(dataset)), 500)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Build model ──
    model = DenoisingGRUAutoencoder()
    print(model)

    # Patch epoch counts from CLI
    import config as cfg
    cfg.EPOCHS_FP32 = args.epochs_fp32
    cfg.EPOCHS_QAT  = args.epochs_qat

    # Monkey-patch train_fp32 to support early stopping via patience
    orig_train_fp32 = train_fp32

    def train_fp32_es(model, train_loader, val_loader, device):
        ckpt_dir  = Path(CHECKPOINT_DIR)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(ckpt_dir / "best_fp32.pth")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FP32)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_fp32, eta_min=1e-6)
        criterion = ReconLoss()
        best_val  = float("inf")
        no_improve = 0

        model.to(device).train()
        for epoch in range(1, args.epochs_fp32 + 1):
            train_loss = 0.0
            model.train()
            for x in tqdm(train_loader,
                          desc=f"[FP32 {epoch:02d}/{args.epochs_fp32}] Train",
                          leave=False):
                x = x.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x, add_noise=True), x)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            scheduler.step()

            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for x in val_loader:
                    val_loss += criterion(model(x.to(device), add_noise=False), x.to(device)).item()
            val_loss /= len(val_loader)
            print(f"  Epoch {epoch:02d} | train={train_loss:.5f} | val={val_loss:.5f}", flush=True)

            if val_loss < best_val:
                best_val   = val_loss
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  Saved best FP32 (val={best_val:.5f})", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                    break

        print(f"\nFP32 training done. Best val={best_val:.5f}")
        return best_path

    # ── Phase 1: FP32 ──
    best_fp32 = train_fp32_es(model, train_loader, val_loader, device)
    model.load_state_dict(torch.load(best_fp32, map_location=device, weights_only=False))

    # ── Threshold calibration ──
    threshold   = calibrate_threshold(model, val_loader, device)
    thresh_path = Path(CHECKPOINT_DIR) / "threshold.npy"
    np.save(str(thresh_path), np.array([threshold]))
    print(f"Threshold saved: {thresh_path}")

    # ── Phase 2: QAT ──
    best_qat = train_qat(model, train_loader, val_loader, device)

    print(f"\nAll done. Artifacts in {CHECKPOINT_DIR}")
    print(f"  FP32 : {best_fp32}")
    print(f"  QAT  : {best_qat}")
    print(f"  Thr  : {thresh_path} ({threshold:.6f})")
    print("Next: python src/export.py")


if __name__ == "__main__":
    main()

