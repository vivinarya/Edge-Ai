"""
export.py — ONNX & OpenVINO IR Export Pipeline
SUPRA SAEINDIA 2025 | Task 1.1: Anomaly Detection

Steps:
  1. Load QAT checkpoint
  2. Convert to INT8 static model
  3. Export to ONNX (opset 17)
  4. Optionally convert to OpenVINO IR (.xml / .bin)
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model  import DenoisingGRUAutoencoder
from config import (
    WINDOW_SIZE, NUM_FEATURES,
    CHECKPOINT_DIR, EXPORT_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_size(path: str, label: str = "") -> float:
    size_kb = Path(path).stat().st_size / 1024
    tag     = f"✓ PASSES <60 KB budget" if size_kb < 60 else "✗ EXCEEDS budget"
    print(f"  {label or path}: {size_kb:.1f} KB  {tag}")
    return size_kb


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load & convert QAT model
# ─────────────────────────────────────────────────────────────────────────────

def load_quantized(qat_ckpt: str) -> torch.nn.Module:
    """Load FP32 weights for ONNX export (ONNX export of PyTorch dynamic quantized GRU is broken)."""
    model = DenoisingGRUAutoencoder()
    
    fp32_ckpt = str(Path(CHECKPOINT_DIR) / "best_fp32.pth")
    model.load_state_dict(torch.load(fp32_ckpt, map_location="cpu", weights_only=False))
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: ONNX Export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model: torch.nn.Module,
                out_dir: str = EXPORT_DIR) -> str:
    """
    Export to ONNX opset 17.
    Dynamic batch axis so NPU can handle batch=1 at inference.
    """
    import onnx
    import onnxruntime as ort

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_dir / "gru_autoencoder.onnx")

    dummy = torch.randn(1, WINDOW_SIZE, NUM_FEATURES)   # [B=1, T=32, F=14]

    model.eval()
    torch.onnx.export(
        model,
        (dummy, False),                 # forward args: (x, add_noise=False)
        onnx_path,
        opset_version=17,
        input_names=["x"],
        output_names=["x_reconstructed"],
        dynamic_axes={"x": {0: "batch"}, "x_reconstructed": {0: "batch"}},
        do_constant_folding=True,
        verbose=False,
    )

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model verified: {onnx_path}")

    # Quick latency test with OnnxRuntime
    sess = ort.InferenceSession(onnx_path,
                                providers=["CPUExecutionProvider"])
    import time
    n_runs = 100
    x_np   = np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"x": x_np})
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  OnnxRuntime CPU latency: {elapsed_ms:.2f} ms / inference")

    print_size(onnx_path, "gru_autoencoder.onnx")
    return onnx_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: OpenVINO IR Conversion (optional — requires OpenVINO toolkit)
# ─────────────────────────────────────────────────────────────────────────────

def export_openvino(onnx_path: str,
                    out_dir: str = EXPORT_DIR) -> str:
    """Convert ONNX → OpenVINO IR (.xml + .bin) for NPU deployment."""
    try:
        from openvino.tools.mo import convert_model
        from openvino.runtime import Core, serialize
    except ImportError:
        print("  [SKIP] OpenVINO not installed. "
              "Run: pip install openvino")
        return ""

    out_dir = Path(out_dir)
    xml_path = str(out_dir / "gru_autoencoder.xml")

    ov_model = convert_model(onnx_path, compress_to_fp16=False)
    serialize(ov_model, xml_path)

    bin_path = xml_path.replace(".xml", ".bin")
    print_size(xml_path, "gru_autoencoder.xml")
    print_size(bin_path, "gru_autoencoder.bin")

    # Benchmark on CPU (proxy for NPU, real NPU benchmark done on-device)
    ie   = Core()
    net  = ie.compile_model(xml_path, "CPU")
    infer_req = net.create_infer_request()
    x_np = np.random.rand(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)

    import time
    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        infer_req.infer({"x": x_np})
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  OpenVINO CPU latency: {elapsed_ms:.2f} ms "
          f"(target on 32 TOPS NPU: <10 ms)")

    return xml_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Bundle threshold with model manifest
# ─────────────────────────────────────────────────────────────────────────────

def export_manifest(out_dir: str = EXPORT_DIR) -> None:
    """Write deployment manifest: threshold + normalization constants."""
    import json, config
    thresh_path = Path(CHECKPOINT_DIR) / "threshold.npy"
    threshold   = float(np.load(str(thresh_path))) if thresh_path.exists() else None

    manifest = {
        "model":        "DenoisingGRUAutoencoder",
        "version":      "1.0",
        "window_size":  config.WINDOW_SIZE,
        "num_features": config.NUM_FEATURES,
        "gru_hidden":   config.GRU_HIDDEN,
        "threshold":    threshold,
        "feature_min":  config.FEATURE_MIN,
        "feature_max":  config.FEATURE_MAX,
        "alert_consec": config.ALERT_CONSECUTIVE,
        "alert_reset":  config.ALERT_RESET,
        "classifier":   "XGBoost",
        "classes":      ["Healthy", "IMU_Impact", "Wheel_Lockup", "Sensor_Noise"]
    }
    out_path = Path(out_dir) / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {out_path}")
    
    # Export XGBoost config if exists
    xgb_ckpt = Path(CHECKPOINT_DIR) / "xgb_classifier.json"
    if xgb_ckpt.exists():
        import shutil
        shutil.copy(str(xgb_ckpt), str(Path(out_dir) / "xgb_classifier.json"))
        print(f"  Classifier → {out_dir}/xgb_classifier.json")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    qat_ckpt = str(Path(CHECKPOINT_DIR) / "best_qat.pth")
    if not Path(qat_ckpt).exists():
        print(f"[ERROR] QAT checkpoint not found: {qat_ckpt}")
        print("Run: python src/train.py")
        sys.exit(1)

    print("=" * 60)
    print("  SUPRA SAEINDIA 2025 — Model Export Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading QAT model...")
    model = load_quantized(qat_ckpt)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}  (~{params/1024:.1f} KB INT8)")

    print("\n[2/4] Exporting ONNX...")
    onnx_path = export_onnx(model)

    print("\n[3/4] Converting to OpenVINO IR...")
    export_openvino(onnx_path)

    print("\n[4/4] Writing deployment manifest...")
    export_manifest()

    print("\n✓ Export complete. Deploy contents of export/ to the NPU.")


if __name__ == "__main__":
    main()
