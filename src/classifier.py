"""
classifier.py — Fault Classification using XGBoost
SUPRA SAEINDIA 2025 | Task 1.2: Fault Classification

This module takes the anomaly embeddings (or raw data) flagged by the GRU autoencoder
and classifies the specific fault type using an XGBoost Classifier.

Usage:
    python src/classifier.py
"""

import os
import sys
import numpy as np
import h5py
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_H5_PATH, CHECKPOINT_DIR, WINDOW_SIZE, NUM_FEATURES

XGB_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "xgb_classifier.json")
LBL_ENCODER_PATH = os.path.join(CHECKPOINT_DIR, "lbl_encoder.pkl")

# Define possible fault classes
# Since we only have healthy data (from pipeline.py), we will simulate fault injections
# to train the classifier, similar to what we did in `infer.py`'s MockCANStream.
#
# 0: Healthy
# 1: IMU Impact (Suspension/Chassis)
# 2: Wheel Lockup (Speed/Velocity drop)
# 3: Sensor Noise (EMI/Connection issues)
FAULT_CLASSES = ["Healthy", "IMU_Impact", "Wheel_Lockup", "Sensor_Noise"]


def inject_faults(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes healthy windows and injects synthetic faults to create a labeled dataset.
    Returns: (X, y)
    """
    print(f"Generating synthetic fault dataset from {len(windows)} healthy windows...")
    
    n_total = len(windows)
    
    # We will quadruple the dataset, adding one copy for each fault type
    X = np.zeros((n_total * 4, WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
    y = np.zeros(n_total * 4, dtype=np.int32)
    
    # 0: Healthy (first quarter)
    X[0:n_total] = windows.copy()
    y[0:n_total] = 0
    
    # 1: IMU Impact (second quarter)
    X_acc = windows.copy()
    # Spike on bot_accel_x (idx 0), top_accel_x (idx 6) randomly within the window
    for i in range(n_total):
        t_spike = np.random.randint(0, WINDOW_SIZE - 2)
        X_acc[i, t_spike:t_spike+2, 0] = np.clip(X_acc[i, t_spike:t_spike+2, 0] * 5.0, 0, 1)
        X_acc[i, t_spike:t_spike+2, 6] = np.clip(X_acc[i, t_spike:t_spike+2, 6] * 5.0, 0, 1)
    
    X[n_total:2*n_total] = X_acc
    y[n_total:2*n_total] = 1
    
    # 2: Wheel Lockup (third quarter)
    X_lock = windows.copy()
    # Drop velocities: bot_vel_x (idx 12), top_vel_x (idx 15), loc_vel_x (idx 18)
    for i in range(n_total):
        t_drop = np.random.randint(0, WINDOW_SIZE - 5)
        # Smoothly drop to near zero
        X_lock[i, t_drop:, 12] *= 0.2
        X_lock[i, t_drop:, 15] *= 0.2
        X_lock[i, t_drop:, 18] *= 0.2
        
    X[2*n_total:3*n_total] = X_lock
    y[2*n_total:3*n_total] = 2
    
    # 3: Sensor Noise (fourth quarter)
    X_noise = windows.copy()
    # Add high frequency noise to random channels
    for i in range(n_total):
        chan = np.random.randint(0, NUM_FEATURES)
        noise = np.random.normal(0, 0.3, WINDOW_SIZE)
        X_noise[i, :, chan] = np.clip(X_noise[i, :, chan] + noise, 0, 1)
        
    X[3*n_total:] = X_noise
    y[3*n_total:] = 3
    
    return X, y


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from the 3D tensor [B, T, F] 
    to create a 2D feature matrix [B, F'] for XGBoost.
    """
    print("Extracting statistical features for XGBoost...")
    # Features: mean, std, min, max per channel over the time window
    # Shape of X: (B, 32, 21) -> Output: (B, 21 * 4) = (B, 84)
    f_mean = np.mean(X, axis=1)
    f_std  = np.std(X, axis=1)
    f_min  = np.min(X, axis=1)
    f_max  = np.max(X, axis=1)
    
    # Also add "jump" (max diff between consecutive samples)
    f_diff = np.max(np.abs(np.diff(X, axis=1)), axis=1)
    
    return np.concatenate([f_mean, f_std, f_min, f_max, f_diff], axis=1)


def train_xgboost():
    print("=" * 60)
    print("  SUPRA SAEINDIA 2025 — Fault Classification (XGBoost)")
    print("=" * 60)

    # 1. Load data
    if not Path(DATA_H5_PATH).exists():
        print(f"[ERROR] H5 Dataset not found: {DATA_H5_PATH}")
        sys.exit(1)
        
    print(f"\n[1/4] Loading healthy data from {DATA_H5_PATH}")
    with h5py.File(DATA_H5_PATH, "r") as f:
        # Load up to 5000 windows to keep it reasonable for XGBoost memory
        n = min(len(f["windows"]), 5000)
        healthy_windows = f["windows"][:n]
        
    # 2. Inject faults to create labels
    print("\n[2/4] Generating labeled dataset...")
    X_raw, y = inject_faults(healthy_windows)
    
    # 3. Feature Extraction
    print(f"\n[3/4] Extracting features (flattening [B, T, F] → [B, F'])")
    X_feat = extract_features(X_raw)
    print(f"  Feature matrix shape: {X_feat.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 4. Train Model
    print("\n[4/4] Training XGBoost Classifier...")
    
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(FAULT_CLASSES),
        n_jobs=-1,
        random_state=42
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )
    
    # 5. Evaluate
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=FAULT_CLASSES))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. Save Model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    clf.save_model(XGB_MODEL_PATH)
    joblib.dump(FAULT_CLASSES, LBL_ENCODER_PATH)
    
    print(f"\n✓ Saved model to {XGB_MODEL_PATH}")
    print(f"✓ Saved classes to {LBL_ENCODER_PATH}")


if __name__ == "__main__":
    train_xgboost()
