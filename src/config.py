"""
config.py — Frozen system constants for Task 1.1 Anomaly Detection.
Updated to full 21-feature dynamic vector from all available /vehicle_8 standard topics.
ALL values fixed at training time — must match firmware exactly.
"""

# ── Data / Hardware ──────────────────────────────────────────────────────────
SAMPLE_RATE_HZ: int = 1000          # Target unified sample rate (1 kHz)
WINDOW_SIZE: int = 32               # Samples per inference window (32 ms)
NUM_FEATURES: int = 21              # Full dynamic feature vector

# ── Model Topology ───────────────────────────────────────────────────────────
GRU_HIDDEN: int = 64
GRU_LAYERS: int = 1

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 512
EPOCHS_FP32: int = 50
EPOCHS_QAT: int = 10
LR_FP32: float = 1e-3
LR_QAT: float = 1e-5
NOISE_STD: float = 0.05             # Denoising corruption strength

# ── Feature vector order ─────────────────────────────────────────────────────
# Idx | Signal            | Topic                      | Hz
#  0  | bot_accel_x (m/s²)| novatel_bottom/rawimux     | 125
#  1  | bot_accel_y       |                             |
#  2  | bot_accel_z       |                             |
#  3  | bot_gyro_x (r/s)  |                             |
#  4  | bot_gyro_y        |                             |
#  5  | bot_gyro_z        |                             |
#  6  | top_accel_x       | novatel_top/rawimux         | 125
#  7  | top_accel_y       |                             |
#  8  | top_accel_z       |                             |
#  9  | top_gyro_x        |                             |
# 10  | top_gyro_y        |                             |
# 11  | top_gyro_z        |                             |
# 12  | bot_vel_x (m/s)   | novatel_bottom/odom         |  60
# 13  | bot_vel_y         |                             |
# 14  | bot_ang_z (r/s)   |                             |
# 15  | top_vel_x         | novatel_top/odom            |  60
# 16  | top_vel_y         |                             |
# 17  | top_ang_z         |                             |
# 18  | loc_vel_x         | local_odometry              |  20
# 19  | loc_vel_y         |                             |
# 20  | loc_ang_z         |                             |

# ── Normalization bounds (frozen — must match firmware) ───────────────────────
FEATURE_MIN = [
    # Bottom IMU
    -50.0, -50.0, -50.0, -15.0, -15.0, -15.0,
    # Top IMU
    -50.0, -50.0, -50.0, -15.0, -15.0, -15.0,
    # Bottom odom
    -70.0, -70.0,  -5.0,
    # Top odom
    -70.0, -70.0,  -5.0,
    # Local odom
    -70.0, -70.0,  -5.0,
]

FEATURE_MAX = [
    # Bottom IMU
     50.0,  50.0,  50.0,  15.0,  15.0,  15.0,
    # Top IMU
     50.0,  50.0,  50.0,  15.0,  15.0,  15.0,
    # Bottom odom
     70.0,  70.0,   5.0,
    # Top odom
     70.0,  70.0,   5.0,
    # Local odom
     70.0,  70.0,   5.0,
]

# ── ROS2 Topics (all standard-type topics with data in M-SOLO-FAST bag) ────────
TOPICS = {
    "/vehicle_8/novatel_bottom/rawimux": 125,   # sensor_msgs/msg/Imu
    "/vehicle_8/novatel_top/rawimux":    125,   # sensor_msgs/msg/Imu
    "/vehicle_8/novatel_bottom/odom":     60,   # nav_msgs/msg/Odometry
    "/vehicle_8/novatel_top/odom":        60,   # nav_msgs/msg/Odometry
    "/vehicle_8/local_odometry":          20,   # nav_msgs/msg/Odometry
    # Excluded: novatel_oem7_msgs types (BESTPOS/BESTVEL/INSPVAX) — custom CDR
    # Excluded: from_can_bus — custom CDR offset (no DBC available)
}

# ── Anomaly Detection ─────────────────────────────────────────────────────────
ANOMALY_THRESHOLD_SIGMA: float = 3.0
ALERT_CONSECUTIVE: int = 3
ALERT_RESET: int = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
BAG_DIR        = "data/M_SOLO_FAST"
DB3_BAG_PATH   = "data/M_SOLO_FAST/M-SOLO-FAST-100-140.db3"
DATA_H5_PATH   = "data/s3_windows.h5"
CHECKPOINT_DIR = "checkpoints/"
EXPORT_DIR     = "export/"
