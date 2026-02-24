import streamlit as st
import time
import numpy as np
import pandas as pd
import torch
import joblib
import plotly.graph_objects as go
import copy
import pickle
import sys
from pathlib import Path

# Add src to path so we can import the GRU model and config
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import DenoisingGRUAutoencoder
from config import WINDOW_SIZE, NUM_FEATURES
from infer import MockCANStream, normalize as gru_normalize

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Raciests BMS",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# PREMIUM DARK UI INJECTION
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Reset ─────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0d0f14 !important; }
section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2130; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Hide default Streamlit chrome ─────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top Nav Bar ──────────────────────────────────── */
.bms-navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    background: #111318;
    border-bottom: 1px solid #1e2130;
    position: sticky; top: 0; z-index: 999;
}
.bms-logo { display: flex; align-items: center; gap: 10px; }
.bms-logo-icon { font-size: 24px; }
.bms-logo-text { font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }
.bms-logo-sub { font-size: 11px; color: #4ade80; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; }
.bms-nav-right { display: flex; align-items: center; gap: 16px; }
.bms-nav-badge {
    background: #1e2130; border: 1px solid #2a2f45;
    border-radius: 8px; padding: 6px 14px;
    color: #94a3b8; font-size: 13px; font-weight: 500; cursor: pointer;
}
.bms-nav-badge.live { background: rgba(74,222,128,0.1); border-color: #4ade80; color: #4ade80; }
.bms-status-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
    display: inline-block; margin-right: 6px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Page Wrapper ─────────────────────────────────── */
.bms-page { padding: 28px 32px; }

/* ── Section Title ────────────────────────────────── */
.bms-section-title {
    font-size: 13px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 28px 0 14px 0;
}

/* ── Stat Cards Row ───────────────────────────────── */
.bms-stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 8px; }
.bms-stat-card {
    background: #111318; border: 1px solid #1e2130;
    border-radius: 14px; padding: 22px 24px;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative; overflow: hidden;
}
.bms-stat-card:hover { border-color: #334155; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
.bms-stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
}
.bms-stat-icon { font-size: 18px; margin-bottom: 12px; }
.bms-stat-label { font-size: 12px; color: #64748b; font-weight: 500; margin-bottom: 8px; }
.bms-stat-value { font-size: 32px; font-weight: 700; color: #f1f5f9; letter-spacing: -1px; line-height: 1; }
.bms-stat-delta { font-size: 12px; margin-top: 8px; display: flex; align-items: center; gap: 4px; }
.bms-stat-delta.up { color: #4ade80; }
.bms-stat-delta.down { color: #f87171; }
.bms-stat-delta.warn { color: #fb923c; }

/* ── Alert Banner ─────────────────────────────────── */
.bms-alert {
    border-radius: 12px; padding: 16px 20px;
    display: flex; align-items: center; gap: 12px;
    font-size: 14px; font-weight: 500;
}
.bms-alert.nominal { background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.25); color: #4ade80; }
.bms-alert.warn    { background: rgba(251,146,60,0.08);  border: 1px solid rgba(251,146,60,0.3);  color: #fb923c; }
.bms-alert.critical{ background: rgba(248,113,113,0.1);  border: 1px solid rgba(248,113,113,0.35); color: #f87171; }

/* ── Gauge + Radar Grid ───────────────────────────── */
.bms-gauge-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.bms-gauge-card {
    background: #111318; border: 1px solid #1e2130;
    border-radius: 14px; padding: 16px 12px;
}
.bms-gauge-title { font-size: 12px; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; text-align: center; margin-bottom: 4px; }

/* ── Chart Card ───────────────────────────────────── */
.bms-chart-card {
    background: #111318; border: 1px solid #1e2130;
    border-radius: 14px; padding: 24px;
    margin-bottom: 16px;
}
.bms-chart-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; }
.bms-chart-title { font-size: 15px; font-weight: 600; color: #e2e8f0; }
.bms-chart-sub { font-size: 12px; color: #64748b; margin-top: 3px; }
.bms-legend { display: flex; gap: 20px; }
.bms-legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #94a3b8; }
.bms-legend-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── Bottom Grid: Chart + Right Panel ────────────── */
.bms-bottom-grid { display: grid; grid-template-columns: 1fr 340px; gap: 16px; }
.bms-right-panel { display: flex; flex-direction: column; gap: 16px; }

/* ── Status Panel ─────────────────────────────────── */
.bms-status-card {
    background: #111318; border: 1px solid #1e2130;
    border-radius: 14px; padding: 20px;
}
.bms-status-card-title { font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 1px; }
.bms-status-row { display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #1e2130; }
.bms-status-row:last-child { border-bottom: none; }
.bms-status-key { font-size: 13px; color: #64748b; }
.bms-status-val { font-size: 13px; font-weight: 600; color: #e2e8f0; }
.bms-status-val.green { color: #4ade80; }
.bms-status-val.yellow { color: #fbbf24; }
.bms-status-val.red { color: #f87171; }

/* ── Latency Bar ──────────────────────────────────── */
.bms-lat-bar-bg { background: #1e2130; border-radius: 99px; height: 6px; overflow: hidden; margin-top: 6px; }
.bms-lat-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #4ade80, #fb923c); transition: width 0.4s; }

/* ── Expander styling ─────────────────────────────── */
.streamlit-expanderHeader { background: #111318 !important; border-radius: 10px !important;
    border: 1px solid #1e2130 !important; color: #94a3b8 !important; font-size: 13px !important; }
.streamlit-expanderContent { background: #0f1117 !important; border: 1px solid #1e2130 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; color: #94a3b8 !important; }

/* ── Plotly chart bg ──────────────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION & STATIC PARAMETERS
# ==========================================
GRU_CHECKPOINT_PATH = "checkpoints/best_fp32.pth"
LSTM_MODEL_PATH     = "battery_model_scripted.pt"
SCALER_PATH         = "scaler.pkl"
GRU_THRESHOLD_PATH  = "checkpoints/threshold.npy"

# ==========================================
# HELPER FUNCTIONS & MOCK LOADERS
# ==========================================

@st.cache_resource
def load_scaler(scaler_path):
    try:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)
    except:
        class MockScaler:
            def transform(self, X):
                return (X - np.array([3.5, 4.0, 2.5, 0.5, 0.95, 0.5])) / np.array([0.5, 0.1, 0.5, 0.5, 0.1, 0.5])
        return MockScaler()


@st.cache_resource
def load_models():
    try:
        lstm_model = torch.jit.load(LSTM_MODEL_PATH)
        lstm_model.eval()
        use_mock_lstm = False
    except:
        lstm_model = None
        use_mock_lstm = True

    try:
        gru_model = DenoisingGRUAutoencoder()
        gru_model.load_state_dict(torch.load(GRU_CHECKPOINT_PATH, map_location="cpu", weights_only=False))
        gru_model.eval()
        use_mock_gru = False
        threshold = 0.05
        if Path(GRU_THRESHOLD_PATH).exists():
            threshold = float(np.load(GRU_THRESHOLD_PATH)[0])
    except Exception as e:
        gru_model = None
        use_mock_gru = True
        threshold = 0.05

    return lstm_model, use_mock_lstm, gru_model, use_mock_gru, threshold


def simulate_daq_inputs(t, emi_noise_std=0.05, can_stream=None):
    base_v = 3.7 - (t % 100) * 0.01
    base_i = 40.0 * np.sin(t * 0.1)
    base_t = 30.0 + (t % 100) * 0.3

    if np.random.rand() > 0.98: base_t += 35.0
    if np.random.rand() > 0.98: base_v -= 1.5

    noisy_v = base_v + np.random.normal(0, emi_noise_std)
    noisy_i = base_i + np.random.normal(0, emi_noise_std * 5)
    noisy_t = base_t + np.random.normal(0, emi_noise_std * 2)

    if np.random.rand() > 0.99:
        noisy_t = -40.0
    elif np.random.rand() > 0.99:
        noisy_v = np.nan

    if can_stream is not None:
        raw_sample = can_stream.read_sample()
        if np.random.rand() > 0.98:
            raw_sample[12] = 85.0
        gru_features = gru_normalize(raw_sample)
    else:
        gru_features = np.random.normal(0.5, 0.1, NUM_FEATURES).astype(np.float32)
        gru_features = np.clip(gru_features, 0.0, 1.0)

    if not np.isnan(noisy_v) and noisy_v < 2.5:
        gru_features[0] = 1.0
        gru_features[6] = 1.0

    return np.array([noisy_v, noisy_i, noisy_t], dtype=np.float32), gru_features


def run_inference(daq_features, seq_buffer_lstm, seq_buffer_gru, scaler, lstm_model, mock_lstm, gru_model, mock_gru, gru_threshold):
    start_time = time.perf_counter()
    anomaly_alert = False
    anomaly_score = 0.0
    diff_sq = np.zeros(NUM_FEATURES, dtype=np.float32)

    if not mock_gru:
        x_gru_tensor = torch.tensor(seq_buffer_gru, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            x_hat = gru_model(x_gru_tensor, add_noise=False)
            diff = (x_gru_tensor - x_hat).numpy()
            diff_sq = (diff ** 2)[0, -1, :]
            anomaly_score = float(diff_sq.mean())
        anomaly_alert = anomaly_score >= gru_threshold
    else:
        anomaly_score = float(np.random.normal(0.004, 0.001))
        diff_sq = np.random.uniform(0, 0.01, NUM_FEATURES).astype(np.float32)
        anomaly_alert = False

    soh_pred = 0.0
    rul_pred = 0.0

    if not mock_lstm:
        x_scaled = scaler.transform(seq_buffer_lstm)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = lstm_model(x_tensor)
            soh_pred = float(np.clip(pred[0, 0].item() * 100.0, 0, 100))
            rul_pred = float(np.clip(pred[0, 1].item() * 100.0, 0, 100))
    else:
        soh_pred = float(np.clip(95 - (daq_features[2] - 30) * 0.3 + np.random.normal(0, 0.5), 0, 100))
        rul_pred = float(np.clip(80 - (seq_buffer_lstm[-1, 5] * 20) + np.random.normal(0, 0.5), 0, 100))

    latency_ms = (time.perf_counter() - start_time) * 1000
    return anomaly_score, anomaly_alert, soh_pred, rul_pred, diff_sq, latency_ms


# ==========================================
# PLOTLY CHART HELPERS
# ==========================================

def make_gauge(value, color, max_val=100):
    value = float(np.clip(value, 0, max_val))
    pct = value / max_val
    if color == "anomaly":
        bar_color = "#4ade80" if pct < 0.5 else ("#fb923c" if pct < 0.8 else "#f87171")
    elif color == "blue":
        bar_color = "#3b82f6"
    elif color == "purple":
        bar_color = "#a855f7"
    else:
        bar_color = "#4ade80"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"color": "#e2e8f0", "size": 28, "family": "Inter"}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 0, "tickcolor": "transparent",
                     "tickfont": {"color": "#334155"}, "color": "#1e2130"},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#0d0f14",
            "borderwidth": 0,
            "steps": [{"range": [0, max_val], "color": "#1e2130"}],
            "threshold": {"line": {"color": bar_color, "width": 2}, "thickness": 0.7, "value": value}
        }
    ))
    fig.update_layout(
        height=180, margin=dict(l=16, r=16, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


def make_radar(feature_errors, is_alert):
    cats = ['Accel-X', 'Accel-Z', 'Gyro-Z', 'Vel-X (Bot)', 'Vel-X (Top)', 'Vel-Y']
    vals = [
        float(feature_errors[0]) * 80,
        float(feature_errors[2]) * 80,
        float(feature_errors[11]) * 80,
        float(feature_errors[12]) * 80,
        float(feature_errors[15]) * 80,
        float(feature_errors[19]) * 80,
    ]
    fill_c = "rgba(248,113,113,0.25)" if is_alert else "rgba(59,130,246,0.2)"
    line_c = "#f87171" if is_alert else "#3b82f6"

    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill='toself', fillcolor=fill_c,
        line=dict(color=line_c, width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d0f14",
            radialaxis=dict(visible=False, range=[0, 1.0]),
            angularaxis=dict(tickfont=dict(size=10, color="#64748b", family="Inter"), linecolor="#1e2130"),
            gridshape="linear",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=180,
        margin=dict(l=18, r=18, t=16, b=16),
        font={"family": "Inter"}
    )
    return fig


def make_history_chart(df):
    fig = go.Figure()
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=df["Time"], y=df["SoH"],
            mode='lines', name='SoH (%)',
            line=dict(width=2.5, color='#3b82f6'),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.07)'
        ))
        fig.add_trace(go.Scatter(
            x=df["Time"], y=df["RUL"],
            mode='lines', name='RUL (%)',
            line=dict(width=2, color='#a855f7', dash='dot'),
            fill='tozeroy', fillcolor='rgba(168,85,247,0.05)'
        ))
        fig.add_trace(go.Scatter(
            x=df["Time"],
            y=(df["Anomaly Score"] / df["Anomaly Score"].max().clip(lower=1e-9) * 100).clip(0, 100),
            mode='lines', name='GRU Shift (%)',
            line=dict(width=1.5, color='#fb923c', dash='dash'),
        ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#334155", tickfont=dict(color="#475569", size=11)),
        yaxis=dict(gridcolor="#1e2130", color="#334155", tickfont=dict(color="#475569", size=11), range=[0, 105]),
        legend=dict(font=dict(color="#94a3b8", size=11, family="Inter"),
                    bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.12),
        margin=dict(l=0, r=0, t=28, b=0),
        font={"family": "Inter"},
        hovermode="x unified"
    )
    return fig


# ==========================================
# MAIN DASHBOARD
# ==========================================

def main():
    # ── Session State ──────────────────────────────────
    if "data_history" not in st.session_state:
        st.session_state.data_history = pd.DataFrame(columns=["Time", "Anomaly Score", "SoH", "RUL", "Latency"])
        st.session_state.tick = 0
    if "seq_buffer_lstm" not in st.session_state:
        st.session_state.seq_buffer_lstm = np.zeros((50, 6), dtype=np.float32)
    if "seq_buffer_gru" not in st.session_state:
        st.session_state.seq_buffer_gru = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
    if "can_stream" not in st.session_state:
        st.session_state.can_stream = MockCANStream(fault_at_sec=999999)
    if "last_v" not in st.session_state:
        st.session_state.last_v = 3.7

    scaler  = load_scaler(SCALER_PATH)
    lstm_model, use_mock_lstm, gru_model, use_mock_gru, gru_threshold = load_models()

    # ── Top Nav ─────────────────────────────────────────
    simulate = st.sidebar.checkbox("▶ Start Live Simulation", value=False)
    if st.sidebar.button("⟳ Clear History"):
        st.session_state.data_history = pd.DataFrame(columns=["Time", "Anomaly Score", "SoH", "RUL", "Latency"])
        st.session_state.tick = 0

    st.markdown("""
    <div class="bms-navbar">
        <div class="bms-logo">
            <span class="bms-logo-icon">🔋</span>
            <div>
                <div class="bms-logo-text">Raciests BMS</div>
                <div class="bms-logo-sub">Edge ML · Formula SAE</div>
            </div>
        </div>
        <div class="bms-nav-right">
            <div class="bms-nav-badge">GRU Autoencoder · INT8</div>
            <div class="bms-nav-badge">AttentiveLSTM · SoH/RUL</div>
            <div class="bms-nav-badge live"><span class="bms-status-dot"></span>NPU &lt;50ms</div>
        </div>
    </div>
    <div class="bms-page">
    """, unsafe_allow_html=True)

    # ── Placeholder containers ───────────────────────────
    stat_row    = st.empty()
    alert_row   = st.empty()
    gauge_row   = st.empty()
    chart_row   = st.empty()
    latency_row = st.empty()

    # ── Docs expanders ───────────────────────────────────
    with st.expander("📝 Task 1 & 3: Anomaly Detection & Edge Deployment", expanded=False):
        st.markdown("""
**1. Architecture — Unsupervised Denoising GRU Autoencoder**
- **Why Unsupervised?** Mechanical failures are unique — no labeled dataset can cover every suspension/battery failure mode. The model learns the *nominal track distribution*; any unseen telemetry state fails to reconstruct cleanly, generating an automatic alert.
- **Why GRU?** Gated Recurrent Units are 3× lighter than LSTMs while preserving sequential memory across the 32-step window. They track vibration resonance frequency shifts (loose wheel ≠ bump).
- **Deviation Radar:** Per-channel reconstruction error ²  is visualized on a polar chart, showing exactly *which* sensor is deviating.

**2. Edge Deployment (INT8, <60 KB, <50 ms)**
- QAT shrinks FP32 weights → INT8 integers, compressing the model to **~49 KB**.
- `torch.onnx.export` (Opset 17) freezes the graph for the NPU compiler (OpenVINO / NXP eIQ).
        """)
    with st.expander("📝 Task 4: Feature Engineering & Racing Context", expanded=False):
        st.markdown("""
**1. Feature Engineering (6 longitudinal features → AttentiveLSTM)**
- `discharge_median_voltage` / `charge_median_voltage`: median-aggregated per charge cycle to remove instantaneous driver-induced noise.
- `energy_efficiency` (discharge_Wh / charge_Wh): implicit internal-resistance proxy — rises as cell ages.
- `cycle_index_norm`: normalized lifespan position for long-horizon SoH trajectory.

**2. Racing Context — High-C-Rate & Regen Braking**
Standard dense models see a violent throttle-induced voltage sag and predict "dead battery". Our *Rolling Sequence Memory* (50-step LSTM window + Temporal Attention) distinguishes transient racing dynamics from permanent capacity fade, maintaining stable SoH even during 8C discharge pulses.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # SIMULATION TICK
    # ==========================================
    if simulate:
        st.session_state.tick += 1
        t = st.session_state.tick

        # ── 1. Acquire & Filter ──────────────────────────
        daq_inputs, gru_features = simulate_daq_inputs(t, can_stream=st.session_state.can_stream)
        v, i, temp = daq_inputs

        sensor_fault_type = None

        # NaN imputation (zero-order hold)
        if np.isnan(v):
            v_imputed = st.session_state.last_v
            sensor_fault_type = "NaN Imputation — Loose Voltage Wire"
        else:
            v_imputed = float(v)
            st.session_state.last_v = v_imputed

        # Garbage filter
        if temp < -20.0 or temp > 150.0:
            temp_imputed = 30.0
            sensor_fault_type = f"Garbage Filtered — Thermistor read {temp:.1f}°C"
        else:
            temp_imputed = float(temp)

        sensor_fault   = sensor_fault_type is not None
        critical_anomaly = (temp_imputed > 60.0 or v_imputed < 2.5) and not sensor_fault

        # ── 2. Sequence Buffers ──────────────────────────
        pseudo_lstm = np.array([v_imputed, 4.0, 2.5, (t % 100)/100.0, 0.95, min(t/1000.0, 1.0)], dtype=np.float32)
        st.session_state.seq_buffer_lstm[:-1] = st.session_state.seq_buffer_lstm[1:]
        st.session_state.seq_buffer_lstm[-1]  = pseudo_lstm
        st.session_state.seq_buffer_gru[:-1]  = st.session_state.seq_buffer_gru[1:]
        st.session_state.seq_buffer_gru[-1]   = gru_features

        # ── 3. Inference ─────────────────────────────────
        score, alert, soh, rul, feat_err, lat = run_inference(
            np.array([v_imputed, float(i), temp_imputed], dtype=np.float32),
            st.session_state.seq_buffer_lstm, st.session_state.seq_buffer_gru,
            scaler, lstm_model, use_mock_lstm, gru_model, use_mock_gru, gru_threshold
        )

        # ── 4. History ───────────────────────────────────
        new_row = pd.DataFrame({"Time":[t],"Anomaly Score":[score],"SoH":[soh],"RUL":[rul],"Latency":[lat]})
        st.session_state.data_history = pd.concat([st.session_state.data_history, new_row], ignore_index=True).tail(60)

        shift_pct = float(min((score / gru_threshold) * 100.0, 100.0))

        # Determine status strings
        if critical_anomaly:
            alert_cls  = "critical"
            alert_icon = "🚨"
            alert_msg  = "CRITICAL PHYSICAL ANOMALY — Thermal Runaway Risk (Temp >60°C or V <2.5V)"
        elif sensor_fault:
            alert_cls  = "warn"
            alert_icon = "⚠️"
            alert_msg  = f"SENSOR FAULT FILTERED — {sensor_fault_type}"
        elif alert:
            alert_cls  = "warn"
            alert_icon = "⚠️"
            alert_msg  = f"GRU INTERRUPT — Unsupervised Telemetry Shift (Score: {score:.5f})"
        else:
            alert_cls  = "nominal"
            alert_icon = "✅"
            alert_msg  = "NOMINAL — Track &amp; Race Conditions Verified"

        soh_delta_cls  = "up" if soh > 80 else ("warn" if soh > 50 else "down")
        rul_delta_cls  = "up" if rul > 60 else ("warn" if rul > 30 else "down")
        v_delta_cls    = "up" if v_imputed >= 3.5 else ("warn" if v_imputed >= 2.8 else "down")
        lat_delta_cls  = "up" if lat < 20 else ("warn" if lat < 40 else "down")

        v_arrow  = "↑" if v_imputed >= 3.5 else "↓"
        s_arrow  = "↑" if soh > 80 else "↓"
        r_arrow  = "↑" if rul > 60 else "↓"
        l_arrow  = "↑" if lat < 20 else "↓"

        # ── 5. Render ─────────────────────────────────────

        # Stat Cards
        stat_row.markdown(f"""
        <div class="bms-section-title">Live Telemetry</div>
        <div class="bms-stat-grid">
            <div class="bms-stat-card" style="--accent: #3b82f6">
                <div class="bms-stat-icon">⚡</div>
                <div class="bms-stat-label">Pack Voltage</div>
                <div class="bms-stat-value">{v_imputed:.2f}<span style="font-size:14px;color:#64748b"> V</span></div>
                <div class="bms-stat-delta {v_delta_cls}">{v_arrow} {abs(v_imputed-3.7):.3f}V from nominal</div>
            </div>
            <div class="bms-stat-card" style="--accent: #a855f7">
                <div class="bms-stat-icon">🌡️</div>
                <div class="bms-stat-label">Pack Temperature</div>
                <div class="bms-stat-value">{temp_imputed:.1f}<span style="font-size:14px;color:#64748b"> °C</span></div>
                <div class="bms-stat-delta {'warn' if temp_imputed>45 else 'up'}">{'↑ Hot' if temp_imputed>45 else '↑ Nominal'} · Δ{temp_imputed-30:.1f}°C</div>
            </div>
            <div class="bms-stat-card" style="--accent: #4ade80">
                <div class="bms-stat-icon">🔋</div>
                <div class="bms-stat-label">State of Health</div>
                <div class="bms-stat-value">{soh:.1f}<span style="font-size:14px;color:#64748b">%</span></div>
                <div class="bms-stat-delta {soh_delta_cls}">{s_arrow} LSTM AttentionNet · Cycle {t}</div>
            </div>
            <div class="bms-stat-card" style="--accent: #fb923c">
                <div class="bms-stat-icon">⏱️</div>
                <div class="bms-stat-label">NPU Latency</div>
                <div class="bms-stat-value">{lat:.1f}<span style="font-size:14px;color:#64748b"> ms</span></div>
                <div class="bms-stat-delta {lat_delta_cls}">{l_arrow} Budget: 50ms · {'✓ PASS' if lat<50 else '✗ BREACH'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Alert Banner
        alert_row.markdown(f"""
        <div style="margin-top:12px">
            <div class="bms-alert {alert_cls}">
                <span style="font-size:18px">{alert_icon}</span>
                <span>{alert_msg}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge Row
        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            st.markdown('<div class="bms-gauge-card"><div class="bms-gauge-title">State of Health</div>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(soh, "blue"), use_container_width=True, key="g_soh")
            st.markdown('</div>', unsafe_allow_html=True)
        with gc2:
            st.markdown('<div class="bms-gauge-card"><div class="bms-gauge-title">Remaining Useful Life</div>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(rul, "purple"), use_container_width=True, key="g_rul")
            st.markdown('</div>', unsafe_allow_html=True)
        with gc3:
            st.markdown('<div class="bms-gauge-card"><div class="bms-gauge-title">GRU Unsupervised Shift</div>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(shift_pct, "anomaly"), use_container_width=True, key="g_shift")
            st.markdown('</div>', unsafe_allow_html=True)
        with gc4:
            st.markdown('<div class="bms-gauge-card"><div class="bms-gauge-title">Deviation Radar (IMU/Odom)</div>', unsafe_allow_html=True)
            st.plotly_chart(make_radar(feat_err, alert), use_container_width=True, key="g_radar")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Bottom Chart Row ──────────────────────────────
        st.markdown('<div class="bms-section-title" style="margin-top:24px">Battery Health Forecasting vs Time</div>', unsafe_allow_html=True)

        ch_col, st_col = st.columns([2, 1])
        with ch_col:
            df = st.session_state.data_history
            st.plotly_chart(make_history_chart(df), use_container_width=True, key="main_chart")

        with st_col:
            lat_pct = min(lat / 50.0, 1.0) * 100
            lat_color_val = "#4ade80" if lat < 20 else ("#fb923c" if lat < 40 else "#f87171")
            cur_i = float(i)

            st.markdown(f"""
            <div class="bms-status-card">
                <div class="bms-status-card-title">System Status</div>
                <div class="bms-status-row">
                    <span class="bms-status-key">GRU Model</span>
                    <span class="bms-status-val green">{'FP32 Active' if not use_mock_gru else 'MOCK'}</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">LSTM Model</span>
                    <span class="bms-status-val green">{'TorchScript' if not use_mock_lstm else 'MOCK'}</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">GRU Threshold</span>
                    <span class="bms-status-val">{gru_threshold:.5f}</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">Anomaly Score</span>
                    <span class="bms-status-val {'red' if alert else 'green'}">{score:.5f}</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">Current Draw</span>
                    <span class="bms-status-val {'red' if abs(cur_i)>50 else 'yellow' if abs(cur_i)>30 else 'green'}">{cur_i:+.1f} A</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">Sensor Fault</span>
                    <span class="bms-status-val {'red' if sensor_fault else 'green'}">{'DETECTED' if sensor_fault else 'NOMINAL'}</span>
                </div>
                <div class="bms-status-row">
                    <span class="bms-status-key">NPU Latency</span>
                    <span class="bms-status-val" style="color:{lat_color_val}">{lat:.2f} ms</span>
                </div>
                <div style="margin-top:12px">
                    <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-bottom:5px">
                        <span>Latency Budget</span><span>{lat_pct:.0f}% / 50ms</span>
                    </div>
                    <div class="bms-lat-bar-bg">
                        <div class="bms-lat-bar-fill" style="width:{lat_pct:.1f}%;background:{lat_color_val}"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.1)
        st.rerun()

    else:
        # ── Idle State ─────────────────────────────────────
        df = st.session_state.data_history
        st.markdown("""
        <div class="bms-section-title">Live Telemetry</div>
        <div class="bms-stat-grid">
            <div class="bms-stat-card" style="--accent:#3b82f6">
                <div class="bms-stat-icon">⚡</div>
                <div class="bms-stat-label">Pack Voltage</div>
                <div class="bms-stat-value">—<span style="font-size:14px;color:#64748b"> V</span></div>
                <div class="bms-stat-delta up">Awaiting simulation</div>
            </div>
            <div class="bms-stat-card" style="--accent:#a855f7">
                <div class="bms-stat-icon">🌡️</div>
                <div class="bms-stat-label">Pack Temperature</div>
                <div class="bms-stat-value">—<span style="font-size:14px;color:#64748b"> °C</span></div>
                <div class="bms-stat-delta up">Awaiting simulation</div>
            </div>
            <div class="bms-stat-card" style="--accent:#4ade80">
                <div class="bms-stat-icon">🔋</div>
                <div class="bms-stat-label">State of Health</div>
                <div class="bms-stat-value">—<span style="font-size:14px;color:#64748b">%</span></div>
                <div class="bms-stat-delta up">LSTM AttentionNet</div>
            </div>
            <div class="bms-stat-card" style="--accent:#fb923c">
                <div class="bms-stat-icon">⏱️</div>
                <div class="bms-stat-label">NPU Latency</div>
                <div class="bms-stat-value">—<span style="font-size:14px;color:#64748b"> ms</span></div>
                <div class="bms-stat-delta up">Budget: 50ms</div>
            </div>
        </div>
        <div style="margin-top:12px">
            <div class="bms-alert nominal">
                <span style="font-size:18px">💤</span>
                <span>Simulation paused — toggle <strong>▶ Start Live Simulation</strong> in the sidebar to begin.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(df) > 1:
            st.markdown('<div class="bms-section-title" style="margin-top:24px">Last Session History</div>', unsafe_allow_html=True)
            st.plotly_chart(make_history_chart(df), use_container_width=True)


if __name__ == "__main__":
    main()
