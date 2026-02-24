import React from 'react';

function ModelCard({ title, color, items, stats }) {
    return (
        <div className="card" style={{ padding: '22px 24px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 18 }}>
                <div style={{ fontSize: 17, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.3px' }}>
                    {title}
                </div>
                <div style={{
                    background: `${color}15`, border: `1px solid ${color}30`,
                    borderRadius: 6, padding: '4px 10px', fontSize: 10, fontWeight: 600, color,
                    fontFamily: 'var(--font-mono)',
                }}>
                    ACTIVE
                </div>
            </div>

            {/* Stat chips */}
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
                {stats.map(s => (
                    <div key={s.label} style={{
                        background: 'var(--surface2)', border: '1px solid var(--border)',
                        borderRadius: 6, padding: '6px 12px',
                    }}>
                        <div style={{ fontSize: 9, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 3 }}>{s.label}</div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)', fontFamily: 'var(--font-mono)' }}>{s.value}</div>
                    </div>
                ))}
            </div>

            {/* Feature list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {items.map((item, i) => (
                    <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                        <div style={{ width: 4, height: 4, borderRadius: '50%', background: color, marginTop: 6, flexShrink: 0 }} />
                        <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
                            <strong style={{ color: 'var(--text)', fontWeight: 600 }}>{item.title}</strong>
                            {item.title && ' — '}{item.desc}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

function LiveValues({ data }) {
    const rows = [
        { k: 'GRU Model', v: data.mock_gru ? 'MOCK (fallback)' : 'FP32 Active', c: data.mock_gru ? 'var(--red)' : 'var(--green)' },
        { k: 'LSTM Model', v: data.mock_lstm ? 'MOCK (fallback)' : 'TorchScript', c: data.mock_lstm ? 'var(--red)' : 'var(--green)' },
        { k: 'GRU Threshold', v: (data.gru_threshold || 0.011).toFixed(5), c: 'var(--text)' },
        { k: 'Last Anomaly Score', v: (data.anomaly_score || 0).toFixed(6), c: data.gru_alert ? 'var(--red)' : 'var(--green)' },
        { k: 'Shift %', v: `${(data.anomaly_shift_pct || 0).toFixed(2)}%`, c: 'var(--text)' },
        { k: 'Last SoH', v: `${(data.soh || 0).toFixed(2)}%`, c: 'var(--blue)' },
        { k: 'Last RUL', v: `${(data.rul || 0).toFixed(2)}%`, c: 'var(--purple)' },
        { k: 'Inference (GRU + LSTM)', v: `${(data.latency_ms || 0).toFixed(2)} ms`, c: data.latency_ms > 40 ? 'var(--orange)' : 'var(--green)' },
    ];
    return (
        <div className="card" style={{ padding: '22px 24px' }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1.2px', marginBottom: 16 }}>
                Live Diagnostics
            </div>
            {rows.map(r => (
                <div key={r.k} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '8px 0', borderBottom: '1px solid var(--border)',
                }}>
                    <span style={{ fontSize: 12, color: 'var(--text3)' }}>{r.k}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: r.c, fontFamily: 'var(--font-mono)', transition: 'color 0.3s' }}>
                        {r.v}
                    </span>
                </div>
            ))}
        </div>
    );
}

export default function MLModels({ data }) {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>ML Models</div>
                <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>
                    Architecture overview and live runtime diagnostics
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <ModelCard
                    title="Denoising GRU Autoencoder"
                    color="var(--blue)"
                    stats={[
                        { label: 'Input', value: '32 × 21' },
                        { label: 'Hidden', value: '64 units' },
                        { label: 'Parameters', value: '~18 K' },
                        { label: 'INT8 Size', value: '< 60 KB' },
                        { label: 'Latency', value: '< 5 ms' },
                        { label: 'Threshold', value: '0.01100' },
                    ]}
                    items={[
                        { title: 'Architecture', desc: 'Single-layer GRU encoder + mirror GRU decoder. Trained to reconstruct clean telemetry from noise-corrupted input (σ=0.05).' },
                        { title: 'Input vector', desc: '21-dimensional: 12 IMU channels (bottom + top Novatel, accel + gyro) and 9 odometry channels (bottom odom, top odom, local odom).' },
                        { title: 'Anomaly score', desc: 'Mean squared reconstruction error across all 21 features at the final timestep. Exceeding threshold=0.011 triggers GRU INTERRUPT.' },
                        { title: 'Edge deployment', desc: 'Post-Training Quantization to INT8. Exported via torch.onnx.export (Opset 17) → OpenVINO / NXX eIQ compiler for NPU execution.' },
                    ]}
                />
                <ModelCard
                    title="AttentiveLSTM"
                    color="var(--purple)"
                    stats={[
                        { label: 'Input', value: '50 × 6' },
                        { label: 'Output', value: 'SoH, RUL' },
                        { label: 'Parameters', value: '~56 K' },
                        { label: 'Dataset', value: 'XJTU' },
                        { label: 'Scaler', value: 'StandardScaler' },
                        { label: 'Activation', value: 'Attention' },
                    ]}
                    items={[
                        { title: 'Architecture', desc: '2-layer LSTM with a temporal attention mechanism. Sequence length of 50 macro-steps gives historical context for degradation trend estimation.' },
                        { title: 'Feature set', desc: 'discharge_median_voltage, charge_median_voltage, discharge_capacity_Ah, charge_time_norm, energy_efficiency (Wh ratio), cycle_index_norm.' },
                        { title: 'SoH output', desc: 'State of Health [0–1] mapped to percentage. Represents remaining battery capacity relative to the rated nominal capacity.' },
                        { title: 'RUL output', desc: 'Remaining Useful Life [0–1] normalized. Estimates the fraction of battery lifetime remaining before capacity falls below 80% of initial.' },
                    ]}
                />
            </div>

            <LiveValues data={data} />
        </div>
    );
}
