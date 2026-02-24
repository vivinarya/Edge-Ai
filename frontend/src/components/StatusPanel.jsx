import React from 'react';

function Row({ label, value, color, mono }) {
    return (
        <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 0', borderBottom: '1px solid var(--border)',
            transition: 'opacity 0.2s ease',
        }}>
            <span style={{ fontSize: 11, color: 'var(--text3)' }}>{label}</span>
            <span style={{
                fontSize: 11, fontWeight: 600,
                color: color || 'var(--text)',
                fontFamily: mono ? 'var(--font-mono)' : 'inherit',
                transition: 'color 0.3s ease',
            }}>{value}</span>
        </div>
    );
}

export default function StatusPanel({ data }) {
    const lat = data.latency_ms || 0;
    const latPct = Math.min(lat / 50, 1);
    const latC = lat < 20 ? 'var(--green)' : lat < 40 ? 'var(--orange)' : 'var(--red)';

    return (
        <div className="card" style={{ padding: '16px 18px' }}>
            <div style={{
                fontSize: 9, fontWeight: 700, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: 14
            }}>
                System Status
            </div>

            <Row
                label="GRU Model"
                value={data.mock_gru ? 'MOCK' : 'FP32 Active'}
                color={data.mock_gru ? 'var(--red)' : 'var(--green)'}
                mono
            />
            <Row
                label="LSTM Model"
                value={data.mock_lstm ? 'MOCK' : 'TorchScript'}
                color={data.mock_lstm ? 'var(--red)' : 'var(--green)'}
                mono
            />
            <Row label="GRU Threshold" value={(data.gru_threshold || 0).toFixed(5)} mono />
            <Row
                label="Anomaly Score"
                value={(data.anomaly_score || 0).toFixed(5)}
                color={data.gru_alert ? 'var(--red)' : 'var(--green)'}
                mono
            />
            <Row
                label="Current Draw"
                value={`${(data.current || 0).toFixed(1)} A`}
                color={Math.abs(data.current || 0) > 50 ? 'var(--red)' : Math.abs(data.current || 0) > 30 ? 'var(--orange)' : 'var(--green)'}
                mono
            />
            <Row
                label="Sensor Fault"
                value={data.sensor_fault ? 'DETECTED' : 'NOMINAL'}
                color={data.sensor_fault ? 'var(--red)' : 'var(--green)'}
                mono
            />
            <div style={{ paddingTop: 2 }}>
                <Row
                    label="Critical / Thermal"
                    value={data.critical_anomaly ? 'ALERT' : 'SAFE'}
                    color={data.critical_anomaly ? 'var(--red)' : 'var(--green)'}
                    mono
                />
            </div>

            {/* Latency bar */}
            <div style={{ marginTop: 14 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text3)', marginBottom: 6 }}>
                    <span>NPU Inference Budget</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: latC, transition: 'color 0.3s' }}>
                        {lat.toFixed(1)} / 50.0 ms
                    </span>
                </div>
                <div style={{ background: 'var(--border)', borderRadius: 99, height: 4, overflow: 'hidden' }}>
                    <div style={{
                        height: '100%', borderRadius: 99,
                        background: `linear-gradient(90deg, var(--green) 0%, ${latC} 100%)`,
                        width: `${latPct * 100}%`,
                        transition: 'width 0.5s cubic-bezier(0.4,0,0.2,1), background 0.4s ease',
                        boxShadow: `0 0 6px ${latC}60`,
                    }} />
                </div>
            </div>

            {/* Pipeline trace */}
            <div style={{
                marginTop: 16, padding: '10px 12px',
                background: 'var(--bg2)', borderRadius: 8,
                border: '1px solid var(--border)',
            }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text3)', marginBottom: 8, letterSpacing: '1.2px', textTransform: 'uppercase' }}>
                    Inference Pipeline
                </div>
                {['DAQ Input', 'NaN Filter', 'Garbage Cap', 'GRU [32×21]', 'LSTM [50×6]', 'Gate → Alert'].map((step, i) => (
                    <div key={step} style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        fontSize: 10, color: 'var(--text3)', marginBottom: 4,
                        animation: `slideIn 0.3s ease ${i * 0.05}s both`,
                    }}>
                        <div style={{
                            width: 4, height: 4, borderRadius: '50%',
                            background: i < 3 ? 'var(--text3)' : i === 4 ? 'var(--purple)' : 'var(--blue)',
                            flexShrink: 0,
                        }} />
                        {step}
                        {i < 5 && (
                            <div style={{ flex: 1, height: 1, background: 'var(--border)', marginLeft: 2 }} />
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
