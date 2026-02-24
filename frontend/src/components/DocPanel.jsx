import React, { useState } from 'react';

function Section({ title, children }) {
    const [open, setOpen] = useState(false);
    return (
        <div style={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 12, overflow: 'hidden', marginBottom: 12 }}>
            <div onClick={() => setOpen(o => !o)} style={{
                padding: '14px 18px', display: 'flex', justifyContent: 'space-between',
                cursor: 'pointer', fontSize: 13, fontWeight: 600, color: '#94a3b8'
            }}>
                <span>{title}</span>
                <span>{open ? '▲' : '▼'}</span>
            </div>
            {open && (
                <div style={{ padding: '0 18px 16px', fontSize: 13, color: '#64748b', lineHeight: 1.8 }}>
                    {children}
                </div>
            )}
        </div>
    );
}

export default function DocPanel() {
    return (
        <div>
            <div style={{
                fontSize: 13, fontWeight: 600, color: '#64748b', textTransform: 'uppercase',
                letterSpacing: '1.2px', marginBottom: 14
            }}>Project Documentation</div>

            <Section title="📡 Task 1 & 3 — Anomaly Detection & Edge Deployment">
                <p><strong style={{ color: '#e2e8f0' }}>Architecture — Unsupervised Denoising GRU Autoencoder</strong></p>
                <p>• <strong style={{ color: '#94a3b8' }}>Why Unsupervised?</strong> Mechanical failures are unique — no labeled dataset covers every suspension/battery failure mode. The model learns the nominal track distribution and flags what it can't reconstruct.</p>
                <p>• <strong style={{ color: '#94a3b8' }}>Why GRU?</strong> Gated Recurrent Units are 3× lighter than LSTMs while preserving sequential memory across the 32-step window, detecting vibration resonance frequency shifts.</p>
                <p>• <strong style={{ color: '#94a3b8' }}>Deviation Radar:</strong> Per-channel reconstruction error² is visualized on a polar chart, showing exactly which sensor is deviating.</p>
                <br />
                <p><strong style={{ color: '#e2e8f0' }}>Edge Deployment (INT8, &lt;60KB, &lt;50ms)</strong></p>
                <p>• QAT shrinks FP32 weights to INT8 integers, compressing to ~49 KB. torch.onnx.export (Opset 17) freezes the graph for the NPU compiler (OpenVINO / NXP eIQ).</p>
            </Section>

            <Section title="🔋 Task 4 — Feature Engineering & Racing Context">
                <p><strong style={{ color: '#e2e8f0' }}>6 Longitudinal Features → AttentiveLSTM</strong></p>
                <p>• discharge_median_voltage / charge_median_voltage — median-aggregated per charge cycle, filtering instantaneous driver-induced noise.</p>
                <p>• energy_efficiency (discharge_Wh / charge_Wh) — implicit internal-resistance proxy.</p>
                <p>• cycle_index_norm — normalized lifespan position for long-horizon SoH trajectory.</p>
                <br />
                <p><strong style={{ color: '#e2e8f0' }}>Racing Context — High-C-Rate & Regen Braking</strong></p>
                <p>• Rolling Sequence Memory (50-step LSTM + Temporal Attention) distinguishes transient racing dynamics from permanent capacity fade, maintaining stable SoH even during 8C discharge pulses.</p>
            </Section>

            <Section title="🛡️ Task 2 — Sensor Fault vs Physical Anomaly">
                <p><strong style={{ color: '#e2e8f0' }}>Preprocessing Pipeline</strong></p>
                <p>• NaN Imputation: Zero-order hold on disconnected sensors (voltage NaN → last valid value).</p>
                <p>• Garbage Filter: Physical boundary checks (Temp &lt; -20°C or &gt; 150°C → impute to 30°C).</p>
                <p>• Wheel Speed: gru_normalize clips signals exceeding FEATURE_MAX, handling 300 km/h CAN spikes.</p>
                <br />
                <p><strong style={{ color: '#e2e8f0' }}>Contextual Decision Gate</strong></p>
                <p>• Critical anomaly only fires when temp_imputed &gt; 60°C AND sensor_fault=False — preventing a broken thermistor from triggering a thermal runaway false alarm on the pit wall.</p>
            </Section>
        </div>
    );
}
