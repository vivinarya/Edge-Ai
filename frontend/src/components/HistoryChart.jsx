import React from 'react';
import {
    ResponsiveContainer, ComposedChart, Area, Line,
    XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: 'var(--surface2)', border: '1px solid var(--border2)',
            borderRadius: 8, padding: '10px 14px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            animation: 'fadeIn 0.15s ease',
        }}>
            <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 8, fontFamily: 'var(--font-mono)' }}>
                T-{String(label).padStart(5, '0')}
            </div>
            {payload.map(p => (
                <div key={p.name} style={{
                    display: 'flex', justifyContent: 'space-between', gap: 20,
                    fontSize: 12, marginBottom: 3,
                    color: p.color,
                }}>
                    <span style={{ color: 'var(--text2)', fontSize: 11 }}>{p.name}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}%
                    </span>
                </div>
            ))}
        </div>
    );
};

export default function HistoryChart({ history }) {
    const data = (history || []).map(h => ({
        tick: h.tick,
        'SoH': h.soh,
        'RUL': h.rul,
        'GRU Shift': h.score ? Math.min((h.score / 0.011) * 100, 100) : 0,
    }));

    return (
        <div className="card" style={{ padding: '18px 18px 10px' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
                <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>Battery Health Forecasting</div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>
                        AttentiveLSTM (SoH, RUL) · GRU Autoencoder Shift — last 60 ticks
                    </div>
                </div>
                <div style={{ display: 'flex', gap: 14 }}>
                    {[['SoH', 'var(--blue)'], ['RUL', 'var(--purple)'], ['GRU Shift', 'var(--orange)']].map(([n, c]) => (
                        <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: 'var(--text2)', fontWeight: 500 }}>
                            <div style={{ width: 6, height: 6, borderRadius: n === 'RUL' ? 1 : '50%', background: c, opacity: 0.9 }} />
                            {n}
                        </div>
                    ))}
                </div>
            </div>

            <ResponsiveContainer width="100%" height={220}>
                <ComposedChart data={data} margin={{ top: 4, right: 0, bottom: 0, left: -20 }}>
                    <defs>
                        <linearGradient id="gSoH" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b7ef6" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#3b7ef6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gRUL" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1} />
                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="2 4" stroke="var(--border)" vertical={false} />
                    <XAxis
                        dataKey="tick" hide={data.length < 2}
                        tick={{ fill: 'var(--text3)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                        axisLine={false} tickLine={false}
                    />
                    <YAxis
                        domain={[0, 105]}
                        tick={{ fill: 'var(--text3)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                        axisLine={false} tickLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border2)', strokeWidth: 1, strokeDasharray: '3 3' }} />
                    <ReferenceLine y={80} stroke="rgba(34,197,94,0.15)" strokeDasharray="4 4" />

                    <Area dataKey="SoH" type="monotoneX" stroke="#3b7ef6" strokeWidth={2} fill="url(#gSoH)" dot={false} isAnimationActive={false} />
                    <Area dataKey="RUL" type="monotoneX" stroke="#8b5cf6" strokeWidth={1.5} fill="url(#gRUL)" dot={false} strokeDasharray="5 3" isAnimationActive={false} />
                    <Line dataKey="GRU Shift" type="monotoneX" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="3 2" isAnimationActive={false} />
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
}
