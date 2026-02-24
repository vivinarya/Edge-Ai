import React from 'react';

const LEVEL_STYLE = {
    nominal: { color: 'var(--green)', bg: 'rgba(34,197,94,0.08)', border: 'rgba(34,197,94,0.2)', label: 'NOMINAL' },
    warn: { color: 'var(--orange)', bg: 'rgba(249,115,22,0.08)', border: 'rgba(249,115,22,0.22)', label: 'WARNING' },
    critical: { color: 'var(--red)', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.22)', label: 'CRITICAL' },
};

function fmt(ts) {
    const d = new Date(ts);
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

export default function AnomalyLog({ events }) {
    const reversed = [...events].reverse();

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>Anomaly Log</div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>
                        Rolling 30-second event capture · {events.length} event{events.length !== 1 ? 's' : ''} stored
                    </div>
                </div>
                <div style={{
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '5px 12px', fontSize: 11, color: 'var(--text2)',
                    fontFamily: 'var(--font-mono)',
                }}>
                    {events.filter(e => e.level === 'critical').length} critical &nbsp;·&nbsp;
                    {events.filter(e => e.level === 'warn').length} warn &nbsp;·&nbsp;
                    {events.filter(e => e.level === 'nominal').length} nominal
                </div>
            </div>

            {/* Table */}
            {events.length === 0 ? (
                <div className="card" style={{ padding: '40px 20px', textAlign: 'center' }}>
                    <div style={{ color: 'var(--text3)', fontSize: 13 }}>
                        No events yet. Start the simulation and events will appear here.
                    </div>
                    <div style={{ color: 'var(--text3)', fontSize: 11, marginTop: 8 }}>
                        Non-nominal events and every 5th nominal tick are logged for the last 30 seconds.
                    </div>
                </div>
            ) : (
                <div className="card" style={{ overflow: 'hidden' }}>
                    {/* Table head */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '80px 60px 80px 90px 68px 68px 68px 68px 68px 1fr',
                        padding: '10px 16px', borderBottom: '1px solid var(--border)',
                        fontSize: 9, fontWeight: 700, color: 'var(--text3)',
                        textTransform: 'uppercase', letterSpacing: '1px',
                        gap: 8,
                    }}>
                        <span>Time</span>
                        <span>Tick</span>
                        <span>Level</span>
                        <span>Score</span>
                        <span>Shift %</span>
                        <span>SoH %</span>
                        <span>RUL %</span>
                        <span>Volt V</span>
                        <span>Temp °C</span>
                        <span>Message</span>
                    </div>

                    {/* Rows */}
                    <div style={{ maxHeight: 'calc(100vh - 280px)', overflowY: 'auto' }}>
                        {reversed.map((e, i) => {
                            const st = LEVEL_STYLE[e.level] || LEVEL_STYLE.nominal;
                            return (
                                <div key={i} style={{
                                    display: 'grid',
                                    gridTemplateColumns: '80px 60px 80px 90px 68px 68px 68px 68px 68px 1fr',
                                    padding: '9px 16px',
                                    borderBottom: i < reversed.length - 1 ? '1px solid var(--border)' : 'none',
                                    fontSize: 11, gap: 8, alignItems: 'center',
                                    background: e.level !== 'nominal' ? st.bg : 'transparent',
                                    transition: 'background 0.2s ease',
                                    animation: 'fadeIn 0.2s ease both',
                                }}>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text3)' }}>{fmt(e.ts)}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text2)' }}>{e.tick}</span>
                                    <span>
                                        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                            <div style={{ width: 5, height: 5, borderRadius: '50%', background: st.color, flexShrink: 0 }} />
                                            <span style={{
                                                fontSize: 9, fontWeight: 700, color: st.color,
                                                fontFamily: 'var(--font-mono)', letterSpacing: '0.8px',
                                            }}>{st.label}</span>
                                        </span>                  </span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: e.level !== 'nominal' ? st.color : 'var(--text2)' }}>
                                        {e.score.toFixed(5)}
                                    </span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text2)' }}>{e.shift.toFixed(1)}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--blue)' }}>{e.soh.toFixed(1)}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--purple)' }}>{e.rul.toFixed(1)}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text2)' }}>{e.voltage.toFixed(3)}</span>
                                    <span style={{ fontFamily: 'var(--font-mono)', color: e.temp > 50 ? 'var(--orange)' : 'var(--text2)' }}>
                                        {e.temp.toFixed(1)}
                                    </span>
                                    <span style={{ color: 'var(--text3)', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {e.msg}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
