import React from 'react';

function IconPlay() { return <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>; }
function IconStop() { return <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" /></svg>; }
function IconReset() { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-4.5" /></svg>; }

const PAGE_TITLES = {
    dashboard: { title: 'Dashboard', sub: 'Real-time Battery Management System' },
    anomaly: { title: 'Anomaly Log', sub: 'Rolling 30-second event capture' },
    models: { title: 'ML Models', sub: 'GRU Autoencoder · AttentiveLSTM' },
    docs: { title: 'Docs', sub: 'Technical project reference' },
};

const Chip = ({ children, color, dim }) => (
    <div style={{
        background: dim, border: `1px solid ${color}28`,
        borderRadius: 6, padding: '5px 10px',
        fontSize: 11, fontWeight: 600, color,
        display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0,
        transition: 'all var(--transition)',
    }}>{children}</div>
);

export default function Topbar({ running, setRunning, onReset, apiReady, tick, page }) {
    const pg = PAGE_TITLES[page] || PAGE_TITLES.dashboard;
    return (
        <header style={{
            background: 'var(--surface)',
            borderBottom: '1px solid var(--border)',
            padding: '0 24px',
            height: 56, flexShrink: 0,
            display: 'flex', alignItems: 'center',
            justifyContent: 'space-between', gap: 12,
        }}>
            {/* Left */}
            <div>
                <h1 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.4px', lineHeight: 1 }}>
                    {pg.title}
                </h1>
                <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 2 }}>
                    {pg.sub}
                </p>
            </div>

            {/* Right */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                {/* API status */}
                <Chip
                    color={apiReady ? '#22c55e' : '#ef4444'}
                    dim={apiReady ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)'}
                >
                    <span className="status-dot" style={{
                        width: 6, height: 6, borderRadius: '50%',
                        background: apiReady ? 'var(--green)' : 'var(--red)',
                        animation: apiReady ? 'pulseGreen 2s infinite' : 'pulseRed 1.2s infinite',
                        display: 'inline-block',
                    }} />
                    {apiReady ? 'API Online' : 'API Offline'}
                </Chip>

                {/* Tick */}
                <div style={{
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '5px 10px',
                    fontSize: 11, color: 'var(--text2)',
                    fontFamily: 'var(--font-mono)',
                }}>
                    T-{String(tick).padStart(5, '0')}
                </div>

                {/* Model badges */}
                <Chip color="var(--blue)" dim="var(--blue-dim)">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="3" /><path d="M6.3 6.3a8 8 0 0 0 0 11.4M17.7 17.7a8 8 0 0 0 0-11.4" /></svg>
                    GRU Autoencoder
                </Chip>
                <Chip color="var(--purple)" dim="var(--purple-dim)">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
                    AttentiveLSTM
                </Chip>

                {/* Reset */}
                <button onClick={onReset} style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 6, padding: '5px 12px',
                    fontSize: 11, color: 'var(--text2)', cursor: 'pointer',
                    fontFamily: 'inherit', fontWeight: 500,
                    transition: 'all var(--transition)',
                }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text)'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text2)'; }}
                >
                    <IconReset /> Reset
                </button>

                {/* Start/Stop */}
                <button onClick={() => setRunning(r => !r)} style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    background: running ? 'rgba(239,68,68,0.1)' : 'rgba(34,197,94,0.1)',
                    border: `1px solid ${running ? 'rgba(239,68,68,0.25)' : 'rgba(34,197,94,0.25)'}`,
                    borderRadius: 6, padding: '5px 14px',
                    fontSize: 11, fontWeight: 700, cursor: 'pointer',
                    color: running ? 'var(--red)' : 'var(--green)',
                    fontFamily: 'inherit', transition: 'all var(--transition)',
                    letterSpacing: '0.3px', textTransform: 'uppercase',
                }}>
                    {running ? <><IconStop /> Stop</> : <><IconPlay /> Live</>}
                </button>
            </div>
        </header>
    );
}
