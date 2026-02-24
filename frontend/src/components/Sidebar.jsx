import React from 'react';

const NAV = [
    { id: 'dashboard', label: 'Dashboard', icon: <IconGrid /> },
    { id: 'raceratio', label: 'Features', icon: <IconGear /> },
    { id: 'anomaly', label: 'Anomaly Log', icon: <IconAlert /> },
    { id: 'models', label: 'ML Models', icon: <IconCpu /> },
    { id: 'docs', label: 'Docs', icon: <IconDoc /> },
];

function IconGrid() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /></svg>; }
function IconAlert() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>; }
function IconCpu() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></svg>; }
function IconDoc() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>; }
function IconGear() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" /></svg>; }

export default function Sidebar({ page, setPage }) {
    return (
        <aside style={{
            width: 220, flexShrink: 0,
            background: 'var(--surface)',
            borderRight: '1px solid var(--border)',
            display: 'flex', flexDirection: 'column',
        }}>
            {/* Logo — no emojis */}
            <div style={{
                padding: '18px 16px 14px',
                borderBottom: '1px solid var(--border)',
                display: 'flex', alignItems: 'center', gap: 10,
            }}>
                <div style={{
                    width: 28, height: 28, borderRadius: 7, flexShrink: 0,
                    background: 'linear-gradient(135deg, #3b7ef6 0%, #8b5cf6 100%)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    <div style={{ width: 10, height: 10, border: '2px solid rgba(255,255,255,0.9)', borderRadius: 2 }} />
                </div>
                <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.2px' }}>
                        Edge AI Dashboard
                    </div>
                    <div style={{ fontSize: 9, color: 'var(--blue)', fontWeight: 600, letterSpacing: '1.4px', textTransform: 'uppercase', marginTop: 1 }}>
                        Raceists
                    </div>
                </div>
            </div>

            {/* Nav */}
            <nav style={{ flex: 1, padding: '10px 8px', overflowY: 'auto' }}>
                <div style={{ fontSize: 9, color: 'var(--text3)', fontWeight: 700, letterSpacing: '1.5px', textTransform: 'uppercase', padding: '6px 8px 8px' }}>
                    Navigation
                </div>
                {NAV.map(n => {
                    const active = page === n.id;
                    return (
                        <button key={n.id} onClick={() => setPage(n.id)} style={{
                            display: 'flex', alignItems: 'center', gap: 9,
                            width: '100%', padding: '8px 10px', marginBottom: 2,
                            borderRadius: 8, cursor: 'pointer',
                            transition: 'all var(--transition)',
                            background: active ? 'rgba(59,126,246,0.1)' : 'transparent',
                            color: active ? 'var(--blue)' : 'var(--text2)',
                            border: active ? '1px solid rgba(59,126,246,0.18)' : '1px solid transparent',
                            fontWeight: active ? 600 : 400, fontSize: 13,
                            fontFamily: 'inherit', textAlign: 'left',
                        }}
                            onMouseEnter={e => { if (!active) { e.currentTarget.style.background = 'var(--surface2)'; e.currentTarget.style.color = 'var(--text)'; } }}
                            onMouseLeave={e => { if (!active) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text2)'; } }}
                        >
                            {n.icon}
                            <span>{n.label}</span>
                            {active && <div style={{ marginLeft: 'auto', width: 4, height: 4, borderRadius: '50%', background: 'var(--blue)' }} />}
                        </button>
                    );
                })}
            </nav>

            {/* Bottom */}
            <div style={{ padding: '8px 8px 14px', borderTop: '1px solid var(--border)' }}>
                <button style={{
                    display: 'flex', alignItems: 'center', gap: 9, width: '100%',
                    padding: '8px 10px', borderRadius: 8, cursor: 'pointer',
                    background: 'transparent', border: '1px solid transparent',
                    color: 'var(--text3)', fontSize: 13, fontFamily: 'inherit',
                    transition: 'all var(--transition)',
                }}
                    onMouseEnter={e => { e.currentTarget.style.color = 'var(--text2)'; e.currentTarget.style.background = 'var(--surface2)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = 'var(--text3)'; e.currentTarget.style.background = 'transparent'; }}
                >
                    <IconGear /><span>Settings</span>
                </button>
            </div>
        </aside>
    );
}
