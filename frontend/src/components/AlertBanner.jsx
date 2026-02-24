import React from 'react';

const CONFIG = {
    nominal: {
        bg: 'rgba(34,197,94,0.06)',
        border: 'rgba(34,197,94,0.18)',
        color: 'var(--green)',
        dot: '#22c55e',
        label: 'NOMINAL',
        pulse: false,
    },
    warn: {
        bg: 'rgba(249,115,22,0.07)',
        border: 'rgba(249,115,22,0.22)',
        color: 'var(--orange)',
        dot: '#f97316',
        label: 'WARNING',
        pulse: true,
    },
    critical: {
        bg: 'rgba(239,68,68,0.08)',
        border: 'rgba(239,68,68,0.25)',
        color: 'var(--red)',
        dot: '#ef4444',
        label: 'CRITICAL',
        pulse: true,
    },
};

// Minimal geometric status indicator — no emojis, no tick/triangle
function StatusIndicator({ level, color }) {
    if (level === 'nominal') {
        // Solid filled circle
        return (
            <div style={{
                width: 8, height: 8, borderRadius: '50%',
                background: color, flexShrink: 0,
                boxShadow: `0 0 6px ${color}80`,
            }} />
        );
    }
    if (level === 'critical') {
        // Solid square — severity indicator
        return (
            <div style={{
                width: 8, height: 8, borderRadius: 2,
                background: color, flexShrink: 0,
                boxShadow: `0 0 8px ${color}90`,
                animation: 'pulseRed 0.9s infinite',
            }} />
        );
    }
    // warn — ring (hollow circle)
    return (
        <div style={{
            width: 8, height: 8, borderRadius: '50%',
            border: `2px solid ${color}`, flexShrink: 0,
            animation: 'pulseRed 1.3s infinite',
        }} />
    );
}

export default function AlertBanner({ level, msg }) {
    const cfg = CONFIG[level] || CONFIG.nominal;

    return (
        <div className="alert-banner" style={{
            background: cfg.bg,
            border: `1px solid ${cfg.border}`,
            borderRadius: 10,
            padding: '11px 16px',
            display: 'flex', alignItems: 'center', gap: 10,
            transition: 'all 0.35s ease',
        }}>
            <StatusIndicator level={level} color={cfg.dot} />

            {/* Level badge */}
            <span style={{
                fontSize: 9, fontWeight: 800, color: cfg.color,
                textTransform: 'uppercase', letterSpacing: '1.5px',
                background: `${cfg.dot}18`,
                border: `1px solid ${cfg.dot}30`,
                padding: '2px 7px', borderRadius: 4, flexShrink: 0,
                fontFamily: 'var(--font-mono)',
            }}>
                {cfg.label}
            </span>

            <span style={{
                fontSize: 12, color: 'var(--text2)', fontWeight: 500,
                transition: 'color 0.3s',
            }}>
                {msg}
            </span>
        </div>
    );
}
