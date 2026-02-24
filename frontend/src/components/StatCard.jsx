import React, { useRef, useEffect, useState } from 'react';

// Smooth animated number display
function AnimatedValue({ value, decimals = 2, suffix = '', style = {} }) {
    const [display, setDisplay] = useState(value);
    const animRef = useRef(null);
    const prevRef = useRef(value);

    useEffect(() => {
        const from = prevRef.current;
        const to = value;
        prevRef.current = to;
        if (from === to) return;

        const duration = 400;
        const start = performance.now();
        const step = (now) => {
            const t = Math.min((now - start) / duration, 1);
            const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
            setDisplay(from + (to - from) * ease);
            if (t < 1) animRef.current = requestAnimationFrame(step);
        };
        cancelAnimationFrame(animRef.current);
        animRef.current = requestAnimationFrame(step);
        return () => cancelAnimationFrame(animRef.current);
    }, [value]);

    return (
        <span style={style}>
            {display.toFixed(decimals)}{suffix}
        </span>
    );
}

export default function StatCard({ icon, label, value, rawValue, decimals = 2, suffix = '', delta, deltaDir, accent }) {
    const deltaColor = { up: 'var(--green)', warn: 'var(--orange)', down: 'var(--red)' }[deltaDir] || 'var(--text3)';
    const deltaArrow = { up: '↑', warn: '↑', down: '↓' }[deltaDir] || '';

    return (
        <div className="card" style={{ padding: '18px 20px', position: 'relative', overflow: 'hidden' }}>
            {/* Accent top stripe */}
            <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                background: accent, borderRadius: '14px 14px 0 0',
            }} />

            {/* Top row: icon + label */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <div style={{
                    color: accent, opacity: 0.85,
                    display: 'flex', alignItems: 'center',
                }}>
                    {icon}
                </div>
                <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                    {label}
                </span>
            </div>

            {/* Value */}
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                <AnimatedValue
                    value={rawValue !== undefined ? rawValue : parseFloat(value)}
                    decimals={decimals}
                    style={{
                        fontSize: 28, fontWeight: 800, color: 'var(--text)',
                        letterSpacing: '-1px', lineHeight: 1,
                        fontFamily: 'var(--font-mono)',
                    }}
                />
                <span style={{ fontSize: 13, color: 'var(--text3)', fontWeight: 500 }}>{suffix}</span>
            </div>

            {/* Delta */}
            <div style={{
                fontSize: 11, marginTop: 8, color: deltaColor,
                display: 'flex', alignItems: 'center', gap: 3,
                transition: 'color 0.3s ease',
            }}>
                <span style={{ fontSize: 10 }}>{deltaArrow}</span>
                <span>{delta}</span>
            </div>

            {/* Background glow */}
            <div style={{
                position: 'absolute', bottom: -20, right: -20,
                width: 80, height: 80, borderRadius: '50%',
                background: accent,
                opacity: 0.035,
                pointerEvents: 'none',
                filter: 'blur(20px)',
            }} />
        </div>
    );
}
