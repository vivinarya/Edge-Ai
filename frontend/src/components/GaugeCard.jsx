import React, { useEffect, useRef, useState } from 'react';

const CIRCUMFERENCE = 2 * Math.PI * 52; // r=52

export default function GaugeCard({ title, value, color, mode, subtitle }) {
    const [animVal, setAnimVal] = useState(0);
    const animRef = useRef(null);
    const prevRef = useRef(0);

    // Smooth animation
    useEffect(() => {
        const target = Math.min(Math.max(value || 0, 0), 100);
        const from = prevRef.current;
        prevRef.current = target;

        const duration = 500;
        const start = performance.now();
        const step = (now) => {
            const t = Math.min((now - start) / duration, 1);
            const ease = 1 - Math.pow(1 - t, 3); // ease-out cubic
            setAnimVal(from + (target - from) * ease);
            if (t < 1) animRef.current = requestAnimationFrame(step);
        };
        cancelAnimationFrame(animRef.current);
        animRef.current = requestAnimationFrame(step);
        return () => cancelAnimationFrame(animRef.current);
    }, [value]);

    const pct = animVal / 100;
    const offset = CIRCUMFERENCE * (1 - pct);

    // Color for anomaly mode: green → orange → red
    let barColor = color;
    if (mode === 'anomaly') {
        barColor = animVal < 50 ? '#22c55e' : animVal < 80 ? '#f97316' : '#ef4444';
    }

    // Track color
    const trackColor = '#1d2334';

    return (
        <div className="card" style={{ padding: '16px 14px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{
                fontSize: 9, fontWeight: 700, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '1.2px', marginBottom: 8,
            }}>
                {title}
            </div>

            <div style={{ position: 'relative', width: 130, height: 130 }}>
                <svg width="130" height="130" viewBox="0 0 130 130">
                    {/* Background track */}
                    <circle
                        cx="65" cy="65" r="52"
                        fill="none" stroke={trackColor} strokeWidth="9"
                    />
                    {/* Animated value arc */}
                    <circle
                        cx="65" cy="65" r="52"
                        fill="none"
                        stroke={barColor}
                        strokeWidth="9"
                        strokeLinecap="round"
                        strokeDasharray={CIRCUMFERENCE}
                        strokeDashoffset={offset}
                        transform="rotate(-90 65 65)"
                        style={{
                            transition: 'stroke 0.4s ease',
                            filter: `drop-shadow(0 0 8px ${barColor}60)`,
                        }}
                    />
                    {/* Center value */}
                    <text
                        x="65" y="61"
                        textAnchor="middle"
                        fill="var(--text)"
                        fontSize="22"
                        fontWeight="700"
                        fontFamily="JetBrains Mono, monospace"
                        opacity="0.95"
                    >
                        {animVal.toFixed(1)}
                    </text>
                    <text
                        x="65" y="78"
                        textAnchor="middle"
                        fill="var(--text3)"
                        fontSize="10"
                        fontFamily="Inter, sans-serif"
                        fontWeight="500"
                    >
                        %
                    </text>
                </svg>

                {/* Subtle inner glow ring */}
                <div style={{
                    position: 'absolute', inset: 12,
                    borderRadius: '50%',
                    background: `radial-gradient(circle, ${barColor}08 0%, transparent 70%)`,
                    transition: 'background 0.4s ease',
                    pointerEvents: 'none',
                }} />
            </div>

            {/* Mini progress bar */}
            <div style={{ width: '82%', height: 3, background: 'var(--border)', borderRadius: 99, overflow: 'hidden', marginTop: 8 }}>
                <div style={{
                    height: '100%', borderRadius: 99,
                    background: barColor,
                    width: `${animVal}%`,
                    transition: 'width 0.5s cubic-bezier(0.4,0,0.2,1), background 0.4s ease',
                }} />
            </div>

            {subtitle && (
                <div style={{ fontSize: 10, color: 'var(--text3)', marginTop: 6, textAlign: 'center', letterSpacing: '0.3px' }}>
                    {subtitle}
                </div>
            )}
        </div>
    );
}
