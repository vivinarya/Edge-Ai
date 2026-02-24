import React, { useEffect, useRef, useState } from 'react';

const LABELS = ['Accel-X', 'Accel-Z', 'Gyro-Z', 'Vel-X Bot', 'Vel-X Top', 'Vel-Y'];
const IDX = [0, 2, 11, 12, 15, 19];
const N = LABELS.length;
const CX = 75, CY = 75, R = 52;

function polar(angleDeg, r) {
    const rad = (angleDeg - 90) * (Math.PI / 180);
    return { x: CX + r * Math.cos(rad), y: CY + r * Math.sin(rad) };
}

export default function RadarChart({ featErr, isAlert }) {
    const raw = featErr || new Array(21).fill(0);
    const vals = IDX.map(i => Math.min((raw[i] || 0) * 90, 1.0));

    const [anim, setAnim] = useState(vals.map(() => 0));
    const animRef = useRef(null);

    useEffect(() => {
        const from = anim.slice();
        const to = vals;
        const duration = 500;
        const start = performance.now();

        const step = (now) => {
            const t = Math.min((now - start) / duration, 1);
            const e = 1 - Math.pow(1 - t, 3);
            setAnim(to.map((v, i) => from[i] + (v - from[i]) * e));
            if (t < 1) animRef.current = requestAnimationFrame(step);
        };
        cancelAnimationFrame(animRef.current);
        animRef.current = requestAnimationFrame(step);
        return () => cancelAnimationFrame(animRef.current);
    }, [featErr?.join?.(',')]);

    const color = isAlert ? 'var(--red)' : 'var(--blue)';
    const fillC = isAlert ? 'rgba(239,68,68,0.15)' : 'rgba(59,126,246,0.12)';

    const pts = anim.map((v, i) => {
        const { x, y } = polar((360 / N) * i, v * R);
        return `${x},${y}`;
    }).join(' ');

    return (
        <div className="card" style={{ padding: '16px 14px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{
                fontSize: 9, fontWeight: 700, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '1.2px', marginBottom: 8,
            }}>
                Deviation Radar
            </div>

            <svg width={150} height={150} viewBox="0 0 150 150">
                {/* Grid rings */}
                {[0.25, 0.5, 0.75, 1.0].map(f => {
                    const gpts = Array.from({ length: N }, (_, i) => {
                        const { x, y } = polar((360 / N) * i, R * f);
                        return `${x},${y}`;
                    }).join(' ');
                    return <polygon key={f} points={gpts} fill="none" stroke="var(--border)" strokeWidth={f === 1 ? 1.2 : 0.8} />;
                })}
                {/* Spokes */}
                {Array.from({ length: N }, (_, i) => {
                    const { x, y } = polar((360 / N) * i, R);
                    return <line key={i} x1={CX} y1={CY} x2={x} y2={y} stroke="var(--border)" strokeWidth={0.8} />;
                })}
                {/* Data polygon */}
                <polygon
                    points={pts}
                    fill={fillC}
                    stroke={color}
                    strokeWidth={1.5}
                    style={{ transition: 'fill 0.4s ease, stroke 0.4s ease', filter: `drop-shadow(0 0 4px ${color}50)` }}
                />
                {/* Dot at each vertex */}
                {anim.map((v, i) => {
                    const { x, y } = polar((360 / N) * i, v * R);
                    return <circle key={i} cx={x} cy={y} r={2.5} fill={color} opacity={0.9}
                        style={{ transition: 'cx 0.5s ease, cy 0.5s ease' }}
                    />;
                })}
                {/* Labels */}
                {LABELS.map((l, i) => {
                    const { x, y } = polar((360 / N) * i, R + 14);
                    return (
                        <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
                            fill="var(--text3)" fontSize={8} fontFamily="Inter, sans-serif" fontWeight="500">
                            {l}
                        </text>
                    );
                })}
            </svg>
        </div>
    );
}
