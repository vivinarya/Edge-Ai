import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';

function Slider({ label, value, min, max, step, unit, onChange }) {
    const pct = ((value - min) / (max - min)) * 100;
    return (
        <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 11, color: 'var(--text2)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                <span>{label}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'white', fontSize: 13 }}>{value} <span style={{ fontSize: 9, color: 'var(--text3)' }}>{unit}</span></span>
            </div>
            <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center' }}>
                <input type="range" className="rr-slider" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))} />
                <div className="rr-slider-progress" style={{ width: `${pct}%` }} />
            </div>
        </div>
    );
}

function Switch({ checked, onChange }) {
    return (
        <div className={`rr-switch ${checked ? 'on' : ''}`} onClick={() => onChange(!checked)}>
            <div className="thumb" />
        </div>
    );
}

export default function RaceRatio() {
    const [tab, setTab] = useState('Engine');
    const [optView, setOptView] = useState('Baseline');
    const [results, setResults] = useState(null);

    const [cfg, setCfg] = useState({
        rain_mode: false,
        mass: 330, wheel_radius: 0.23, tire_grip: 1.45, cd: 1.0, frontal_area: 1.3, crr: 0.017, drivetrain_eff: 0.92,
        rpm_min: 3000, rpm_max: 14000, torque_curve: [40, 50, 60, 65, 70, 75, 70, 65, 60, 50],
        final_drive: 4.62, gears: [2.91, 2.1, 1.62, 1.3, 1.08, 0.92],
        driver_profile: 'Aggressive', traction_usage: 0.95, shift_time: 0.15,
    });

    useEffect(() => {
        fetch('http://localhost:8005/api/raceratio', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg)
        }).then(r => r.json()).then(setResults).catch(console.error);
    }, [cfg]);

    const update = (k, v) => setCfg(prev => ({ ...prev, [k]: v }));

    const chartData = results ? [
        { name: 'Acceleration', base: results.baseline.accel_0_75, opt: results.optimized.accel_0_75 },
        { name: 'Skidpad', base: results.baseline.skidpad, opt: results.optimized.skidpad },
        { name: 'Autocross', base: results.baseline.autocross, opt: results.optimized.autocross },
    ] : [];

    const rrColors = { bg: "rgba(255,255,255,0.03)", red: "#ef4444", green: "#10b981", black: "rgba(0, 0, 0, 0.8)" };

    return (
        <div style={{ display: 'flex', gap: 20, height: '100%', color: 'white' }}>

            {/* LEFT COLUMN - CONFIGURATION */}
            <div style={{ flex: '0 0 380px', display: 'flex', flexDirection: 'column', gap: 12 }}>

                {/* Header Block */}
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '12px 16px', borderRadius: 12, display: 'flex', alignItems: 'center', gap: 14 }}>
                    <div style={{ width: 28, height: 28, background: rrColors.red, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 10px rgba(239, 68, 68, 0.4)' }}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                    </div>
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: '0.5px' }}>FEATURES</div>
                    </div>
                </div>

                <div className="rr-block" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                    <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <svg width="14" height="14" fill="none" stroke={rrColors.red} strokeWidth="2"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
                        Configuration
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: rrColors.bg, padding: '10px 14px', borderRadius: 8, marginBottom: 16 }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 8 }}>
                            <svg width="14" height="14" fill="none" stroke="#10b981" strokeWidth="2"><path d="M20 16.2A8 8 0 1 1 12 3v9" /></svg> Rain Mode
                        </span>
                        <Switch checked={cfg.rain_mode} onChange={v => update('rain_mode', v)} />
                    </div>

                    <div style={{ display: 'flex', background: rrColors.bg, borderRadius: 8, padding: 4, marginBottom: 20 }}>
                        {['Vehicle', 'Engine', 'Gears', 'Driver'].map(t => (
                            <button key={t} className={`rr-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                                {t}
                            </button>
                        ))}
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', paddingRight: 8 }} className="custom-scroll">
                        {tab === 'Vehicle' && <>
                            <Slider label="Mass" value={cfg.mass} min={200} max={500} step={5} unit="kg" onChange={v => update('mass', v)} />
                            <Slider label="Wheel Radius" value={cfg.wheel_radius} min={0.2} max={0.3} step={0.01} unit="m" onChange={v => update('wheel_radius', v)} />
                            <Slider label="Tire Grip (mu)" value={cfg.tire_grip} min={0.8} max={2.0} step={0.05} onChange={v => update('tire_grip', v)} />
                            <Slider label="Drag Coeff (Cd)" value={cfg.cd} min={0.5} max={1.5} step={0.05} onChange={v => update('cd', v)} />
                            <Slider label="Frontal Area" value={cfg.frontal_area} min={0.8} max={2.0} step={0.1} unit="m²" onChange={v => update('frontal_area', v)} />
                            <Slider label="Rolling Resistance" value={cfg.crr} min={0.01} max={0.05} step={0.001} onChange={v => update('crr', v)} />
                            <Slider label="Drivetrain Eff." value={cfg.drivetrain_eff} min={0.8} max={1.0} step={0.01} onChange={v => update('drivetrain_eff', v)} />
                        </>}

                        {tab === 'Engine' && <>
                            <div style={{ fontSize: 12, fontWeight: 600, color: 'white', marginBottom: 12 }}>Torque Curve (Nm)</div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginBottom: 24 }}>
                                {cfg.torque_curve.map((tq, i) => (
                                    <div key={i} style={{ background: rrColors.red, color: 'white', fontSize: 12, fontWeight: 700, textAlign: 'center', padding: '12px 0', borderRadius: 6, boxShadow: '0 2px 5px rgba(239,68,68,0.3)' }}>
                                        {tq}
                                    </div>
                                ))}
                            </div>
                            <Slider label="RPM Min" value={cfg.rpm_min} min={1000} max={6000} step={100} unit="rpm" onChange={v => update('rpm_min', v)} />
                            <Slider label="RPM Max" value={cfg.rpm_max} min={8000} max={16000} step={100} unit="rpm" onChange={v => update('rpm_max', v)} />
                        </>}

                        {tab === 'Gears' && <>
                            <div style={{ display: 'flex', gap: 8, marginBottom: 24, background: rrColors.bg, padding: 4, borderRadius: 8 }}>
                                <button className={`rr-pill ${optView === 'Baseline' ? 'active' : ''}`} onClick={() => setOptView('Baseline')}>Baseline</button>
                                <button className={`rr-pill ${optView === 'Optimized' ? 'active opt' : ''}`} onClick={() => setOptView('Optimized')}>Optimized</button>
                            </div>
                            <Slider label="Final Drive" value={cfg.final_drive} min={2.5} max={6.0} step={0.01} onChange={v => update('final_drive', v)} />
                            {cfg.gears.map((g, i) => (
                                <Slider key={i} label={`Gear ${i + 1}`} value={g} min={0.5} max={4.0} step={0.01} onChange={v => {
                                    const newG = [...cfg.gears]; newG[i] = v; update('gears', newG);
                                }} />
                            ))}
                        </>}

                        {tab === 'Driver' && <>
                            <div style={{ fontSize: 12, fontWeight: 600, color: 'white', marginBottom: 8 }}>Driver Profile</div>
                            <select style={{ width: '100%', background: 'rgba(255,255,255,0.05)', color: 'white', border: '1px solid rgba(255,255,255,0.1)', padding: '10px 12px', borderRadius: 8, marginBottom: 24, outline: 'none', fontSize: 13 }}>
                                <option>Aggressive</option>
                                <option>Smooth</option>
                            </select>
                            <Slider label="Traction Usage" value={cfg.traction_usage} min={0.7} max={1.0} step={0.01} onChange={v => update('traction_usage', v)} />
                            <Slider label="Shift Time" value={cfg.shift_time} min={0.05} max={0.5} step={0.01} unit="s" onChange={v => update('shift_time', v)} />

                            <div style={{ background: rrColors.bg, padding: 14, borderRadius: 8, marginTop: 24 }}>
                                <div style={{ fontSize: 13, fontWeight: 700, color: 'white', marginBottom: 6 }}>Aggressive Profile</div>
                                <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5 }}>Later shifts, higher traction utilization, maximum performance.</div>
                            </div>
                        </>}
                    </div>

                </div>
            </div>

            {/* RIGHT COLUMN - RESULTS */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>

                {/* Top Metric Row */}
                {results && (
                    <div style={{ display: 'flex', gap: 12 }}>
                        <div className="rr-block" style={{ flex: 1 }}>
                            <div style={{ fontSize: 11, color: 'var(--text2)', textTransform: 'uppercase', fontWeight: 700, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                0-75m Time <svg width="14" height="14" stroke={rrColors.red} strokeWidth="2.5" fill="none"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                            </div>
                            <div style={{ fontSize: 32, fontWeight: 800, color: 'white', marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                                {results[optView.toLowerCase()]?.accel_0_75.toFixed(2)}<span style={{ fontSize: 18, color: 'var(--text3)' }}>s</span>
                            </div>
                            <div style={{ fontSize: 11, color: rrColors.green, fontWeight: 700, marginTop: 6 }}>
                                ↗ {(Math.abs(results.baseline.accel_0_75 - results.optimized.accel_0_75)).toFixed(2)}s faster
                            </div>
                        </div>
                        <div className="rr-block" style={{ flex: 1 }}>
                            <div style={{ fontSize: 11, color: 'var(--text2)', textTransform: 'uppercase', fontWeight: 700, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                Skidpad Lap <svg width="14" height="14" stroke={rrColors.red} strokeWidth="2.5" fill="none"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                            </div>
                            <div style={{ fontSize: 32, fontWeight: 800, color: 'white', marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                                {results[optView.toLowerCase()]?.skidpad.toFixed(2)}<span style={{ fontSize: 18, color: 'var(--text3)' }}>s</span>
                            </div>
                            <div style={{ fontSize: 11, color: rrColors.green, fontWeight: 700, marginTop: 6 }}>
                                ↗ {(Math.abs(results.baseline.skidpad - results.optimized.skidpad)).toFixed(2)}s faster
                            </div>
                        </div>
                        <div className="rr-block" style={{ flex: 1 }}>
                            <div style={{ fontSize: 11, color: 'var(--text2)', textTransform: 'uppercase', fontWeight: 700, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                Autocross <svg width="14" height="14" stroke={rrColors.red} strokeWidth="2.5" fill="none"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                            </div>
                            <div style={{ fontSize: 32, fontWeight: 800, color: 'white', marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                                {results[optView.toLowerCase()]?.autocross.toFixed(2)}<span style={{ fontSize: 18, color: 'var(--text3)' }}>s</span>
                            </div>
                            <div style={{ fontSize: 11, color: rrColors.green, fontWeight: 700, marginTop: 6 }}>
                                ↗ {(Math.abs(results.baseline.autocross - results.optimized.autocross)).toFixed(2)}s faster
                            </div>
                        </div>
                    </div>
                )}

                {/* Tab-like Headers */}
                <div style={{ display: 'flex', gap: 24, borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: 12, marginTop: 8, fontSize: 12, fontWeight: 700, color: 'var(--text3)' }}>
                    <div style={{ color: 'white', borderBottom: `2px solid white`, paddingBottom: 11, marginBottom: -13 }}>Tractive Force</div>
                    <div>Acceleration</div>
                    <div>Gear Shifts</div>
                    <div>Event Scoring</div>
                    <div>Optimization</div>
                    <div>Comparison</div>
                </div>

                {/* Chart & Analysis Area */}
                <div style={{ flex: 1, display: 'flex', gap: 12 }}>

                    <div className="rr-block" style={{ flex: 2.5, display: 'flex', flexDirection: 'column' }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'white', marginBottom: 20 }}>Baseline vs Optimized Comparison</div>
                        <div style={{ flex: 1, minHeight: 0 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="name" stroke="none" tick={{ fill: 'var(--text2)', fontSize: 12, fontWeight: 600 }} dy={10} />
                                    <YAxis stroke="none" tick={{ fill: 'var(--text3)', fontSize: 11 }} />
                                    <Bar dataKey="base" fill={rrColors.black} radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="opt" fill={rrColors.red} radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div style={{ background: rrColors.bg, padding: '14px', borderRadius: 8, marginTop: 16, textAlign: 'center', color: 'white', fontWeight: 800, fontSize: 14 }}>
                            Total Score <span style={{ color: rrColors.red, fontSize: 18, paddingLeft: 6 }}>{results ? (298 - results.baseline.accel_0_75 * 2).toFixed(1) : "243.5"}</span><span style={{ color: 'var(--text3)', fontSize: 13 }}> / 298</span>
                        </div>
                    </div>

                    <div className="rr-block" style={{ flex: 1.5 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'white', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div style={{ width: 10, height: 10, borderRadius: '50%', background: rrColors.green, boxShadow: `0 0 8px ${rrColors.green}` }} />
                            Skidpad Analysis
                        </div>

                        {results && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                                <div style={{ background: rrColors.bg, padding: 16, borderRadius: 8 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 8, fontWeight: 600 }}>Max Corner Speed</div>
                                    <div style={{ fontSize: 18, fontWeight: 800, color: 'white', fontFamily: 'var(--font-mono)' }}>{results.skidpad_analysis.max_corner_speed} <span style={{ fontSize: 12, color: 'var(--text3)' }}>km/h</span></div>
                                </div>
                                <div style={{ background: rrColors.bg, padding: 16, borderRadius: 8 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 8, fontWeight: 600 }}>Optimal Gear</div>
                                    <div style={{ fontSize: 20, fontWeight: 800, color: rrColors.red, fontFamily: 'var(--font-mono)' }}>{results.skidpad_analysis.optimal_gear}</div>
                                </div>
                                <div style={{ background: rrColors.bg, padding: 16, borderRadius: 8 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 8, fontWeight: 600 }}>RPM in Corner</div>
                                    <div style={{ fontSize: 18, fontWeight: 800, color: 'white', fontFamily: 'var(--font-mono)' }}>{results.skidpad_analysis.rpm_in_corner}</div>
                                </div>
                                <div style={{ background: rrColors.bg, padding: 16, borderRadius: 8 }}>
                                    <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 8, fontWeight: 600 }}>Lap Time</div>
                                    <div style={{ fontSize: 18, fontWeight: 800, color: 'white', fontFamily: 'var(--font-mono)' }}>{results.skidpad_analysis.lap_time} <span style={{ fontSize: 12, color: 'var(--text3)' }}>s</span></div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
