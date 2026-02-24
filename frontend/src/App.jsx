import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import StatCard from './components/StatCard';
import GaugeCard from './components/GaugeCard';
import RadarChart from './components/RadarChart';
import HistoryChart from './components/HistoryChart';
import StatusPanel from './components/StatusPanel';
import AlertBanner from './components/AlertBanner';
import AnomalyLog from './components/AnomalyLog';
import MLModels from './components/MLModels';
import Docs from './components/Docs';
import RaceRatio from './components/RaceRatio';
import './index.css';

const API = 'http://localhost:8005';

// Safe default — models start unknown until status check resolves
const INIT_DATA = {
  tick: 0, voltage: 3.7, current: 0, temperature: 30,
  soh: 0, rul: 0, anomaly_score: 0, anomaly_shift_pct: 0,
  gru_alert: false, sensor_fault: false, critical_anomaly: false,
  alert_level: 'nominal', alert_msg: 'Waiting for simulation',
  latency_ms: 0, gru_threshold: 0.011,
  mock_gru: false, mock_lstm: false,   // assume real until proven otherwise
  feat_err: new Array(21).fill(0), history: []
};

// SVG icons
const IconVoltage = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
  </svg>
);
const IconTemp = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
  </svg>
);
const IconBattery = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="1" y="6" width="18" height="12" rx="2" />
    <line x1="23" y1="13" x2="23" y2="11" />
    <line x1="5" y1="10" x2="5" y2="14" /><line x1="9" y1="10" x2="9" y2="14" />
  </svg>
);
const IconClock = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);

export default function App() {
  const [data, setData] = useState(INIT_DATA);
  const [running, setRunning] = useState(false);
  const [apiReady, setApiReady] = useState(false);
  const [page, setPage] = useState('dashboard');
  // Anomaly log — persists across resets, keeps last 30 s worth of events
  const [eventLog, setEventLog] = useState([]);
  const tickRef = useRef(null);

  // ── Check API and get real model status ─────────────────────────────────────
  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/status`);
      const s = await r.json();
      setApiReady(s.ready);
      // Persist the real mock flags so reset never shows "MOCK"
      setData(prev => ({
        ...prev,
        mock_gru: s.mock_gru,
        mock_lstm: s.mock_lstm,
        gru_threshold: s.gru_threshold,
      }));
    } catch {
      setApiReady(false);
    }
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  // ── Tick ────────────────────────────────────────────────────────────────────
  const tick = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/tick`, { method: 'POST' });
      const json = await res.json();
      setData(json);

      // Add to event log (keep last 30 seconds = ~75 ticks at 400ms)
      const now = Date.now();
      setEventLog(prev => {
        const cutoff = now - 30000;
        const pruned = prev.filter(e => e.ts > cutoff);
        // Only log non-nominal events (or every 5th nominal for reference)
        if (json.alert_level !== 'nominal' || json.tick % 5 === 0) {
          return [...pruned, {
            ts: now,
            tick: json.tick,
            level: json.alert_level,
            msg: json.alert_msg,
            score: json.anomaly_score,
            shift: json.anomaly_shift_pct,
            soh: json.soh,
            rul: json.rul,
            voltage: json.voltage,
            temp: json.temperature,
            latency: json.latency_ms,
          }];
        }
        return pruned;
      });
    } catch { /* backend offline */ }
  }, []);

  useEffect(() => {
    if (running) {
      tickRef.current = setInterval(tick, 400);
    } else {
      clearInterval(tickRef.current);
    }
    return () => clearInterval(tickRef.current);
  }, [running, tick]);

  // ── Reset: only reset sim history, NOT the model status labels ─────────────
  const handleReset = async () => {
    await fetch(`${API}/api/reset`, { method: 'POST' });
    // Re-fetch status to get accurate mock_gru / mock_lstm
    const r = await fetch(`${API}/api/status`);
    const s = await r.json();
    setData({
      ...INIT_DATA,
      mock_gru: s.mock_gru,
      mock_lstm: s.mock_lstm,
      gru_threshold: s.gru_threshold,
    });
    // Do NOT clear eventLog — logs persist across resets
  };

  const history = data.history || [];
  const vDelta = (data.voltage - 3.7);
  const tDelta = (data.temperature - 30);
  const latPass = data.latency_ms < 50;

  // ── Page routing ────────────────────────────────────────────────────────────
  const renderPage = () => {
    switch (page) {
      case 'anomaly':
        return <AnomalyLog events={eventLog} />;
      case 'models':
        return <MLModels data={data} />;
      case 'raceratio':
        return <RaceRatio />;
      case 'docs':
        return <Docs />;
      default:
        return (
          <>
            <AlertBanner level={data.alert_level} msg={data.alert_msg} />

            <div>
              <div className="section-label">Live Telemetry</div>
              <div className="stat-grid">
                <StatCard icon={<IconVoltage />} label="Pack Voltage" accent="var(--blue)"
                  rawValue={data.voltage} decimals={3} suffix=" V"
                  delta={`${vDelta >= 0 ? '+' : ''}${vDelta.toFixed(3)} V from nominal`}
                  deltaDir={data.voltage >= 3.5 ? 'up' : data.voltage >= 2.8 ? 'warn' : 'down'}
                />
                <StatCard icon={<IconTemp />} label="Pack Temperature" accent="var(--purple)"
                  rawValue={data.temperature} decimals={1} suffix=" °C"
                  delta={`Δ${tDelta >= 0 ? '+' : ''}${tDelta.toFixed(1)} °C from idle`}
                  deltaDir={data.temperature > 55 ? 'down' : data.temperature > 40 ? 'warn' : 'up'}
                />
                <StatCard icon={<IconBattery />} label="State of Health" accent="var(--green)"
                  rawValue={data.soh} decimals={1} suffix=" %"
                  delta={`AttentiveLSTM · Cycle ${data.tick}`}
                  deltaDir={data.soh > 80 ? 'up' : data.soh > 50 ? 'warn' : 'down'}
                />
                <StatCard icon={<IconClock />} label="NPU Latency" accent="var(--orange)"
                  rawValue={data.latency_ms} decimals={1} suffix=" ms"
                  delta={`50 ms budget · ${latPass ? 'PASS' : 'BREACH'}`}
                  deltaDir={data.latency_ms < 20 ? 'up' : data.latency_ms < 40 ? 'warn' : 'down'}
                />
              </div>
            </div>

            <div>
              <div className="section-label">Model Outputs</div>
              <div className="gauge-grid">
                <GaugeCard title="State of Health" value={data.soh} color="var(--blue)" mode="normal" subtitle="AttentiveLSTM" />
                <GaugeCard title="Remaining Useful Life" value={data.rul} color="var(--purple)" mode="normal" subtitle="AttentiveLSTM" />
                <GaugeCard title="GRU Unsupervised Shift" value={data.anomaly_shift_pct} color="var(--green)" mode="anomaly" subtitle="Autoencoder MSE" />
                <RadarChart featErr={data.feat_err} isAlert={data.gru_alert} />
              </div>
            </div>

            <div className="bottom-grid">
              <HistoryChart history={history} />
              <StatusPanel data={data} />
            </div>
          </>
        );
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: 'var(--bg)' }}>
      <Sidebar page={page} setPage={setPage} />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
        <Topbar
          running={running}
          setRunning={setRunning}
          onReset={handleReset}
          apiReady={apiReady}
          tick={data.tick}
          page={page}
        />

        <main className="main-content" style={{
          flex: 1, overflowY: 'auto',
          padding: '16px 20px',
          display: 'flex', flexDirection: 'column', gap: 12,
        }}>
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
