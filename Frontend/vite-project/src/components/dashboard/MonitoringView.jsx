import { useCallback, useEffect, useMemo, useState } from 'react';
import DonutChart from '../DonutChart';
import { useToast } from '../ToastContext';
import { useAuth } from '../../context/auth-context';
import * as gateway from '../../api/gateway';
import './MonitoringView.css';

const PERIODS = ['1h', '24h', '7d', '30d'];
const POLL_MS = 3000;

// detection layer label -> phase value stored by the backend
const LAYERS = [
  { label: 'Semantic Search', phase: 'semantic_search' },
  { label: 'Autoencoder', phase: 'autoencoder' },
  { label: 'Ensemble BERT', phase: 'ensemble_bert' },
  { label: 'LLM Judge', phase: 'llm_judge' },
];

function layerLabel(phase) {
  const found = LAYERS.find((l) => l.phase === phase);
  if (found) return found.label;
  return phase === 'error' ? 'Pipeline error' : '—';
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function verdictStyle(v) {
  if (v === 'blocked') return { bg: 'rgba(239,68,68,0.14)', color: '#ef4444' };
  if (v === 'flagged') return { bg: 'rgba(245,158,11,0.14)', color: '#f59e0b' };
  if (v === 'bypassed') return { bg: 'rgba(139,150,165,0.16)', color: '#8b96a5' };
  return { bg: 'rgba(34,197,94,0.14)', color: '#22c55e' };
}
function confColor(c) {
  return c > 70 ? '#ef4444' : c > 35 ? '#f59e0b' : '#22c55e';
}
function valueColor(tone) {
  return tone === 'danger' ? '#ef4444' : tone === 'warn' ? '#f59e0b' : tone === 'good' ? '#22c55e' : '#e8ecf1';
}

function fmtNum(n) {
  return n.toLocaleString('en-US');
}
function fmtTime(iso) {
  return new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' });
}
function fmtBucket(iso, period) {
  const d = new Date(iso);
  if (period === '1h' || period === '24h') return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Turn a list of numbers into an svg path "M x,y L x,y ..."
function buildPath(values, w, h, max, pad = 4) {
  const step = (w - pad * 2) / Math.max(values.length - 1, 1);
  return values.map((v, i) => {
    const x = pad + i * step;
    const y = h - pad - (v / max) * (h - pad * 2);
    return (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
}

// Build everything the trend chart needs from the backend buckets
function buildTrend(buckets, period) {
  const w = 640, h = 200, pad = 4;
  const attempts = buckets.map((b) => b.attempts);
  const blocked = buckets.map((b) => b.blocked);
  const max = Math.max(1, ...attempts) * 1.15;
  const attemptsPath = buildPath(attempts, w, h, max);
  const blockedPath = buildPath(blocked, w, h, max);
  const attemptsArea = attemptsPath + ` L${w - pad},${h - pad} L${pad},${h - pad} Z`;
  const step = (w - pad * 2) / Math.max(buckets.length - 1, 1);
  const points = buckets.map((b, i) => ({
    x: +(pad + i * step).toFixed(1),
    y: +(h - pad - (b.attempts / max) * (h - pad * 2)).toFixed(1),
    yBlocked: +(h - pad - (b.blocked / max) * (h - pad * 2)).toFixed(1),
    attemptVal: b.attempts,
    blockedVal: b.blocked,
    label: fmtBucket(b.start, period),
  }));
  // about 5 labels under the chart
  const every = Math.max(1, Math.floor(buckets.length / 5));
  const labels = points.filter((_, i) => i % every === 0).map((p) => p.label);
  return {
    attemptsPath, blockedPath, attemptsArea, points, labels,
    maxLabel: fmtNum(Math.round(max)), midLabel: fmtNum(Math.round(max / 2)),
  };
}

function buildDonut(summary) {
  const screened = summary.blocked + summary.flagged + summary.passed;
  const pct = screened === 0 ? 0 : ((summary.blocked + summary.flagged) / screened) * 100;
  const circumference = 2 * Math.PI * 54;
  return {
    pctBlocked: pct.toFixed(1),
    dashArray: ((pct / 100) * circumference).toFixed(1) + ' ' + circumference.toFixed(1),
    blockedCount: fmtNum(summary.blocked + summary.flagged),
    passedCount: fmtNum(summary.passed),
  };
}

const EMPTY_SUMMARY = { total: 0, blocked: 0, flagged: 0, passed: 0, bypassed: 0, malicious_pct: 0, avg_latency_ms: 0, high_risk_users: 0 };

export default function MonitoringView() {
  const showToast = useToast();
  const { token } = useAuth();
  const [period, setPeriod] = useState('24h');
  const [keys, setKeys] = useState([]);
  const [keyId, setKeyId] = useState('all');
  const [summary, setSummary] = useState(EMPTY_SUMMARY);
  const [buckets, setBuckets] = useState([]);
  const [events, setEvents] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [search, setSearch] = useState('');
  const [verdictFilter, setVerdictFilter] = useState('all');
  const [layerFilter, setLayerFilter] = useState('all');
  const [sortMode, setSortMode] = useState('time'); // time | conf-desc | conf-asc
  const [expandedId, setExpandedId] = useState(null);
  const [feedback, setFeedback] = useState({});
  const [hoverIndex, setHoverIndex] = useState(null);

  // fetch everything the page needs from the backend
  const loadData = useCallback(async () => {
    const selected = keyId === 'all' ? undefined : keyId;
    try {
      const [keyList, sum, trendData, eventList] = await Promise.all([
        gateway.listKeys(token),
        gateway.fetchSummary(token, { period, keyId: selected }),
        gateway.fetchTrend(token, { period, keyId: selected }),
        gateway.fetchEvents(token, {
          period,
          keyId: selected,
          verdict: verdictFilter === 'all' ? '' : verdictFilter,
          phase: layerFilter === 'all' ? '' : layerFilter,
          q: search,
        }),
      ]);
      setKeys(keyList);
      setSummary(sum);
      setBuckets(trendData.buckets);
      setEvents(eventList);
      setLoaded(true);
    } catch {
      // keep showing the old numbers; the next poll will try again
    }
  }, [token, period, keyId, verdictFilter, layerFilter, search]);

  // load now, then every 3 seconds
  useEffect(() => {
    const first = setTimeout(loadData, 0);
    const timer = setInterval(loadData, POLL_MS);
    return () => { clearTimeout(first); clearInterval(timer); };
  }, [loadData]);

  // the keys this page is looking at (one key, or all of them)
  const selectedKeys = keyId === 'all' ? keys : keys.filter((k) => String(k.id) === String(keyId));
  const activeCount = selectedKeys.filter((k) => k.status === 'active').length;
  const detectionOn = selectedKeys.length > 0 && activeCount === selectedKeys.length;
  const detectionOff = activeCount === 0;

  const toggleDetection = async () => {
    const next = detectionOn ? 'stopped' : 'active';
    try {
      await Promise.all(selectedKeys.map((k) => gateway.setKeyStatus(token, k.id, next)));
      showToast(next === 'active' ? 'Detection is ON.' : 'Detection is OFF. Prompts are passing through unscreened.', 'info');
      loadData();
    } catch (err) {
      showToast(err.message, 'warn');
    }
  };

  const kpiList = [
    { key: 'total', label: 'Total Prompts Processed', value: fmtNum(summary.total), sub: period, tone: 'default' },
    { key: 'malicious', label: '% Malicious / Flagged', value: summary.malicious_pct + '%', sub: 'of screened traffic', tone: 'warn' },
    { key: 'blocked', label: 'Blocked vs Passed', value: fmtNum(summary.blocked), sub: fmtNum(summary.passed) + ' passed', tone: 'danger' },
    { key: 'latency', label: 'Avg Detection Latency', value: Math.round(summary.avg_latency_ms) + ' ms', sub: 'cascade end-to-end', tone: 'default' },
    {
      key: 'status', label: 'Detection status',
      value: selectedKeys.length === 0 ? 'No key' : detectionOn ? 'ON' : detectionOff ? 'OFF' : 'Partial',
      sub: activeCount + ' of ' + selectedKeys.length + ' keys active',
      tone: detectionOn ? 'good' : detectionOff ? 'danger' : 'warn',
    },
    { key: 'bypassed', label: 'Bypassed', value: fmtNum(summary.bypassed), sub: 'passed while detection was OFF', tone: summary.bypassed > 0 ? 'warn' : 'default' },
    { key: 'highrisk', label: 'High-Risk Users', value: summary.high_risk_users, sub: '3+ blocked prompts', tone: 'danger' },
  ];

  const trend = useMemo(() => buildTrend(buckets, period), [buckets, period]);
  const donut = buildDonut(summary);

  // the server sends newest first; the user can also sort by confidence
  const rows = useMemo(() => {
    const list = events.map((e) => ({ ...e, confidencePct: Math.round(e.confidence * 100) }));
    if (sortMode === 'conf-desc') list.sort((a, b) => b.confidencePct - a.confidencePct);
    if (sortMode === 'conf-asc') list.sort((a, b) => a.confidencePct - b.confidencePct);
    return list;
  }, [events, sortMode]);

  const nextSortMode = () => setSortMode((m) => (m === 'time' ? 'conf-desc' : m === 'conf-desc' ? 'conf-asc' : 'time'));
  const sortArrow = sortMode === 'conf-desc' ? ' ↓' : sortMode === 'conf-asc' ? ' ↑' : '';

  const noKeys = loaded && keys.length === 0;
  const noTraffic = loaded && keys.length > 0 && summary.total === 0 && !search && verdictFilter === 'all' && layerFilter === 'all';

  const hoverPoint = hoverIndex !== null ? trend.points[hoverIndex] : null;

  const onChartMove = (e) => {
    if (trend.points.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fracX = (e.clientX - rect.left) / rect.width;
    const idx = Math.max(0, Math.min(trend.points.length - 1, Math.round(fracX * (trend.points.length - 1))));
    setHoverIndex(idx);
  };

  const handleFeedback = (id, val) => {
    setFeedback((f) => ({ ...f, [id]: val }));
    showToast('Feedback recorded — thank you.', 'success');
  };

  const download = (format) => {
    const header = ['Timestamp', 'Sender', 'Detection Layer', 'Confidence', 'Verdict', 'Prompt Preview'];
    const escape = (v) => '"' + String(v).replace(/"/g, '""') + '"';
    const lines = [header.map(escape).join(',')].concat(
      rows.map((r) => [fmtTime(r.time), r.end_user || '', layerLabel(r.phase), r.confidencePct, r.verdict, r.prompt_preview].map(escape).join(','))
    );
    const csv = lines.join('\n');
    const mime = format === 'xls' ? 'application/vnd.ms-excel' : 'text/csv';
    const blob = new Blob([csv], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prompt-analysis.' + format;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="monitoring">
      <div className="monitoring-toolbar">
        <div className="monitoring-toolbar-left">
          <div className="monitoring-subtitle">Live security insights, refreshed every 3 seconds</div>
          <div className="monitoring-vdivider" />
          <div className="monitoring-project">
            <span>Key</span>
            <select value={keyId} onChange={(e) => setKeyId(e.target.value)}>
              <option value="all">All keys</option>
              {keys.map((k) => <option key={k.id} value={k.id}>{k.name}</option>)}
            </select>
          </div>
        </div>
        <div className="period-switch">
          {PERIODS.map((p) => (
            <button key={p} className={period === p ? 'active' : ''} onClick={() => setPeriod(p)}>{p}</button>
          ))}
        </div>
      </div>

      <div className="detection-switch" style={{ borderColor: detectionOn ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)' }}>
        <div className="toggle detection-toggle" style={{ background: detectionOn ? '#22c55e' : '#232b38', opacity: selectedKeys.length ? 1 : 0.4 }}
          onClick={selectedKeys.length ? toggleDetection : undefined}>
          <div className="toggle-knob detection-knob" style={{ left: detectionOn ? '27px' : '3px' }} />
        </div>
        <div>
          <div className="detection-switch-title">Detection {detectionOn ? 'ON' : 'OFF'}</div>
          <div className="detection-switch-sub">
            {detectionOn ? 'Prompts are screened before they reach your model.' : 'Prompts pass through without being screened.'}
          </div>
        </div>
        <span className="detection-badge" style={{
          background: detectionOn ? 'rgba(34,197,94,0.14)' : 'rgba(239,68,68,0.14)',
          color: detectionOn ? '#22c55e' : '#ef4444',
        }}>
          {selectedKeys.length === 0 ? 'No key' : detectionOn ? 'Protected' : detectionOff ? 'Unprotected' : 'Partially protected'}
        </span>
      </div>

      <div className="kpi-grid">
        {kpiList.map((k) => (
          <div key={k.key} className="kpi-card">
            <div className="kpi-card-top">
              <div className="kpi-card-label">{k.label}</div>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#5b6472" strokeWidth="2"><path d="M9 6l6 6-6 6" /></svg>
            </div>
            <div className="kpi-card-value" style={{ color: valueColor(k.tone) }}>{k.value}</div>
            <div className="kpi-card-sub">{k.sub}</div>
          </div>
        ))}
      </div>

      {noKeys && <div className="monitoring-empty">You have no API keys yet. Create one on the API Generation page to start screening prompts.</div>}
      {noTraffic && (
        <div className="monitoring-empty">
          No prompts have been screened in the last {period}. Send a request to <code>POST /v1/screen</code> with your key and it will show up here.
        </div>
      )}

      {!noKeys && !noTraffic && (<>
      <div className="monitoring-charts">
        <div className="trend-card">
          <div className="trend-card-header">
            <div className="trend-card-title">Malicious Prompt Trend</div>
            <div className="trend-legend">
              <div className="trend-legend-item"><span className="dot" style={{ background: '#4f7fff' }} />Attempts</div>
              <div className="trend-legend-item"><span className="dot" style={{ background: '#ef4444' }} />Blocked</div>
            </div>
          </div>
          <div className="trend-chart-wrap">
            <svg viewBox="0 0 640 200" className="trend-svg" preserveAspectRatio="none"
              onMouseMove={onChartMove} onMouseLeave={() => setHoverIndex(null)}>
              <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4f7fff" stopOpacity="0.32" />
                  <stop offset="100%" stopColor="#4f7fff" stopOpacity="0" />
                </linearGradient>
              </defs>
              <line x1="4" y1="4" x2="636" y2="4" stroke="#1a212c" strokeWidth="1" />
              <line x1="4" y1="100" x2="636" y2="100" stroke="#1a212c" strokeWidth="1" />
              <line x1="4" y1="196" x2="636" y2="196" stroke="#1a212c" strokeWidth="1" />
              <text x="638" y="8" fontSize="10" fill="#5b6472" textAnchor="end">{trend.maxLabel}</text>
              <text x="638" y="104" fontSize="10" fill="#5b6472" textAnchor="end">{trend.midLabel}</text>
              <text x="638" y="194" fontSize="10" fill="#5b6472" textAnchor="end">0</text>
              <path d={trend.attemptsArea} fill="url(#areaGrad)" stroke="none" />
              <path d={trend.attemptsPath} fill="none" stroke="#4f7fff" strokeWidth="2.5" />
              <path d={trend.blockedPath} fill="none" stroke="#ef4444" strokeWidth="2" strokeDasharray="4 3" />
              {hoverPoint && (
                <>
                  <line x1={hoverPoint.x} y1="4" x2={hoverPoint.x} y2="196" stroke="#3a4658" strokeWidth="1" strokeDasharray="3 3" />
                  <circle cx={hoverPoint.x} cy={hoverPoint.y} r="4" fill="#4f7fff" stroke="#0a0e14" strokeWidth="2" />
                  <circle cx={hoverPoint.x} cy={hoverPoint.yBlocked} r="4" fill="#ef4444" stroke="#0a0e14" strokeWidth="2" />
                </>
              )}
            </svg>
            {hoverPoint && (
              <div className="trend-tooltip" style={{ left: Math.min(78, Math.max(0, (hoverPoint.x / 640) * 100)) + '%' }}>
                <div className="trend-tooltip-label">{hoverPoint.label}</div>
                <div style={{ color: '#6d94ff' }}>Attempts: {hoverPoint.attemptVal}</div>
                <div style={{ color: '#ff8080' }}>Blocked: {hoverPoint.blockedVal}</div>
              </div>
            )}
          </div>
          <div className="trend-labels">
            {trend.labels.map((l, i) => <span key={i}>{l}</span>)}
          </div>
        </div>

        <DonutChart
          pctBlocked={donut.pctBlocked}
          dashArray={donut.dashArray}
          blockedCount={donut.blockedCount}
          passedCount={donut.passedCount}
        />
      </div>

      <div className="prompt-analysis">
        <div className="prompt-analysis-header">
          <div className="trend-card-title">Prompt Analysis</div>
          <div className="prompt-analysis-actions">
            <input
              type="text"
              placeholder="Search end user or prompt text..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="search-input"
            />
            <button className="export-btn" onClick={() => download('csv')}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v12M7 10l5 5 5-5M4 21h16" /></svg>CSV
            </button>
            <button className="export-btn" onClick={() => download('xls')}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3v12M7 10l5 5 5-5M4 21h16" /></svg>Excel
            </button>
          </div>
        </div>

        <div className="prompt-filters">
          <select value={verdictFilter} onChange={(e) => setVerdictFilter(e.target.value)}>
            <option value="all">All verdicts</option>
            <option value="blocked">Blocked</option>
            <option value="flagged">Flagged</option>
            <option value="passed">Passed</option>
            <option value="bypassed">Bypassed</option>
          </select>
          <select value={layerFilter} onChange={(e) => setLayerFilter(e.target.value)}>
            <option value="all">All detection layers</option>
            {LAYERS.map((l) => <option key={l.phase} value={l.phase}>{l.label}</option>)}
          </select>
        </div>

        <div className="table-head-row">
          <div>Timestamp</div>
          <div>Sender</div>
          <div>Detection Layer</div>
          <div className="sortable" onClick={nextSortMode}>
            Confidence{sortArrow}
          </div>
          <div>Verdict</div>
          <div />
        </div>

        {rows.map((row) => {
          const vs = verdictStyle(row.verdict);
          const isExpanded = expandedId === row.id;
          const fb = feedback[row.id];
          return (
            <div key={row.id}>
              <div className="table-row" onClick={() => setExpandedId(isExpanded ? null : row.id)}>
                <div className="cell-mono">{fmtTime(row.time)}</div>
                <div className="cell-sender">{row.end_user || '—'}</div>
                <div className="cell-muted">{layerLabel(row.phase)}</div>
                <div className="cell-confidence">
                  <div className="confidence-track">
                    <div className="confidence-fill" style={{ width: row.confidencePct + '%', background: confColor(row.confidencePct) }} />
                  </div>
                  <span className="cell-mono">{row.confidencePct}</span>
                </div>
                <div>
                  <span className="verdict-pill" style={{ background: vs.bg, color: vs.color }}>{capitalize(row.verdict)}</span>
                </div>
                <div className="cell-expand">{isExpanded ? '▾' : '▸'}</div>
              </div>
              {isExpanded && (
                <div className="row-expand">
                  <div className="row-expand-text">"{row.prompt_preview}"</div>
                  <div className="row-expand-feedback">
                    <span>Was this correct?</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleFeedback(row.id, 'up'); }}
                      style={{ background: fb === 'up' ? 'rgba(34,197,94,0.2)' : '#12171f', color: fb === 'up' ? '#22c55e' : '#8b96a5' }}
                    >{'\u{1F44D}'}</button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleFeedback(row.id, 'down'); }}
                      style={{ background: fb === 'down' ? 'rgba(239,68,68,0.2)' : '#12171f', color: fb === 'down' ? '#ef4444' : '#8b96a5' }}
                    >{'\u{1F44E}'}</button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {rows.length === 0 && <div className="no-results">No prompts match these filters.</div>}
      </div>
      </>)}
    </div>
  );
}
