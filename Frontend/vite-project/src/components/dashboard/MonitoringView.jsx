import { useMemo, useState } from 'react';
import DonutChart from '../DonutChart';
import { useToast } from '../ToastContext';
import {
  ALL_PROMPTS, KPI_BY_PERIOD, LAYER_OPTIONS,
  buildTrend, buildDonut, buildKpiList, fmtTime, now,
} from '../../data/mockData';
import './MonitoringView.css';

const PERIODS = ['24h', '7d', '30d'];
const PROJECT_OPTIONS = ['Production', 'Staging'];

function verdictStyle(v) {
  if (v === 'Blocked') return { bg: 'rgba(239,68,68,0.14)', color: '#ef4444' };
  if (v === 'Flagged') return { bg: 'rgba(245,158,11,0.14)', color: '#f59e0b' };
  return { bg: 'rgba(34,197,94,0.14)', color: '#22c55e' };
}
function confColor(c) {
  return c > 70 ? '#ef4444' : c > 35 ? '#f59e0b' : '#22c55e';
}
function valueColor(tone) {
  return tone === 'danger' ? '#ef4444' : tone === 'warn' ? '#f59e0b' : tone === 'good' ? '#22c55e' : '#e8ecf1';
}

export default function MonitoringView() {
  const showToast = useToast();
  const [period, setPeriod] = useState('24h');
  const [project, setProject] = useState('Production');
  const [search, setSearch] = useState('');
  const [dateRange, setDateRange] = useState('all');
  const [verdictFilter, setVerdictFilter] = useState('all');
  const [layerFilter, setLayerFilter] = useState('all');
  const [sortDir, setSortDir] = useState('desc');
  const [expandedId, setExpandedId] = useState(null);
  const [feedback, setFeedback] = useState({});
  const [hoverIndex, setHoverIndex] = useState(null);

  const kpiRaw = KPI_BY_PERIOD[period];
  const kpiList = useMemo(() => buildKpiList(kpiRaw, period), [kpiRaw, period]);
  const trend = useMemo(() => buildTrend(period), [period]);
  const donut = useMemo(() => buildDonut(kpiRaw), [kpiRaw]);

  const rows = useMemo(() => {
    const nowMs = now();
    const rangeMs = { '24h': 86400000, '7d': 7 * 86400000, '30d': 30 * 86400000 }[dateRange];
    let filtered = ALL_PROMPTS.filter((p) => {
      if (search && !(p.sender.toLowerCase().includes(search.toLowerCase()) || p.preview.toLowerCase().includes(search.toLowerCase()))) return false;
      if (verdictFilter !== 'all' && p.verdict !== verdictFilter) return false;
      if (layerFilter !== 'all' && p.layer !== layerFilter) return false;
      if (rangeMs && nowMs - p.timestamp > rangeMs) return false;
      return true;
    });
    filtered = filtered.slice().sort((a, b) => (sortDir === 'desc' ? b.confidence - a.confidence : a.confidence - b.confidence));
    return filtered.map((p) => ({ ...p, timeStr: fmtTime(p.timestamp) }));
  }, [search, dateRange, verdictFilter, layerFilter, sortDir]);

  const hoverPoint = hoverIndex !== null ? trend.points[hoverIndex] : null;

  const onChartMove = (e) => {
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
      rows.map((r) => [r.timeStr, r.sender, r.layer, r.confidence, r.verdict, r.preview].map(escape).join(','))
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
          <div className="monitoring-subtitle">Security insights, most important first</div>
          <div className="monitoring-vdivider" />
          <div className="monitoring-project">
            <span>Project</span>
            <select value={project} onChange={(e) => setProject(e.target.value)}>
              {PROJECT_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>
        <div className="period-switch">
          {PERIODS.map((p) => (
            <button key={p} className={period === p ? 'active' : ''} onClick={() => setPeriod(p)}>{p}</button>
          ))}
        </div>
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
              placeholder="Search sender or prompt text..."
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
          <select value={dateRange} onChange={(e) => setDateRange(e.target.value)}>
            <option value="all">All time</option>
            <option value="24h">Last 24h</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </select>
          <select value={verdictFilter} onChange={(e) => setVerdictFilter(e.target.value)}>
            <option value="all">All verdicts</option>
            <option value="Blocked">Blocked</option>
            <option value="Flagged">Flagged</option>
            <option value="Passed">Passed</option>
          </select>
          <select value={layerFilter} onChange={(e) => setLayerFilter(e.target.value)}>
            <option value="all">All detection layers</option>
            {LAYER_OPTIONS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>

        <div className="table-head-row">
          <div>Timestamp</div>
          <div>Sender</div>
          <div>Detection Layer</div>
          <div className="sortable" onClick={() => setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))}>
            Confidence {sortDir === 'desc' ? '↓' : '↑'}
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
                <div className="cell-mono">{row.timeStr}</div>
                <div className="cell-sender">{row.sender}</div>
                <div className="cell-muted">{row.layer}</div>
                <div className="cell-confidence">
                  <div className="confidence-track">
                    <div className="confidence-fill" style={{ width: row.confidence + '%', background: confColor(row.confidence) }} />
                  </div>
                  <span className="cell-mono">{row.confidence}</span>
                </div>
                <div>
                  <span className="verdict-pill" style={{ background: vs.bg, color: vs.color }}>{row.verdict}</span>
                </div>
                <div className="cell-expand">{isExpanded ? '▾' : '▸'}</div>
              </div>
              {isExpanded && (
                <div className="row-expand">
                  <div className="row-expand-text">"{row.preview}"</div>
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
    </div>
  );
}
