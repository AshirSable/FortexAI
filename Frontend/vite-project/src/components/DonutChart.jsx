import './DonutChart.css';

export default function DonutChart({
  title = 'Blocked vs Passed',
  pctBlocked,
  dashArray,
  blockedCount,
  passedCount,
  blockedLabel = 'blocked',
  passedLabel = 'passed',
  trackColor = '#1c2430',
  blockedColor = '#ef4444',
  passedColor = '#22c55e',
  size = 150,
}) {
  return (
    <div className="donut-chart-card">
      {title && <div className="donut-chart-title">{title}</div>}
      <svg viewBox="0 0 140 140" width={size} height={size}>
        <circle cx="70" cy="70" r="54" fill="none" stroke={trackColor} strokeWidth="16" />
        <circle
          cx="70"
          cy="70"
          r="54"
          fill="none"
          stroke={blockedColor}
          strokeWidth="16"
          strokeDasharray={dashArray}
          strokeLinecap="round"
          transform="rotate(-90 70 70)"
        >
          <title>{blockedCount} blocked/flagged ({pctBlocked}%)</title>
        </circle>
        <text x="70" y="65" textAnchor="middle" fontSize="22" fontWeight="800" fill="#e8ecf1" fontFamily="JetBrains Mono, monospace">
          {pctBlocked}%
        </text>
        <text x="70" y="83" textAnchor="middle" fontSize="10.5" fill="#8b96a5">
          blocked+flagged
        </text>
      </svg>
      <div className="donut-chart-legend">
        <div className="donut-chart-legend-item" title={`${blockedCount} blocked or flagged`}>
          <div className="donut-chart-legend-value" style={{ color: blockedColor }}>{blockedCount}</div>
          <div className="donut-chart-legend-label">{blockedLabel}</div>
        </div>
        <div className="donut-chart-legend-item" title={`${passedCount} passed clean`}>
          <div className="donut-chart-legend-value" style={{ color: passedColor }}>{passedCount}</div>
          <div className="donut-chart-legend-label">{passedLabel}</div>
        </div>
      </div>
    </div>
  );
}
