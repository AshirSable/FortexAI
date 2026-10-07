import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../context/auth-context';
import * as gateway from '../../api/gateway';
import { useToast } from '../ToastContext';
import { fmtDuration } from './usageFormat';
import './PlaceholderView.css';

function Meter({ label, used, limit, resetsIn }) {
  const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  const full = limit > 0 && used >= limit;
  const fill = full ? '#ef4444' : pct >= 80 ? '#f59e0b' : undefined;
  return (
    <div style={{ marginBottom: 18 }}>
      <div className="usage-header">
        <div className="placeholder-card-title" style={{ marginBottom: 0 }}>{label}</div>
        <div className="usage-count">{used.toLocaleString()} / {limit > 0 ? limit.toLocaleString() : 'unlimited'}</div>
      </div>
      <div className="usage-bar-track">
        <div className="usage-bar-fill" style={{ width: pct + '%', ...(fill ? { background: fill } : {}) }} />
      </div>
      <div className="usage-footnote">
        {full ? `Limit fully used. Requests are rejected (429) until it resets in ${fmtDuration(resetsIn)}.` : `Resets in ${fmtDuration(resetsIn)}`}
      </div>
    </div>
  );
}

// Limits are enforced per API key; this page adds the keys up to show the account total.
export default function UsageView() {
  const { token } = useAuth();
  const showToast = useToast();
  const [keys, setKeys] = useState(null);

  const load = useCallback(async () => {
    try {
      setKeys(await gateway.listKeys(token));
    } catch (err) {
      showToast(err.message, 'warn');
    }
  }, [token, showToast]);

  useEffect(() => {
    const first = setTimeout(load, 0);
    return () => clearTimeout(first);
  }, [load]);

  if (keys === null) return <div className="placeholder-intro">Loading usage…</div>;

  const sum = (field) => keys.reduce((total, k) => total + (k.usage?.[field] ?? 0), 0);
  const first = keys[0]?.usage;
  const perMinute = first?.per_minute_limit;

  return (
    <div>
      <div className="placeholder-intro">Each API key has its own daily, monthly and per-minute limit. Totals below add up all your keys.</div>
      {keys.length > 0 && (
        <div className="placeholder-card" style={{ padding: 26, marginBottom: 18 }}>
          <Meter label="Requests today (all keys)" used={sum('daily_used')} limit={sum('daily_limit')} resetsIn={first.daily_resets_in_seconds} />
          <Meter label="Requests this month (all keys)" used={sum('monthly_used')} limit={sum('monthly_limit')} resetsIn={first.monthly_resets_in_seconds} />
        </div>
      )}
      <div className="usage-stats-grid" style={{ marginBottom: 18 }}>
        <div className="placeholder-card"><div className="stat-label">Rate limit (per key)</div><div className="stat-value">{perMinute > 0 ? `${perMinute} req/min` : 'unlimited'}</div></div>
        <div className="placeholder-card"><div className="stat-label">Active API keys</div><div className="stat-value">{keys.filter((k) => k.status === 'active').length} / {keys.length}</div></div>
      </div>
      {keys.map((k) => (
        <div className="placeholder-card" style={{ padding: 22, marginBottom: 14 }} key={k.id}>
          <div className="placeholder-card-title">{k.name} <span style={{ color: '#5b6472', fontWeight: 400 }}>{k.key_prefix}••••</span></div>
          <Meter label="Today" used={k.usage.daily_used} limit={k.usage.daily_limit} resetsIn={k.usage.daily_resets_in_seconds} />
          <Meter label="This month" used={k.usage.monthly_used} limit={k.usage.monthly_limit} resetsIn={k.usage.monthly_resets_in_seconds} />
        </div>
      ))}
      {keys.length === 0 && <div className="placeholder-intro">No API keys yet. Generate one on the API Keys page.</div>}
    </div>
  );
}
