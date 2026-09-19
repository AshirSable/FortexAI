import { useState } from 'react';
import { useAuth } from '../../context/auth-context';
import './PlaceholderView.css';

const DOCS_CARDS = [
  { title: 'Quickstart', desc: 'Generate a key and screen your first prompt in under five minutes.' },
  { title: 'API Reference', desc: 'Full endpoint, request, and response schema documentation.' },
  { title: 'SDKs', desc: 'Official client libraries for Python, Node.js, and Go.' },
  { title: 'Webhooks', desc: 'Subscribe to block events and high-risk-user alerts in real time.' },
];

const INVOICES = [
  { date: 'Jul 1, 2026', amount: '$199.00', status: 'Paid' },
  { date: 'Jun 1, 2026', amount: '$199.00', status: 'Paid' },
];

function Toggle({ on, onToggle }) {
  return (
    <div className="setting-toggle" style={{ background: on ? '#4f7fff' : '#232b38' }} onClick={onToggle}>
      <div className="setting-toggle-knob" style={{ left: on ? '21px' : '3px' }} />
    </div>
  );
}

export default function PlaceholderView({ kind }) {
  const { user } = useAuth();
  const [settings, setSettings] = useState({ emailAlerts: true, weeklyDigest: true, slackAlerts: false });
  const toggle = (field) => setSettings((s) => ({ ...s, [field]: !s[field] }));

  if (kind === 'docs') {
    return (
      <div>
        <div className="placeholder-intro">Everything you need to integrate and operate FortexAI in your own stack.</div>
        <div className="docs-grid">
          {DOCS_CARDS.map((c) => (
            <div className="placeholder-card" key={c.title}>
              <div className="placeholder-card-title">{c.title}</div>
              <div className="placeholder-card-desc">{c.desc}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (kind === 'usage') {
    return (
      <div>
        <div className="placeholder-intro">Track how much of your plan's monthly allowance has been used.</div>
        <div className="placeholder-card" style={{ padding: 26, marginBottom: 18 }}>
          <div className="usage-header">
            <div className="placeholder-card-title" style={{ marginBottom: 0 }}>Prompts screened this cycle</div>
            <div className="usage-count">512,300 / 1,000,000</div>
          </div>
          <div className="usage-bar-track">
            <div className="usage-bar-fill" style={{ width: '51.2%' }} />
          </div>
          <div className="usage-footnote">Resets in 6 days &middot; Pro plan</div>
        </div>
        <div className="usage-stats-grid">
          <div className="placeholder-card"><div className="stat-label">Rate limit</div><div className="stat-value">500 req/min</div></div>
          <div className="placeholder-card"><div className="stat-label">Active API keys</div><div className="stat-value">2</div></div>
          <div className="placeholder-card"><div className="stat-label">Vector DB size</div><div className="stat-value">2.4 GB</div></div>
        </div>
      </div>
    );
  }

  if (kind === 'billing') {
    return (
      <div>
        <div className="billing-grid">
          <div className="placeholder-card" style={{ padding: 26 }}>
            <div className="billing-eyebrow">Current plan</div>
            <div className="billing-plan-name">Pro &mdash; $199/mo</div>
            <div className="billing-plan-desc">1M prompts/mo, unlimited API keys, priority support.</div>
            <button className="btn-outline">Change Plan</button>
          </div>
          <div className="placeholder-card" style={{ padding: 26 }}>
            <div className="billing-eyebrow" style={{ color: '#8b96a5' }}>Payment method</div>
            <div className="billing-payment"><span className="card-badge">VISA</span> &middot;&middot;&middot;&middot; 4242</div>
          </div>
        </div>
        <div className="placeholder-card" style={{ overflow: 'hidden' }}>
          <div className="invoices-header">Invoices</div>
          <div className="invoices-row invoices-head">
            <div>Date</div><div>Amount</div><div>Status</div><div />
          </div>
          {INVOICES.map((inv) => (
            <div className="invoices-row" key={inv.date}>
              <div>{inv.date}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace" }}>{inv.amount}</div>
              <div><span style={{ color: '#22c55e', fontWeight: 600 }}>{inv.status}</span></div>
              <div><a href="#">Download</a></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (kind === 'settings') {
    return (
      <div style={{ maxWidth: 640 }}>
        <div className="placeholder-card" style={{ padding: 26, marginBottom: 18 }}>
          <div className="placeholder-card-title" style={{ marginBottom: 18 }}>Account</div>
          <div className="settings-field">
            <label>Name</label>
            <input value={user?.name || ''} readOnly />
          </div>
          <div className="settings-field">
            <label>Email</label>
            <input value={user?.email || ''} readOnly />
          </div>
        </div>
        <div className="placeholder-card" style={{ padding: 26, marginBottom: 18 }}>
          <div className="placeholder-card-title" style={{ marginBottom: 18 }}>Notifications</div>
          <div className="settings-toggle-row">
            <div><div className="settings-toggle-title">Email alerts on blocked attacks</div><div className="settings-toggle-desc">Real-time notification per high-confidence block</div></div>
            <Toggle on={settings.emailAlerts} onToggle={() => toggle('emailAlerts')} />
          </div>
          <div className="settings-toggle-row">
            <div><div className="settings-toggle-title">Weekly digest</div><div className="settings-toggle-desc">Summary of trends and high-risk users</div></div>
            <Toggle on={settings.weeklyDigest} onToggle={() => toggle('weeklyDigest')} />
          </div>
          <div className="settings-toggle-row">
            <div><div className="settings-toggle-title">Slack alerts</div><div className="settings-toggle-desc">Post blocks to a #security-alerts channel</div></div>
            <Toggle on={settings.slackAlerts} onToggle={() => toggle('slackAlerts')} />
          </div>
        </div>
        <div className="danger-zone">
          <div className="danger-zone-title">Danger zone</div>
          <div className="danger-zone-desc">Permanently delete this workspace and all logged data.</div>
          <button className="btn-danger-outline">Delete Workspace</button>
        </div>
      </div>
    );
  }

  return null;
}
