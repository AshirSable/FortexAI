import { Fragment, useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../context/auth-context';
import * as gateway from '../../api/gateway';
import { useToast } from '../ToastContext';
import { fmtDuration } from './usageFormat';
import './ApiGenView.css';

// "3 min ago", "2 days ago", or "never"
function timeAgo(iso) {
  if (!iso) return 'never';
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return Math.floor(seconds / 60) + ' min ago';
  if (seconds < 86400) return Math.floor(seconds / 3600) + ' h ago';
  return Math.floor(seconds / 86400) + ' days ago';
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

// one usage bar: "Today  1,200 / 10,000"
function UsageBar({ label, used, limit }) {
  const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  const color = pct >= 100 ? '#ef4444' : pct >= 80 ? '#f59e0b' : '#4f7fff';
  return (
    <div className="ukey-bar">
      <div className="ukey-bar-top">
        <span>{label}</span>
        <span>{used.toLocaleString()} / {limit > 0 ? limit.toLocaleString() : 'unlimited'}</span>
      </div>
      <div className="ukey-bar-track"><div className="ukey-bar-fill" style={{ width: pct + '%', background: color }} /></div>
    </div>
  );
}

function UsageBlock({ usage }) {
  if (!usage) return null;
  const dayFull = usage.daily_limit > 0 && usage.daily_used >= usage.daily_limit;
  const monthFull = usage.monthly_limit > 0 && usage.monthly_used >= usage.monthly_limit;
  return (
    <div className="apigen-usage">
      <UsageBar label="Today" used={usage.daily_used} limit={usage.daily_limit} />
      <UsageBar label="This month" used={usage.monthly_used} limit={usage.monthly_limit} />
      {(monthFull || dayFull) && (
        <div className="usage-alert">
          {monthFull
            ? `All ${usage.monthly_limit.toLocaleString()} requests for this month have been used. Detection is paused for this key and requests are rejected (429). It resets in ${fmtDuration(usage.monthly_resets_in_seconds)}.`
            : `All ${usage.daily_limit.toLocaleString()} requests for today have been used. Detection is paused for this key and requests are rejected (429). It resets in ${fmtDuration(usage.daily_resets_in_seconds)} (00:00 UTC).`}
        </div>
      )}
    </div>
  );
}

const CURL_SNIPPET = `curl ${gateway.API_URL}/v1/screen \\
  -H "Authorization: Bearer ftx_live_..." \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "summarize this document", "end_user": "user-123"}'`;

const PYTHON_SNIPPET = `import requests

FORTEX_URL = "${gateway.API_URL}/v1/screen"
FORTEX_KEY = "ftx_live_..."  # keep this in an environment variable

def ask_llm_safely(user_prompt, user_id=None):
    # 1. send the prompt to FortexAI first
    r = requests.post(
        FORTEX_URL,
        headers={"Authorization": f"Bearer {FORTEX_KEY}"},
        json={"prompt": user_prompt, "end_user": user_id},
        timeout=10,
    )
    if r.status_code == 429:
        # key hit its rate / daily / monthly limit; Retry-After says how many seconds to wait
        return "The service is busy right now, please try again later."
    r.raise_for_status()
    verdict = r.json()

    # 2. suspicious -> stop here, never reaches your LLM
    if not verdict["allowed"]:
        return "Sorry, this looks like a suspicious prompt, so we can't process it."

    # 3. safe -> forward to your LLM as usual
    return call_your_llm(user_prompt)`;

export default function ApiGenView() {
  const showToast = useToast();
  const { token } = useAuth();
  const [keys, setKeys] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [keyNameInput, setKeyNameInput] = useState('');
  const [newlyGenerated, setNewlyGenerated] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  // load the keys from the backend
  const loadKeys = useCallback(async () => {
    try {
      setKeys(await gateway.listKeys(token));
    } catch (err) {
      showToast(err.message, 'warn');
    }
  }, [token, showToast]);

  useEffect(() => {
    const first = setTimeout(loadKeys, 0);
    return () => clearTimeout(first);
  }, [loadKeys]);

  const openModal = () => { setShowModal(true); setKeyNameInput(''); setNewlyGenerated(null); };
  const closeModal = () => setShowModal(false);

  const generateKey = async () => {
    try {
      const created = await gateway.createKey(token, keyNameInput.trim() || 'New Key');
      setNewlyGenerated(created); // holds the full key - shown only once
      loadKeys();
    } catch (err) {
      showToast(err.message, 'warn');
    }
  };

  const copyKey = (text) => {
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
    showToast('Copied to clipboard.', 'success');
  };

  const toggleStatus = async (key) => {
    const next = key.status === 'active' ? 'stopped' : 'active';
    try {
      await gateway.setKeyStatus(token, key.id, next);
      showToast(next === 'active' ? 'Detection is ON for this key.' : 'Detection is OFF for this key.', 'info');
      loadKeys();
    } catch (err) {
      showToast(err.message, 'warn');
    }
  };

  const confirmDelete = async () => {
    try {
      await gateway.deleteKey(token, confirmDeleteId);
      showToast('API key deleted.', 'warn');
    } catch (err) {
      showToast(err.message, 'warn');
    }
    setConfirmDeleteId(null);
    loadKeys();
  };

  const confirmKeyName = keys.find((k) => k.id === confirmDeleteId)?.name;

  return (
    <div className="apigen">
      <div className="apigen-header">
        <div className="apigen-intro">Generate keys to authenticate your application against the FortexAI screening endpoint. Treat every key like a password.</div>
        <button className="btn-primary" onClick={openModal}>+ Generate New Key</button>
      </div>

      <div className="apigen-table">
        <div className="apigen-table-head">
          <div>Name</div><div>Key</div><div>Created</div><div>Last Used</div><div>Status</div><div />
        </div>
        {keys.map((k) => {
          const active = k.status === 'active';
          return (
            <Fragment key={k.id}>
            <div className="apigen-table-row">
              <div className="apigen-key-name">{k.name}</div>
              <div className="apigen-key-value">
                {k.key_prefix}••••••••
              </div>
              <div className="apigen-key-meta">{fmtDate(k.created_at)}</div>
              <div className="apigen-key-meta">{timeAgo(k.last_used_at)}</div>
              <div className="apigen-status">
                <div className="toggle" style={{ background: active ? '#4f7fff' : '#232b38' }} onClick={() => toggleStatus(k)}>
                  <div className="toggle-knob" style={{ left: active ? '19px' : '3px' }} />
                </div>
                <span style={{ color: active ? '#22c55e' : '#8b96a5', fontWeight: 600, fontSize: 12 }}>{active ? 'Active' : 'Stopped'}</span>
              </div>
              <div>
                <button className="delete-btn" title="Delete key" onClick={() => setConfirmDeleteId(k.id)}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" /></svg>
                </button>
              </div>
            </div>
            <UsageBlock usage={k.usage} />
            </Fragment>
          );
        })}
        {keys.length === 0 && <div className="apigen-key-meta" style={{ padding: 20 }}>No API keys yet. Generate one to get started.</div>}
      </div>

      <div className="apigen-card">
        <div className="apigen-card-title">Quick integration</div>
        <div className="apigen-card-desc">Send every user prompt to FortexAI <b>before</b> it reaches your LLM. If the reply has <code>allowed: true</code>, forward the prompt to your LLM. If it has <code>allowed: false</code>, do not forward it &mdash; show the user a warning instead.</div>
        <div className="apigen-card-desc" style={{ marginTop: 14 }}>1. Test your key from a terminal:</div>
        <pre className="code-block" style={{ margin: 0 }}>{CURL_SNIPPET}</pre>
        <div className="apigen-card-desc" style={{ marginTop: 14 }}>2. Add this to your application (Python):</div>
        <pre className="code-block" style={{ margin: 0 }}>{PYTHON_SNIPPET}</pre>
        <div className="apigen-card-desc" style={{ marginTop: 14 }}>The response looks like <code>{'{"allowed": true, "verdict": "passed", "phase": "semantic_search", "confidence": 0.99, "latency_ms": 12.3}'}</code>. <code>verdict</code> is <code>passed</code>, <code>flagged</code> or <code>blocked</code>; only <code>blocked</code> sets <code>allowed</code> to false.</div>
      </div>

      <div className="apigen-card">
        <div className="apigen-card-title">Integrating by environment</div>
        <div className="apigen-card-desc">Use a separate key per environment so you can rotate or stop one without affecting the others.</div>
        <div className="env-grid">
          <div className="env-card">
            <div className="env-card-head"><span className="env-dot" style={{ background: '#22c55e' }} /><div className="env-name">Production</div></div>
            <div className="env-desc">Use a <code>ftx_live_*</code> key. Enable full blocking &mdash; malicious prompts are stopped before they reach your model.</div>
          </div>
          <div className="env-card">
            <div className="env-card-head"><span className="env-dot" style={{ background: '#f59e0b' }} /><div className="env-name">Staging</div></div>
            <div className="env-desc">Use a <code>ftx_test_*</code> key. Run in monitor-only mode to validate detection before promoting to production.</div>
          </div>
          <div className="env-card">
            <div className="env-card-head"><span className="env-dot" style={{ background: '#6d94ff' }} /><div className="env-name">Local / CI</div></div>
            <div className="env-desc">Point at a self-hosted FortexAI instance via Docker &mdash; no key rotation needed across test runs.</div>
          </div>
        </div>
      </div>

      {showModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            {newlyGenerated ? (
              <>
                <div className="modal-title">Key generated</div>
                <div className="modal-desc">Copy it now &mdash; you won't be able to see it again.</div>
                <div className="modal-key-value">{newlyGenerated.key}</div>
                <div className="modal-actions">
                  <button className="btn-primary flex" onClick={() => copyKey(newlyGenerated.key)}>Copy Key</button>
                  <button className="btn-secondary" onClick={closeModal}>Done</button>
                </div>
              </>
            ) : (
              <>
                <div className="modal-title" style={{ marginBottom: 18 }}>Generate new API key</div>
                <label className="modal-label">Key name</label>
                <input
                  type="text"
                  placeholder="e.g. Production"
                  value={keyNameInput}
                  onChange={(e) => setKeyNameInput(e.target.value)}
                  className="modal-input"
                />
                <div className="modal-actions">
                  <button className="btn-primary flex" onClick={generateKey}>Generate</button>
                  <button className="btn-secondary" onClick={closeModal}>Cancel</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {confirmDeleteId && (
        <div className="modal-overlay" onClick={() => setConfirmDeleteId(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-title">Delete this API key?</div>
            <div className="modal-desc" style={{ marginBottom: 22 }}>
              Any application still using <b style={{ color: '#e8ecf1' }}>{confirmKeyName}</b> will immediately stop being able to authenticate. This can't be undone.
            </div>
            <div className="modal-actions">
              <button className="btn-danger flex" onClick={confirmDelete}>Delete Key</button>
              <button className="btn-secondary" onClick={() => setConfirmDeleteId(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
