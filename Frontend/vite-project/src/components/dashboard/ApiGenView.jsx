import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../context/auth-context';
import * as gateway from '../../api/gateway';
import { useToast } from '../ToastContext';
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
            <div className="apigen-table-row" key={k.id}>
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
          );
        })}
        {keys.length === 0 && <div className="apigen-key-meta" style={{ padding: 20 }}>No API keys yet. Generate one to get started.</div>}
      </div>

      <div className="apigen-card">
        <div className="apigen-card-title">Quick integration</div>
        <div className="apigen-card-desc">Send the prompt before it reaches your model. If the answer has <code>allowed: false</code>, do not forward it.</div>
        <div className="code-block">
          <div style={{ color: '#5b6472' }}># screen a prompt before it reaches your model</div>
          <div>curl {gateway.API_URL}/v1/screen \</div>
          <div>&nbsp;&nbsp;-H "Authorization: Bearer <span style={{ color: '#8fa9ff' }}>ftx_live_...</span>" \</div>
          <div>&nbsp;&nbsp;-H "Content-Type: application/json" \</div>
          <div>&nbsp;&nbsp;-d '{'{"prompt": "summarize this document", "end_user": "user-123"}'}'</div>
        </div>
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
