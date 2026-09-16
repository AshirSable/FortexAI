import { useState } from 'react';
import { useToast } from '../Toast';
import './ApiGenView.css';

const INITIAL_KEYS = [
  { id: 'k1', name: 'Production', key: 'ftx_live_8f2a9c4e1b6d47a0', created: 'Jun 2, 2026', lastUsed: '3 min ago', status: 'active' },
  { id: 'k2', name: 'Staging', key: 'ftx_test_3c7e91b0d5f2468a', created: 'May 14, 2026', lastUsed: '2 days ago', status: 'active' },
];

function maskKey(key) {
  return key.slice(0, 9) + '••••••••' + key.slice(-4);
}

export default function ApiGenView() {
  const showToast = useToast();
  const [keys, setKeys] = useState(INITIAL_KEYS);
  const [showModal, setShowModal] = useState(false);
  const [keyNameInput, setKeyNameInput] = useState('');
  const [newlyGenerated, setNewlyGenerated] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);

  const openModal = () => { setShowModal(true); setKeyNameInput(''); setNewlyGenerated(null); };
  const closeModal = () => setShowModal(false);

  const generateKey = () => {
    const name = keyNameInput.trim() || 'New Key';
    const rnd = Math.random().toString(16).slice(2, 18).padEnd(16, '0');
    const key = 'ftx_live_' + rnd;
    const entry = { id: 'k' + Date.now(), name, key, created: 'today', lastUsed: 'never', status: 'active' };
    setKeys((k) => [entry, ...k]);
    setNewlyGenerated(entry);
  };

  const copyKey = (text) => {
    if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
    showToast('Copied to clipboard.', 'success');
  };

  const toggleStatus = (id) => {
    setKeys((k) => k.map((item) => (item.id === id ? { ...item, status: item.status === 'active' ? 'stopped' : 'active' } : item)));
    showToast('Key status updated.', 'info');
  };

  const confirmDelete = () => {
    setKeys((k) => k.filter((item) => item.id !== confirmDeleteId));
    setConfirmDeleteId(null);
    showToast('API key deleted.', 'warn');
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
                {maskKey(k.key)}
                <button className="copy-btn" onClick={() => copyKey(k.key)}>Copy</button>
              </div>
              <div className="apigen-key-meta">{k.created}</div>
              <div className="apigen-key-meta">{k.lastUsed}</div>
              <div className="apigen-status">
                <div className="toggle" style={{ background: active ? '#4f7fff' : '#232b38' }} onClick={() => toggleStatus(k.id)}>
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
      </div>

      <div className="apigen-card">
        <div className="apigen-card-title">Quick integration</div>
        <div className="apigen-card-desc">Same key, any environment &mdash; point the request at your active project's endpoint.</div>
        <div className="code-block">
          <div style={{ color: '#5b6472' }}># screen a prompt before it reaches your model</div>
          <div>curl https://api.fortexai.dev/v1/screen \</div>
          <div>&nbsp;&nbsp;-H "Authorization: Bearer <span style={{ color: '#8fa9ff' }}>ftx_live_...</span>" \</div>
          <div>&nbsp;&nbsp;-d '{'{"prompt": "summarize this document"}'}'</div>
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
