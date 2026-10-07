const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function parseResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = data?.detail || 'Something went wrong. Please try again.';
    throw new Error(typeof message === 'string' ? message : 'Something went wrong. Please try again.');
  }
  return data;
}

function authHeaders(token) {
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}

// Turn { period: '24h', key_id: 3, q: '' } into "?period=24h&key_id=3" (empty values are skipped)
function toQuery(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([name, value]) => {
    if (value !== undefined && value !== null && value !== '') search.set(name, value);
  });
  const text = search.toString();
  return text ? `?${text}` : '';
}

// ---- API keys ----

export async function listKeys(token) {
  const res = await fetch(`${API_BASE_URL}/keys`, { headers: authHeaders(token) });
  return parseResponse(res);
}

export async function createKey(token, name) {
  const res = await fetch(`${API_BASE_URL}/keys`, {
    method: 'POST',
    headers: authHeaders(token),
    body: JSON.stringify({ name }),
  });
  return parseResponse(res);
}

export async function setKeyStatus(token, keyId, status) {
  const res = await fetch(`${API_BASE_URL}/keys/${keyId}/status`, {
    method: 'PATCH',
    headers: authHeaders(token),
    body: JSON.stringify({ status }),
  });
  return parseResponse(res);
}

export async function deleteKey(token, keyId) {
  const res = await fetch(`${API_BASE_URL}/keys/${keyId}`, {
    method: 'DELETE',
    headers: authHeaders(token),
  });
  return parseResponse(res);
}

// ---- Metrics ----

export async function fetchSummary(token, { period, keyId }) {
  const res = await fetch(`${API_BASE_URL}/metrics/summary${toQuery({ period, key_id: keyId })}`, {
    headers: authHeaders(token),
  });
  return parseResponse(res);
}

export async function fetchTrend(token, { period, keyId }) {
  const res = await fetch(`${API_BASE_URL}/metrics/trend${toQuery({ period, key_id: keyId })}`, {
    headers: authHeaders(token),
  });
  return parseResponse(res);
}

export async function fetchEvents(token, { period, keyId, verdict, phase, q }) {
  const query = toQuery({ period, key_id: keyId, verdict, phase, q });
  const res = await fetch(`${API_BASE_URL}/metrics/events${query}`, { headers: authHeaders(token) });
  return parseResponse(res);
}

export const API_URL = API_BASE_URL;
