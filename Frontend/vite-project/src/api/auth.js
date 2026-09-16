const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function parseResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = data?.detail || 'Something went wrong. Please try again.';
    throw new Error(typeof message === 'string' ? message : 'Something went wrong. Please try again.');
  }
  return data;
}

export async function signup({ name, email, company, password }) {
  const res = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email, company, password }),
  });
  return parseResponse(res);
}

export async function login({ email, password }) {
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  return parseResponse(res);
}

export async function logout(token) {
  const res = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
  });
  return parseResponse(res);
}

export async function fetchMe(token) {
  const res = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return parseResponse(res);
}
