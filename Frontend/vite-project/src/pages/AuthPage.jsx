import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import fortexLogo from '../assets/fortexai-logo.png';
import { useToast } from '../components/ToastContext';
import { useAuth } from '../context/auth-context';
import './AuthPage.css';

function EyeIcon({ open }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      {open ? (
        <>
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" />
          <circle cx="12" cy="12" r="3" />
        </>
      ) : (
        <>
          <path d="M3 3l18 18" />
          <path d="M10.6 5.2A10.6 10.6 0 0112 5c6.5 0 10 7 10 7a17.9 17.9 0 01-3.6 4.6M6.6 6.6C4 8.3 2 12 2 12s3.5 7 10 7a10.4 10.4 0 004.4-.9" />
          <path d="M9.9 9.9a3 3 0 004.2 4.2" />
        </>
      )}
    </svg>
  );
}

const CASCADE_STEPS = [
  { text: 'Autoencoder anomaly detection', good: false },
  { text: 'Semantic search vs. attack memory', good: false },
  { text: 'Ensemble BERT classification', good: false },
  { text: 'LLM judge for ambiguous cases', good: false },
  { text: 'Cosine similarity output check', good: true },
];

export default function AuthPage({ mode }) {
  const isLogin = mode === 'login';
  const navigate = useNavigate();
  const showToast = useToast();
  const { login, signup } = useAuth();

  const [loginForm, setLoginForm] = useState({ email: '', password: '' });
  const [registerForm, setRegisterForm] = useState({ name: '', email: '', company: '', password: '' });
  const [submitting, setSubmitting] = useState(false);
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [showRegisterPassword, setShowRegisterPassword] = useState(false);

  const submitLogin = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      await login(loginForm);
      navigate('/dashboard');
      showToast('Welcome back — signed in successfully.', 'success');
    } catch (err) {
      showToast(err.message, 'warn');
    } finally {
      setSubmitting(false);
    }
  };

  const submitRegister = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      await signup(registerForm);
      navigate('/dashboard');
      showToast('Account created. Welcome to FortexAI!', 'success');
    } catch (err) {
      showToast(err.message, 'warn');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-form-side">
        <div className="auth-form-wrap">
          <div className="auth-brand" onClick={() => navigate('/')}>
            <img src={fortexLogo} alt="FortexAI" className="auth-brand-logo" />
            <span className="auth-brand-name">FortexAI</span>
          </div>

          {isLogin ? (
            <div>
              <h1 className="auth-title">Welcome back</h1>
              <p className="auth-subtitle">Sign in to your FortexAI dashboard.</p>
              <form onSubmit={submitLogin} className="auth-form">
                <div>
                  <label className="auth-label">Work email</label>
                  <input
                    type="email" required placeholder="you@company.com"
                    value={loginForm.email}
                    onChange={(e) => setLoginForm((f) => ({ ...f, email: e.target.value }))}
                    className="auth-input"
                  />
                </div>
                <div>
                  <label className="auth-label">Password</label>
                  <div className="auth-password-wrap">
                    <input
                      type={showLoginPassword ? 'text' : 'password'} required placeholder="••••••••"
                      value={loginForm.password}
                      onChange={(e) => setLoginForm((f) => ({ ...f, password: e.target.value }))}
                      className="auth-input"
                    />
                    <button
                      type="button"
                      className="auth-password-toggle"
                      onClick={() => setShowLoginPassword((v) => !v)}
                      aria-label={showLoginPassword ? 'Hide password' : 'Show password'}
                      tabIndex={-1}
                    >
                      <EyeIcon open={showLoginPassword} />
                    </button>
                  </div>
                </div>
                <div className="auth-forgot"><a href="#">Forgot password?</a></div>
                <button type="submit" className="auth-submit" disabled={submitting}>
                  {submitting ? 'Signing in…' : 'Sign In'}
                </button>
              </form>
              <div className="auth-switch">No account? <a href="#" onClick={(e) => { e.preventDefault(); navigate('/signup'); }}>Create one free</a></div>
            </div>
          ) : (
            <div>
              <h1 className="auth-title">Create your account</h1>
              <p className="auth-subtitle">Start screening prompts in minutes.</p>
              <form onSubmit={submitRegister} className="auth-form">
                <div>
                  <label className="auth-label">Full name</label>
                  <input
                    type="text" required placeholder="Ada Lovelace"
                    value={registerForm.name}
                    onChange={(e) => setRegisterForm((f) => ({ ...f, name: e.target.value }))}
                    className="auth-input"
                  />
                </div>
                <div>
                  <label className="auth-label">Work email</label>
                  <input
                    type="email" required placeholder="you@company.com"
                    value={registerForm.email}
                    onChange={(e) => setRegisterForm((f) => ({ ...f, email: e.target.value }))}
                    className="auth-input"
                  />
                </div>
                <div>
                  <label className="auth-label">Company</label>
                  <input
                    type="text" placeholder="Acme Inc."
                    value={registerForm.company}
                    onChange={(e) => setRegisterForm((f) => ({ ...f, company: e.target.value }))}
                    className="auth-input"
                  />
                </div>
                <div>
                  <label className="auth-label">Password</label>
                  <div className="auth-password-wrap">
                    <input
                      type={showRegisterPassword ? 'text' : 'password'} required placeholder="••••••••"
                      value={registerForm.password}
                      onChange={(e) => setRegisterForm((f) => ({ ...f, password: e.target.value }))}
                      className="auth-input"
                    />
                    <button
                      type="button"
                      className="auth-password-toggle"
                      onClick={() => setShowRegisterPassword((v) => !v)}
                      aria-label={showRegisterPassword ? 'Hide password' : 'Show password'}
                      tabIndex={-1}
                    >
                      <EyeIcon open={showRegisterPassword} />
                    </button>
                  </div>
                </div>
                <button type="submit" className="auth-submit" disabled={submitting}>
                  {submitting ? 'Creating account…' : 'Create Account'}
                </button>
              </form>
              <div className="auth-switch">Already have an account? <a href="#" onClick={(e) => { e.preventDefault(); navigate('/login'); }}>Sign in</a></div>
            </div>
          )}
        </div>
      </div>

      <div className="auth-visual-side">
        <div className="auth-visual-glow" />
        <div className="auth-visual-content">
          <div className="auth-visual-eyebrow">Five-stage cascade</div>
          <div className="auth-visual-headline">Cheap checks first. The expensive judge only for the hard cases.</div>
          <div className="auth-cascade-list">
            {CASCADE_STEPS.map((step) => (
              <div key={step.text} className={'auth-cascade-step' + (step.good ? ' good' : '')}>
                <span className="auth-cascade-dot" style={{ background: step.good ? '#22c55e' : '#4f7fff' }} />
                <span>{step.text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
