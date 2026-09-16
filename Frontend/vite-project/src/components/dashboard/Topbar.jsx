import { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import './Topbar.css';

const TITLES = {
  '': 'Monitoring',
  'api-keys': 'API Keys',
  docs: 'Documentation',
  usage: 'Usage & Limits',
  billing: 'Billing',
  settings: 'Settings',
};

function getInitials(name) {
  if (!name) return 'U';
  const parts = name.trim().split(/\s+/);
  const initials = parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : parts[0].slice(0, 2);
  return initials.toUpperCase();
}

export default function Topbar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const segment = location.pathname.replace(/^\/dashboard\/?/, '').split('/')[0] || '';
  const title = TITLES[segment] || 'Monitoring';

  useEffect(() => {
    function handleClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSignOut = async () => {
    setMenuOpen(false);
    await logout();
    navigate('/login');
  };

  return (
    <div className="topbar">
      <div>
        <div className="topbar-eyebrow">Dashboard</div>
        <div className="topbar-title">{title}</div>
      </div>
      <div className="topbar-actions">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#a3adba" strokeWidth="1.8" className="topbar-bell">
          <path d="M18 8a6 6 0 10-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.7 21a2 2 0 01-3.4 0" />
        </svg>
        <div className="topbar-divider" />
        <div className="topbar-profile" ref={menuRef}>
          <div
            className="topbar-avatar"
            onClick={() => setMenuOpen((v) => !v)}
            title="Account menu"
          >
            {getInitials(user?.name)}
          </div>
          {menuOpen && (
            <div className="topbar-menu">
              <button
                type="button"
                className="topbar-menu-item"
                onClick={() => {
                  setMenuOpen(false);
                  navigate('/dashboard/settings');
                }}
              >
                Settings
              </button>
              <button type="button" className="topbar-menu-item topbar-menu-item-danger" onClick={handleSignOut}>
                Sign Out
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
