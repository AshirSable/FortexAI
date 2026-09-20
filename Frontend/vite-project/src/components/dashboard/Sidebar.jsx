import { NavLink, useNavigate } from 'react-router-dom';
import fortexLogo from '../../assets/fortexai-logo.png';
import { useAuth } from '../../context/auth-context';
import './Sidebar.css';

function getInitials(name) {
  if (!name) return 'U';
  const parts = name.trim().split(/\s+/);
  const initials = parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : parts[0].slice(0, 2);
  return initials.toUpperCase();
}

const NAV_ITEMS = [
  {
    id: 'api-keys', label: 'API Keys', end: false,
    icon: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="3" />
        <path d="M8 2v4M16 2v4M4 10h16" />
      </>
    ),
  },
  {
    id: 'docs', label: 'Documentation',
    icon: (
      <>
        <path d="M7 3h7l4 4v14H7z" />
        <path d="M14 3v4h4" />
        <path d="M9.5 12h5M9.5 15.5h5" />
      </>
    ),
  },
  {
    id: '', label: 'Monitoring', end: true,
    icon: <path d="M3 20V6M3 20h18M7 16l3-4 3 3 4-6" />,
  },
  {
    id: 'usage', label: 'Usage & Limits',
    icon: (
      <>
        <circle cx="12" cy="12" r="8" />
        <path d="M12 8v4l2.5 2.5" />
      </>
    ),
  },
  {
    id: 'billing', label: 'Billing',
    icon: (
      <>
        <rect x="3" y="6" width="18" height="13" rx="2" />
        <path d="M3 10.5h18M7 15h4" />
      </>
    ),
  },
  {
    id: 'settings', label: 'Settings',
    icon: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1" />
      </>
    ),
  },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const { user } = useAuth();

  return (
    <div className="sidebar">
      <div className="sidebar-brand" onClick={() => navigate('/')}>
        <img src={fortexLogo} alt="FortexAI" className="sidebar-brand-logo" />
        <span className="sidebar-brand-name">FortexAI</span>
      </div>

      <div className="sidebar-section-label">Workspace</div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.label}
            to={item.id}
            end={item.end}
            className={({ isActive }) => 'sidebar-nav-item' + (isActive ? ' active' : '')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              {item.icon}
            </svg>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-status">
        <span className="sidebar-status-dot" />
        <div className="sidebar-status-text">All systems operational</div>
      </div>

      <div className="sidebar-profile">
        <div className="sidebar-profile-avatar">{getInitials(user?.name)}</div>
        <div className="sidebar-profile-info">
          <div className="sidebar-profile-name">{user?.name || 'Loading…'}</div>
          <div className="sidebar-profile-org">{user?.company || user?.email || ''}</div>
        </div>
      </div>
    </div>
  );
}
