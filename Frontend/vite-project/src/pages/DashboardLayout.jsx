import { Outlet } from 'react-router-dom';
import Sidebar from '../components/dashboard/Sidebar';
import Topbar from '../components/dashboard/Topbar';
import './DashboardLayout.css';

export default function DashboardLayout() {
  return (
    <div className="dashboard-shell">
      <Sidebar />
      <div className="dashboard-main">
        <Topbar />
        <div className="dashboard-content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
