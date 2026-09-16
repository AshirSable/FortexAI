import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ToastProvider } from './components/Toast';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import AuthPage from './pages/AuthPage';
import DashboardLayout from './pages/DashboardLayout';
import MonitoringView from './components/dashboard/MonitoringView';
import ApiGenView from './components/dashboard/ApiGenView';
import PlaceholderView from './components/dashboard/PlaceholderView';

function App() {
  return (
    <ToastProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<AuthPage mode="login" />} />
            <Route path="/signup" element={<AuthPage mode="register" />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <DashboardLayout />
                </ProtectedRoute>
              }
            >
              <Route index element={<MonitoringView />} />
              <Route path="api-keys" element={<ApiGenView />} />
              <Route path="docs" element={<PlaceholderView kind="docs" />} />
              <Route path="usage" element={<PlaceholderView kind="usage" />} />
              <Route path="billing" element={<PlaceholderView kind="billing" />} />
              <Route path="settings" element={<PlaceholderView kind="settings" />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </ToastProvider>
  );
}

export default App;
