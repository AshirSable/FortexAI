import { useCallback, useRef, useState } from 'react';
import { ToastContext } from './ToastContext';
import './Toast.css';

const PALETTE = {
  success: { bg: '#12271c', border: '#1f5c3a', icon: '✓' },
  warn: { bg: '#2b2013', border: '#6b4a15', icon: '⚠' },
  info: { bg: '#12171f', border: '#2c3646', icon: 'ℹ' },
};

export function ToastProvider({ children }) {
  const [toast, setToast] = useState(null);
  const timerRef = useRef(null);

  const showToast = useCallback((msg, type = 'info') => {
    setToast({ msg, type });
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setToast(null), 3000);
  }, []);

  const palette = PALETTE[toast?.type || 'info'];

  return (
    <ToastContext.Provider value={showToast}>
      {children}
      <div className="toast-host">
        {toast && (
          <div
            className="toast"
            style={{ background: palette.bg, borderColor: palette.border }}
          >
            <span>{palette.icon}</span>
            <span>{toast.msg}</span>
          </div>
        )}
      </div>
    </ToastContext.Provider>
  );
}
