/**
 * StatusBadge.jsx – displays the API health status returned from the backend.
 */
import React from "react";
import "./StatusBadge.css";

export default function StatusBadge({ status, message, loading, error }) {
  if (loading) {
    return (
      <div className="status-badge status-loading" aria-live="polite" aria-busy="true">
        <span className="status-dot pulse" />
        <span>Connecting to backend…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="status-badge status-error" role="alert">
        <span className="status-dot" />
        <span>Backend unreachable — {error}</span>
      </div>
    );
  }

  return (
    <div className="status-badge status-ok" aria-live="polite">
      <span className="status-dot" />
      <span>
        <strong>{status?.toUpperCase()}</strong> — {message}
      </span>
    </div>
  );
}
