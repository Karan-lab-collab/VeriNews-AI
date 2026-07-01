/**
 * api.js – centralised service layer for VeriNews AI API calls.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

/**
 * Generic fetch wrapper with JSON parsing and error handling.
 */
async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API error ${response.status}: ${text}`);
  }

  return response.json();
}

// ── Health ────────────────────────────────────────────────────────────────────
export const fetchHealth = () => request("/health");
