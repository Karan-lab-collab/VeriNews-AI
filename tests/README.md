# Tests

This directory will contain integration and end-to-end tests for VeriNews AI.

## Planned Test Coverage

### Backend (pytest)
- `test_health.py` — verify `/api/v1/health` returns 200 and correct JSON
- `test_predict.py` — test the fake-news detection endpoint (once implemented)

### Frontend (Vitest / Playwright)
- Unit tests for hooks and components
- E2E tests for the full prediction flow

## Running Tests

```bash
# Backend (from backend/)
pytest tests/ -v

# Frontend (from frontend/)
npm test
```
