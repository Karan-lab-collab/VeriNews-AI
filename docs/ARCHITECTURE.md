# Architecture Overview

## System Design

VeriNews AI follows a decoupled client-server architecture:

```
┌──────────────────┐     HTTP / REST      ┌─────────────────────┐
│  React + Vite    │ ──────────────────▶  │  FastAPI backend    │
│  (port 5173)     │ ◀──────────────────  │  (port 8000)        │
└──────────────────┘                      └─────────────────────┘
                                                    │
                                           ┌────────▼────────┐
                                           │  ML Services    │
                                           │  (planned)      │
                                           └─────────────────┘
```

## Data Flow (planned)

1. User submits a news article URL or raw text via the React UI.
2. Frontend calls `POST /api/v1/predict` with the payload.
3. FastAPI validates the request with Pydantic models.
4. The `DetectionService` preprocesses text and runs inference.
5. A confidence score and label are returned to the frontend.
6. The UI renders the verdict with an explanation.

## Key Design Decisions

- **Monorepo layout**: frontend and backend are co-located for simplicity.
- **Vite dev proxy**: `/api` requests are proxied to the backend during development, eliminating CORS issues in the browser.
- **Explicit CORS config**: the FastAPI app also configures CORS headers for production deployments where the proxy is absent.
- **Pydantic v2**: all request/response schemas use Pydantic 2 for fast validation and automatic OpenAPI generation.
