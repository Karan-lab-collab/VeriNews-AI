from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health

app = FastAPI(
    title="VeriNews AI",
    description="AI-powered Fake News Detection API",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS – allow the Vite dev server (and any localhost port) to call the API
# ---------------------------------------------------------------------------
origins = [
    "http://localhost",
    "http://localhost:5173",   # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
