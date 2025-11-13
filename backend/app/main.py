"""FastAPI entry point for the tiny reasoning backend."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

allowed_origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""

    return {"status": "ok"}


app.include_router(api_router, prefix="/api", tags=["Completions"])


__all__ = ["app"]
