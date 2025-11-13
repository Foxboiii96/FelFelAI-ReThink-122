"""Application settings module."""
from functools import lru_cache
from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field("Tiny Reasoning API", env="APP_NAME")
    model_weights_path: Path = Field(
        Path(__file__).resolve().parent / "models" / "artifacts" / "tiny_reasoning_model.pt",
        env="MODEL_WEIGHTS_PATH",
    )
    max_sequence_length: int = Field(128, env="MAX_SEQUENCE_LENGTH")
    allowed_origins: str = Field("*", env="CORS_ALLOWED_ORIGINS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""

    settings = Settings()
    weights_dir = settings.model_weights_path.parent
    weights_dir.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["get_settings", "Settings"]
