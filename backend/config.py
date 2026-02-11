"""
Application configuration for the Adaptive Forecast system.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from . import BASE_DIR

load_dotenv()


class Settings(BaseSettings):
    """Typed application settings with sensible defaults."""

    environment: str = Field(default="development")
    # database_url removed - using MongoDB only
    models_dir: Path = Field(default=BASE_DIR / "artifacts")
    logs_dir: Path = Field(default=BASE_DIR / "logs")
    forecast_horizon: int = Field(default=12, ge=1)
    lag_window: int = Field(default=24, ge=4)
    min_history: int = Field(default=200, ge=24)
    evaluation_window: int = Field(default=100, ge=1)
    portfolio_initial_cash: float = Field(default=10_000.0, gt=0)
    error_threshold: float = Field(default=0.02, ge=0.0)
    dashboard_refresh_seconds: int = Field(default=10, ge=1)
    symbol: str = Field(default="AAPL")
    allow_external_data: bool = Field(default=False)
    mongo_uri: str = Field(default="mongodb://localhost:27017/")
    mongo_db: str = Field(default="forecast_db")
    use_mongo: bool = Field(default=True)
    continuous_eval_interval_minutes: int = Field(default=15, ge=1)
    auto_start_retraining: bool = Field(default=False)

    model_config = {
        "env_prefix": "FORECAST_",
        "case_sensitive": False,
    }


class Paths(BaseModel):
    """Helper container for key filesystem locations."""

    data_dir: Path = Field(default=BASE_DIR / "data")
    diagrams_dir: Path = Field(default=BASE_DIR.parent / "docs")

    def ensure(self) -> None:
        """Ensure all important directories exist."""
        for path in (self.data_dir, self.diagrams_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    paths = Paths()
    paths.ensure()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Return cached paths instance."""
    paths = Paths()
    paths.ensure()
    return paths


__all__ = ["Settings", "get_settings", "get_paths"]

