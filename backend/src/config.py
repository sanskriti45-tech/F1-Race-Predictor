"""
Central configuration for the F1 race predictor.

All paths and environment-driven settings live here so the rest of the
codebase never hard-codes a path or reads os.environ directly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    fastf1_cache_dir: Path = PROJECT_ROOT / os.getenv("FASTF1_CACHE_DIR", "data/cache")
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def ensure_dirs(self) -> None:
        for d in (
            self.fastf1_cache_dir,
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured consistently across the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger
