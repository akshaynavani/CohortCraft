"""Runtime configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Config:
    seed: int = field(default_factory=lambda: _env_int("COHORTCRAFT_SEED", 42))
    n_customers: int = field(default_factory=lambda: _env_int("COHORTCRAFT_N_CUSTOMERS", 5000))
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("COHORTCRAFT_DB_PATH", "outputs/cohortcraft.db"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("COHORTCRAFT_OUTPUT_DIR", "outputs"))
    )
    # Observation window end date for feature snapshots and censoring.
    snapshot_date: str = field(
        default_factory=lambda: os.getenv("COHORTCRAFT_SNAPSHOT_DATE", "2025-12-31")
    )
    test_size: float = field(default_factory=lambda: _env_float("COHORTCRAFT_TEST_SIZE", 0.25))
    # Lifetime value settings
    clv_horizon_months: int = field(default_factory=lambda: _env_int("COHORTCRAFT_CLV_HORIZON", 36))
    annual_discount_rate: float = field(
        default_factory=lambda: _env_float("COHORTCRAFT_DISCOUNT_RATE", 0.10)
    )
    gross_margin: float = field(default_factory=lambda: _env_float("COHORTCRAFT_MARGIN", 0.35))
    # Segmentation
    k_min: int = 2
    k_max: int = 8

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
