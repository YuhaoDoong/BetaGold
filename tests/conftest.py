"""Shared pytest fixtures for the GLD/SLV trading system.

Tests must be offline-reproducible: no live yfinance, no Moomoo OpenD,
no network access. Reference data lives under ``tests/fixtures/`` as
small parquet/csv snippets committed to the repo.
"""
from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import pytest

# Make GoldDash root importable as the top-level package root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Path to the cached test fixtures directory."""
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def fixed_today() -> pd.Timestamp:
    """A deterministic 'today' anchor so tests do not depend on wall clock."""
    return pd.Timestamp("2026-05-22").normalize()
