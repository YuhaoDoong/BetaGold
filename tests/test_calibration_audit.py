"""Regression tests for v3.7.243: DL Range OOS calibration audit.

Lock the label definition (5-day forward H/L vs t-day close, per parquet's
``actual_*_pct`` columns) and exercise the per-month / per-regime grouping +
edge cases.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.eval.model_calibration_audit import (
    REQUIRED_COLUMNS,
    compute_calibration_metrics,
    run_audit,
)


@pytest.fixture
def synthetic_oos(tmp_path):
    """Hand-crafted OOS rows with known analytical metrics.

    Spans 60 trading days across 3 months, with predictable pred/actual
    relationships so width-ratio and coverage are checkable by hand.
    """
    dates = pd.bdate_range("2026-01-05", periods=60)
    df = pd.DataFrame({
        # pred wider than actual: ratio = 2.0 on both sides
        "pred_upper_pct":   [+4.0] * 60,
        "pred_lower_pct":   [-4.0] * 60,
        "actual_upper_pct": [+2.0] * 60,
        "actual_lower_pct": [-2.0] * 60,
    }, index=dates)
    p = tmp_path / "oos.parquet"
    df.to_parquet(p)
    return str(p)


def test_compute_metrics_analytical(synthetic_oos):
    df = pd.read_parquet(synthetic_oos)
    m = compute_calibration_metrics(df)
    # All rows: pred=±4, actual=±2 → width ratio 2.0; coverage 100% on both sides
    assert m["n"] == 60
    assert m["pred_upper_mean"] == 4.0
    assert m["pred_lower_mean"] == -4.0
    assert m["actual_upper_mean"] == 2.0
    assert m["actual_lower_mean"] == -2.0
    assert m["width_ratio_upper"] == 2.0
    assert m["width_ratio_lower"] == 2.0  # (-4)/(-2) = 2.0
    assert m["coverage_upper"] == 1.0  # 2.0 <= 4.0 always
    assert m["coverage_lower"] == 1.0  # -2.0 >= -4.0 always
    assert m["coverage_both"] == 1.0


def test_per_month_grouping(synthetic_oos):
    df = run_audit("GLD", start="2026-01-01", end="2026-04-01",
                     parquet_path=synthetic_oos)
    # 60 bdates from 2026-01-05 spans Jan, Feb, Mar partial; plus one ALL row
    months = df["__month"].tolist()
    assert "ALL" in months
    non_all = [m for m in months if m != "ALL"]
    assert all(m.startswith("2026-") for m in non_all)
    assert 2 <= len(non_all) <= 4  # Jan + Feb + Mar (+ Apr partial maybe)


def test_per_regime_grouping(tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=30)
    df = pd.DataFrame({
        "pred_upper_pct":   [+5.0] * 30,
        "pred_lower_pct":   [-5.0] * 30,
        "actual_upper_pct": [+2.5] * 30,
        "actual_lower_pct": [-2.5] * 30,
        "regime": (["Bull"] * 10 + ["Bear"] * 10 + ["Sideways"] * 10),
    }, index=dates)
    p = tmp_path / "oos_regime.parquet"
    df.to_parquet(p)
    summary = run_audit("GLD", start="2026-01-01", end="2026-03-01",
                          regime_col="regime", parquet_path=str(p))
    assert "regime" in summary.columns
    # We expect rows for at least 3 (month, regime) pairs + an ALL row
    assert len(summary[summary["__month"] != "ALL"]) >= 3


def test_empty_window_returns_empty(synthetic_oos):
    summary = run_audit("GLD", start="2030-01-01", end="2030-02-01",
                          parquet_path=synthetic_oos)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0


def test_missing_actual_columns_raises(tmp_path):
    """Hard-fail when the parquet lacks the authoritative actual_* labels.

    Critical: we refuse to substitute single-day overnight returns because
    that was the original v3.7.232 draft's bug.
    """
    dates = pd.bdate_range("2026-01-05", periods=5)
    df_bad = pd.DataFrame({
        "pred_upper_pct": [+4.0] * 5,
        "pred_lower_pct": [-4.0] * 5,
        # Intentionally omit actual_upper_pct + actual_lower_pct
    }, index=dates)
    p = tmp_path / "oos_bad.parquet"
    df_bad.to_parquet(p)
    with pytest.raises(ValueError,
                          match="5-day forward H/L per src/models/train_dl_range"):
        run_audit("GLD", start="2026-01-01", end="2026-02-01",
                    parquet_path=str(p))


def test_safe_ratio_with_zero_actual(tmp_path):
    """If actual mean is near zero, width_ratio returns NaN (not Inf)."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    df = pd.DataFrame({
        "pred_upper_pct":   [+4.0] * 5,
        "pred_lower_pct":   [-4.0] * 5,
        "actual_upper_pct": [0.0] * 5,
        "actual_lower_pct": [0.0] * 5,
    }, index=dates)
    p = tmp_path / "oos_zero.parquet"
    df.to_parquet(p)
    summary = run_audit("GLD", start="2026-01-01", end="2026-02-01",
                          parquet_path=str(p))
    agg = summary[summary["__month"] == "ALL"].iloc[0]
    assert pd.isna(agg["width_ratio_upper"])
    assert pd.isna(agg["width_ratio_lower"])
    # Coverage rates are still well-defined
    assert agg["coverage_upper"] == 1.0  # 0 <= 4
    assert agg["coverage_lower"] == 1.0  # 0 >= -4


def test_coverage_formula_matches_eval_range(tmp_path):
    """Coverage definition must match src/models/train_dl_range.eval_range:
    upper_covered = actual_upper <= pred_upper
    lower_covered = actual_lower >= pred_lower
    """
    dates = pd.bdate_range("2026-01-05", periods=4)
    df = pd.DataFrame({
        "pred_upper_pct":   [+5.0, +5.0, +5.0, +5.0],
        "pred_lower_pct":   [-5.0, -5.0, -5.0, -5.0],
        # Row 0: both inside (covered)
        # Row 1: upper breaches (not covered)
        # Row 2: lower breaches (not covered)
        # Row 3: both breach
        "actual_upper_pct": [+3.0, +6.0, +3.0, +6.0],
        "actual_lower_pct": [-3.0, -3.0, -6.0, -6.0],
    }, index=dates)
    p = tmp_path / "oos_cov.parquet"
    df.to_parquet(p)
    summary = run_audit("GLD", start="2026-01-01", end="2026-02-01",
                          parquet_path=str(p))
    agg = summary[summary["__month"] == "ALL"].iloc[0]
    assert agg["coverage_upper"] == 0.5  # rows 0,2 cover; 1,3 breach
    assert agg["coverage_lower"] == 0.5  # rows 0,1 cover; 2,3 breach
    assert agg["coverage_both"] == 0.25  # only row 0


def test_required_columns_constant_is_definitive():
    """Locking the contract: future refactors must not silently drop a label column."""
    assert set(REQUIRED_COLUMNS) == {
        "pred_upper_pct", "pred_lower_pct",
        "actual_upper_pct", "actual_lower_pct",
    }
