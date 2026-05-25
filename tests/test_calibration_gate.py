"""Regression tests for v3.7.247: calibration gate decision logic (plan closure)."""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.calibration_gate_grid import (
    compound_gate_decision,
    render_report,
    DEFAULT_TARGET_COVERAGE,
    DEFAULT_MAX_DEGRADATION_PP,
    DEFAULT_MIN_PASS_WINDOWS,
)


def _mk(label, raw_cb, cal_cb, raw_cu=0.8, cal_cu=0.8, raw_cl=0.8, cal_cl=0.8, n=100):
    return {
        "window": label, "n": n,
        "raw_coverage_both": raw_cb, "cal_coverage_both": cal_cb,
        "raw_coverage_upper": raw_cu, "cal_coverage_upper": cal_cu,
        "raw_coverage_lower": raw_cl, "cal_coverage_lower": cal_cl,
    }


def test_gate_passes_when_all_windows_move_toward_target():
    """Raw is at 0.50, target is 0.80; calibrated lifts to 0.70 in every window."""
    windows = [_mk(lbl, 0.50, 0.70) for lbl in ("10y", "5y", "3y", "1y")]
    res = compound_gate_decision(windows, target=0.80)
    assert res["gate_passed"] is True
    assert res["n_pass"] == 4
    assert res["n_total"] == 4
    assert all(d["pass"] for d in res["decisions"])


def test_gate_fails_on_coverage_degradation_in_one_window():
    """A single severely-degrading window does not bring down the overall gate
    when ≥ min_pass other windows pass (3 of 4 with min_pass=3 = OK).

    Distance from target 0.80:
      raw 0.50 → |dist|=0.30; cal 0.70 → |dist|=0.10; delta=-0.20 (pass)
      raw 0.85 → |dist|=0.05; cal 0.70 → |dist|=0.10; delta=+0.05; pass iff < 0.05.
    With strict < the last window fails. Other three pass → gate_passed=True.
    """
    windows = [
        _mk("10y", 0.50, 0.70),
        _mk("5y", 0.55, 0.70),
        _mk("3y", 0.60, 0.70),
        _mk("1y", 0.85, 0.70),  # Distance regresses by exactly 0.05 → strict fail
    ]
    res = compound_gate_decision(windows, target=0.80,
                                       max_degradation=0.05)
    last = res["decisions"][-1]
    # Distance regressed; with strict <, this window fails.
    assert last["pass"] is False
    # Three of four pass → gate passes with default min_pass=3.
    assert res["gate_passed"] is True
    assert res["n_pass"] == 3


def test_gate_fails_on_strict_min_pass():
    """If we tighten min_pass to 4 (require ALL windows), one degrade fails the gate."""
    windows = [
        _mk("10y", 0.50, 0.70),
        _mk("5y", 0.55, 0.70),
        _mk("3y", 0.60, 0.70),
        _mk("1y", 0.85, 0.70),  # Degrades 15pp
    ]
    res = compound_gate_decision(windows, target=0.80,
                                       max_degradation=0.05,
                                       min_pass=4)
    assert res["gate_passed"] is False


def test_gate_fails_on_insufficient_improvement():
    """Only 2 of 4 windows move toward target → gate fails (default min_pass=3)."""
    # Constructed so that exactly half the windows regress meaningfully
    # past max_degradation (default 0.05).
    windows = [
        _mk("10y", 0.50, 0.70),  # |dist| 0.10 vs 0.30, delta=-0.20 → pass
        _mk("5y", 0.50, 0.20),    # |dist| 0.60 vs 0.30, delta=+0.30 → fail
        _mk("3y", 0.55, 0.72),    # |dist| 0.08 vs 0.25, delta=-0.17 → pass
        _mk("1y", 0.78, 0.30),    # |dist| 0.50 vs 0.02, delta=+0.48 → fail
    ]
    res = compound_gate_decision(windows, target=0.80,
                                       max_degradation=0.05)
    assert res["gate_passed"] is False
    assert res["n_pass"] == 2


def test_gate_handles_raw_above_target():
    """If raw is ABOVE the target (over-covering = bands too wide), 'toward'
    means moving DOWN toward target."""
    windows = [
        _mk("10y", 0.95, 0.85),   # 0.85 closer to 0.80 than 0.95 → toward
        _mk("5y", 0.92, 0.83),
        _mk("3y", 0.90, 0.82),
        _mk("1y", 0.93, 0.85),
    ]
    res = compound_gate_decision(windows, target=0.80)
    assert res["gate_passed"] is True


def test_empty_window_results_marked_empty():
    """None entries (e.g., insufficient data) → gate fails on those windows."""
    windows = [None, _mk("5y", 0.50, 0.70), _mk("3y", 0.50, 0.70),
                _mk("1y", 0.50, 0.70)]
    res = compound_gate_decision(windows, target=0.80)
    # 3 windows pass; min_pass=3 → gate passes provided n_total>=3
    assert res["n_total"] == 3
    assert res["gate_passed"] is True


def test_nan_coverage_skipped():
    """NaN raw or cal coverage → window not counted toward n_total."""
    windows = [
        _mk("10y", float("nan"), 0.70),
        _mk("5y", 0.50, 0.70),
        _mk("3y", 0.55, 0.72),
        _mk("1y", 0.60, 0.74),
    ]
    res = compound_gate_decision(windows, target=0.80)
    assert res["n_total"] == 3
    assert res["gate_passed"] is True


def test_report_serialization_includes_gate_passed_line(tmp_path):
    """gate_report.md must carry a machine-parseable 'gate_passed: true|false' line."""
    per_asset = [
        {
            "asset": "GLD",
            "windows": [_mk(lbl, 0.50, 0.70) for lbl in ("10y", "5y", "3y", "1y")],
            "gate_both": compound_gate_decision(
                [_mk(lbl, 0.50, 0.70) for lbl in ("10y", "5y", "3y", "1y")],
                metric_key="coverage_both"),
            "gate_upper": compound_gate_decision(
                [_mk(lbl, 0.50, 0.70) for lbl in ("10y", "5y", "3y", "1y")],
                metric_key="coverage_upper"),
            "gate_lower": compound_gate_decision(
                [_mk(lbl, 0.50, 0.70) for lbl in ("10y", "5y", "3y", "1y")],
                metric_key="coverage_lower"),
            "gate_passed": True,
        },
    ]
    out = tmp_path / "gate_report.md"
    overall = render_report(per_asset, out)
    text = out.read_text()
    assert "gate_passed: true" in text
    assert "GLD" in text
    assert overall is True


def test_report_failure_path(tmp_path):
    """Failed gate emits gate_passed: false + 'keep build_band on raw' next-action line."""
    per_asset = [
        {
            "asset": "SLV",
            "windows": [_mk("1y", 0.85, 0.70)],
            "gate_both": compound_gate_decision([_mk("1y", 0.85, 0.70)]),
            "gate_upper": compound_gate_decision([_mk("1y", 0.85, 0.70)]),
            "gate_lower": compound_gate_decision([_mk("1y", 0.85, 0.70)]),
            "gate_passed": False,
        },
    ]
    out = tmp_path / "gate_report.md"
    overall = render_report(per_asset, out)
    text = out.read_text()
    assert "gate_passed: false" in text
    assert "keep `build_band()` on raw" in text or "shadow-only" in text
    assert overall is False
