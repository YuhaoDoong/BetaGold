"""Regression tests for v3.7.249: Dashboard `run_backtest` parity harness.

Verifies the parity contract for the deprecation wrapper:
- intraday exit-event counts match within ±1 between legacy and unified paths
- signal-column drift recorded with documented attribution
- catastrophic exit-event count divergence reported as failure
- DeprecationWarning emitted on legacy `run_backtest` import path
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import pytest

from core.dashboard_parity import (
    _replay_one_pass,
    parity_check,
    ParityVerdict,
    EXIT_TYPES,
)


def _synthetic_fixture(n=60, seed=0):
    """60 trading days with deterministic close/high/low + reasonable bands."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=n)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, n)), index=dates)
    high = close + 0.5 + rng.uniform(0, 0.3, n)
    low = close - 0.5 - rng.uniform(0, 0.3, n)
    upper_band = close + 2.5
    lower_band = close - 2.5
    return close, high, low, upper_band, lower_band


# -- Identical-pipeline equivalence -------------------------------------------


def test_identical_signals_produce_identical_event_counts():
    """When legacy and unified are fed the SAME signals, event counts match."""
    close, high, low, ub, lb = _synthetic_fixture()
    buy_sig = pd.Series(False, index=close.index)
    # Inject 3 entries at known dates so the replay produces deterministic trades
    buy_sig.iloc[5] = True
    buy_sig.iloc[20] = True
    buy_sig.iloc[40] = True
    exit_sig = pd.Series(False, index=close.index)

    legacy_trades = _replay_one_pass(close, high, low, ub, lb,
                                          buy_sig, exit_sig)
    unified_trades = _replay_one_pass(close, high, low, ub, lb,
                                            buy_sig, exit_sig)
    verdict = parity_check(legacy_trades, unified_trades,
                              legacy_buy_signal=buy_sig,
                              unified_buy_signal=buy_sig)
    assert verdict.status == "PASS"
    assert verdict.max_count_drift == 0


# -- PASS_WITH_DRIFT path (signal-column drift, but exit events still match) --


def test_signal_drift_recorded_in_attribution_csv(tmp_path):
    """When the unified path admits a signal that legacy didn't, parity status
    becomes PASS_WITH_DRIFT and the attribution CSV is written."""
    close, high, low, ub, lb = _synthetic_fixture()
    legacy_buy = pd.Series(False, index=close.index)
    legacy_buy.iloc[5] = True
    unified_buy = legacy_buy.copy()
    unified_buy.iloc[5] = False         # unified rejected what legacy accepted
    unified_buy.iloc[6] = True           # unified accepted what legacy rejected
    exit_sig = pd.Series(False, index=close.index)

    legacy_trades = _replay_one_pass(close, high, low, ub, lb,
                                          legacy_buy, exit_sig)
    unified_trades = _replay_one_pass(close, high, low, ub, lb,
                                            unified_buy, exit_sig)
    attr = tmp_path / "signal_drift_attribution.csv"
    verdict = parity_check(legacy_trades, unified_trades,
                              legacy_buy_signal=legacy_buy,
                              unified_buy_signal=unified_buy,
                              drift_attribution_path=str(attr))
    # Event counts only differ by entry timing (legacy entered at 5, unified
    # at 6); both produce one trade typed Timeout (max_hold_days=30 with
    # synthetic close so no StopLoss/BandExit/Pullback fires) → count drift 0.
    assert verdict.status == "PASS_WITH_DRIFT"
    assert attr.exists()
    df = pd.read_csv(attr)
    # Two drifted dates: index 5 (legacy True, unified False) and 6 (vice versa)
    assert len(df) == 2
    assert set(df.columns) == {
        "signal_date", "legacy_buy_signal", "unified_buy_signal", "attribution"
    }


def test_signal_drift_without_attribution_path_does_not_write():
    """Optional attribution path: omitting it means no file is created."""
    close, high, low, ub, lb = _synthetic_fixture()
    legacy_buy = pd.Series(False, index=close.index); legacy_buy.iloc[5] = True
    unified_buy = pd.Series(False, index=close.index); unified_buy.iloc[6] = True
    exit_sig = pd.Series(False, index=close.index)
    legacy_trades = _replay_one_pass(close, high, low, ub, lb, legacy_buy, exit_sig)
    unified_trades = _replay_one_pass(close, high, low, ub, lb, unified_buy, exit_sig)
    verdict = parity_check(legacy_trades, unified_trades,
                              legacy_buy_signal=legacy_buy,
                              unified_buy_signal=unified_buy)
    assert verdict.status == "PASS_WITH_DRIFT"
    assert verdict.attribution_path is None


# -- FAIL_EXIT_EVENT_COUNT path -----------------------------------------------


def test_excess_event_count_drift_fails():
    """If exit-event counts differ by more than ±1, status is FAIL."""
    # Construct fake trade lists with very different exit-type counts
    legacy_trades = [{"exit_type": "StopLoss"} for _ in range(2)]
    unified_trades = [{"exit_type": "StopLoss"} for _ in range(8)]
    verdict = parity_check(legacy_trades, unified_trades)
    assert verdict.status == "FAIL_EXIT_EVENT_COUNT"
    assert verdict.max_count_drift == 6


def test_drift_at_threshold_passes():
    """Exactly ±1 drift is the AC contract — must still PASS."""
    legacy_trades = [{"exit_type": "Pullback"} for _ in range(5)]
    unified_trades = [{"exit_type": "Pullback"} for _ in range(6)]
    verdict = parity_check(legacy_trades, unified_trades)
    assert verdict.status == "PASS"
    assert verdict.max_count_drift == 1


# -- Schema lock --------------------------------------------------------------


def test_exit_types_constant_is_complete():
    """Future event-type additions must update the EXIT_TYPES tuple AND the tests."""
    assert set(EXIT_TYPES) == {"StopLoss", "BandExit", "Pullback", "Timeout"}


def test_event_count_dict_has_every_known_type():
    verdict = parity_check([], [])
    for t in EXIT_TYPES:
        assert t in verdict.legacy_event_counts
        assert t in verdict.unified_event_counts
        assert verdict.legacy_event_counts[t] == 0


# -- DeprecationWarning on legacy run_backtest -------------------------------


def test_legacy_run_backtest_emits_deprecation_warning():
    """Importing core.signals_v2 and calling run_backtest emits the warning."""
    from core.signals_v2 import run_backtest
    close, high, low, ub, lb = _synthetic_fixture()
    regime = pd.Series("Bull", index=close.index)
    rv_pctile = pd.Series(0.5, index=close.index)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            run_backtest(close, high, low, ub, lb, regime, rv_pctile)
        except Exception:
            # Body may error on the synthetic fixture; the DeprecationWarning
            # fires BEFORE any body execution, so we only need to verify the warning.
            pass
        deps = [rec for rec in w
                if issubclass(rec.category, DeprecationWarning)
                and "run_backtest" in str(rec.message)]
        assert deps, (
            "expected DeprecationWarning from legacy run_backtest; got: "
            + str([str(r.message) for r in w])
        )
