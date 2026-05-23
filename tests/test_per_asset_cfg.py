"""Regression tests for v3.7.238: per-asset option-exit config resolver.

Verifies that ``get_option_exit_config(asset, strategy)`` returns the right
dataclass type, applies per-asset overrides where registered, and that
``simulate_option_exit`` emits a DeprecationWarning when called without an
``asset`` argument.
"""
from __future__ import annotations
import warnings
import pandas as pd
import pytest

from core.strategy_config import get_option_exit_config
from core.strategies.buy_call import BCConfig
from core.strategies.sell_put import SPConfig
from core.strategies.straddle import StraddleConfig
from core.strategies.short_vol import ShortVolConfig
from core.paper_positions import simulate_option_exit


def test_default_bc_config_no_per_asset_override():
    gld = get_option_exit_config("GLD", "BUY CALL")
    slv = get_option_exit_config("SLV", "BUY CALL")
    assert isinstance(gld, BCConfig) and isinstance(slv, BCConfig)
    # BC has no per-asset overrides; both should equal the dataclass default.
    assert gld.profit_target_mult == BCConfig().profit_target_mult
    assert slv.profit_target_mult == BCConfig().profit_target_mult


def test_sp_per_asset_override_gld_vs_slv():
    gld = get_option_exit_config("GLD", "SELL PUT")
    slv = get_option_exit_config("SLV", "SELL PUT")
    assert isinstance(gld, SPConfig) and isinstance(slv, SPConfig)
    # v3.7.184 per-asset split: GLD pt=70, SLV pt=30. Resolver carries this.
    assert gld.profit_target_credit_pct == 70.0
    assert slv.profit_target_credit_pct == 30.0
    # Other fields stay at dataclass defaults
    assert gld.stop_loss_margin_pct == SPConfig().stop_loss_margin_pct
    assert slv.base_dte == SPConfig().base_dte


def test_straddle_and_short_vol_return_right_types():
    s = get_option_exit_config("GLD", "STRADDLE")
    sv = get_option_exit_config("SLV", "SHORT_VOL")
    assert isinstance(s, StraddleConfig)
    assert isinstance(sv, ShortVolConfig)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError, match="Unknown option strategy"):
        get_option_exit_config("GLD", "NOT_A_STRATEGY")


def test_simulate_option_exit_without_asset_warns():
    """Calling without asset must DeprecationWarn (v3.7.238 contract)."""
    entry_pricing = {"legs": [("long_call", "US.GLD260515C400000", 400.0, 1)],
                      "entry_price": 1.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simulate_option_exit(entry_pricing, pd.Timestamp("2026-05-22"),
                                "BUY CALL", pd.Timestamp("2026-05-22"))
        assert any(issubclass(rec.category, DeprecationWarning) and
                    "asset" in str(rec.message).lower()
                    for rec in w), (
            "expected DeprecationWarning mentioning asset when "
            "simulate_option_exit called without asset")


def test_simulate_option_exit_with_asset_no_warn():
    """Calling with asset must NOT emit DeprecationWarning."""
    entry_pricing = {"legs": [("long_call", "US.GLD260515C400000", 400.0, 1)],
                      "entry_price": 1.0}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simulate_option_exit(entry_pricing, pd.Timestamp("2026-05-22"),
                                "BUY CALL", pd.Timestamp("2026-05-22"),
                                asset="GLD")
        dep = [rec for rec in w if issubclass(rec.category, DeprecationWarning)
                and "asset" in str(rec.message).lower()]
        assert not dep, f"unexpected DeprecationWarning(s): {[str(r.message) for r in dep]}"
