"""v3.7.229 方向性期权 Layer 2: BC + SP, trailing 1y/6m/3m.

数据: kline_db (2025-04-29 → 2026-05-06), 1y 限制
跨窗: 1y / 6m / 3m, 看 BC pt/sl, SP pt 参数稳定性, 各 tier 信号下哪种工具最优.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.backtest.framework import (build_raw_universe, trailing_slice,
                                              LAYER2_WINDOWS,
                                              run_layer2_backtest_with_disposition)
from core.paper_positions import price_strategy_at, simulate_option_exit


_PATCHED = {}

def patch_class(cls, overrides):
    if cls in _PATCHED: return
    orig = cls.__init__
    _PATCHED[cls] = orig
    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        for k, v in overrides.items():
            if hasattr(self, k): setattr(self, k, v)
    cls.__init__ = patched


def unpatch_all():
    for cls, orig in list(_PATCHED.items()):
        cls.__init__ = orig
        del _PATCHED[cls]


def backtest_option(dates, ohlc, asset, strategy, overrides=None):
    """v3.7.242: returns (closed_df, disposition_dict).

    Uses run_layer2_backtest_with_disposition so unclosed/stale positions are
    counted explicitly rather than silently dropped.
    """
    if overrides:
        if strategy == "BUY CALL":
            from core.strategies.buy_call import BCConfig
            patch_class(BCConfig, overrides)
        elif strategy == "SELL PUT":
            from core.strategies.sell_put import SPConfig
            patch_class(SPConfig, overrides)
    try:
        closed_df, disp = run_layer2_backtest_with_disposition(
            dates, ohlc, asset, strategy,
            price_fn=price_strategy_at, exit_fn=simulate_option_exit,
            dte_target=30)
        # Backward compat: keep 'pnl' as the column name some downstream
        # code expects (score(df['pnl']) below).
        if "pnl_pct" in closed_df.columns:
            closed_df = closed_df.rename(columns={"pnl_pct": "pnl",
                                                       "exit_reason": "reason"})
    finally:
        if overrides: unpatch_all()
    return closed_df, disp


def score(s):
    if not len(s): return {"n": 0, "scoreB": 0}
    n = len(s); wr = (s > 0).mean()
    return {"n": n, "WR": round(wr*100, 1), "mean": round(s.mean(), 2),
              "sum": round(s.sum(), 1), "max_loss": round(s.min(), 1),
              "scoreB": round((wr**2) * math.log(1+n) * s.mean(), 2)}


def get_dates(raw, tier):
    bs = raw["buy_signal"].fillna(False).astype(bool)
    st = raw["signal_tier"].fillna("")
    if tier == "ALL": return raw.index[bs].tolist()
    if tier == "S+A": return raw.index[bs & st.isin(["S","A"])].tolist()
    return raw.index[bs & (st == tier)].tolist()


def grid_bc_pt(asset, ohlc, dates):
    """BC pt grid in given dates. Disposition of the LAST grid entry is
    kept (all grid entries share the same date list, so disposition is
    identical across pt values)."""
    rows = []; last_disp = None
    for pt in [1.5, 2.0, 2.5, 3.0, 4.0]:
        df, disp = backtest_option(dates, ohlc, asset, "BUY CALL",
                                  {"profit_target_mult": pt})
        last_disp = disp
        if not len(df): continue
        s = score(df["pnl"]); s["pt"] = pt; rows.append(s)
    return pd.DataFrame(rows), last_disp


def grid_sp_pt(asset, ohlc, dates):
    rows = []; last_disp = None
    for pt in [30, 40, 50, 70, 90]:
        df, disp = backtest_option(dates, ohlc, asset, "SELL PUT",
                                  {"profit_target_credit_pct": pt})
        last_disp = disp
        if not len(df): continue
        s = score(df["pnl"]); s["pt_pct"] = pt; rows.append(s)
    return pd.DataFrame(rows), last_disp


def main():
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history/v3.7.229_layer2_directional")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset} 方向性期权 (trailing 1y/6m/3m)\n{'='*100}")
        raw, ohlc = build_raw_universe(asset)
        for tier in ["S+A", "B", "ALL"]:
            print(f"\n--- tier={tier} ---")
            for win_label, win_days in LAYER2_WINDOWS:
                sub_raw = trailing_slice(raw, win_days)
                dates = get_dates(sub_raw, tier)
                if len(dates) < 3:
                    print(f"  {win_label}: n={len(dates)} 不足跳过"); continue
                # BC pt grid
                df_bc, bc_disp = grid_bc_pt(asset, ohlc, dates)
                # SP pt grid
                df_sp, sp_disp = grid_sp_pt(asset, ohlc, dates)
                # 选 BC best
                bc_best = df_bc.loc[df_bc["scoreB"].idxmax()] if len(df_bc) else None
                sp_best = df_sp.loc[df_sp["scoreB"].idxmax()] if len(df_sp) else None
                bc_str = (f"pt={bc_best['pt']} n={bc_best['n']} WR={bc_best['WR']}% "
                            f"mean={bc_best['mean']}% sum={bc_best['sum']}%") if bc_best is not None else "n/a"
                sp_str = (f"pt={sp_best['pt_pct']}% n={sp_best['n']} WR={sp_best['WR']}% "
                            f"mean={sp_best['mean']}% sum={sp_best['sum']}%") if sp_best is not None else "n/a"
                # v3.7.242: disposition print so survivorship bias is visible
                if bc_disp:
                    print(f"  {win_label} signals={len(dates)} "
                            f"[entered={bc_disp['n_entered']} closed={bc_disp['n_closed']} "
                            f"open={bc_disp['n_open']} skip_stale={bc_disp['n_skipped_stale']} "
                            f"skip_no_contract={bc_disp['n_skipped_no_contract']}]: "
                            f"BC[{bc_str}]  SP[{sp_str}]")
                else:
                    print(f"  {win_label} signals={len(dates)}: BC[{bc_str}]  SP[{sp_str}]")
                _row = {
                    "asset": asset, "tier": tier, "window": win_label,
                    "n_signal": len(dates),
                    "bc_best_pt": bc_best["pt"] if bc_best is not None else None,
                    "bc_n": bc_best["n"] if bc_best is not None else 0,
                    "bc_WR": bc_best["WR"] if bc_best is not None else None,
                    "bc_sum": bc_best["sum"] if bc_best is not None else None,
                    "bc_scoreB": bc_best["scoreB"] if bc_best is not None else 0,
                    "sp_best_pt": sp_best["pt_pct"] if sp_best is not None else None,
                    "sp_n": sp_best["n"] if sp_best is not None else 0,
                    "sp_WR": sp_best["WR"] if sp_best is not None else None,
                    "sp_sum": sp_best["sum"] if sp_best is not None else None,
                    "sp_scoreB": sp_best["scoreB"] if sp_best is not None else 0,
                }
                # v3.7.242 AC-7: full disposition columns
                if bc_disp:
                    _row.update({
                        "n_entered": bc_disp["n_entered"],
                        "n_closed": bc_disp["n_closed"],
                        "n_open": bc_disp["n_open"],
                        "n_skipped_stale": bc_disp["n_skipped_stale"],
                        "n_skipped_no_contract": bc_disp["n_skipped_no_contract"],
                    })
                summary_rows.append(_row)

    df = pd.DataFrame(summary_rows)
    print(f"\n\n=== 方向性期权多窗总览 ===")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "directional_trailing_windows.csv", index=False)


if __name__ == "__main__":
    main()
