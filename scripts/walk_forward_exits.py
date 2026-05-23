"""v3.7.225: 退出策略参数 walk-forward — BC/SP/Futures exit params.

参数 (生产值):
  BUY CALL:    pt=2.5x, sl=0.3x, hold_max=30d
  SELL PUT:    pt=70% credit (GLD) / 30% (SLV), sl=100% margin
  FUTURES:     leverage=5x (B) / 10x (S/A), hold_max=45d, TP=200%, SL=100% margin

方法:
  对每个 SLV-S 信号日 (n=23) + GLD 自家 BUY 信号 (n=93), 跑参数 grid,
  walk-forward train/test, 看 OOS scoreB.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.paper_positions import price_strategy_at, simulate_option_exit
from core.strategies.buy_call import BCConfig
from core.strategies.sell_put import SPConfig


def get_buy_signals(asset):
    cfg = load_config()
    if asset == "GLD":
        oos = load_oos_predictions(cfg)
    else:
        oos = pd.read_parquet(Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet")
    feat_path = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
                  if asset == "GLD" else
                  "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    feat = pd.read_parquet(feat_path)
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high = ohlc.loc[common,"High"]; low = ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier(min_hold_days=1).classify(
        feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset=asset, gvz_series=gvz["Close"])
    return sig[sig["buy_signal"]], ohlc


_PATCHED_CLASSES = {}    # cls -> orig_init
_INSTANCE_SAVES = []     # [(instance, key, old_value)]


def _patch_init(cls, overrides):
    if cls in _PATCHED_CLASSES:
        return
    orig_init = cls.__init__
    _PATCHED_CLASSES[cls] = orig_init
    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
    cls.__init__ = patched


def _save_and_set(instance, k, v):
    if hasattr(instance, k):
        _INSTANCE_SAVES.append((instance, k, getattr(instance, k)))
        setattr(instance, k, v)


def _restore_all():
    for cls, orig in list(_PATCHED_CLASSES.items()):
        cls.__init__ = orig
        del _PATCHED_CLASSES[cls]
    for instance, k, v in _INSTANCE_SAVES:
        setattr(instance, k, v)
    _INSTANCE_SAVES.clear()


def _apply_cfg_overrides(strategy, asset, overrides):
    if not overrides: return False
    if strategy == "BUY CALL":
        from core.strategies.buy_call import BCConfig
        _patch_init(BCConfig, overrides)
    elif strategy == "SELL PUT":
        from core.strategies.sell_put import SPConfig
        _patch_init(SPConfig, overrides)
        from core.strategy_configs import SELL_PUT_GLD, SELL_PUT_SLV
        target = SELL_PUT_GLD if asset == "GLD" else SELL_PUT_SLV
        for k, v in overrides.items():
            _save_and_set(target, k, v)
    elif strategy == "STRADDLE":
        from core.strategies.straddle import StraddleConfig
        _patch_init(StraddleConfig, overrides)
    return True


def _restore_overrides(_unused):
    _restore_all()


def backtest_strategy(buy_dates, ohlc, strategy, asset, cfg_overrides=None):
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    pnls = []
    saved = _apply_cfg_overrides(strategy, asset, cfg_overrides)
    try:
        for d in buy_dates:
            if d not in ohlc.index: continue
            eO = float(ohlc.loc[d,"Open"]); eC = float(ohlc.loc[d,"Close"])
            eH = float(ohlc.loc[d,"High"]); eL = float(ohlc.loc[d,"Low"])
            ent = price_strategy_at(asset, strategy, d,
                                          d + pd.Timedelta(hours=9, minutes=30),
                                          eO, eO, eC, eH, eL, dte_target=30)
            if not ent.get("legs"): continue
            sim = simulate_option_exit(ent, d, strategy, today,
                                              live_spot=eC, live_high=eH, live_low=eL)
            if sim.get("is_closed"):
                pnls.append({"date": d, "pnl": float(sim.get("pnl_pct", 0) or 0),
                              "hold": int(sim.get("hold_days", 0) or 0),
                              "reason": sim.get("exit_reason", "")})
    finally:
        _restore_overrides(saved)
    return pnls


def score(pnls):
    if not pnls: return {"n": 0, "WR": None, "mean": None, "sum": None, "scoreB": 0}
    s = pd.Series([p["pnl"] for p in pnls])
    wr = (s > 0).mean()
    mean = s.mean()
    return {"n": len(pnls), "WR": round(wr*100, 1), "mean": round(mean, 2),
              "sum": round(s.sum(), 1), "scoreB": round((wr**2) * math.log(1+len(pnls)) * mean, 2),
              "max_loss": round(s.min(), 1)}


def walk_forward_exit_param(asset, strategy, param, grid, train_years=4):
    buy_sig, ohlc = get_buy_signals(asset)
    print(f"  baseline: {asset} {strategy} 全期 BUY 信号 {len(buy_sig)} 笔")
    all_years = sorted(set(buy_sig.index.year))
    if len(all_years) < train_years + 1:
        print(f"  ⚠️ 年份不足 ({len(all_years)})")
        return None
    folds = []
    for test_year in range(all_years[train_years], all_years[-1] + 1):
        train_start = test_year - train_years
        train_dates = buy_sig.index[
            (buy_sig.index.year >= train_start) & (buy_sig.index.year < test_year)]
        test_dates = buy_sig.index[buy_sig.index.year == test_year]
        if len(train_dates) < 8 or len(test_dates) < 3:
            continue
        # train grid
        train_results = []
        for v in grid:
            pnls = backtest_strategy(train_dates, ohlc, strategy, asset, {param: v})
            train_results.append({"v": v, **score(pnls)})
        valid = [r for r in train_results if r["n"] >= 5]
        if not valid: continue
        best = max(valid, key=lambda r: r["scoreB"])
        # test
        test_pnls = backtest_strategy(test_dates, ohlc, strategy, asset,
                                            {param: best["v"]})
        test_perf = score(test_pnls)
        # prod baseline
        prod_pnls = backtest_strategy(test_dates, ohlc, strategy, asset, {})
        prod_perf = score(prod_pnls)
        folds.append({
            "test_year": test_year, "train": f"{train_start}-{test_year-1}",
            "best_v": best["v"], "train_n": best["n"], "train_WR": best["WR"],
            "train_scoreB": best["scoreB"],
            "test_n": test_perf["n"], "test_WR": test_perf["WR"],
            "test_sum": test_perf["sum"], "test_scoreB": test_perf["scoreB"],
            "prod_n": prod_perf["n"], "prod_WR": prod_perf["WR"],
            "prod_scoreB": prod_perf["scoreB"],
        })
    return pd.DataFrame(folds)


TESTS = [
    # (asset, strategy, param, grid)
    ("GLD", "BUY CALL", "profit_target_mult", [1.5, 2.0, 2.5, 3.0, 4.0]),
    ("GLD", "BUY CALL", "stop_loss_mult", [0.2, 0.3, 0.5, 0.7]),
    ("GLD", "BUY CALL", "hold_max_days", [10, 15, 21, 30, 45]),
    ("GLD", "SELL PUT", "profit_target_credit_pct", [30, 50, 70, 90]),
    ("SLV", "BUY CALL", "profit_target_mult", [1.5, 2.0, 2.5, 3.0]),
    ("SLV", "SELL PUT", "profit_target_credit_pct", [20, 30, 50, 70]),
]


def main():
    print("=" * 100)
    print("Task C: 退出策略 walk-forward (BC / SP exit params)")
    print("=" * 100)
    for asset, strat, param, grid in TESTS:
        print(f"\n--- {asset} {strat} :: {param} ---")
        print(f"   grid: {grid}")
        df = walk_forward_exit_param(asset, strat, param, grid)
        if df is None or not len(df):
            print("  ⚠️ skip"); continue
        print(df.to_string(index=False))
        best_dist = df["best_v"].value_counts().to_dict()
        avg_test = df[df["test_n"] > 0]["test_scoreB"].mean()
        avg_prod = df[df["prod_n"] > 0]["prod_scoreB"].mean()
        print(f"  跨 fold best 分布: {best_dist}")
        print(f"  avg test scoreB: {avg_test:.2f}  vs prod: {avg_prod:.2f}")
        out = Path(f"/Users/yhdong/Gold/data/backtest_history/"
                     f"wf_exit_{asset}_{strat.replace(' ','')}_{param}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
