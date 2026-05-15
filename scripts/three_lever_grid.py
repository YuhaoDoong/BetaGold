"""v3.7.200 三 lever 过滤 grid (GLD only).

Lever:
  L1: regime smooth_window [20, 30, 45, 60]  (越小, regime 切 Bear 越快)
  L2: regime bear_threshold [-0.05, -0.10, -0.15, -0.20]  (越大, 早判 Bear)
  L3: ma_trend_threshold [0.975, 0.98, 0.985, 0.99, 0.995]  (越高, BC 越严)
  L4: iv_filter_high_min [25, 26, 27, 28]  (越低, 高 IV 路径越早触发)

Truth: 全历史 BC 信号 × 5日 spot P&L (用 generate_daily_signals 各种 cfg 组合)

输出每个变种:
  n_BC_kept, WR, sum%, mean%, max_loss
  blocked_2026q1_count (本应 13 笔, 越多越好)
  blocked_5_12_14 (本应 3 笔, 越多越好)
  to_SP_count (BC 转 SP, 高 IV 深破)

测试方式: independent (其他 lever 保持当前默认)
"""
from __future__ import annotations
import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import yfinance as yf

from core.data import load_oos_predictions, load_config
import core.signals_v2 as sv2
import core.strategy_config as sc
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier


def build_inputs():
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high = ohlc.loc[common,"High"]; low = ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    return close, high, low, upper, lower, rv_p, feat, feat_cols, ohlc, gvz["Close"]


def evaluate_signals(bc: pd.DataFrame, ohlc: pd.DataFrame) -> dict:
    """5日 spot P&L 评估 BC 信号集. 单独跟踪 Q1 / 5/12-14 是否在 bc 集合里."""
    pnls = []
    q1_kept = 0; may_kept = 0
    # 先统计 kept (不管有没有 5d future), 后单独算 spot P&L
    for sig_d in bc.index:
        if "2026-01-01" <= str(sig_d.date()) <= "2026-03-31":
            q1_kept += 1
        if str(sig_d.date()) in ("2026-05-12", "2026-05-13", "2026-05-14"):
            may_kept += 1
        if sig_d not in ohlc.index: continue
        idx = ohlc.index.get_loc(sig_d)
        if idx + 5 >= len(ohlc): continue  # 无 5d future → 不算 PnL 但已计 kept
        entry = float(ohlc.iloc[idx]["Open"])
        exit_p = float(ohlc.iloc[idx+5]["Close"])
        pnls.append((sig_d, (exit_p/entry - 1) * 100))
    ps = pd.Series([p for _,p in pnls]) if pnls else pd.Series([], dtype=float)
    return {
        "n_BC_total": len(bc),                # 包含没 5d future 的
        "n_BC_with_pnl": len(ps),             # 算 PnL 的
        "WR%": round((ps>0).mean()*100, 1) if len(ps) else None,
        "sum%": round(ps.sum(), 1) if len(ps) else None,
        "mean%": round(ps.mean(), 2) if len(ps) else None,
        "max_loss%": round(ps.min(), 1) if len(ps) else None,
        "blocked_Q1": 13 - q1_kept,           # 13 笔 Q1 baseline (no IV filter)
        "blocked_5_12_14": 3 - may_kept,      # 3 笔 May baseline
        "Q1_kept": q1_kept, "may_kept": may_kept,
    }


def run_variant(lever_name, value, inputs, baseline_kwargs):
    close, high, low, upper, lower, rv_p, feat, feat_cols, ohlc, gvz = inputs
    # regime: rebuild with current params
    rc_kwargs = {"bull_threshold": 0.2, "bear_threshold": -0.2,
                  "smooth_window": 60, "min_hold_days": 20}
    if lever_name == "smooth_window": rc_kwargs["smooth_window"] = value
    if lever_name == "bear_threshold": rc_kwargs["bear_threshold"] = value
    regime = RegimeClassifier(**rc_kwargs).classify(feat.loc[close.index, feat_cols])["regime"]

    # monkeypatch GLD cfg per lever
    import core.strategy_config as scfg
    orig_gld = copy.deepcopy(scfg.ASSET_CONFIGS["GLD"])
    if lever_name == "ma_trend_threshold":
        scfg.ASSET_CONFIGS["GLD"].ma_trend_threshold = value
    if lever_name == "iv_filter_high_min":
        scfg.ASSET_CONFIGS["GLD"].iv_filter_high_min = value
    # signals_v2 用 _ac = ASSET_CONFIGS[asset], 改全局 cfg 即可

    sig = sv2.generate_daily_signals(close, high, low, upper, lower,
                                            regime, rv_p, asset="GLD",
                                            gvz_series=gvz)
    bc = sig[(sig["buy_type"] == "BUY CALL") & sig["buy_signal"]]
    res = evaluate_signals(bc, ohlc)
    res["lever"] = lever_name; res["value"] = value
    # 还原
    scfg.ASSET_CONFIGS["GLD"] = orig_gld
    return res


def main():
    inputs = build_inputs()
    print("Loading inputs...done\n")

    # 0. NO-FILTER baseline (不带 gvz, regime default, 看原始信号集大小)
    close, high, low, upper, lower, rv_p, feat, feat_cols, ohlc, gvz = inputs
    regime0 = RegimeClassifier().classify(feat.loc[close.index, feat_cols])["regime"]
    sig_nofilter = sv2.generate_daily_signals(close, high, low, upper, lower,
                                                       regime0, rv_p, asset="GLD",
                                                       gvz_series=None)
    bc_nofilter = sig_nofilter[(sig_nofilter["buy_type"]=="BUY CALL") & sig_nofilter["buy_signal"]]
    nf = evaluate_signals(bc_nofilter, ohlc)
    print(f"NO-FILTER (gvz=None, current other defaults): {nf}\n")

    # 1. Current production baseline
    base = run_variant("BASELINE", "current", inputs, {})
    print(f"PROD-BASELINE (with IV filter, current cfg): {base}\n")

    rows = [base]
    grids = {
        "smooth_window": [20, 30, 45, 60],
        "bear_threshold": [-0.05, -0.10, -0.15, -0.20],
        "ma_trend_threshold": [0.975, 0.98, 0.985, 0.99, 0.995],
        "iv_filter_high_min": [25, 26, 27, 28],
    }
    for lever, values in grids.items():
        print(f"--- {lever} grid ---")
        for v in values:
            r = run_variant(lever, v, inputs, {})
            print(f"  {lever}={v}: n={r.get('n_BC_kept','?')} WR={r.get('WR%','?')}% "
                  f"sum={r.get('sum%','?')}% blocked_Q1={r.get('blocked_Q1','?')}/13 "
                  f"blocked_5_12_14={r.get('blocked_5_12_14','?')}/3 "
                  f"max_loss={r.get('max_loss%','?')}%")
            rows.append(r)
        print()

    df = pd.DataFrame(rows)
    print("=" * 100)
    print(df[["lever","value","n_BC_kept","WR%","sum%","mean%","max_loss%",
                 "blocked_Q1","blocked_5_12_14"]].to_string(index=False))
    out = "/Users/yhdong/Gold/data/backtest_history/three_lever_grid.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
