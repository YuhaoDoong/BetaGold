"""v3.7.209 多窗口验证所有 cfg 参数

5y / 3y / 1y / 6m / 3m 窗口分别测:
  1. 信号 filter cutoff (rv_pctile_max, ret_20d_min)
  2. IV filter (iv_filter_high_min)
  3. Tier 边界 (S/A 准入)
  4. BC pt/sl (5y BS+real grid)
  5. SP pt/sl
  6. 期货 lev per-tier
  7. 期货 hold_max

目标: 确认局势是否变化, 各窗口最优是否一致.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf
import numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategies.futures_long import FuturesConfig, simulate_long_position
import core.strategy_config as sc
import copy

WINDOWS = [('5y',5*365),('3y',3*365),('1y',365),('6m',180),('3m',90)]


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
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    return (cfg, close, high, low, upper, lower, rv_p, regime,
            gvz["Close"], feat, feat_cols, ohlc)


def windowed_signal_stats(sig: pd.DataFrame, today, label_func):
    """每窗口跑统计 — sig 必须已含 buy_signal."""
    out = {}
    for w_name, w_days in WINDOWS:
        sub = sig[(sig.index >= today - pd.Timedelta(days=w_days)) & sig['buy_signal']]
        out[w_name] = label_func(sub)
    return out


def fmt_row(d):
    """格式化一行结果."""
    return '  '.join(f'{str(v)[:18]:>18}' for v in d.values())


def main():
    (cfg, close, high, low, upper, lower, rv_p, regime, gvz_s,
        feat, feat_cols, ohlc) = build_inputs()
    today = ohlc.index.max()

    # ── 1. 信号 filter cutoff: rv_pctile_max 多窗口 ──
    print("=" * 130)
    print("1. rv_pctile_max grid × 多窗口  (评判: 5d spot WR / sum)")
    print("=" * 130)
    print(f'{"cutoff":>8}  ' + '  '.join(f'{w:>22}' for w,_ in WINDOWS))

    def spot_eval(sig_buy):
        """spot 5d hold 评估."""
        if not len(sig_buy): return 'n=0'
        pnls = []
        for sig_d in sig_buy.index:
            if sig_d not in ohlc.index: continue
            idx = ohlc.index.get_loc(sig_d)
            if idx + 5 >= len(ohlc): continue
            entry = float(ohlc.iloc[idx]['Open'])
            exit_p = float(ohlc.iloc[idx+5]['Close'])
            pnls.append((exit_p/entry - 1) * 100)
        if not pnls: return 'n=0'
        p = pd.Series(pnls)
        return f'n={len(p)} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}%'

    for rv_max in [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5]:
        orig = copy.deepcopy(sc.ASSET_CONFIGS["GLD"])
        sc.ASSET_CONFIGS["GLD"].rv_pctile_max_hard = rv_max
        sig_t = generate_daily_signals(close, high, low, upper, lower,
                                              regime, rv_p, asset="GLD",
                                              gvz_series=gvz_s)
        line = f'rv<{rv_max}  '
        for w_name, w_days in WINDOWS:
            sub = sig_t[(sig_t.index >= today - pd.Timedelta(days=w_days)) & sig_t['buy_signal']]
            line += f'  {spot_eval(sub):>22}'
        print(line)
        sc.ASSET_CONFIGS["GLD"] = orig

    # ── 2. ret_20d_min ──
    print("\n" + "=" * 130)
    print("2. ret_20d_min grid × 多窗口")
    print("=" * 130)
    print(f'{"cutoff":>8}  ' + '  '.join(f'{w:>22}' for w,_ in WINDOWS))
    for ret_min in [-1.0, -0.10, -0.05, -0.03, -0.01, 0.0]:
        orig = copy.deepcopy(sc.ASSET_CONFIGS["GLD"])
        sc.ASSET_CONFIGS["GLD"].ret_20d_min_hard = ret_min
        sig_t = generate_daily_signals(close, high, low, upper, lower,
                                              regime, rv_p, asset="GLD",
                                              gvz_series=gvz_s)
        line = f'ret>{ret_min*100:+.0f}%  '
        for w_name, w_days in WINDOWS:
            sub = sig_t[(sig_t.index >= today - pd.Timedelta(days=w_days)) & sig_t['buy_signal']]
            line += f'  {spot_eval(sub):>22}'
        print(line)
        sc.ASSET_CONFIGS["GLD"] = orig

    # ── 3. IV filter high_min ──
    print("\n" + "=" * 130)
    print("3. iv_filter_high_min grid × 多窗口")
    print("=" * 130)
    print(f'{"thr":>8}  ' + '  '.join(f'{w:>22}' for w,_ in WINDOWS))
    for ivh in [40, 32, 28, 25, 22]:
        orig = copy.deepcopy(sc.ASSET_CONFIGS["GLD"])
        sc.ASSET_CONFIGS["GLD"].iv_filter_high_min = float(ivh)
        sig_t = generate_daily_signals(close, high, low, upper, lower,
                                              regime, rv_p, asset="GLD",
                                              gvz_series=gvz_s)
        line = f'GVZ>={ivh}  '
        for w_name, w_days in WINDOWS:
            sub = sig_t[(sig_t.index >= today - pd.Timedelta(days=w_days)) & sig_t['buy_signal']]
            line += f'  {spot_eval(sub):>22}'
        print(line)
        sc.ASSET_CONFIGS["GLD"] = orig

    # ── 4. 期货 leverage × 多窗口 (固定 hold=30 无 TP) ──
    print("\n" + "=" * 130)
    print("4. 期货 leverage × 多窗口 (hold=30, sl=100%, 无 early_tp_lock)")
    print("=" * 130)
    gc = yf.Ticker("GC=F").history(period="6y", auto_adjust=True)
    gc.index = pd.to_datetime(gc.index).tz_localize(None).normalize()
    gc = gc[["Open","High","Low","Close"]]
    sig_curr = generate_daily_signals(close, high, low, upper, lower,
                                              regime, rv_p, asset="GLD",
                                              gvz_series=gvz_s)
    buy_curr = sig_curr[sig_curr['buy_signal']]

    def fut_eval(buy_w, lev):
        cfg_f = FuturesConfig(leverage=lev, hold_max_days=30,
                                    funding_rate_8h=-0.00002,
                                    tp_margin_pct=200, sl_margin_pct=100,
                                    early_tp_locks=())
        rows = []
        for sig_d in buy_w.index:
            gc_after = gc.loc[gc.index >= sig_d]
            if not len(gc_after): continue
            es = float(gc_after.iloc[0]['Open'])
            res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                                ohlc=gc_after, today=today, cfg=cfg_f)
            if not res.get('closed'): continue
            rows.append((max(-100, min(500, res.get('net_pnl_pct',0))),
                          res.get('is_liquidation', False)))
        if not rows: return 'n=0'
        p = pd.Series([r[0] for r in rows])
        nl = sum(1 for r in rows if r[1])
        return f'n={len(p)} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}% liq={nl}'

    print(f'{"lev":>6}  ' + '  '.join(f'{w:>25}' for w,_ in WINDOWS))
    for lev in [3, 5, 10, 15, 20]:
        line = f'  {lev:>2}x  '
        for w_name, w_days in WINDOWS:
            buy_w = buy_curr[buy_curr.index >= today - pd.Timedelta(days=w_days)]
            line += f'  {fut_eval(buy_w, lev):>25}'
        print(line)

    # ── 5. 期货 hold_max × 多窗口 (lev=10 固定) ──
    print("\n" + "=" * 130)
    print("5. 期货 hold_max × 多窗口 (lev=10 全, sl=100%, 无 early_tp_lock)")
    print("=" * 130)
    print(f'{"hold":>6}  ' + '  '.join(f'{w:>25}' for w,_ in WINDOWS))
    def hold_eval(buy_w, hold):
        cfg_f = FuturesConfig(leverage=10, hold_max_days=hold,
                                    funding_rate_8h=-0.00002,
                                    tp_margin_pct=200, sl_margin_pct=100,
                                    early_tp_locks=())
        rows = []
        for sig_d in buy_w.index:
            gc_after = gc.loc[gc.index >= sig_d]
            if not len(gc_after): continue
            es = float(gc_after.iloc[0]['Open'])
            res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                                ohlc=gc_after, today=today, cfg=cfg_f)
            if not res.get('closed'): continue
            rows.append((max(-100, min(500, res.get('net_pnl_pct',0))),
                          res.get('is_liquidation', False)))
        if not rows: return 'n=0'
        p = pd.Series([r[0] for r in rows])
        nl = sum(1 for r in rows if r[1])
        return f'n={len(p)} WR={(p>0).mean()*100:.0f}% sum={p.sum():+.0f}% liq={nl}'

    for hold in [15, 20, 30, 45, 60, 90]:
        line = f'  {hold:>2}d  '
        for w_name, w_days in WINDOWS:
            buy_w = buy_curr[buy_curr.index >= today - pd.Timedelta(days=w_days)]
            line += f'  {hold_eval(buy_w, hold):>25}'
        print(line)


if __name__ == "__main__":
    main()
