"""v3.7.209 期货真实 TP 机制 grid

之前 v3.7.208 撤 early_tp_lock 后, 期货 TP 只剩:
  - tp_margin_pct=200 (= spot +20%@lev10, 几乎不触发)
  - signal_reversal (bp_high > 0.85) — 但 ledger 没传 bp_high_series, 死代码
  - hold_max=30d timeout

测真实 TP 机制:
  A. baseline: 无 TP 只 hold=30 + SL (现 v3.7.208)
  B. enable signal_reversal (传 bp_high_series, band exit)
  C. pullback: 峰值 +X%, 回撤 Y% 平 (真 trailing stop)
  D. 硬 TP spot: +10% / +15% / +20%
  E. 组合: pullback + signal_reversal
"""
from __future__ import annotations
import sys, copy, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf
import numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategies.futures_long import FuturesConfig, simulate_long_position


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
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz["Close"])
    buy = sig[sig["buy_signal"]].copy()
    # bp_high series for signal_reversal
    return buy, sig["bp_high"], close, high, low, ohlc


def simulate_with_pullback(sig_d, entry_spot, gc_after, today, cfg,
                              peak_threshold=0.05, dd_threshold=0.02,
                              signal_tier="B"):
    """自定义 simulate: 加 pullback (peak>X% + dd>=Y% 平) 逻辑."""
    # 用同款 base 但自己跑 — 加 pullback 检测
    # 用 per-tier lev
    lev = cfg.leverage
    if signal_tier == "S" and cfg.tier_s_leverage: lev = cfg.tier_s_leverage
    elif signal_tier == "A" and cfg.tier_a_leverage: lev = cfg.tier_a_leverage
    elif signal_tier == "B" and cfg.tier_b_leverage: lev = cfg.tier_b_leverage

    liq_pct = -(1.0 / lev - cfg.maintenance_margin_rate) * 100
    sl_spot = cfg.sl_margin_pct / lev
    tp_spot = cfg.tp_margin_pct / lev

    peak_pct = 0.0
    later = gc_after.index[gc_after.index > sig_d]
    hold = 0
    for d in later:
        if d > today: break
        hold += 1
        H = float(gc_after.loc[d, "High"])
        L = float(gc_after.loc[d, "Low"])
        C = float(gc_after.loc[d, "Close"])
        rL = (L / entry_spot - 1) * 100
        rH = (H / entry_spot - 1) * 100
        rC = (C / entry_spot - 1) * 100

        # 1. liq
        if rL <= liq_pct:
            return {"ret_lev": -100, "is_liq": True, "hold": hold, "reason": "爆仓"}
        # 2. SL
        if rL <= -sl_spot:
            return {"ret_lev": -sl_spot * lev, "is_liq": False, "hold": hold,
                     "reason": f"SL {-sl_spot:.1f}%"}
        # 3. TP (rare)
        if rH >= tp_spot:
            return {"ret_lev": tp_spot * lev, "is_liq": False, "hold": hold,
                     "reason": f"TP {tp_spot:.1f}%"}
        # 4. Pullback (峰值更新)
        peak_pct = max(peak_pct, rH)
        if peak_pct >= peak_threshold*100:
            # 已达到触发条件, 回撤 dd_threshold 即平
            dd = peak_pct - rC
            if dd >= dd_threshold*100:
                # 用 dd 时刻 close 价
                return {"ret_lev": rC * lev, "is_liq": False, "hold": hold,
                         "reason": f"pullback peak {peak_pct:.1f}% dd {dd:.1f}%"}
        # 5. Hold timeout
        if hold >= cfg.hold_max_days:
            return {"ret_lev": rC * lev, "is_liq": False, "hold": hold,
                     "reason": f"{cfg.hold_max_days}d 时间"}
    # OPEN
    return None


def run(buy, gc, today, cfg, mode, **kwargs):
    rows = []
    for sig_d in buy.index:
        gc_after = gc.loc[gc.index >= sig_d]
        if not len(gc_after): continue
        es = float(gc_after.iloc[0]["Open"])
        tier = buy.loc[sig_d, "signal_tier"] or "B"

        if mode == "baseline":
            res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                                ohlc=gc_after, today=today, cfg=cfg,
                                                signal_tier=tier)
            if not res.get("closed"): continue
            ret = max(-100, min(500, res.get("net_pnl_pct", 0)))
            is_liq = res.get("is_liquidation", False)
        elif mode == "signal_reversal":
            # 传 bp_high_series
            bp_high = kwargs.get("bp_high")
            res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                                ohlc=gc_after, today=today, cfg=cfg,
                                                signal_tier=tier, bp_high_series=bp_high)
            if not res.get("closed"): continue
            ret = max(-100, min(500, res.get("net_pnl_pct", 0)))
            is_liq = res.get("is_liquidation", False)
        elif mode == "pullback":
            r = simulate_with_pullback(gc_after.index[0], es, gc_after, today, cfg,
                                              peak_threshold=kwargs["peak"],
                                              dd_threshold=kwargs["dd"],
                                              signal_tier=tier)
            if r is None: continue
            ret = max(-100, min(500, r["ret_lev"]))
            is_liq = r["is_liq"]
        else:
            continue
        rows.append({"sig_d": sig_d.date(), "tier": tier, "ret": ret, "is_liq": is_liq})
    return rows


def summarize(rows, label):
    if not rows: return f"{label}: n=0"
    p = pd.Series([r["ret"] for r in rows])
    n_liq = sum(1 for r in rows if r.get("is_liq"))
    sc = (p.gt(0).mean())**2 * math.log(1+len(p)) * p.mean()
    return ("%s: n=%3d WR=%5.1f%% sum=%+7.0f%% mean=%+6.1f%% max_loss=%+6.1f%% liq=%d sB=%.0f"
            % (label, len(p), (p>0).mean()*100, p.sum(), p.mean(), p.min(), n_liq, sc))


def main():
    buy, bp_high, close, high, low, ohlc = build_inputs()
    gc = yf.Ticker("GC=F").history(period="6y", auto_adjust=True)
    gc.index = pd.to_datetime(gc.index).tz_localize(None).normalize()
    gc = gc[["Open","High","Low","Close"]]
    today = gc.index.max()
    buy = buy[buy.index >= today - pd.Timedelta(days=5*365)]
    print(f"5y BUY 信号: {len(buy)} 笔\n")

    base_cfg = dict(leverage=5, tier_s_leverage=10, tier_a_leverage=10,
                       tier_b_leverage=5, hold_max_days=30,
                       funding_rate_8h=-0.00002, tp_margin_pct=200,
                       sl_margin_pct=100, early_tp_locks=())

    # ── A. baseline (现 v3.7.208 cfg) ──
    cfg_base = FuturesConfig(**base_cfg)
    print("=" * 90)
    rA = run(buy, gc, today, cfg_base, "baseline")
    print(summarize(rA, "A. baseline (无 TP, hold=30)"))

    # ── B. signal_reversal (band exit) ──
    print()
    rB = run(buy, gc, today, cfg_base, "signal_reversal", bp_high=bp_high)
    print(summarize(rB, "B. signal_reversal (bp_high>0.85 + 利润)"))

    # ── C. pullback grid ──
    print()
    print("--- C. pullback (peak >X% + drawdown >=Y% 平) ---")
    for peak, dd in [(0.03,0.01),(0.03,0.015),(0.05,0.015),(0.05,0.02),
                          (0.05,0.025),(0.08,0.02),(0.08,0.03),(0.10,0.03)]:
        r = run(buy, gc, today, cfg_base, "pullback", peak=peak, dd=dd)
        print(summarize(r, f"   peak>={peak*100:.1f}% dd>={dd*100:.1f}%"))

    # ── top pullback per-tier ──
    print()
    print("--- per-tier 拆 (top 3 pullback combos) ---")
    for peak, dd in [(0.05, 0.02), (0.08, 0.025), (0.05, 0.025)]:
        rows = run(buy, gc, today, cfg_base, "pullback", peak=peak, dd=dd)
        print(f"\n  peak>={peak*100:.1f}% dd>={dd*100:.1f}%:")
        for t in ['S','A','B']:
            sub = [r for r in rows if r['tier']==t]
            print("  " + summarize(sub, f"  tier {t}"))


if __name__ == "__main__":
    main()
