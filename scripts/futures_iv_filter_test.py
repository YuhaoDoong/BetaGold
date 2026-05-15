"""v3.7.208 期货 IV filter 必要性测试

期货没 theta/vega, IV crush 不直接伤期货. 但高 IV 可能是 regime risk 代理.
对比同 cfg 下: WITH IV filter vs WITHOUT IV filter

5y GLD BUY 信号 × GC=F 期货 simulate.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategies.futures_long import FuturesConfig, simulate_long_position


def build_buys(with_iv: bool):
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
    gvz_raw = yf.Ticker("^GVZ").history(period="5y")
    gvz_raw.index = pd.to_datetime(gvz_raw.index).tz_localize(None).normalize()
    gvz_s = gvz_raw["Close"] if with_iv else None
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz_s)
    return sig[sig["buy_signal"]].copy()


def run(buy, gc, cfg, today):
    rows = []
    for sig_d in buy.index:
        gc_after = gc.loc[gc.index >= sig_d]
        if not len(gc_after): continue
        es = float(gc_after.iloc[0]["Open"])
        tier = buy.loc[sig_d, "signal_tier"] or "B"  # 没 tier 当 B
        res = simulate_long_position(entry_d=gc_after.index[0], entry_spot=es,
                                            ohlc=gc_after, today=today, cfg=cfg,
                                            live_spot=float(gc.iloc[-1]["Close"]),
                                            signal_tier=tier)
        if not res.get("closed"): continue
        rows.append({"sig_d": sig_d.date(), "tier": tier,
                       "ret": max(-100, min(500, res.get("net_pnl_pct", 0))),
                       "is_liq": res.get("is_liquidation", False)})
    return rows


def summarize(rows, label):
    if not rows: return f"{label}: n=0"
    p = pd.Series([r["ret"] for r in rows])
    n_liq = sum(1 for r in rows if r.get("is_liq"))
    return ("%s: n=%3d WR=%5.1f%% sum=%+7.1f%% mean=%+6.2f%% "
            "max_loss=%+6.1f%% liq=%d"
            % (label, len(p), (p>0).mean()*100, p.sum(), p.mean(), p.min(), n_liq))


def main():
    gc = yf.Ticker("GC=F").history(period="6y", auto_adjust=True)
    gc.index = pd.to_datetime(gc.index).tz_localize(None).normalize()
    gc = gc[["Open","High","Low","Close"]]
    today = gc.index.max()
    cutoff = today - pd.Timedelta(days=5*365)

    # 用现有 GLD cfg (per-tier lev S=10 A=10 B=5, hold=30)
    cfg = FuturesConfig(leverage=5, tier_s_leverage=10, tier_a_leverage=10,
                              tier_b_leverage=5, hold_max_days=30,
                              funding_rate_8h=-0.00002)

    # ── A. WITH IV filter (current) ──
    buy_a = build_buys(with_iv=True)
    buy_a = buy_a[buy_a.index >= cutoff]
    print(f"A. WITH IV filter: {len(buy_a)} 笔 BUY 信号")
    print(f"   tier: {buy_a['signal_tier'].value_counts().to_dict()}")

    # ── B. WITHOUT IV filter ──
    buy_b = build_buys(with_iv=False)
    buy_b = buy_b[buy_b.index >= cutoff]
    print(f"B. WITHOUT IV filter: {len(buy_b)} 笔 BUY 信号")
    print(f"   tier: {buy_b['signal_tier'].value_counts().to_dict()}")
    print(f"   差异: {len(buy_b)-len(buy_a)} 笔 (IV filter 拦的)")

    # 跑模拟
    res_a = run(buy_a, gc, cfg, today)
    res_b = run(buy_b, gc, cfg, today)

    print("\n" + "=" * 80)
    print("结果对比 (期货 cfg: per-tier S=10 A=10 B=5, hold=30):")
    print("=" * 80)
    print(summarize(res_a, "A. WITH IV filter (当前)"))
    print(summarize(res_b, "B. NO IV filter (期货专用)"))

    # 拆分 IV filter 拦下的那些
    a_dates = set(r["sig_d"] for r in res_a)
    extra = [r for r in res_b if r["sig_d"] not in a_dates]
    print("\n--- IV filter 拦下的笔 (在 B 不在 A) ---")
    print(summarize(extra, "  仅 B"))
    print(f"  这些 trade 详情 (前 10):")
    for r in extra[:10]:
        sign = "✓" if r["ret"] > 0 else "✗"
        liq = " LIQ" if r["is_liq"] else ""
        print(f"    {r['sig_d']} tier={r['tier']} ret={r['ret']:+.1f}%{liq} {sign}")
    if len(extra) > 10:
        print(f"    ... 还有 {len(extra)-10} 笔")

    # per-tier 对比
    print("\n--- per-tier 对比 ---")
    for t in ['S','A','B']:
        a_t = [r for r in res_a if r['tier']==t]
        b_t = [r for r in res_b if r['tier']==t]
        print(f"  tier {t}:")
        print('    ' + summarize(a_t, "  A (WITH IV)"))
        print('    ' + summarize(b_t, "  B (NO IV)"))


if __name__ == "__main__":
    main()
