"""v3.7.205 期权 per-tier 验证

用 v3.7.201 信号过滤 + S/A/B tier, 在 kline_db 1y 跑 BC + SP, 分 tier 统计.

问:
  1. 当前 BC pt=1.5x/sl=0.5x/DTE=30 在新信号集上还是最优吗?
  2. S/A 期权表现 vs B 差距?
  3. sp_score routing (BC vs SP) 在 per-tier 下合理吗?

输出:
  - per-tier BC/SP P&L 分布
  - BC TP/SL grid (现 cfg vs 候选)
  - SP TP/SL grid (现 cfg vs 候选)
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
from core.paper_positions import (_load_kline_db, pick_liquid_monthly_option,
                                      interpolate_option_intraday)
from core.strategies.buy_call import simulate_bc_position, BCConfig
from core.strategies.sell_put import simulate_sp_position, SPConfig


def build_buy_signals():
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
    buy = buy.join(ohlc[["Open","High","Low","Close"]], how="inner")
    return buy


def run_bc(buy: pd.DataFrame, db: pd.DataFrame, today, cfg: BCConfig):
    """对 buy 信号集跑 BC, 仅算 is_closed=True."""
    rows = []
    for sig_d, r in buy.iterrows():
        eO,eC,eH,eL = float(r["Open"]),float(r["Close"]),float(r["High"]),float(r["Low"])
        lc = pick_liquid_monthly_option("GLD", sig_d, eO, "C",
                                              dte_target=cfg.base_dte, min_dte=14)
        if not lc: continue
        entry = interpolate_option_intraday(lc, eO, eC, eO, eH, eL)
        ent = {"legs": [("long_call", lc["code"], lc["strike"], 1)],
                "entry_price": entry,
                "leg_prices": [("long_call", entry)]}
        res = simulate_bc_position(ent, sig_d, today, db, cfg)
        if not res.get("is_closed"): continue
        rows.append({"sig_d": sig_d.date(), "tier": r["signal_tier"],
                      "pnl_pct": max(-100, min(500, float(res.get("pnl_pct", 0)))),
                      "exit_reason": res.get("exit_reason", ""),
                      "hold": int(res.get("hold_days", 0))})
    return rows


def run_sp(buy: pd.DataFrame, db: pd.DataFrame, today, cfg: SPConfig):
    """对 buy 信号集跑 SP credit spread (-ATM put / +ATM-5% put)."""
    from core.paper_positions import price_strategy_at
    rows = []
    for sig_d, r in buy.iterrows():
        eO,eC,eH,eL = float(r["Open"]),float(r["Close"]),float(r["High"]),float(r["Low"])
        ent = price_strategy_at("GLD", "SELL PUT", sig_d,
                                       sig_d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=cfg.base_dte)
        if not ent.get("legs"): continue
        res = simulate_sp_position(ent, sig_d, today, db, cfg)
        if not res.get("is_closed"): continue
        rows.append({"sig_d": sig_d.date(), "tier": r["signal_tier"],
                      "pnl_pct": max(-100, min(150, float(res.get("pnl_pct", 0)))),
                      "exit_reason": res.get("exit_reason", ""),
                      "hold": int(res.get("hold_days", 0))})
    return rows


def summarize(rows, label):
    if not rows: return f"{label}: n=0"
    p = pd.Series([r["pnl_pct"] for r in rows])
    n_w = (p>0).sum(); n_l = (p<=0).sum()
    pf = (p[p>0].sum() / abs(p[p<=0].sum())) if n_l else float('inf')
    return ("%s: n=%3d WR=%5.1f%% sum=%+7.1f%% mean=%+6.2f%% "
            "max_loss=%+6.1f%% PF=%.2f"
            % (label, len(p), n_w/len(p)*100, p.sum(), p.mean(), p.min(), pf))


def main():
    db = _load_kline_db()
    if db is None: return print("kline_db 不存在")
    today = db["date"].max()
    print(f"kline_db: {db['date'].min().date()} → {today.date()}")
    buy_all = build_buy_signals()
    # 只看 kline_db 内的信号
    buy = buy_all[(buy_all.index >= db["date"].min()) & (buy_all.index <= today)]
    print(f"BUY 信号 (kline_db 时间窗): {len(buy)} 笔 "
          f"(tier: {buy['signal_tier'].value_counts().to_dict()})\n")

    # 1. 现 BC cfg per-tier
    print("=" * 70)
    print("1. BC 现 cfg (pt=1.5x sl=0.5x DTE=30) per-tier")
    print("=" * 70)
    bc_rows = run_bc(buy, db, today, BCConfig())
    print(summarize(bc_rows, "BC 全部"))
    for t in ['S','A','B']:
        print("  " + summarize([r for r in bc_rows if r['tier']==t], f"BC tier {t}"))

    # 2. 现 SP cfg per-tier
    print("\n" + "=" * 70)
    print("2. SP 现 cfg (GLD pt=50% sl=100% DTE=30) per-tier")
    print("=" * 70)
    sp_rows = run_sp(buy, db, today, SPConfig(profit_target_credit_pct=50.0,
                                                            stop_loss_margin_pct=100.0,
                                                            base_dte=30))
    print(summarize(sp_rows, "SP 全部"))
    for t in ['S','A','B']:
        print("  " + summarize([r for r in sp_rows if r['tier']==t], f"SP tier {t}"))

    # 3. BC pt/sl grid
    print("\n" + "=" * 70)
    print("3. BC pt/sl grid (DTE=30 固定)")
    print("=" * 70)
    print(f"{'pt':>5} {'sl':>5} {'n':>4} {'WR%':>6} {'sum%':>8} {'mean%':>7} {'PF':>5}")
    for pt in [1.3, 1.5, 1.8, 2.0, 2.5]:
        for sl in [0.3, 0.5, 0.7]:
            cfg = BCConfig(profit_target_mult=pt, stop_loss_mult=sl, base_dte=30)
            r = run_bc(buy, db, today, cfg)
            if not r: continue
            p = pd.Series([x["pnl_pct"] for x in r])
            n_w = (p>0).sum(); n_l = (p<=0).sum()
            pf = (p[p>0].sum()/abs(p[p<=0].sum())) if n_l else 99
            print(f"{pt:>5.1f} {sl:>5.1f} {len(p):>4d} {n_w/len(p)*100:>5.1f}% "
                  f"{p.sum():>+7.1f}% {p.mean():>+6.2f}% {pf:>5.2f}")

    # 4. SP pt/sl grid
    print("\n" + "=" * 70)
    print("4. SP pt/sl grid (DTE=30 固定)")
    print("=" * 70)
    print(f"{'pt':>5} {'sl':>5} {'n':>4} {'WR%':>6} {'sum%':>8} {'mean%':>7} {'PF':>5}")
    for pt in [30, 50, 70]:
        for sl in [50, 100, 150]:
            cfg = SPConfig(profit_target_credit_pct=pt,
                                stop_loss_margin_pct=sl, base_dte=30)
            r = run_sp(buy, db, today, cfg)
            if not r: continue
            p = pd.Series([x["pnl_pct"] for x in r])
            n_w = (p>0).sum(); n_l = (p<=0).sum()
            pf = (p[p>0].sum()/abs(p[p<=0].sum())) if n_l else 99
            print(f"{pt:>5d} {sl:>5d} {len(p):>4d} {n_w/len(p)*100:>5.1f}% "
                  f"{p.sum():>+7.1f}% {p.mean():>+6.2f}% {pf:>5.2f}")

    # 5. per-tier 决策建议
    print("\n" + "=" * 70)
    print("5. Per-tier: BC vs SP 哪个更优")
    print("=" * 70)
    for t in ['S','A','B']:
        bc_t = [r for r in bc_rows if r['tier']==t]
        sp_t = [r for r in sp_rows if r['tier']==t]
        bc_s = sum(r['pnl_pct'] for r in bc_t)/len(bc_t) if bc_t else 0
        sp_s = sum(r['pnl_pct'] for r in sp_t)/len(sp_t) if sp_t else 0
        bc_w = sum(1 for r in bc_t if r['pnl_pct']>0)/len(bc_t)*100 if bc_t else 0
        sp_w = sum(1 for r in sp_t if r['pnl_pct']>0)/len(sp_t)*100 if sp_t else 0
        better = "BC" if bc_s > sp_s else "SP"
        print(f"  tier {t}: BC n={len(bc_t)} WR={bc_w:.0f}% mean={bc_s:+.1f}% | "
              f"SP n={len(sp_t)} WR={sp_w:.0f}% mean={sp_s:+.1f}% → {better}")


if __name__ == "__main__":
    main()
