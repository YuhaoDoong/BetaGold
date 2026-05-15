"""v3.7.205 期权 5y 统一回测 (kline_db 真实 + BS proxy)

≤ 1y kline_db 内: 用真实期权 OHLC + 完整 TP/SL
> 1y kline_db 外: BS + GVZ (IV) 每日 MTM, 同款 TP/SL 规则

3y/5y 窗口, per-tier 拆分, 让 tier 表用更大样本.
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
                                      interpolate_option_intraday, bs_price)
from core.strategies.buy_call import simulate_bc_position, BCConfig
from core.strategies.sell_put import simulate_sp_position, SPConfig


# ── BS proxy simulator (kline_db 外用) ──

def simulate_bc_bs(sig_d: pd.Timestamp, entry_spot: float,
                      gld_ohlc: pd.DataFrame, gvz: pd.Series,
                      cfg: BCConfig, today: pd.Timestamp) -> dict:
    """BS-based BC 日 MTM 模拟 — ATM call, DTE=base_dte, IV=GVZ.

    每日重算 BS 价 (DTE 减少, GVZ 变化), 检查 TP/SL/expiry.
    """
    DTE = cfg.base_dte
    K = round(entry_spot)
    # 入场 IV (用入场日 GVZ)
    iv0 = float(gvz.get(sig_d, 0.18)) / 100 if sig_d in gvz.index else 0.18
    iv0 = max(0.10, min(0.50, iv0))
    T0 = DTE / 365.0
    ent_price = bs_price(entry_spot, K, T0, 0.04, iv0, "C")
    if ent_price <= 0.01: return {"is_closed": False, "reason": "BS price 0"}
    tp = ent_price * cfg.profit_target_mult
    sl = ent_price * cfg.stop_loss_mult
    # 后续日 MTM
    days_after = gld_ohlc.index[gld_ohlc.index > sig_d]
    hold = 0
    cur_price = ent_price
    for d in days_after:
        if d > today: break
        hold += 1
        if hold > DTE: break
        spot_d = float(gld_ohlc.loc[d, "Close"])
        T_d = (DTE - hold) / 365.0
        if T_d <= 0:
            cur_price = max(spot_d - K, 0)  # 内在
            pnl = (cur_price / ent_price - 1) * 100
            return {"is_closed": True, "exit_date": d, "exit_value": cur_price,
                     "exit_reason": "expiry (BS)", "pnl_pct": pnl,
                     "hold_days": hold}
        iv_d = float(gvz.get(d, iv0*100)) / 100 if d in gvz.index else iv0
        iv_d = max(0.10, min(0.50, iv_d))
        cur_price = bs_price(spot_d, K, T_d, 0.04, iv_d, "C")
        pnl = (cur_price / ent_price - 1) * 100
        if cur_price >= tp:
            return {"is_closed": True, "exit_date": d, "exit_value": cur_price,
                     "exit_reason": "+100% TP (BS)", "pnl_pct": pnl,
                     "hold_days": hold}
        if cur_price <= sl:
            return {"is_closed": True, "exit_date": d, "exit_value": cur_price,
                     "exit_reason": "-50% SL (BS)", "pnl_pct": pnl,
                     "hold_days": hold}
    # 没触发, OPEN
    pnl_open = (cur_price / ent_price - 1) * 100
    return {"is_closed": False, "current_value": cur_price, "hold_days": hold,
             "pnl_pct": pnl_open}


def simulate_sp_bs(sig_d: pd.Timestamp, entry_spot: float,
                      gld_ohlc: pd.DataFrame, gvz: pd.Series,
                      cfg: SPConfig, today: pd.Timestamp) -> dict:
    """BS-based SP credit spread (-ATM put / +OTM-5% put)."""
    DTE = cfg.base_dte
    K_s = round(entry_spot)           # short ATM
    K_l = round(entry_spot * 0.95)    # long OTM -5%
    width = K_s - K_l
    iv0 = float(gvz.get(sig_d, 0.18)) / 100 if sig_d in gvz.index else 0.18
    iv0 = max(0.10, min(0.50, iv0))
    T0 = DTE / 365.0
    sp0 = bs_price(entry_spot, K_s, T0, 0.04, iv0, "P")
    lp0 = bs_price(entry_spot, K_l, T0, 0.04, iv0, "P")
    credit0 = sp0 - lp0
    if credit0 <= 0.01: return {"is_closed": False, "reason": "credit 0"}
    max_risk = width - credit0
    if max_risk <= 0: return {"is_closed": False, "reason": "neg risk"}
    # 退出阈值
    tp_credit = credit0 * (1 - cfg.profit_target_credit_pct / 100)  # 收 X% credit 即平
    sl_credit = credit0 + cfg.stop_loss_margin_pct / 100 * max_risk
    days_after = gld_ohlc.index[gld_ohlc.index > sig_d]
    hold = 0
    cur_credit = credit0
    for d in days_after:
        if d > today: break
        hold += 1
        if hold > DTE: break
        spot_d = float(gld_ohlc.loc[d, "Close"])
        T_d = (DTE - hold) / 365.0
        if T_d <= 0:
            # expiry: cur_credit = intrinsic spread
            cur_credit = (max(K_s - spot_d, 0) - max(K_l - spot_d, 0))
            pnl = (credit0 - cur_credit) / max_risk * 100
            return {"is_closed": True, "exit_date": d, "exit_value": cur_credit,
                     "exit_reason": "expiry (BS)", "pnl_pct": pnl,
                     "hold_days": hold}
        iv_d = float(gvz.get(d, iv0*100)) / 100 if d in gvz.index else iv0
        iv_d = max(0.10, min(0.50, iv_d))
        sp_d = bs_price(spot_d, K_s, T_d, 0.04, iv_d, "P")
        lp_d = bs_price(spot_d, K_l, T_d, 0.04, iv_d, "P")
        cur_credit = sp_d - lp_d
        pnl = (credit0 - cur_credit) / max_risk * 100
        if cur_credit <= tp_credit:
            return {"is_closed": True, "exit_date": d, "exit_value": cur_credit,
                     "exit_reason": f"+{cfg.profit_target_credit_pct:.0f}% TP (BS)",
                     "pnl_pct": pnl, "hold_days": hold}
        if cur_credit >= sl_credit:
            return {"is_closed": True, "exit_date": d, "exit_value": cur_credit,
                     "exit_reason": f"-{cfg.stop_loss_margin_pct:.0f}% SL (BS)",
                     "pnl_pct": pnl, "hold_days": hold}
    pnl_open = (credit0 - cur_credit) / max_risk * 100
    return {"is_closed": False, "current_value": cur_credit, "hold_days": hold,
             "pnl_pct": pnl_open}


# ── 统一 dispatcher ──

def run_unified(buy, gld_ohlc, db, db_min, today, gvz, strategy="BC",
                  bc_cfg=None, sp_cfg=None):
    if bc_cfg is None: bc_cfg = BCConfig()
    if sp_cfg is None: sp_cfg = SPConfig(profit_target_credit_pct=50.0,
                                                stop_loss_margin_pct=100.0, base_dte=30)
    from core.paper_positions import price_strategy_at
    rows = []
    for sig_d, r in buy.iterrows():
        eO = float(r["Open"]); eC = float(r["Close"])
        eH = float(r["High"]); eL = float(r["Low"])
        use_real = sig_d >= db_min
        if strategy == "BC":
            if use_real:
                lc = pick_liquid_monthly_option("GLD", sig_d, eO, "C",
                                                      dte_target=bc_cfg.base_dte, min_dte=14)
                if not lc: continue
                entry = interpolate_option_intraday(lc, eO, eC, eO, eH, eL)
                ent = {"legs": [("long_call", lc["code"], lc["strike"], 1)],
                        "entry_price": entry, "leg_prices": [("long_call", entry)]}
                res = simulate_bc_position(ent, sig_d, today, db, bc_cfg)
            else:
                res = simulate_bc_bs(sig_d, eO, gld_ohlc, gvz, bc_cfg, today)
        else:  # SP
            if use_real:
                ent = price_strategy_at("GLD", "SELL PUT", sig_d,
                                              sig_d + pd.Timedelta(hours=9,minutes=30),
                                              eO, eO, eC, eH, eL,
                                              dte_target=sp_cfg.base_dte)
                if not ent.get("legs"): continue
                res = simulate_sp_position(ent, sig_d, today, db, sp_cfg)
            else:
                res = simulate_sp_bs(sig_d, eO, gld_ohlc, gvz, sp_cfg, today)
        if not res.get("is_closed"): continue
        clip = 500 if strategy == "BC" else 150
        rows.append({"sig_d": sig_d.date(), "tier": r["signal_tier"],
                      "source": "real" if use_real else "BS",
                      "pnl_pct": max(-100, min(clip, float(res.get("pnl_pct", 0)))),
                      "exit_reason": res.get("exit_reason",""),
                      "hold": int(res.get("hold_days", 0))})
    return rows


def summarize(rows, label):
    if not rows: return f"{label}: n=0"
    p = pd.Series([r["pnl_pct"] for r in rows])
    n_w = (p>0).sum(); n_l = (p<=0).sum()
    pf = (p[p>0].sum() / abs(p[p<=0].sum())) if (n_l and abs(p[p<=0].sum())>0) else 99
    return ("%s: n=%3d WR=%5.1f%% sum=%+8.1f%% mean=%+6.2f%% "
            "max_loss=%+6.1f%% PF=%.2f"
            % (label, len(p), n_w/len(p)*100, p.sum(), p.mean(), p.min(), pf))


def main():
    db = _load_kline_db()
    db_min = db["date"].min() if db is not None else pd.Timestamp("2026-01-01")
    cfg = load_config()
    oos = load_oos_predictions(cfg)
    feat = pd.read_parquet("/Users/yhdong/Gold/data/processed/features_all.parquet")
    ohlc = pd.read_csv("/Users/yhdong/Gold/data/raw/market/gld.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common,"Close"]; high=ohlc.loc[common,"High"]; low=ohlc.loc[common,"Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common,"rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat.loc[common, feat_cols])["regime"]
    gvz_raw = yf.Ticker("^GVZ").history(period="5y")
    gvz_raw.index = pd.to_datetime(gvz_raw.index).tz_localize(None).normalize()
    gvz_s = gvz_raw["Close"]
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset="GLD", gvz_series=gvz_s)
    buy_all = sig[sig["buy_signal"]].copy().join(
        ohlc[["Open","High","Low","Close"]], how="inner")
    today = ohlc.index.max()

    for years in [3, 5]:
        cutoff = today - pd.Timedelta(days=years*365)
        buy = buy_all[buy_all.index >= cutoff]
        n_real = sum(buy.index >= db_min); n_bs = sum(buy.index < db_min)
        print(f"\n{'=' * 80}")
        print(f"窗口: {years}y ({cutoff.date()} → {today.date()})")
        print(f"  BUY 信号 {len(buy)} 笔: kline_db 真实 {n_real} | BS proxy {n_bs}")
        print(f"  tier: {buy['signal_tier'].value_counts().to_dict()}")
        print('=' * 80)

        for strat in ["BC", "SP"]:
            print(f"\n--- {strat} ---")
            rows = run_unified(buy, ohlc, db, db_min, today, gvz_s, strategy=strat)
            print(summarize(rows, "  全部"))
            for t in ['S','A','B']:
                sub = [r for r in rows if r['tier']==t]
                print('  ' + summarize(sub, f"  tier {t} (exclusive)"))
            # nested
            print('  -- nested (S⊂A⊂B 累计) --')
            sub_S = [r for r in rows if r['tier']=='S']
            sub_A = [r for r in rows if r['tier'] in ('S','A')]
            print('  ' + summarize(sub_S, "  S only"))
            print('  ' + summarize(sub_A, "  A (含S)"))
            print('  ' + summarize(rows, "  B (全部含S/A)"))


if __name__ == "__main__":
    main()
