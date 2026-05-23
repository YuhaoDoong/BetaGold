"""v3.7.229 波动率期权 Layer 2: STRADDLE + SHORT_VOL, trailing 1y/6m/3m.

STRADDLE: 自己的 detect_straddle_signal 触发 (long vol)
SHORT_VOL: 自己的 detect_short_vol_signal 触发 (short vol)
跨窗看 P&L 稳健性.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts.backtest.framework import (build_raw_universe, trailing_slice,
                                              LAYER2_WINDOWS)
from core.paper_positions import price_strategy_at, simulate_option_exit
from core.events import detect_straddle_signal, detect_short_vol_signal


def backtest_strategy(dates, ohlc, asset, strategy):
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    rows = []
    for d in dates:
        if d not in ohlc.index: continue
        eO = float(ohlc.loc[d, "Open"]); eC = float(ohlc.loc[d, "Close"])
        eH = float(ohlc.loc[d, "High"]); eL = float(ohlc.loc[d, "Low"])
        dte = 14 if strategy == "STRADDLE" else 30
        ent = price_strategy_at(asset, strategy, d,
                                      d + pd.Timedelta(hours=9, minutes=30),
                                      eO, eO, eC, eH, eL, dte_target=dte)
        if not ent.get("legs"): continue
        sim = simulate_option_exit(ent, d, strategy, today,
                                          live_spot=eC, live_high=eH, live_low=eL)
        if sim.get("is_closed"):
            rows.append({"date": d, "pnl_pct": float(sim.get("pnl_pct", 0) or 0)})
    return pd.DataFrame(rows)


def score(pnls):
    if not len(pnls): return {"n": 0, "scoreB": 0}
    s = pnls if isinstance(pnls, pd.Series) else pd.Series(pnls)
    n = len(s); wr = (s > 0).mean()
    return {"n": n, "WR": round(wr*100, 1), "mean": round(s.mean(), 2),
              "sum": round(s.sum(), 1), "max_loss": round(s.min(), 1),
              "scoreB": round((wr**2) * math.log(1+n) * s.mean(), 2)}


def find_vol_signals(asset, raw):
    """返回 (strad_dates, sv_dates) — 自己的信号."""
    rv_raw = (raw["close"].pct_change().rolling(10).std() * (252**0.5)) * 100
    strad = detect_straddle_signal(rv_raw, raw.index,
                                          rv_pctile=raw["rv_pctile"],
                                          close=raw["close"], high=raw["high"],
                                          low=raw["low"], asset=asset)
    sv = detect_short_vol_signal(rv_raw, raw["rv_pctile"], raw.index,
                                        regime=raw["regime"],
                                        close=raw["close"], high=raw["high"],
                                        low=raw["low"], asset=asset)
    strad_dates = strad.index[strad["straddle_signal"]].tolist() if len(strad) else []
    sv_dates = sv.index[sv["short_vol_signal"]].tolist() if len(sv) else []
    return strad_dates, sv_dates


def main():
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history/v3.7.229_layer2_vol_options")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset} 波动率期权 (trailing 1y/6m/3m)\n{'='*100}")
        raw, ohlc = build_raw_universe(asset)
        strad_dates, sv_dates = find_vol_signals(asset, raw)
        print(f"  全期: STRADDLE 信号 {len(strad_dates)} 笔, SHORT_VOL 信号 {len(sv_dates)} 笔")

        for strategy_name, dates in [("STRADDLE", strad_dates),
                                            ("SHORT_VOL", sv_dates)]:
            print(f"\n--- {strategy_name} ---")
            for win_label, win_days in LAYER2_WINDOWS:
                sub_raw = trailing_slice(raw, win_days)
                start = sub_raw.index.min()
                # 信号也限制到该窗口
                win_dates = [d for d in dates if d >= start]
                if len(win_dates) < 3:
                    print(f"  {win_label}: n_signal={len(win_dates)} 不足跳过"); continue
                df = backtest_strategy(win_dates, ohlc, asset, strategy_name)
                s = score(df["pnl_pct"]) if len(df) else {"n": 0}
                print(f"  {win_label} ({len(win_dates)} signals, "
                      f"{start.date()} → today): "
                      f"closed n={s.get('n')} WR={s.get('WR')}% "
                      f"mean={s.get('mean')}% sum={s.get('sum')}% "
                      f"max_loss={s.get('max_loss')}")
                all_rows.append({
                    "asset": asset, "strategy": strategy_name,
                    "window": win_label, "n_signals": len(win_dates),
                    **{k: s.get(k) for k in ["n","WR","mean","sum","max_loss","scoreB"]}
                })

    df = pd.DataFrame(all_rows)
    print(f"\n\n=== 波动率期权多窗总览 ===")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "vol_options_trailing_windows.csv", index=False)


if __name__ == "__main__":
    main()
