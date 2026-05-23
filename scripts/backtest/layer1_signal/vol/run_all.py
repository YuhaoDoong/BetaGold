"""v3.7.230 Layer 1 波动率信号验证 — trailing 10y/5y/3y/1y.

修正后的判定指标:
  STRADDLE 主胜率: abs_r > BE (entry premium 距离, = IV * sqrt(h/252))
  SHORT_VOL 主胜率: abs_r < 短腿距离 (= 1.6 * BE, ATM IC 配置)
  辅助: iv_change (vega 方向), rv_fwd vs iv_entry
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from scripts.backtest.framework import (build_raw_universe, trailing_slice,
                                              LAYER1_WINDOWS,
                                              score_straddle, score_short_vol)
from core.events import detect_straddle_signal, detect_short_vol_signal


def find_vol_signal_dates(asset, raw):
    rv_raw = (raw["close"].pct_change().rolling(10).std() * (252**0.5)) * 100
    strad = detect_straddle_signal(rv_raw, raw.index,
                                          rv_pctile=raw["rv_pctile"],
                                          close=raw["close"], high=raw["high"],
                                          low=raw["low"], asset=asset)
    sv = detect_short_vol_signal(rv_raw, raw["rv_pctile"], raw.index,
                                        regime=raw["regime"],
                                        close=raw["close"], high=raw["high"],
                                        low=raw["low"], asset=asset)
    return (strad.index[strad["straddle_signal"]] if len(strad) else pd.DatetimeIndex([]),
              sv.index[sv["short_vol_signal"]] if len(sv) else pd.DatetimeIndex([]))


def main():
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history/v3.7.230_layer1_vol")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset} 波动率信号 (修正后指标: abs_r vs BE)\n{'='*100}")
        raw, _ = build_raw_universe(asset)
        strad_dates, sv_dates = find_vol_signal_dates(asset, raw)
        print(f"  全期: STRADDLE 信号 {len(strad_dates)} 笔, SHORT_VOL {len(sv_dates)} 笔")

        for h in [10, 14, 20]:  # 加 14 跟 STRADDLE_CONFIG.hold_max=14 一致
            print(f"\n--- horizon={h}d ---")
            # STRADDLE 信号验证
            print(f"\n  [STRADDLE 信号 — WR_abs_gt_BE 主指标]")
            for win_label, win_days in LAYER1_WINDOWS:
                sub_raw = trailing_slice(raw, win_days)
                sig_in = [d for d in strad_dates if d in sub_raw.index]
                if len(sig_in) < 5: continue
                sub = sub_raw.loc[sig_in]
                s = score_straddle(sub, h)
                # baseline = Bull-only
                bull = sub_raw[sub_raw["regime"] == "Bull"]
                s_base = score_straddle(bull, h)
                uplift_be = s.get("WR_abs_gt_BE") - s_base.get("WR_abs_gt_BE", 0) \
                            if s.get("n") and s_base.get("n") else None
                print(f"    {win_label} (n={s.get('n')}): "
                      f"WR(abs_r>BE)={s.get('WR_abs_gt_BE')}% vs 基线 {s_base.get('WR_abs_gt_BE')}%  "
                      f"uplift {uplift_be:+.1f}pp" if uplift_be is not None else "")
                print(f"      abs_r={s.get('mean_abs_r')}% (BE={s.get('mean_BE')}%), "
                      f"IV={s.get('mean_iv_entry')}, IV_chg={s.get('mean_iv_change')}, "
                      f"WR_max_gt_BE={s.get('WR_max_gt_BE')}%")
                all_rows.append({
                    "asset": asset, "signal": "STRADDLE", "horizon": h,
                    "window": win_label, "n": s.get("n"),
                    "WR_abs_gt_BE": s.get("WR_abs_gt_BE"),
                    "baseline_WR": s_base.get("WR_abs_gt_BE"),
                    "uplift_pp": uplift_be,
                    "WR_max_gt_BE": s.get("WR_max_gt_BE"),
                    "mean_abs_r": s.get("mean_abs_r"),
                    "mean_BE": s.get("mean_BE"),
                    "mean_iv_entry": s.get("mean_iv_entry"),
                    "mean_iv_change": s.get("mean_iv_change"),
                })

            # SHORT_VOL 信号验证
            print(f"\n  [SHORT_VOL 信号 — WR_abs_lt_strike 主指标]")
            for win_label, win_days in LAYER1_WINDOWS:
                sub_raw = trailing_slice(raw, win_days)
                sig_in = [d for d in sv_dates if d in sub_raw.index]
                if len(sig_in) < 5: continue
                sub = sub_raw.loc[sig_in]
                s = score_short_vol(sub, h)
                bull = sub_raw[sub_raw["regime"] == "Bull"]
                s_base = score_short_vol(bull, h)
                uplift_strike = s.get("WR_abs_lt_strike") - s_base.get("WR_abs_lt_strike", 0) \
                                if s.get("n") and s_base.get("n") else None
                print(f"    {win_label} (n={s.get('n')}): "
                      f"WR(abs_r<strike)={s.get('WR_abs_lt_strike')}% vs 基线 {s_base.get('WR_abs_lt_strike')}%  "
                      f"uplift {uplift_strike:+.1f}pp" if uplift_strike is not None else "")
                print(f"      abs_r={s.get('mean_abs_r')}% (strike={s.get('mean_strike_dist')}%), "
                      f"IV={s.get('mean_iv_entry')}, IV_chg={s.get('mean_iv_change')}")
                all_rows.append({
                    "asset": asset, "signal": "SHORT_VOL", "horizon": h,
                    "window": win_label, "n": s.get("n"),
                    "WR_abs_lt_strike": s.get("WR_abs_lt_strike"),
                    "baseline_WR": s_base.get("WR_abs_lt_strike"),
                    "uplift_pp": uplift_strike,
                    "mean_abs_r": s.get("mean_abs_r"),
                    "mean_strike_dist": s.get("mean_strike_dist"),
                    "mean_iv_entry": s.get("mean_iv_entry"),
                    "mean_iv_change": s.get("mean_iv_change"),
                })

    df = pd.DataFrame(all_rows)
    print(f"\n\n=== Vol 信号 Layer 1 总览 (修正指标) ===")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "vol_signals_trailing_v2.csv", index=False)


if __name__ == "__main__":
    main()
