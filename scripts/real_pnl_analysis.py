"""用真实期权 P&L 重做 RV %tile grid search."""
import pandas as pd
import numpy as np

for asset in ["GLD", "SLV"]:
    fpath = f"/Users/yhdong/Gold/data/real_options_backtest/{asset}_real_pnl_hold5d.csv"
    df = pd.read_csv(fpath)
    print(f"\n{'='*80}\n  {asset} 真实期权 P&L 聚合 ({len(df)} 信号)\n{'='*80}")

    # 各信号类型 + 工具组合
    print(f"\n按 RV %tile 分桶 (步长 0.1):")
    print(f"{'bucket':<12} {'n':<4} {'BUY_CALL':<25} {'STRADDLE':<25} {'SELL_PUT':<25}")
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        sub = df[(df["rv_pctile"] >= lo) & (df["rv_pctile"] < hi)]
        if len(sub) == 0:
            continue
        bc = sub[sub["signal_type"] == "BUY CALL"]
        bc = bc[bc["long_call_pnl_pct"].notna()]
        sp = sub[sub["signal_type"] == "SELL PUT"]
        sp = sp[sp["short_put_pnl_pct"].notna()]
        st = sub[sub["signal_type"] == "STRADDLE"]
        st = st[st["straddle_pnl_pct"].notna()]
        bc_str = (f"{len(bc)}n {(bc['long_call_pnl_pct']>0).mean()*100:.0f}% "
                  f"avg{bc['long_call_pnl_pct'].mean():+.1f}%"
                  if len(bc) else "—")
        sp_str = (f"{len(sp)}n {(sp['short_put_pnl_pct']>0).mean()*100:.0f}% "
                  f"avg{sp['short_put_pnl_pct'].mean():+.1f}%"
                  if len(sp) else "—")
        st_str = (f"{len(st)}n {(st['straddle_pnl_pct']>0).mean()*100:.0f}% "
                  f"avg{st['straddle_pnl_pct'].mean():+.1f}%"
                  if len(st) else "—")
        print(f"[{lo:.1f},{hi:.1f})    {len(sub):<4} "
              f"{bc_str:<25} {st_str:<25} {sp_str:<25}")

    print(f"\n📊 重做 grid: 不同 RV %tile 阈值下 STRADDLE 真实 P&L:")
    st_all = df[(df["signal_type"] == "STRADDLE")
                  & (df["straddle_pnl_pct"].notna())].copy()
    bc_all = df[(df["signal_type"] == "BUY CALL")
                  & (df["long_call_pnl_pct"].notna())].copy()
    sp_all = df[(df["signal_type"] == "SELL PUT")
                  & (df["short_put_pnl_pct"].notna())].copy()

    if len(st_all) > 0:
        print(f"\nSTRADDLE pctile_max 阈值 (持<阈值入场):")
        for th in np.arange(0.3, 1.05, 0.1):
            f = st_all[st_all["rv_pctile"] < th]
            if len(f) == 0:
                continue
            wr = (f["straddle_pnl_pct"] > 0).mean()
            tot = f["straddle_pnl_pct"].sum()
            avg = f["straddle_pnl_pct"].mean()
            sharpe = f["straddle_pnl_pct"].mean() / (f["straddle_pnl_pct"].std() + 1e-9)
            print(f"  < {th:.2f}: {len(f):>3} 笔 胜{wr*100:>3.0f}% "
                  f"总{tot:+6.1f}% avg{avg:+5.1f}% Sharpe{sharpe:.2f}")
        print(f"\nBUY CALL pctile 阈值:")
        for th in [0.30, 0.40, 0.50, 0.60, 0.70, 1.00]:
            f = bc_all[bc_all["rv_pctile"] < th]
            if len(f) == 0:
                continue
            wr = (f["long_call_pnl_pct"] > 0).mean()
            tot = f["long_call_pnl_pct"].sum()
            avg = f["long_call_pnl_pct"].mean()
            print(f"  < {th:.2f}: {len(f):>3} 笔 胜{wr*100:>3.0f}% "
                  f"总{tot:+6.1f}% avg{avg:+5.1f}%")

    # 事件邻近分析
    print(f"\n📅 距 FOMC ≤ 5 天 vs > 5 天 (STRADDLE):")
    if len(st_all) > 0:
        near = st_all[st_all["days_to_fomc"] <= 5]
        far = st_all[st_all["days_to_fomc"] > 5]
        if len(near) > 0:
            print(f"  ≤5天: {len(near)} 笔, 胜{(near['straddle_pnl_pct']>0).mean()*100:.0f}%, "
                  f"avg{near['straddle_pnl_pct'].mean():+.1f}%, "
                  f"max_avg{near['straddle_max_pnl_pct'].mean():+.1f}%")
        if len(far) > 0:
            print(f"  >5天: {len(far)} 笔, 胜{(far['straddle_pnl_pct']>0).mean()*100:.0f}%, "
                  f"avg{far['straddle_pnl_pct'].mean():+.1f}%")

    # 当前过滤效果对比
    print(f"\n🔄 v3.7.32 过滤前后 (STRADDLE pctile < 0.42 GLD / 0.20 SLV):")
    cur_th = 0.42 if asset == "GLD" else 0.20
    if len(st_all) > 0:
        kept = st_all[st_all["rv_pctile"] < cur_th]
        rejected = st_all[st_all["rv_pctile"] >= cur_th]
        print(f"  保留 (< {cur_th}): {len(kept)} 笔, 总 {kept['straddle_pnl_pct'].sum():+.1f}%, "
              f"avg {kept['straddle_pnl_pct'].mean():+.1f}%" if len(kept) else "  保留: 0")
        if len(rejected) > 0:
            print(f"  屏蔽 (≥ {cur_th}): {len(rejected)} 笔, 若入场 总 "
                  f"{rejected['straddle_pnl_pct'].sum():+.1f}%, "
                  f"avg {rejected['straddle_pnl_pct'].mean():+.1f}%, "
                  f"max_avg {rejected['straddle_max_pnl_pct'].mean():+.1f}%")
            print(f"  → 屏蔽损失: {-rejected['straddle_pnl_pct'].sum():+.1f}% 末日 / "
                  f"{-rejected['straddle_max_pnl_pct'].sum():+.1f}% 上帝视角")
