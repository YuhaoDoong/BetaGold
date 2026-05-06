"""ma_trend 阈值不同时, 信号月度分布 + 累计 PnL.

用户疑虑: 4月 GLD / 3月 SLV 几乎零方向性信号, ma_trend 是否过严?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def add_ma_trend(df, asset):
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                        index_col=0, parse_dates=True)
    ohlc["ma_trend"] = ohlc["Close"].rolling(20).mean() / ohlc["Close"].rolling(50).mean()
    df = df.copy()
    df["ma_trend"] = df["signal_date"].map(
        lambda d: ohlc["ma_trend"].get(pd.Timestamp(d).normalize(), np.nan))
    return df


from core.strategy_config import get_config

for asset in ["GLD", "SLV"]:
    cfg = get_config(asset)
    print(f"\n{'='*78}")
    print(f"{asset} ma_trend 阈值灵敏度 + 月度信号分布 (cfg thr = {cfg.ma_trend_threshold})")
    print(f"{'='*78}")
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    df = add_ma_trend(df, asset)
    df = df.dropna(subset=["ma_trend"])

    # 抽 BUY CALL (paired = 1 行/信号日)
    bc = df[df["strategy"]=="BUY CALL"].copy()
    print(f"\n所有 BC 信号 paired (5y): {len(bc)} 笔")
    print(f"\n{'thr':>6}{'n_kept':>8}{'每月平均':>10}"
           f"{'累计 sum':>11}{'平均/笔':>10}{'最长零信号月数':>15}")

    # ma_trend 阈值扫描
    for thr in [0.90, 0.93, 0.95, 0.97, 0.975, 0.99, 1.00]:
        sub = bc[bc["ma_trend"] >= thr].copy()
        if not len(sub): continue
        # 月度分布
        sub["ym"] = sub["signal_date"].dt.to_period("M")
        n_months_with_signal = sub["ym"].nunique()
        # 5y window 月数 ~ 60
        all_months = pd.period_range(bc["signal_date"].min().to_period("M"),
                                       bc["signal_date"].max().to_period("M"),
                                       freq="M")
        # 最长零信号月数
        sig_set = set(sub["ym"].unique())
        max_dry = 0; cur = 0
        for m in all_months:
            if m in sig_set: cur = 0
            else: cur += 1; max_dry = max(max_dry, cur)
        avg_per_mo = len(sub) / len(all_months)
        sum_pnl = sub["pnl_pct"].sum()
        avg_pnl = sub["pnl_pct"].mean()
        marker = " ←现行" if abs(thr - cfg.ma_trend_threshold) < 0.001 else ""
        print(f"  ≥{thr:>5.3f}{len(sub):>8}{avg_per_mo:>9.2f}"
              f"{sum_pnl:>+10.0f}%{avg_pnl:>+9.1f}%{max_dry:>14}{marker}")

    # 月度详细 (用现行 thr)
    print(f"\n现行 thr={cfg.ma_trend_threshold} 下月度信号数 (5y):")
    cur_sub = bc[bc["ma_trend"] >= cfg.ma_trend_threshold].copy()
    cur_sub["ym"] = cur_sub["signal_date"].dt.to_period("M")
    monthly = cur_sub.groupby("ym").size().reset_index(name="n_sigs")
    monthly = monthly.sort_values("ym")
    # 显示空月
    all_months = pd.period_range(bc["signal_date"].min().to_period("M"),
                                   bc["signal_date"].max().to_period("M"),
                                   freq="M")
    month_dict = {m: 0 for m in all_months}
    for _, r in monthly.iterrows(): month_dict[r["ym"]] = r["n_sigs"]
    # 最近 12 月
    recent_12 = list(month_dict.items())[-12:]
    print(f"  {'月':<10}{'信号数':>8}")
    for m, n in recent_12:
        print(f"  {str(m):<10}{n:>8}")
