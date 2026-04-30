"""定期重测 — Per-asset 阈值 grid search.

用法:
    python scripts/tune_thresholds.py --asset GLD
    python scripts/tune_thresholds.py --asset SLV
    python scripts/tune_thresholds.py --asset GLD --param rv_filter --years 5
    python scripts/tune_thresholds.py --asset SLV --param short_vol --years 8

参数:
    --asset: GLD / SLV / (未来 QQQ 等)
    --param: rv_filter / short_vol / all (默认 all)
    --years: 回测窗口年数 (默认 5)
    --step: 网格步长 (默认 0.025)

输出:
    1. Top 10 配置 (按 Sharpe / 总收益)
    2. 当前 strategy_config 在该资产上的表现
    3. 推荐参数 (与当前对比)
    4. 详细 CSV 保存到 tmp/

定期建议:
    - 每月跑一次 (新月数据足够建立趋势)
    - 重大事件后 (regime 切换/新资产) 立即跑
    - 参数变化 > 10% 才考虑切换 (避免噪音)
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from core.data import load_features, load_config
from core.signals import compute_rv_pctile, build_band
from core.signals_v2 import run_backtest
from core.events import backtest_short_vol
from core.regime import RegimeClassifier
from core.strategy_config import get_config, ASSET_CONFIGS


def load_asset(asset: str):
    """加载某资产的全部数据 (close/high/low, OOS, regime, RV)."""
    cfg = load_config()
    fname = "gld.csv" if asset == "GLD" else "slv.csv"
    path = f"/Users/yhdong/Gold/data/raw/market/{fname}"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    features = load_features(cfg)
    common = features.index.intersection(df.index)
    features = features.loc[common]
    close = df["Close"][common]
    high = df["High"][common]
    low = df["Low"][common]

    ret = close.pct_change()
    rv_10d = ret.rolling(10).std() * (252 ** 0.5) * 100
    rv_pct = compute_rv_pctile(rv_10d)

    oos_path = f"/Users/yhdong/Gold/data/models/dl_range_{asset.lower()}_oos.parquet"
    if not os.path.exists(oos_path):
        oos_path = "/Users/yhdong/Gold/data/models/dl_range_oos.parquet"
    oos = pd.read_parquet(oos_path)
    upper, lower, _ = build_band(oos, close)

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    return {
        "close": close, "high": high, "low": low,
        "rv_10d": rv_10d, "rv_pct": rv_pct,
        "upper": upper, "lower": lower,
        "regime": regime, "features": features,
    }


def grid_directional(data, years, step):
    """方向性 RV_LOW × RV_HIGH 网格."""
    start = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
    lows = np.arange(0.30, 0.575, step)
    highs = np.arange(0.65, 0.975, step)
    results = []
    for lo in lows:
        for hi in highs:
            if lo >= hi - 0.05:
                continue
            trades = run_backtest(
                data["close"], data["high"], data["low"],
                data["upper"], data["lower"],
                data["regime"], data["rv_pct"],
                start_date=start, entry_price_mode="close",
                rv_low=round(lo, 3), rv_high=round(hi, 3),
            )
            closed = [t for t in trades if not t.get("active")]
            if not closed:
                continue
            pnls = [t["gain"] for t in closed]
            wins = sum(1 for t in closed if t["gain"] > 0)
            nav = [100.0]
            for g in pnls:
                nav.append(nav[-1] * (1 + g / 100))
            nav = np.array(nav)
            peak = np.maximum.accumulate(nav)
            max_dd = ((nav - peak) / peak).min() * 100
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
            results.append({
                "lo": round(lo, 3), "hi": round(hi, 3),
                "n": len(closed), "wr": wins / len(closed),
                "total": sum(pnls), "avg": np.mean(pnls),
                "sharpe": sharpe, "max_dd": max_dd,
            })
    return pd.DataFrame(results)


def grid_straddle_pctile(data, years, step=0.01):
    """STRADDLE RV %tile 上限网格 (步长统一 0.01).

    在 baseline (无 RV %tile 过滤) 基础上, 测试不同上限阈值的影响.
    """
    from core.events import backtest_straddle
    start = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
    dates = data["features"].index[data["features"].index >= start]
    trades_base = backtest_straddle(
        data["close"], data["high"], data["low"],
        data["rv_10d"], dates,
    )
    if not trades_base:
        return pd.DataFrame()

    rv_pct = data["rv_pct"]
    results = []
    for th in np.arange(0.20, 1.01, step):
        th = round(th, 3)
        filtered = [t for t in trades_base
                    if rv_pct.get(t["entry_date"], 0.5) < th]
        if not filtered:
            continue
        pnls = [t["pnl_pct"] for t in filtered]
        wins = sum(1 for t in filtered if t["pnl_pct"] > 0)
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
        results.append({
            "th": th, "n": len(filtered), "wr": wins / len(filtered),
            "total": sum(pnls), "sharpe": sharpe,
        })
    return pd.DataFrame(results)


def grid_short_vol(data, years, step):
    """SHORT_VOL Iron Condor RV %tile 中位窄带网格."""
    start = pd.Timestamp.now() - pd.Timedelta(days=years * 365)
    los = np.arange(0.20, 0.475, step)
    his = np.arange(0.55, 0.825, step)
    results = []
    daily_range = (data["high"] - data["low"]) / data["close"] * 100
    for lo in los:
        for hi in his:
            if lo >= hi - 0.10:
                continue
            trades = backtest_short_vol(
                data["close"], data["high"], data["low"],
                data["rv_10d"], data["rv_pct"],
                data["features"].index[data["features"].index >= start],
                rv_pctile_lo=round(lo, 3), rv_pctile_hi=round(hi, 3),
                regime=data["regime"], daily_range=daily_range,
            )
            if not trades:
                continue
            pnls = [t["pnl_pct"] for t in trades]
            wins = sum(1 for t in trades if t["win"])
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
            results.append({
                "lo": round(lo, 3), "hi": round(hi, 3),
                "n": len(trades), "wr": wins / len(trades),
                "total": sum(pnls), "sharpe": sharpe,
            })
    return pd.DataFrame(results)


def report(df, label, current_lo, current_hi):
    """打印 Top 配置 + 与当前配置对比."""
    if len(df) == 0:
        print(f"  [{label}] 无有效配置")
        return
    print(f"\n=== {label}: Top 5 by Sharpe ===")
    print(df.nlargest(5, "sharpe").to_string(index=False))
    print(f"\n=== {label}: Top 5 by Total ===")
    print(df.nlargest(5, "total").to_string(index=False))
    cur = df[(df["lo"] == current_lo) & (df["hi"] == current_hi)]
    if len(cur):
        r = cur.iloc[0]
        print(f"\n[当前 {current_lo}/{current_hi}]: "
              f"n={r['n']:.0f} 胜{r['wr']*100:.0f}% "
              f"总{r['total']:+.1f}% Sharpe{r['sharpe']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="GLD", choices=["GLD", "SLV"])
    parser.add_argument("--param", default="all",
                         choices=["rv_filter", "short_vol", "straddle", "all"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--out-dir", default="/tmp")
    args = parser.parse_args()

    print(f"加载 {args.asset} 数据 (近 {args.years} 年)...")
    data = load_asset(args.asset)
    cur = get_config(args.asset)
    print(f"当前配置: rv_filter={cur.rv_filter_low}/{cur.rv_filter_high}, "
          f"short_vol={cur.short_vol_rv_pctile_lo}/{cur.short_vol_rv_pctile_hi}")

    if args.param in ("rv_filter", "all"):
        print(f"\n[1/2] 方向性 RV 阈值网格 (步长 {args.step})...")
        df_dir = grid_directional(data, args.years, args.step)
        report(df_dir, "方向性 RV", cur.rv_filter_low, cur.rv_filter_high)
        out_path = f"{args.out_dir}/grid_{args.asset}_directional.csv"
        df_dir.to_csv(out_path, index=False)
        print(f"\n详细数据 → {out_path}")

    if args.param in ("short_vol", "all"):
        print(f"\n[2/3] SHORT_VOL Iron Condor 网格 (步长 {args.step})...")
        df_sv = grid_short_vol(data, args.years, args.step)
        report(df_sv, "SHORT_VOL", cur.short_vol_rv_pctile_lo,
                cur.short_vol_rv_pctile_hi)
        out_path = f"{args.out_dir}/grid_{args.asset}_shortvol.csv"
        df_sv.to_csv(out_path, index=False)
        print(f"\n详细数据 → {out_path}")

    if args.param in ("straddle", "all"):
        # STRADDLE 用统一步长 0.01 (v3.7.33+ 全局精度规范)
        from core.strategy_config import GRID_PRECISION
        print(f"\n[3/3] STRADDLE RV %tile 上限网格 (步长 {GRID_PRECISION['rv_pctile']})...")
        df_st = grid_straddle_pctile(data, args.years,
                                       GRID_PRECISION["rv_pctile"])
        if len(df_st) > 0:
            print(f"\n=== STRADDLE: Top 5 by Sharpe (n ≥ 25) ===")
            sub = df_st[df_st["n"] >= 25]
            if len(sub) > 0:
                print(sub.nlargest(5, "sharpe").to_string(index=False))
            cur_st = df_st[df_st["th"] == cur.straddle_rv_pctile_max]
            if len(cur_st):
                r = cur_st.iloc[0]
                print(f"\n[当前 < {cur.straddle_rv_pctile_max}]: "
                      f"n={r['n']:.0f} 胜{r['wr']*100:.0f}% "
                      f"总{r['total']:+.1f}% Sharpe{r['sharpe']:.3f}")
            out_path = f"{args.out_dir}/grid_{args.asset}_straddle.csv"
            df_st.to_csv(out_path, index=False)
            print(f"\n详细数据 → {out_path}")

    print(f"\n下一步: 如果某 Top 配置 vs 当前 > 5% 改进, 编辑")
    print(f"    core/strategy_config.py ASSET_CONFIGS['{args.asset}']")
    print(f"    更新 last_tuned + notes")


if __name__ == "__main__":
    main()
