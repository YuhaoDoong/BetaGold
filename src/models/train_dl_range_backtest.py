"""
DL Range 区间交易回测

交易逻辑:
    用 DL Range 预测构建动态价格区间
    价格接近下沿 → 买入 (预期反弹)
    价格接近上沿 → 卖出 (预期回落/止盈)

核心指标:
    单笔交易胜率 (期权杠杆需要高胜率)
    每笔 P&L 分布
    按 Regime 分段胜率

用法:
    conda activate gold
    python src/models/train_dl_range_backtest.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")


def load_data():
    """加载 DL Range 预测 + GLD 价格 + Regime。"""
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    return gld, range_df, regime


def build_band(range_df, gld_close):
    """
    用滞后 DL Range 预测构建价格区间。

    每天 t 的区间 = 过去 1~3 天预测的平均 (这些预测的5天窗口覆盖今天):
        upper = mean(close[t-k] × (1 + pred_upper[t-k]/100))  k=1,2,3
        lower = mean(close[t-k] × (1 + pred_lower[t-k]/100))  k=1,2,3
    """
    close = gld_close.reindex(range_df.index)

    uppers, lowers = [], []
    for lag in [1, 2, 3]:
        cl = close.shift(lag)
        pu = range_df["pred_upper_pct"].shift(lag)
        pl = range_df["pred_lower_pct"].shift(lag)
        uppers.append(cl * (1 + pu / 100))
        lowers.append(cl * (1 + pl / 100))

    upper_band = pd.concat(uppers, axis=1).mean(axis=1)
    lower_band = pd.concat(lowers, axis=1).mean(axis=1)
    bp = (close - lower_band) / (upper_band - lower_band)

    return upper_band, lower_band, bp


def run_backtest(close, bp, regime,
                 buy_thresh=0.15, sell_thresh=0.85,
                 max_hold=10, regime_filter="bull_mixed"):
    """
    区间交易回测 — 逐笔跟踪。

    Entry: bp < buy_thresh (且满足 regime 过滤)
    Exit:  bp > sell_thresh OR 持仓 >= max_hold 个交易日

    regime_filter:
        "none"       — 不用 regime
        "bull_mixed"  — Bull 或 Mixed 时才买入
        "bull_only"   — 只在 Bull 买入
    """
    trades = []
    in_trade = False
    entry_date = entry_price = entry_bp = entry_regime = None

    dates = bp.dropna().index

    for date in dates:
        price = close.get(date)
        bp_val = bp.get(date)
        reg = regime.get(date, "Mixed") if date in regime.index else "Mixed"

        if price is None or np.isnan(bp_val):
            continue

        if not in_trade:
            buy_ok = bp_val < buy_thresh
            if regime_filter == "bull_only":
                buy_ok = buy_ok and reg == "Bull"
            elif regime_filter == "bull_mixed":
                buy_ok = buy_ok and reg in ("Bull", "Mixed")

            if buy_ok:
                in_trade = True
                entry_date = date
                entry_price = price
                entry_bp = bp_val
                entry_regime = reg
        else:
            days_held = int(np.busday_count(
                entry_date.date(), date.date()))
            sell_signal = bp_val > sell_thresh
            timeout = days_held >= max_hold

            if sell_signal or timeout:
                pnl_pct = (price / entry_price - 1) * 100
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "days_held": days_held,
                    "entry_bp": entry_bp,
                    "exit_bp": bp_val,
                    "entry_regime": entry_regime,
                    "exit_reason": "signal" if sell_signal else "timeout",
                })
                in_trade = False

    return pd.DataFrame(trades)


def analyze_trades(trades_df):
    """计算交易统计。"""
    if len(trades_df) == 0:
        return {}

    n = len(trades_df)
    pnl = trades_df["pnl_pct"]
    wins = pnl > 0

    # 盈亏
    avg_win = pnl[wins].mean() if wins.any() else 0
    avg_loss = pnl[~wins].mean() if (~wins).any() else 0
    total_win = pnl[wins].sum() if wins.any() else 0
    total_loss = abs(pnl[~wins].sum()) if (~wins).any() else 0
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

    # 连胜连亏
    max_cw = max_cl = cw = cl = 0
    for w in wins:
        if w:
            cw += 1
            cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1
            cw = 0
            max_cl = max(max_cl, cl)

    # 按 Regime
    regime_stats = {}
    for reg in sorted(trades_df["entry_regime"].unique()):
        mask = trades_df["entry_regime"] == reg
        rt = trades_df[mask]
        regime_stats[reg] = {
            "n": len(rt),
            "win_rate": (rt["pnl_pct"] > 0).mean(),
            "avg_pnl": rt["pnl_pct"].mean(),
            "avg_win": rt.loc[rt["pnl_pct"] > 0, "pnl_pct"].mean()
            if (rt["pnl_pct"] > 0).any() else 0,
        }

    # 按退出类型
    for reason in ["signal", "timeout"]:
        mask = trades_df["exit_reason"] == reason
        if mask.sum() > 0:
            rt = trades_df[mask]
            regime_stats[f"exit_{reason}"] = {
                "n": len(rt),
                "win_rate": (rt["pnl_pct"] > 0).mean(),
                "avg_pnl": rt["pnl_pct"].mean(),
            }

    # 年化
    if n > 1:
        first = trades_df["entry_date"].min()
        last = trades_df["exit_date"].max()
        years = (last - first).days / 365.25
        trades_per_year = n / years if years > 0 else 0
        annual_pnl = pnl.mean() * trades_per_year
    else:
        trades_per_year = annual_pnl = 0

    # 累计PnL回撤
    cum_pnl = pnl.cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    return {
        "n_trades": n,
        "win_rate": wins.mean(),
        "avg_pnl": pnl.mean(),
        "median_pnl": pnl.median(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_consec_win": max_cw,
        "max_consec_loss": max_cl,
        "avg_hold_days": trades_df["days_held"].mean(),
        "cum_pnl": cum_pnl.iloc[-1],
        "max_dd": max_dd,
        "trades_per_year": trades_per_year,
        "annual_pnl": annual_pnl,
        "regime_stats": regime_stats,
    }


def print_results(stats, trades_df, name):
    """打印单个配置的详细结果。"""
    print(f"\n  {'='*58}")
    print(f"  {name}")
    print(f"  {'='*58}")

    if not stats:
        print("  0 trades")
        return

    s = stats
    print(f"  交易次数: {s['n_trades']}  "
          f"({s['trades_per_year']:.1f} trades/year)")
    print(f"  胜率:     {s['win_rate']:.1%}")
    print(f"  平均收益: {s['avg_pnl']:+.2f}%  "
          f"(median: {s['median_pnl']:+.2f}%)")
    print(f"  平均盈利: {s['avg_win']:+.2f}%")
    print(f"  平均亏损: {s['avg_loss']:+.2f}%")
    print(f"  盈亏比:   {s['profit_factor']:.2f}")
    print(f"  持仓天数: {s['avg_hold_days']:.1f}d")
    print(f"  连胜/连亏: {s['max_consec_win']} / {s['max_consec_loss']}")
    print(f"  累计PnL:  {s['cum_pnl']:+.1f}%  "
          f"(年化: {s['annual_pnl']:+.1f}%)")
    print(f"  最大回撤: {s['max_dd']:+.1f}% (累计PnL)")

    # Regime
    print(f"\n  按 Regime:")
    for key, rs in s["regime_stats"].items():
        if key.startswith("exit_"):
            continue
        print(f"    {key:8s}: n={rs['n']:3d}, "
              f"胜率={rs['win_rate']:.1%}, "
              f"avg={rs['avg_pnl']:+.2f}%, "
              f"avg_win={rs['avg_win']:+.2f}%")

    # Exit reason
    print(f"\n  按退出方式:")
    for key, rs in s["regime_stats"].items():
        if not key.startswith("exit_"):
            continue
        label = key.replace("exit_", "")
        print(f"    {label:8s}: n={rs['n']:3d}, "
              f"胜率={rs['win_rate']:.1%}, "
              f"avg={rs['avg_pnl']:+.2f}%")

    # PnL distribution
    pnl = trades_df["pnl_pct"]
    print(f"\n  收益分布:")
    print(f"    min={pnl.min():+.2f}%, "
          f"Q25={pnl.quantile(0.25):+.2f}%, "
          f"Q75={pnl.quantile(0.75):+.2f}%, "
          f"max={pnl.max():+.2f}%")

    # 最近 10 笔交易
    print(f"\n  最近 10 笔交易:")
    print(f"    {'Entry':>10s} {'Exit':>10s} {'EntPx':>7s} {'ExtPx':>7s} "
          f"{'PnL':>7s} {'Hold':>4s} {'BP_in':>5s} {'BP_out':>6s} "
          f"{'Regime':>6s} {'Reason':>7s}")
    for _, t in trades_df.tail(10).iterrows():
        print(f"    {t['entry_date'].strftime('%Y-%m-%d'):>10s} "
              f"{t['exit_date'].strftime('%Y-%m-%d'):>10s} "
              f"{t['entry_price']:7.1f} {t['exit_price']:7.1f} "
              f"{t['pnl_pct']:+7.2f}% {t['days_held']:4d}d "
              f"{t['entry_bp']:5.2f} {t['exit_bp']:6.2f} "
              f"{t['entry_regime']:>6s} {t['exit_reason']:>7s}")


def main():
    print("=" * 60)
    print("  DL Range 区间交易回测")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    gld, range_df, regime = load_data()
    gld_close = gld["Close"]

    upper_band, lower_band, bp = build_band(range_df, gld_close)

    valid = bp.dropna()
    print(f"  数据: {valid.index[0].date()} ~ {valid.index[-1].date()}")
    print(f"  样本: {len(valid)}")
    print(f"  Band Position: mean={bp.mean():.3f}, std={bp.std():.3f}")
    print(f"  bp < 0.15: {(bp < 0.15).sum()} days ({(bp < 0.15).mean():.1%})")
    print(f"  bp < 0.20: {(bp < 0.20).sum()} days ({(bp < 0.20).mean():.1%})")
    print(f"  bp > 0.85: {(bp > 0.85).sum()} days ({(bp > 0.85).mean():.1%})")

    # ============================================================
    # 多配置回测
    # ============================================================
    configs = [
        # (name, buy_thresh, sell_thresh, max_hold, regime_filter)
        ("A: No Regime filter (0.15/0.85, 10d)",
         0.15, 0.85, 10, "none"),
        ("B: Bull+Mixed (0.15/0.85, 10d)",
         0.15, 0.85, 10, "bull_mixed"),
        ("C: Bull Only (0.15/0.85, 10d)",
         0.15, 0.85, 10, "bull_only"),
        ("D: Bull+Mixed wider (0.20/0.80, 10d)",
         0.20, 0.80, 10, "bull_mixed"),
        ("E: Bull+Mixed 5d max",
         0.15, 0.85, 5, "bull_mixed"),
        ("F: Bull+Mixed aggressive (0.25/0.75, 8d)",
         0.25, 0.75, 8, "bull_mixed"),
        ("G: Bull Only aggressive (0.25/0.75, 8d)",
         0.25, 0.75, 8, "bull_only"),
    ]

    all_results = {}
    for name, buy_t, sell_t, max_h, reg_f in configs:
        trades = run_backtest(
            gld_close, bp, regime,
            buy_thresh=buy_t, sell_thresh=sell_t,
            max_hold=max_h, regime_filter=reg_f)
        stats = analyze_trades(trades)
        print_results(stats, trades, name)
        all_results[name] = {"stats": stats, "trades": trades}

    # ============================================================
    # 汇总表
    # ============================================================
    print(f"\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    print(f"\n  {'Config':42s} {'N':>4s} {'Win%':>6s} {'AvgPnL':>7s} "
          f"{'MedPnL':>7s} {'PF':>5s} {'Hold':>5s} {'Ann%':>6s}")
    print(f"  {'-'*82}")

    best_name = None
    best_score = -1

    for name, res in all_results.items():
        s = res["stats"]
        if not s:
            continue
        # 评分: 胜率 × sqrt(交易次数) × avg_pnl (如果正)
        score = s["win_rate"] * np.sqrt(s["n_trades"])
        if s["avg_pnl"] > 0:
            score *= (1 + s["avg_pnl"] / 2)
        if score > best_score:
            best_score = score
            best_name = name

        print(f"  {name:42s} {s['n_trades']:4d} {s['win_rate']:6.1%} "
              f"{s['avg_pnl']:+7.2f}% {s['median_pnl']:+7.2f}% "
              f"{s['profit_factor']:5.2f} {s['avg_hold_days']:5.1f}d "
              f"{s['annual_pnl']:+6.1f}%")

    if best_name:
        print(f"\n  推荐: {best_name}")
        best_trades = all_results[best_name]["trades"]
        best_stats = all_results[best_name]["stats"]

        # 年度分析
        if len(best_trades) > 0:
            best_trades = best_trades.copy()
            best_trades["year"] = best_trades["entry_date"].dt.year
            print(f"\n  年度分析 ({best_name}):")
            print(f"    {'Year':>6s} {'N':>4s} {'Win%':>6s} {'AvgPnL':>7s} "
                  f"{'CumPnL':>7s}")
            print(f"    {'-'*35}")
            for yr, grp in best_trades.groupby("year"):
                w = (grp["pnl_pct"] > 0).mean()
                print(f"    {yr:6d} {len(grp):4d} {w:6.1%} "
                      f"{grp['pnl_pct'].mean():+7.2f}% "
                      f"{grp['pnl_pct'].sum():+7.1f}%")

    # ============================================================
    # 保存
    # ============================================================
    if best_name and len(all_results[best_name]["trades"]) > 0:
        out_dir = os.path.join(PROJECT_ROOT, "data", "models")
        out_path = os.path.join(out_dir, "dl_range_backtest_trades.parquet")
        all_results[best_name]["trades"].to_parquet(out_path)
        print(f"\n  保存: {out_path}")


if __name__ == "__main__":
    main()
