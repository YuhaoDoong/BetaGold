"""
信号优化对比分析:
  1. Bull 区间阈值上移 (bp<0.20 → bp<0.30)
  2. RV 辅助方向性买卖信号
  3. RV 独立波动率信号 (Phase 4B)
  4. 综合优化 vs 基线对比

用法:
    conda activate gold
    python src/models/analysis_optimized_signals.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "regime_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 数据加载
# ============================================================

def load_all():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))

    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]

    rv_20d = features["rv_20d"] if "rv_20d" in features.columns else None

    return gld, range_df, regime, rv_20d, features


def build_band(range_df, gld_close):
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


def compute_rv_percentile(rv_20d, window=252):
    """RV 在过去 window 天的百分位。"""
    return rv_20d.rolling(window, min_periods=60).rank(pct=True)


def compute_forward(gld, dates):
    """前瞻收益 + MAE + 前瞻RV变化。"""
    close = gld["Close"]
    high = gld["High"]
    low = gld["Low"]
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(20).std() * np.sqrt(252) * 100

    result = pd.DataFrame(index=dates)

    for h in [5, 10, 15, 20]:
        result[f"fwd_ret_{h}d"] = ((close.shift(-h) / close - 1) * 100).reindex(dates)

    # 5d / 10d MAE
    for h in [5, 10]:
        mae = pd.Series(dtype=float, index=dates)
        for d in dates:
            loc = close.index.get_loc(d)
            end_loc = min(loc + h, len(close) - 1)
            if end_loc <= loc:
                continue
            min_low = low.iloc[loc + 1: end_loc + 1].min()
            mae.loc[d] = (min_low / close.iloc[loc] - 1) * 100
        result[f"mae_{h}d"] = mae

    # 前瞻 RV 变化
    rv_fwd_20d = rv.shift(-20)
    result["rv_now"] = rv.reindex(dates)
    result["rv_fwd_20d"] = rv_fwd_20d.reindex(dates)
    result["rv_change_20d"] = ((rv_fwd_20d / rv - 1) * 100).reindex(dates)

    return result


# ============================================================
# 信号生成
# ============================================================

def generate_signals(bp, regime, rv_pctile, rv_20d):
    """生成所有买卖信号。"""
    dates = bp.dropna().index
    reg = regime.reindex(dates)
    bp_s = bp.reindex(dates)
    rv_p = rv_pctile.reindex(dates)
    rv_raw = rv_20d.reindex(dates)

    is_bull = reg == "Bull"

    signals = pd.DataFrame(index=dates)

    # === 方向性买入信号 ===

    # A: 基线 — Bull + bp<0.20 (OLD catch knife)
    bp_cross_down_020 = (bp_s < 0.20) & (bp_s.shift(1) >= 0.20)
    signals["buy_A_baseline"] = is_bull & bp_cross_down_020

    # B: 阈值上移 — Bull + bp<0.30
    bp_cross_down_030 = (bp_s < 0.30) & (bp_s.shift(1) >= 0.30)
    signals["buy_B_wider"] = is_bull & bp_cross_down_030

    # C: 阈值上移+RV过滤 — Bull + bp<0.30 + RV不在极高位
    signals["buy_C_wider_rv"] = is_bull & bp_cross_down_030 & (rv_p < 0.75)

    # D: RV极低辅助 — Bull + RV_pctile<0.15 + bp<0.50
    rv_enter_low = (rv_p < 0.15) & (rv_p.shift(1) >= 0.15)
    signals["buy_D_rv_low"] = is_bull & rv_enter_low & (bp_s < 0.50)

    # E: 综合 — C 或 D (两条入场通道)
    signals["buy_E_combined"] = signals["buy_C_wider_rv"] | signals["buy_D_rv_low"]

    # === 方向性卖出信号 ===

    # 基线: bp>0.80
    bp_cross_up_080 = (bp_s > 0.80) & (bp_s.shift(1) <= 0.80)
    signals["sell_baseline"] = bp_cross_up_080

    # 上移: bp>0.90
    bp_cross_up_090 = (bp_s > 0.90) & (bp_s.shift(1) <= 0.90)
    signals["sell_bp090"] = bp_cross_up_090

    # RV 极高卖出: RV_pctile > 0.80
    rv_enter_high = (rv_p > 0.80) & (rv_p.shift(1) <= 0.80)
    signals["sell_rv_high"] = rv_enter_high

    # Regime 退出 Bull
    bull_exit = is_bull.shift(1) & (~is_bull)
    signals["sell_regime_exit"] = bull_exit

    # 综合卖出: RV极高 或 Regime退出
    signals["sell_combined"] = signals["sell_rv_high"] | signals["sell_regime_exit"]

    # === RV 独立波动率信号 (Phase 4B) ===

    # 波动率做多: RV极低 → 预期波动扩大 → 买入straddle/strangle
    signals["vol_buy"] = (rv_p < 0.15) & (rv_p.shift(1) >= 0.15)

    # 波动率做空: RV极高 → 预期波动收缩 → 卖出straddle/iron condor
    signals["vol_sell"] = (rv_p > 0.85) & (rv_p.shift(1) <= 0.85)

    # 也记录连续状态 (用于持仓区间)
    signals["rv_pctile"] = rv_p
    signals["rv_raw"] = rv_raw
    signals["bp"] = bp_s
    signals["regime"] = reg

    return signals


# ============================================================
# 评估函数
# ============================================================

def eval_signal(name, signal_mask, fwd_df, horizons=[5, 10, 15]):
    """评估一组信号的前瞻表现。"""
    idx = signal_mask[signal_mask].index
    idx = idx.intersection(fwd_df.index)
    n = len(idx)
    if n < 3:
        return {"name": name, "n": n}

    result = {"name": name, "n": n}
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col not in fwd_df.columns:
            continue
        rets = fwd_df.loc[idx, col].dropna()
        if len(rets) < 3:
            continue
        result[f"wr_{h}d"] = (rets > 0).mean()
        result[f"avg_{h}d"] = rets.mean()
        result[f"med_{h}d"] = rets.median()
        wins = rets[rets > 0].sum()
        losses = abs(rets[rets <= 0].sum())
        result[f"pf_{h}d"] = wins / losses if losses > 0 else float("inf")

    # MAE
    for h in [5, 10]:
        col = f"mae_{h}d"
        if col in fwd_df.columns:
            mae = fwd_df.loc[idx, col].dropna()
            if len(mae) > 0:
                result[f"mae_{h}d"] = mae.mean()

    # RV 变化 (for vol signals)
    if "rv_change_20d" in fwd_df.columns:
        rv_chg = fwd_df.loc[idx, "rv_change_20d"].dropna()
        if len(rv_chg) > 0:
            result["rv_chg_20d_mean"] = rv_chg.mean()
            result["rv_chg_20d_med"] = rv_chg.median()
            result["rv_decline_rate"] = (rv_chg < 0).mean()

    # 年化频率
    if n > 1:
        span_years = (idx[-1] - idx[0]).days / 365.25
        result["per_year"] = n / span_years if span_years > 0 else 0

    return result


def print_signal_table(results_list, title, show_rv=False):
    """打印信号对比表。"""
    print(f"\n  === {title} ===")

    if show_rv:
        print(f"  {'Signal':30s} {'N':>4s} {'次/年':>5s} "
              f"{'5dWR':>6s} {'5dAvg':>7s} {'5dPF':>5s} "
              f"{'10dWR':>6s} {'10dAvg':>7s} "
              f"{'5dMAE':>6s} "
              f"{'RV↓率':>5s} {'RVchg':>7s}")
        print(f"  {'-' * 100}")
    else:
        print(f"  {'Signal':30s} {'N':>4s} {'次/年':>5s} "
              f"{'5dWR':>6s} {'5dAvg':>7s} {'5dPF':>5s} "
              f"{'10dWR':>6s} {'10dAvg':>7s} {'10dPF':>6s} "
              f"{'5dMAE':>6s} {'10dMAE':>7s}")
        print(f"  {'-' * 105}")

    for r in results_list:
        if r["n"] < 3:
            print(f"  {r['name']:30s} {r['n']:4d}  (样本不足)")
            continue

        py = r.get("per_year", 0)
        if show_rv:
            rv_decline = r.get("rv_decline_rate", np.nan)
            rv_chg = r.get("rv_chg_20d_mean", np.nan)
            print(f"  {r['name']:30s} {r['n']:4d} {py:5.1f} "
                  f"{r.get('wr_5d', np.nan):6.1%} {r.get('avg_5d', np.nan):+7.2f}% "
                  f"{r.get('pf_5d', np.nan):5.2f} "
                  f"{r.get('wr_10d', np.nan):6.1%} {r.get('avg_10d', np.nan):+7.2f}% "
                  f"{r.get('mae_5d', np.nan):+6.2f}% "
                  f"{rv_decline:5.0%} {rv_chg:+7.1f}%")
        else:
            print(f"  {r['name']:30s} {r['n']:4d} {py:5.1f} "
                  f"{r.get('wr_5d', np.nan):6.1%} {r.get('avg_5d', np.nan):+7.2f}% "
                  f"{r.get('pf_5d', np.nan):5.2f} "
                  f"{r.get('wr_10d', np.nan):6.1%} {r.get('avg_10d', np.nan):+7.2f}% "
                  f"{r.get('pf_10d', np.nan):6.2f} "
                  f"{r.get('mae_5d', np.nan):+6.2f}% {r.get('mae_10d', np.nan):+7.2f}%")


# ============================================================
# 可视化
# ============================================================

def fig_comparison(signals, fwd_df, gld, range_df):
    """生成4面板对比图。"""
    close = gld["Close"]

    fig, axes = plt.subplots(4, 1, figsize=(20, 22),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]},
                             sharex=True)

    # 近2年
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2026-03-31")

    common = signals.index[(signals.index >= start) & (signals.index <= end)]
    close_s = close.reindex(common)

    # --- Panel 1: 价格 + 买卖信号对比 ---
    ax = axes[0]
    ax.plot(common, close_s, color="black", linewidth=1.2, label="GLD")

    # Regime 着色
    reg = signals["regime"].reindex(common)
    bull_mask = reg == "Bull"
    starts = bull_mask & (~bull_mask.shift(1, fill_value=False))
    ends = bull_mask & (~bull_mask.shift(-1, fill_value=False))
    for s, e in zip(common[starts], common[ends]):
        ax.axvspan(s, e, alpha=0.1, color="green")

    # 基线买入 (A)
    buy_a = signals["buy_A_baseline"].reindex(common).fillna(False)
    if buy_a.sum() > 0:
        ax.scatter(common[buy_a], close_s[buy_a],
                   marker="^", s=120, color="blue", zorder=5, label="Baseline buy (bp<0.20)")

    # 优化买入 (E = C+D)
    buy_e = signals["buy_E_combined"].reindex(common).fillna(False)
    # 只画 E 中不在 A 里的新信号
    buy_new = buy_e & (~buy_a)
    if buy_new.sum() > 0:
        ax.scatter(common[buy_new], close_s[buy_new],
                   marker="^", s=120, color="lime", zorder=5,
                   edgecolors="darkgreen", label="New buy (bp<0.30/RV low)")

    # RV极低辅助买入 (D)
    buy_d = signals["buy_D_rv_low"].reindex(common).fillna(False)
    buy_d_only = buy_d & (~buy_a) & (~signals["buy_C_wider_rv"].reindex(common).fillna(False))
    if buy_d_only.sum() > 0:
        ax.scatter(common[buy_d_only], close_s[buy_d_only] * 0.998,
                   marker="D", s=80, color="cyan", zorder=5,
                   edgecolors="darkblue", label="RV low buy")

    # 卖出信号
    sell_base = signals["sell_baseline"].reindex(common).fillna(False)
    sell_comb = signals["sell_combined"].reindex(common).fillna(False)

    if sell_base.sum() > 0:
        ax.scatter(common[sell_base], close_s[sell_base],
                   marker="v", s=100, color="red", alpha=0.5, zorder=5, label="Baseline sell (bp>0.80)")

    sell_new = sell_comb & (~sell_base)
    if sell_new.sum() > 0:
        ax.scatter(common[sell_new], close_s[sell_new],
                   marker="v", s=100, color="orange", zorder=5,
                   edgecolors="red", label="New sell (RV high/regime exit)")

    ax.set_ylabel("GLD Price ($)")
    ax.set_title("Signal Comparison: Baseline vs Optimized (2024-2025)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Band Position ---
    ax = axes[1]
    bp = signals["bp"].reindex(common)
    ax.plot(common, bp, color="steelblue", linewidth=0.8)
    ax.axhline(0.20, color="blue", linewidth=1, linestyle="--", alpha=0.6, label="bp=0.20 (old)")
    ax.axhline(0.30, color="green", linewidth=1, linestyle="--", alpha=0.6, label="bp=0.30 (new)")
    ax.axhline(0.80, color="red", linewidth=1, linestyle="--", alpha=0.6, label="bp=0.80 (old sell)")
    ax.axhline(0.90, color="orange", linewidth=1, linestyle="--", alpha=0.6, label="bp=0.90 (new sell)")
    ax.fill_between(common, 0, 0.30, alpha=0.05, color="green")
    ax.fill_between(common, 0.80, 1.2, alpha=0.05, color="red")
    ax.set_ylabel("Band Position")
    ax.set_ylim(-0.2, 1.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: RV Percentile ---
    ax = axes[2]
    rv_p = signals["rv_pctile"].reindex(common)
    ax.plot(common, rv_p, color="darkorange", linewidth=1)
    ax.axhline(0.15, color="cyan", linewidth=1.5, linestyle="--", label="RV pctile=0.15 (vol buy)")
    ax.axhline(0.85, color="red", linewidth=1.5, linestyle="--", label="RV pctile=0.85 (vol sell)")
    ax.fill_between(common, 0, 0.15, alpha=0.1, color="cyan")
    ax.fill_between(common, 0.85, 1, alpha=0.1, color="red")

    # 波动率独立信号
    vol_buy = signals["vol_buy"].reindex(common).fillna(False)
    vol_sell = signals["vol_sell"].reindex(common).fillna(False)
    if vol_buy.sum() > 0:
        ax.scatter(common[vol_buy], rv_p[vol_buy],
                   marker="^", s=100, color="cyan", edgecolors="blue",
                   zorder=5, label=f"Vol BUY (n={vol_buy.sum()})")
    if vol_sell.sum() > 0:
        ax.scatter(common[vol_sell], rv_p[vol_sell],
                   marker="v", s=100, color="red", edgecolors="darkred",
                   zorder=5, label=f"Vol SELL (n={vol_sell.sum()})")
    ax.set_ylabel("RV Percentile (252d)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: RV raw ---
    ax = axes[3]
    rv_raw = signals["rv_raw"].reindex(common)
    ax.plot(common, rv_raw, color="steelblue", linewidth=1, label="RV 20d")
    ax.axhline(rv_raw.mean(), color="red", linewidth=0.5, linestyle="--",
               label=f"Mean={rv_raw.mean():.1f}")
    ax.set_ylabel("RV 20d")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "09_signal_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def fig_bar_comparison(results_buy, results_sell, results_vol):
    """柱状图对比: 基线 vs 优化。"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- 买入信号对比 ---
    ax = axes[0]
    names = [r["name"] for r in results_buy if r["n"] >= 3]
    wr5 = [r.get("wr_5d", 0) for r in results_buy if r["n"] >= 3]
    wr10 = [r.get("wr_10d", 0) for r in results_buy if r["n"] >= 3]
    counts = [r["n"] for r in results_buy if r["n"] >= 3]

    x = np.arange(len(names))
    w = 0.3
    bars1 = ax.bar(x - w / 2, [v * 100 for v in wr5], w, label="5d Win%", color="steelblue", alpha=0.7)
    bars2 = ax.bar(x + w / 2, [v * 100 for v in wr10], w, label="10d Win%", color="darkgreen", alpha=0.7)

    for i, (b1, b2, n) in enumerate(zip(bars1, bars2, counts)):
        ax.annotate(f"n={n}", xy=(i, max(b1.get_height(), b2.get_height()) + 1),
                    fontsize=8, ha="center")

    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    short_names = [n.split("_", 2)[-1] if len(n) > 15 else n for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Buy Signals: Win Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- 卖出信号对比 ---
    ax = axes[1]
    names_s = [r["name"] for r in results_sell if r["n"] >= 3]
    # 卖出信号看的是 5d 后下跌的概率
    decline5 = [1 - r.get("wr_5d", 0.5) for r in results_sell if r["n"] >= 3]
    decline10 = [1 - r.get("wr_10d", 0.5) for r in results_sell if r["n"] >= 3]
    counts_s = [r["n"] for r in results_sell if r["n"] >= 3]

    x = np.arange(len(names_s))
    bars1 = ax.bar(x - w / 2, [v * 100 for v in decline5], w, label="5d Decline%", color="salmon", alpha=0.7)
    bars2 = ax.bar(x + w / 2, [v * 100 for v in decline10], w, label="10d Decline%", color="darkred", alpha=0.7)

    for i, (b1, b2, n) in enumerate(zip(bars1, bars2, counts_s)):
        ax.annotate(f"n={n}", xy=(i, max(b1.get_height(), b2.get_height()) + 1),
                    fontsize=8, ha="center")

    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    short_s = [n.split("_", 1)[-1] if len(n) > 15 else n for n in names_s]
    ax.set_xticks(x)
    ax.set_xticklabels(short_s, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Decline Rate (%)")
    ax.set_title("Sell Signals: Post-Signal Decline Rate")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- RV 波动率信号 ---
    ax = axes[2]
    names_v = [r["name"] for r in results_vol if r["n"] >= 3]
    rv_chg = [r.get("rv_chg_20d_mean", 0) for r in results_vol if r["n"] >= 3]
    rv_correct = [r.get("rv_decline_rate", 0.5) for r in results_vol if r["n"] >= 3]
    counts_v = [r["n"] for r in results_vol if r["n"] >= 3]

    x = np.arange(len(names_v))
    colors_v = ["cyan" if "buy" in n.lower() else "red" for n in names_v]
    bars = ax.bar(x, rv_chg, color=colors_v, alpha=0.7, edgecolor="gray")
    for i, (b, n, corr) in enumerate(zip(bars, counts_v, rv_correct)):
        label = f"n={n}\ncorrect={corr:.0%}"
        ax.annotate(label, xy=(i, b.get_height()), fontsize=8, ha="center",
                    va="bottom" if b.get_height() > 0 else "top")

    ax.axhline(0, color="black", linewidth=0.5)
    short_v = [n.replace("vol_", "") for n in names_v]
    ax.set_xticks(x)
    ax.set_xticklabels(short_v, fontsize=9)
    ax.set_ylabel("Avg RV Change 20d (%)")
    ax.set_title("Vol Signals: Future RV Change")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Baseline vs Optimized Signal Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "10_signal_bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  信号优化对比分析")
    print("=" * 70)

    gld, range_df, regime, rv_20d, features = load_all()

    # 构建区间和 RV 百分位
    upper_band, lower_band, bp = build_band(range_df, gld["Close"])
    rv_pctile = compute_rv_percentile(rv_20d)

    # 生成信号
    signals = generate_signals(bp, regime, rv_pctile, rv_20d)
    dates = signals.index

    print(f"  数据: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} 天)")
    print(f"  Bull: {(signals['regime']=='Bull').sum()} 天")

    # 计算前瞻收益
    print(f"  计算前瞻收益 + MAE ...")
    fwd = compute_forward(gld, dates)

    # ============================================================
    # 1. 买入信号对比
    # ============================================================
    print(f"\n{'='*70}")
    print("  1. 买入信号对比")
    print(f"{'='*70}")

    buy_signals = [
        ("buy_A: Bull+bp<0.20 (基线)", "buy_A_baseline"),
        ("buy_B: Bull+bp<0.30 (上移)", "buy_B_wider"),
        ("buy_C: Bull+bp<0.30+RV<75%", "buy_C_wider_rv"),
        ("buy_D: Bull+RV_low+bp<0.50", "buy_D_rv_low"),
        ("buy_E: C+D 综合", "buy_E_combined"),
    ]

    results_buy = []
    for name, col in buy_signals:
        r = eval_signal(name, signals[col], fwd)
        results_buy.append(r)

    print_signal_table(results_buy, "买入信号对比")

    # 增量分析: E 中新增的信号 (不在 A 中) 质量如何
    new_in_e = signals["buy_E_combined"] & (~signals["buy_A_baseline"])
    r_new = eval_signal("E中新增 (非A)", new_in_e, fwd)
    r_old = eval_signal("E中保留 (与A重叠)", signals["buy_E_combined"] & signals["buy_A_baseline"], fwd)
    print_signal_table([r_old, r_new], "综合信号E的增量分析")

    # ============================================================
    # 2. 卖出信号对比
    # ============================================================
    print(f"\n{'='*70}")
    print("  2. 卖出信号对比")
    print(f"{'='*70}")

    sell_signals = [
        ("sell: bp>0.80 (基线)", "sell_baseline"),
        ("sell: bp>0.90 (上移)", "sell_bp090"),
        ("sell: RV_pctile>0.80", "sell_rv_high"),
        ("sell: Regime退出Bull", "sell_regime_exit"),
        ("sell: RV高+Regime退出", "sell_combined"),
    ]

    results_sell = []
    for name, col in sell_signals:
        r = eval_signal(name, signals[col], fwd)
        results_sell.append(r)

    print_signal_table(results_sell, "卖出信号对比 (看下跌率)")

    # ============================================================
    # 3. RV 独立波动率信号 (Phase 4B)
    # ============================================================
    print(f"\n{'='*70}")
    print("  3. RV 独立波动率信号 (Phase 4B)")
    print(f"{'='*70}")

    vol_signals = [
        ("vol_buy: RV<15% (做多波动)", "vol_buy"),
        ("vol_sell: RV>85% (做空波动)", "vol_sell"),
    ]

    results_vol = []
    for name, col in vol_signals:
        r = eval_signal(name, signals[col], fwd)
        results_vol.append(r)

    print_signal_table(results_vol, "RV 波动率信号", show_rv=True)

    # 更详细的 RV 信号分析
    print(f"\n  === RV 波动率信号详细 ===")
    for name, col in vol_signals:
        idx = signals[col][signals[col]].index
        idx = idx.intersection(fwd.index)
        if len(idx) < 3:
            continue
        print(f"\n  {name} (n={len(idx)}):")
        rv_now = fwd.loc[idx, "rv_now"]
        rv_fwd = fwd.loc[idx, "rv_fwd_20d"]
        rv_chg = fwd.loc[idx, "rv_change_20d"]
        print(f"    触发时RV: mean={rv_now.mean():.1f}, med={rv_now.median():.1f}")
        print(f"    20d后RV:  mean={rv_fwd.mean():.1f}, med={rv_fwd.median():.1f}")
        print(f"    RV变化:   mean={rv_chg.mean():+.1f}%, med={rv_chg.median():+.1f}%")

        if "buy" in col:
            print(f"    RV扩大率: {(rv_chg > 0).mean():.0%} (正确方向)")
            print(f"    RV扩大>20%率: {(rv_chg > 20).mean():.0%}")
        else:
            print(f"    RV收缩率: {(rv_chg < 0).mean():.0%} (正确方向)")
            print(f"    RV收缩>20%率: {(rv_chg < -20).mean():.0%}")

        # 价格方向 (辅助)
        ret_5d = fwd.loc[idx, "fwd_ret_5d"]
        ret_10d = fwd.loc[idx, "fwd_ret_10d"]
        print(f"    价格5d:  wr={( ret_5d > 0).mean():.0%}, avg={ret_5d.mean():+.2f}%")
        print(f"    价格10d: wr={(ret_10d > 0).mean():.0%}, avg={ret_10d.mean():+.2f}%")

    # ============================================================
    # 4. 可视化
    # ============================================================
    print(f"\n{'='*70}")
    print("  4. 生成可视化")
    print(f"{'='*70}")

    fig_comparison(signals, fwd, gld, range_df)
    fig_bar_comparison(results_buy, results_sell, results_vol)

    # ============================================================
    # 5. 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("  5. 最终汇总")
    print(f"{'='*70}")

    a = next(r for r in results_buy if "基线" in r["name"])
    e = next(r for r in results_buy if "综合" in r["name"])

    print(f"\n  买入信号:")
    print(f"    基线 (A):  {a['n']:3d} 信号, {a.get('per_year',0):.1f}/年, "
          f"5d胜率={a.get('wr_5d',0):.1%}, 10d胜率={a.get('wr_10d',0):.1%}")
    print(f"    优化 (E):  {e['n']:3d} 信号, {e.get('per_year',0):.1f}/年, "
          f"5d胜率={e.get('wr_5d',0):.1%}, 10d胜率={e.get('wr_10d',0):.1%}")
    print(f"    信号增加:  {e['n']-a['n']:+d} ({(e['n']/a['n']-1)*100:+.0f}%)")

    s_base = next(r for r in results_sell if "基线" in r["name"])
    s_comb = next(r for r in results_sell if "RV高" in r["name"])

    print(f"\n  卖出信号:")
    print(f"    基线 (bp>0.80):     {s_base['n']:3d}, "
          f"5d下跌率={1-s_base.get('wr_5d',0.5):.1%}")
    print(f"    优化 (RV高+退出):   {s_comb['n']:3d}, "
          f"5d下跌率={1-s_comb.get('wr_5d',0.5):.1%}")

    print(f"\n  图表保存在: {OUT_DIR}")
    print(f"    09_signal_comparison.png — 近2年信号对比 (4面板)")
    print(f"    10_signal_bar_comparison.png — 胜率/下跌率柱状图")


if __name__ == "__main__":
    main()
