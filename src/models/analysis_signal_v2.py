"""
信号优化 V2: 并集逻辑 + 纯前瞻收益评估 (不考虑持仓)

买入: (Bull + bp<0.30) OR (Bull + RV极低)
卖出: (Regime退出Bull) OR (RV极高)
RV独立波动率信号

用法:
    conda activate gold
    python src/models/analysis_signal_v2.py
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
    return gld, range_df, regime, rv_20d


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


def compute_rv_pctile(rv_20d, window=252):
    return rv_20d.rolling(window, min_periods=60).rank(pct=True)


def compute_forward(gld, dates):
    close = gld["Close"]
    result = pd.DataFrame(index=dates)
    for h in [5, 10, 15, 20]:
        result[f"fwd_{h}d"] = ((close.shift(-h) / close - 1) * 100).reindex(dates)
    # RV forward
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(20).std() * np.sqrt(252) * 100
    result["rv_now"] = rv.reindex(dates)
    result["rv_fwd_20d"] = rv.shift(-20).reindex(dates)
    result["rv_chg_20d"] = ((rv.shift(-20) / rv - 1) * 100).reindex(dates)
    return result


def eval_signal(name, mask, fwd):
    """纯前瞻评估: 信号触发日后的 5/10/15/20d 收益。"""
    idx = mask[mask].index.intersection(fwd.index)
    n = len(idx)
    if n < 3:
        return {"name": name, "n": n}

    r = {"name": name, "n": n}
    span = (idx[-1] - idx[0]).days / 365.25
    r["per_year"] = n / span if span > 0 else 0

    for h in [5, 10, 15, 20]:
        col = f"fwd_{h}d"
        if col not in fwd.columns:
            continue
        rets = fwd.loc[idx, col].dropna()
        if len(rets) < 3:
            continue
        r[f"wr_{h}"] = (rets > 0).mean()
        r[f"avg_{h}"] = rets.mean()
        r[f"med_{h}"] = rets.median()
        wins = rets[rets > 0].sum()
        loss = abs(rets[rets <= 0].sum())
        r[f"pf_{h}"] = wins / loss if loss > 0 else float("inf")
        # 未大涨率: P(fwd < +1%) — 卖出信号用
        r[f"nr1_{h}"] = (rets < 1.0).mean()
        r[f"nr2_{h}"] = (rets < 2.0).mean()

    # RV change (for vol signals)
    if "rv_chg_20d" in fwd.columns:
        rv_chg = fwd.loc[idx, "rv_chg_20d"].dropna()
        if len(rv_chg) > 0:
            r["rv_expand"] = (rv_chg > 0).mean()
            r["rv_chg_mean"] = rv_chg.mean()

    return r


def print_table(results, title, mode="buy"):
    print(f"\n  === {title} ===")
    if mode == "vol":
        print(f"  {'Signal':35s} {'N':>4s} {'/yr':>5s} "
              f"{'RV扩大率':>7s} {'RVchg':>7s} "
              f"{'5dWR':>5s} {'10dWR':>6s} {'5dAvg':>7s} {'10dAvg':>7s}")
        print(f"  {'-' * 90}")
        for r in results:
            if r["n"] < 3:
                print(f"  {r['name']:35s} {r['n']:4d}  样本不足")
                continue
            print(f"  {r['name']:35s} {r['n']:4d} {r.get('per_year',0):5.1f} "
                  f"{r.get('rv_expand', np.nan):7.0%} {r.get('rv_chg_mean', np.nan):+7.1f}% "
                  f"{r.get('wr_5', np.nan):5.0%} {r.get('wr_10', np.nan):6.0%} "
                  f"{r.get('avg_5', np.nan):+7.2f}% {r.get('avg_10', np.nan):+7.2f}%")
    elif mode == "sell":
        # 卖出信号: 用"未大涨率" P(fwd < +1%) 评估
        print(f"  {'Signal':35s} {'N':>4s} {'/yr':>5s} "
              f"{'5d跌率':>6s} {'5d<1%':>6s} {'5d<2%':>6s} "
              f"{'10d跌率':>7s} {'10d<1%':>7s} "
              f"{'20d跌率':>7s} {'20d<1%':>7s} {'5dAvg':>7s}")
        print(f"  {'-' * 110}")
        for r in results:
            if r["n"] < 3:
                print(f"  {r['name']:35s} {r['n']:4d}  样本不足")
                continue
            wr5 = 1 - r.get('wr_5', np.nan)
            wr10 = 1 - r.get('wr_10', np.nan)
            wr20 = 1 - r.get('wr_20', np.nan)
            nr1_5 = r.get('nr1_5', np.nan)
            nr1_10 = r.get('nr1_10', np.nan)
            nr1_20 = r.get('nr1_20', np.nan)
            nr2_5 = r.get('nr2_5', np.nan)
            avg5 = r.get('avg_5', np.nan)
            print(f"  {r['name']:35s} {r['n']:4d} {r.get('per_year',0):5.1f} "
                  f"{wr5:6.0%} {nr1_5:6.0%} {nr2_5:6.0%} "
                  f"{wr10:7.0%} {nr1_10:7.0%} "
                  f"{wr20:7.0%} {nr1_20:7.0%} {avg5:+7.2f}%")
    else:
        print(f"  {'Signal':35s} {'N':>4s} {'/yr':>5s} "
              f"{'5d涨率':>6s} {'5dAvg':>7s} {'5dPF':>5s} "
              f"{'10d涨率':>7s} {'10dAvg':>7s} {'10dPF':>6s} "
              f"{'15d涨率':>7s} {'20d涨率':>7s} {'20dAvg':>7s}")
        print(f"  {'-' * 110}")
        for r in results:
            if r["n"] < 3:
                print(f"  {r['name']:35s} {r['n']:4d}  样本不足")
                continue
            wr5 = r.get('wr_5', np.nan)
            wr10 = r.get('wr_10', np.nan)
            wr15 = r.get('wr_15', np.nan)
            wr20 = r.get('wr_20', np.nan)
            avg5 = r.get('avg_5', np.nan)
            avg10 = r.get('avg_10', np.nan)
            avg20 = r.get('avg_20', np.nan)
            print(f"  {r['name']:35s} {r['n']:4d} {r.get('per_year',0):5.1f} "
                  f"{wr5:6.0%} {avg5:+7.2f}% {r.get('pf_5', np.nan):5.2f} "
                  f"{wr10:7.0%} {avg10:+7.2f}% {r.get('pf_10', np.nan):6.2f} "
                  f"{wr15:7.0%} {wr20:7.0%} {avg20:+7.2f}%")


# ============================================================
# 可视化
# ============================================================

def make_plots(signals, fwd, gld, results_buy, results_sell, results_vol):
    close = gld["Close"]

    # ============================================================
    # Fig 11: 近2年4面板信号图
    # ============================================================
    fig, axes = plt.subplots(4, 1, figsize=(20, 20),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2]},
                             sharex=True)

    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2026-03-31")
    common = signals.index[(signals.index >= start) & (signals.index <= end)]
    cl = close.reindex(common)
    reg = signals["regime"].reindex(common)

    # --- Panel 1: Price + signals ---
    ax = axes[0]
    ax.plot(common, cl, color="black", linewidth=1.2)

    # Bull shading
    bull = reg == "Bull"
    s_ = bull & (~bull.shift(1, fill_value=False))
    e_ = bull & (~bull.shift(-1, fill_value=False))
    for s, e in zip(common[s_], common[e_]):
        ax.axvspan(s, e, alpha=0.1, color="green")

    # Baseline buy (A)
    ba = signals["buy_A"].reindex(common).fillna(False)
    if ba.sum() > 0:
        ax.scatter(common[ba], cl[ba], marker="^", s=130, color="blue",
                   zorder=5, label=f"A: Bull+bp<0.20 (n={ba.sum()})")

    # Optimized buy (F) — new signals only
    bf = signals["buy_F"].reindex(common).fillna(False)
    bf_new = bf & (~ba)
    if bf_new.sum() > 0:
        ax.scatter(common[bf_new], cl[bf_new], marker="^", s=130, color="lime",
                   edgecolors="darkgreen", linewidths=1.5, zorder=5,
                   label=f"F new: bp<0.30∪RV低 (n={bf_new.sum()})")

    # Baseline sell
    sb = signals["sell_A"].reindex(common).fillna(False)
    if sb.sum() > 0:
        ax.scatter(common[sb], cl[sb], marker="v", s=100, color="salmon",
                   alpha=0.5, zorder=5, label=f"sell_A: bp>0.80 (n={sb.sum()})")

    # Optimized sell (G)
    sg = signals["sell_G"].reindex(common).fillna(False)
    sg_new = sg & (~sb)
    if sg_new.sum() > 0:
        ax.scatter(common[sg_new], cl[sg_new], marker="v", s=120, color="red",
                   edgecolors="darkred", linewidths=1.5, zorder=5,
                   label=f"sell_G new: regime退出∪RV高 (n={sg_new.sum()})")

    # Vol signals
    vb = signals["vol_buy"].reindex(common).fillna(False)
    vs = signals["vol_sell"].reindex(common).fillna(False)
    if vb.sum() > 0:
        ax.scatter(common[vb], cl[vb] * 0.99, marker="D", s=60, color="cyan",
                   edgecolors="blue", zorder=4, label=f"Vol buy (n={vb.sum()})")
    if vs.sum() > 0:
        ax.scatter(common[vs], cl[vs] * 1.01, marker="D", s=60, color="orange",
                   edgecolors="red", zorder=4, label=f"Vol sell (n={vs.sum()})")

    ax.set_ylabel("GLD Price ($)")
    ax.set_title("V2 Signal Comparison: Baseline vs Optimized (2024-2025)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: bp ---
    ax = axes[1]
    bp_s = signals["bp"].reindex(common)
    ax.plot(common, bp_s, color="steelblue", linewidth=0.8)
    ax.axhline(0.20, color="blue", linewidth=1, linestyle="--", alpha=0.5, label="0.20 (old)")
    ax.axhline(0.30, color="green", linewidth=1, linestyle="--", alpha=0.5, label="0.30 (new)")
    ax.axhline(0.80, color="red", linewidth=1, linestyle="--", alpha=0.5, label="0.80 (sell)")
    ax.fill_between(common, 0, 0.30, alpha=0.05, color="green")
    ax.set_ylabel("Band Position")
    ax.set_ylim(-0.2, 1.3)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: RV percentile ---
    ax = axes[2]
    rp = signals["rv_pctile"].reindex(common)
    ax.plot(common, rp, color="darkorange", linewidth=1)
    ax.axhline(0.15, color="cyan", linewidth=1.5, linestyle="--", label="0.15 (vol buy / 方向买辅助)")
    ax.axhline(0.85, color="red", linewidth=1.5, linestyle="--", label="0.85 (vol sell / 方向卖辅助)")
    ax.fill_between(common, 0, 0.15, alpha=0.08, color="cyan")
    ax.fill_between(common, 0.85, 1, alpha=0.08, color="red")
    ax.set_ylabel("RV Percentile")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Regime ---
    ax = axes[3]
    colors_map = {"Bull": 1, "Mixed": 0, "Bear": -1}
    reg_num = reg.map(colors_map)
    ax.fill_between(common, 0, reg_num, where=reg_num > 0, color="#2ecc71", alpha=0.5)
    ax.fill_between(common, 0, reg_num, where=reg_num < 0, color="#e74c3c", alpha=0.5)
    ax.set_ylabel("Regime")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Bear", "Mixed", "Bull"])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "11_signal_v2_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ============================================================
    # Fig 12: 对比柱状图
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # --- Buy comparison ---
    ax = axes[0]
    buy_names = [r["name"] for r in results_buy if r["n"] >= 3]
    buy_wr5 = [r.get("wr_5", 0) for r in results_buy if r["n"] >= 3]
    buy_wr10 = [r.get("wr_10", 0) for r in results_buy if r["n"] >= 3]
    buy_wr20 = [r.get("wr_20", 0) for r in results_buy if r["n"] >= 3]
    buy_n = [r["n"] for r in results_buy if r["n"] >= 3]

    x = np.arange(len(buy_names))
    w = 0.25
    ax.bar(x - w, [v * 100 for v in buy_wr5], w, label="5d Win%", color="steelblue", alpha=0.7)
    ax.bar(x, [v * 100 for v in buy_wr10], w, label="10d Win%", color="darkgreen", alpha=0.7)
    ax.bar(x + w, [v * 100 for v in buy_wr20], w, label="20d Win%", color="goldenrod", alpha=0.7)

    for i, n in enumerate(buy_n):
        ax.annotate(f"n={n}", xy=(i, max(buy_wr5[i], buy_wr10[i], buy_wr20[i]) * 100 + 1),
                    fontsize=8, ha="center")

    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    short = [n.replace("Bull+", "").replace(" (基线)", "\n(基线)").replace(" (优化)", "\n(优化)")
             for n in buy_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Buy Signals: Win Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Sell comparison: 未大涨率 P(fwd < +1%) ---
    ax = axes[1]
    sell_names = [r["name"] for r in results_sell if r["n"] >= 3]
    sell_nr5 = [r.get("nr1_5", 0.5) for r in results_sell if r["n"] >= 3]
    sell_nr10 = [r.get("nr1_10", 0.5) for r in results_sell if r["n"] >= 3]
    sell_nr20 = [r.get("nr1_20", 0.5) for r in results_sell if r["n"] >= 3]
    sell_n = [r["n"] for r in results_sell if r["n"] >= 3]

    x = np.arange(len(sell_names))
    ax.bar(x - w, [v * 100 for v in sell_nr5], w, label="5d <+1%", color="salmon", alpha=0.7)
    ax.bar(x, [v * 100 for v in sell_nr10], w, label="10d <+1%", color="red", alpha=0.7)
    ax.bar(x + w, [v * 100 for v in sell_nr20], w, label="20d <+1%", color="darkred", alpha=0.7)

    for i, n in enumerate(sell_n):
        ax.annotate(f"n={n}", xy=(i, max(sell_nr5[i], sell_nr10[i], sell_nr20[i]) * 100 + 1),
                    fontsize=8, ha="center")

    ax.axhline(50, color="gray", linewidth=1, linestyle="--")
    short_s = [n.replace("sell: ", "") for n in sell_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_s, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Not-Rally Rate: P(fwd < +1%)")
    ax.set_title("Sell Signals: 未大涨率 (fwd < +1%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Vol signals ---
    ax = axes[2]
    vol_names = [r["name"] for r in results_vol if r["n"] >= 3]
    vol_rv_expand = [r.get("rv_expand", 0.5) for r in results_vol if r["n"] >= 3]
    vol_rv_chg = [r.get("rv_chg_mean", 0) for r in results_vol if r["n"] >= 3]
    vol_n = [r["n"] for r in results_vol if r["n"] >= 3]

    x = np.arange(len(vol_names))
    colors = ["cyan" if "buy" in n.lower() else "red" for n in vol_names]
    bars = ax.bar(x, [v * 100 for v in vol_rv_expand], color=colors, alpha=0.7, edgecolor="gray")
    ax.axhline(50, color="gray", linewidth=1, linestyle="--")

    for i, (n, chg) in enumerate(zip(vol_n, vol_rv_chg)):
        ax.annotate(f"n={n}\nRV chg={chg:+.0f}%", xy=(i, vol_rv_expand[i] * 100 + 1),
                    fontsize=9, ha="center")

    short_v = [n.replace("vol: ", "") for n in vol_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_v, fontsize=9)
    ax.set_ylabel("Correct Direction Rate (%)")
    ax.set_title("Vol Signals: Direction Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("V2 Optimized: Buy / Sell / Vol Signal Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "12_signal_v2_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  信号优化 V2: 并集逻辑")
    print("=" * 70)

    gld, range_df, regime, rv_20d = load_all()
    upper_band, lower_band, bp = build_band(range_df, gld["Close"])
    rv_pctile = compute_rv_pctile(rv_20d)

    dates = bp.dropna().index
    reg = regime.reindex(dates)
    bp_s = bp.reindex(dates)
    rv_p = rv_pctile.reindex(dates)
    is_bull = reg == "Bull"

    print(f"  数据: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}天)")
    print(f"  Bull: {is_bull.sum()}天 ({is_bull.mean():.0%})")

    # ============================================================
    # 信号定义
    # ============================================================
    signals = pd.DataFrame(index=dates)
    signals["bp"] = bp_s
    signals["rv_pctile"] = rv_p
    signals["regime"] = reg

    # --- 买入 ---
    bp_down_020 = (bp_s < 0.20) & (bp_s.shift(1) >= 0.20)
    bp_down_030 = (bp_s < 0.30) & (bp_s.shift(1) >= 0.30)
    rv_enter_low = (rv_p < 0.15) & (rv_p.shift(1) >= 0.15)

    signals["buy_A"] = is_bull & bp_down_020                          # 基线
    signals["buy_B"] = is_bull & bp_down_030                          # bp上移
    signals["buy_C"] = is_bull & rv_enter_low                         # RV极低 (独立)
    signals["buy_F"] = signals["buy_B"] | signals["buy_C"]            # 并集: B OR C

    # --- 卖出 ---
    bp_up_080 = (bp_s > 0.80) & (bp_s.shift(1) <= 0.80)
    rv_enter_high = (rv_p > 0.85) & (rv_p.shift(1) <= 0.85)
    bull_exit = is_bull.shift(1).fillna(False) & (~is_bull)

    signals["sell_A"] = bp_up_080                                      # 基线
    signals["sell_rv"] = rv_enter_high                                 # RV极高
    signals["sell_exit"] = bull_exit                                    # Regime退出
    signals["sell_G"] = signals["sell_rv"] | signals["sell_exit"]      # 并集

    # --- RV 独立波动率 ---
    signals["vol_buy"] = (rv_p < 0.15) & (rv_p.shift(1) >= 0.15)
    signals["vol_sell"] = (rv_p > 0.85) & (rv_p.shift(1) <= 0.85)

    # ============================================================
    # 前瞻收益
    # ============================================================
    print(f"  计算前瞻收益...")
    fwd = compute_forward(gld, dates)

    # ============================================================
    # 1. 买入信号
    # ============================================================
    print(f"\n{'='*70}")
    print("  1. 买入信号对比")
    print(f"{'='*70}")

    buy_list = [
        ("A: Bull+bp<0.20 (基线)", "buy_A"),
        ("B: Bull+bp<0.30 (上移)", "buy_B"),
        ("C: Bull+RV<15% (RV低)", "buy_C"),
        ("F: B∪C (优化并集)", "buy_F"),
    ]
    results_buy = [eval_signal(n, signals[c], fwd) for n, c in buy_list]
    print_table(results_buy, "买入信号", mode="buy")

    # 增量分析
    f_only_b = signals["buy_F"] & signals["buy_B"] & (~signals["buy_C"])
    f_only_c = signals["buy_F"] & signals["buy_C"] & (~signals["buy_B"])
    f_both = signals["buy_F"] & signals["buy_B"] & signals["buy_C"]
    inc_list = [
        eval_signal("F中: 仅来自B (bp<0.30)", f_only_b, fwd),
        eval_signal("F中: 仅来自C (RV低)", f_only_c, fwd),
        eval_signal("F中: B∩C 同时触发", f_both, fwd),
    ]
    print_table(inc_list, "F 信号来源分解", mode="buy")

    # A vs F: 新增信号质量
    f_new = signals["buy_F"] & (~signals["buy_A"])
    f_old = signals["buy_F"] & signals["buy_A"]
    inc2 = [
        eval_signal("F∩A (与基线重叠)", f_old, fwd),
        eval_signal("F-A (新增信号)", f_new, fwd),
    ]
    print_table(inc2, "F vs A: 新增信号质量", mode="buy")

    # ============================================================
    # 2. 卖出信号
    # ============================================================
    print(f"\n{'='*70}")
    print("  2. 卖出信号对比")
    print(f"{'='*70}")

    sell_list = [
        ("sell: bp>0.80 (基线)", "sell_A"),
        ("sell: RV_pctile>0.85", "sell_rv"),
        ("sell: Regime退出Bull", "sell_exit"),
        ("sell: RV高∪Regime退出 (优化)", "sell_G"),
    ]
    results_sell = [eval_signal(n, signals[c], fwd) for n, c in sell_list]
    print_table(results_sell, "卖出信号 (看下跌率)", mode="sell")

    # ============================================================
    # 3. RV 独立波动率信号
    # ============================================================
    print(f"\n{'='*70}")
    print("  3. RV 独立波动率信号 (Phase 4B)")
    print(f"{'='*70}")

    vol_list = [
        ("vol: RV<15% 做多波动", "vol_buy"),
        ("vol: RV>85% 做空波动", "vol_sell"),
    ]
    results_vol = [eval_signal(n, signals[c], fwd) for n, c in vol_list]
    print_table(results_vol, "RV 波动率信号", mode="vol")

    # 详细
    for name, col in vol_list:
        idx = signals[col][signals[col]].index.intersection(fwd.index)
        if len(idx) < 3:
            continue
        rv_now = fwd.loc[idx, "rv_now"]
        rv_fwd = fwd.loc[idx, "rv_fwd_20d"]
        rv_chg = fwd.loc[idx, "rv_chg_20d"]
        print(f"\n  {name} (n={len(idx)}):")
        print(f"    触发RV: {rv_now.mean():.1f} → 20d后: {rv_fwd.mean():.1f}")
        if "buy" in col:
            print(f"    RV扩大率: {(rv_chg > 0).mean():.0%}, 扩大>20%: {(rv_chg > 20).mean():.0%}")
        else:
            print(f"    RV收缩率: {(rv_chg < 0).mean():.0%}, 收缩>20%: {(rv_chg < -20).mean():.0%}")

    # ============================================================
    # 4. 可视化
    # ============================================================
    print(f"\n{'='*70}")
    print("  4. 生成可视化")
    print(f"{'='*70}")

    make_plots(signals, fwd, gld, results_buy, results_sell, results_vol)

    # ============================================================
    # 5. 汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("  5. 基线 vs 优化 汇总")
    print(f"{'='*70}")

    a = next(r for r in results_buy if "基线" in r["name"])
    f = next(r for r in results_buy if "优化" in r["name"])

    print(f"\n  买入:")
    print(f"    基线 A: {a['n']:3d}信号 ({a.get('per_year',0):.1f}/年) | "
          f"5d {a.get('wr_5',0):.0%} | 10d {a.get('wr_10',0):.0%} | "
          f"20d {a.get('wr_20',0):.0%} | avg10d {a.get('avg_10',0):+.2f}%")
    print(f"    优化 F: {f['n']:3d}信号 ({f.get('per_year',0):.1f}/年) | "
          f"5d {f.get('wr_5',0):.0%} | 10d {f.get('wr_10',0):.0%} | "
          f"20d {f.get('wr_20',0):.0%} | avg10d {f.get('avg_10',0):+.2f}%")
    print(f"    变化: 信号 {f['n']-a['n']:+d} ({(f['n']/a['n']-1)*100:+.0f}%)")

    sa = next(r for r in results_sell if "基线" in r["name"])
    sg = next(r for r in results_sell if "优化" in r["name"])

    print(f"\n  卖出 (未大涨率 = P(fwd < +1%)):")
    print(f"    基线 A: {sa['n']:3d}信号 | 5d {sa.get('nr1_5',0):.0%} | "
          f"10d {sa.get('nr1_10',0):.0%} | 20d {sa.get('nr1_20',0):.0%}")
    print(f"    优化 G: {sg['n']:3d}信号 | 5d {sg.get('nr1_5',0):.0%} | "
          f"10d {sg.get('nr1_10',0):.0%} | 20d {sg.get('nr1_20',0):.0%}")

    print(f"\n  图表路径: {OUT_DIR}/")
    print(f"    11_signal_v2_detail.png — 近2年信号对比 (4面板)")
    print(f"    12_signal_v2_bars.png   — 胜率柱状图对比")


if __name__ == "__main__":
    main()
