"""
E3: GLD 整数关口效应检验
E4: 动量退出信号 (MACD/RSI) 对比

用法:
    conda activate gold
    python src/models/analysis_e3_e4.py
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
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]
    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))
    rv_20d = features["rv_20d"] if "rv_20d" in features.columns else None
    return gld, regime, range_df, rv_20d


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


# ============================================================
# E3: 整数关口效应
# ============================================================

def e3_round_number(gld, regime):
    print("=" * 70)
    print("  E3: GLD 整数关口效应检验")
    print("=" * 70)

    close = gld["Close"]
    high = gld["High"]
    low = gld["Low"]
    reg = regime.reindex(close.index)

    # 距离最近 $5 整数位
    def dist_to_round(price, step):
        nearest = np.round(price / step) * step
        return (price - nearest) / price * 100  # 百分比距离

    dist_5 = close.apply(lambda x: dist_to_round(x, 5))
    dist_10 = close.apply(lambda x: dist_to_round(x, 10))

    # 前瞻收益
    fwd_5d = (close.shift(-5) / close - 1) * 100
    fwd_10d = (close.shift(-10) / close - 1) * 100

    # 距离最近支撑位 (下方最近整数)
    def dist_to_support(price, step):
        support = np.floor(price / step) * step
        return (price - support) / price * 100

    dist_sup_5 = close.apply(lambda x: dist_to_support(x, 5))
    dist_sup_10 = close.apply(lambda x: dist_to_support(x, 10))

    valid = close.index[close.notna() & fwd_5d.notna() & reg.notna()]

    # --- 测试1: 接近 $5 整数位时反弹概率 ---
    print(f"\n  === 测试1: 接近 $5 整数位支撑 ===")
    print(f"  dist_to_support = (price - floor_5) / price × 100")
    print(f"  {'距支撑':>10s} {'N':>6s} {'5dWR':>6s} {'5dAvg':>7s} {'10dWR':>6s} {'10dAvg':>7s}")
    print(f"  {'-' * 48}")

    bins = [0, 0.3, 0.6, 1.0, 1.5, 2.5, 5.0]
    labels = ["<0.3%", "0.3-0.6%", "0.6-1.0%", "1.0-1.5%", "1.5-2.5%", ">2.5%"]
    bucket = pd.cut(dist_sup_5.reindex(valid), bins=bins, labels=labels)

    for lab in labels:
        mask = bucket == lab
        n = mask.sum()
        if n < 20:
            continue
        wr5 = (fwd_5d.reindex(valid)[mask] > 0).mean()
        avg5 = fwd_5d.reindex(valid)[mask].mean()
        wr10 = (fwd_10d.reindex(valid)[mask] > 0).mean()
        avg10 = fwd_10d.reindex(valid)[mask].mean()
        print(f"  {lab:>10s} {n:6d} {wr5:6.1%} {avg5:+7.2f}% {wr10:6.1%} {avg10:+7.2f}%")

    # --- 测试2: Bull中接近整数支撑 ---
    bull_mask = reg.reindex(valid) == "Bull"
    print(f"\n  === 测试2: Bull中接近 $5 整数位支撑 ===")
    print(f"  {'距支撑':>10s} {'N':>6s} {'5dWR':>6s} {'5dAvg':>7s} {'10dWR':>6s} {'10dAvg':>7s}")
    print(f"  {'-' * 48}")

    bucket_bull = pd.cut(dist_sup_5.reindex(valid)[bull_mask], bins=bins, labels=labels)
    for lab in labels:
        mask = bucket_bull == lab
        n = mask.sum()
        if n < 10:
            continue
        idx = valid[bull_mask][mask]
        wr5 = (fwd_5d.reindex(idx) > 0).mean()
        avg5 = fwd_5d.reindex(idx).mean()
        wr10 = (fwd_10d.reindex(idx) > 0).mean()
        avg10 = fwd_10d.reindex(idx).mean()
        print(f"  {lab:>10s} {n:6d} {wr5:6.1%} {avg5:+7.2f}% {wr10:6.1%} {avg10:+7.2f}%")

    # --- 测试3: $10 整数位 ---
    print(f"\n  === 测试3: 接近 $10 整数位支撑 ===")
    dist_sup_10_v = dist_sup_10.reindex(valid)
    bins10 = [0, 0.5, 1.0, 2.0, 3.0, 5.0]
    labels10 = ["<0.5%", "0.5-1%", "1-2%", "2-3%", ">3%"]
    bucket10 = pd.cut(dist_sup_10_v, bins=bins10, labels=labels10)
    print(f"  {'距支撑':>10s} {'N':>6s} {'5dWR':>6s} {'5dAvg':>7s} {'10dWR':>6s}")
    print(f"  {'-' * 40}")
    for lab in labels10:
        mask = bucket10 == lab
        n = mask.sum()
        if n < 30:
            continue
        wr5 = (fwd_5d.reindex(valid)[mask] > 0).mean()
        avg5 = fwd_5d.reindex(valid)[mask].mean()
        wr10 = (fwd_10d.reindex(valid)[mask] > 0).mean()
        print(f"  {lab:>10s} {n:6d} {wr5:6.1%} {avg5:+7.2f}% {wr10:6.1%}")

    # --- 测试4: 是否触碰过整数位后反弹 ---
    print(f"\n  === 测试4: 日内触碰 $5 整数位 (Low 在整数位附近) ===")
    # 日内最低价距最近 $5 的距离
    low_dist = low.apply(lambda x: abs(x % 5 - round(x % 5 / 5) * 5) / x * 100)
    touch_5 = low_dist < 0.3  # 日内低点在 $5 整数位 ±0.3% 以内

    for label, mask_extra in [("全样本", pd.Series(True, index=valid)),
                               ("Bull", bull_mask)]:
        tmask = touch_5.reindex(valid) & mask_extra
        not_tmask = (~touch_5.reindex(valid)) & mask_extra
        n_t = tmask.sum()
        n_nt = not_tmask.sum()
        if n_t < 10 or n_nt < 10:
            continue
        wr5_t = (fwd_5d.reindex(valid)[tmask] > 0).mean()
        wr5_nt = (fwd_5d.reindex(valid)[not_tmask] > 0).mean()
        avg5_t = fwd_5d.reindex(valid)[tmask].mean()
        avg5_nt = fwd_5d.reindex(valid)[not_tmask].mean()
        wr10_t = (fwd_10d.reindex(valid)[tmask] > 0).mean()
        wr10_nt = (fwd_10d.reindex(valid)[not_tmask] > 0).mean()
        print(f"  {label} 触碰$5: n={n_t:4d}, 5dWR={wr5_t:.1%} ({avg5_t:+.2f}%), 10dWR={wr10_t:.1%}")
        print(f"  {label} 未触碰:  n={n_nt:4d}, 5dWR={wr5_nt:.1%} ({avg5_nt:+.2f}%), 10dWR={wr10_nt:.1%}")
        print(f"  差异: 5dWR {(wr5_t-wr5_nt)*100:+.1f}pp, 10dWR {(wr10_t-wr10_nt)*100:+.1f}pp")
        print()

    return dist_sup_5


# ============================================================
# E4: 动量退出信号
# ============================================================

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def e4_momentum_exit(gld, regime, bp):
    print(f"\n{'='*70}")
    print("  E4: 动量退出信号对比")
    print("=" * 70)

    close = gld["Close"]
    reg = regime.reindex(close.index)

    # 计算技术指标
    macd_12_26, sig_12_26, hist_12_26 = compute_macd(close, 12, 26, 9)
    macd_8_17, sig_8_17, hist_8_17 = compute_macd(close, 8, 17, 6)
    macd_6_13, sig_6_13, hist_6_13 = compute_macd(close, 6, 13, 4)
    rsi_14 = compute_rsi(close, 14)
    rsi_7 = compute_rsi(close, 7)

    # 前瞻收益
    fwd_5d = (close.shift(-5) / close - 1) * 100
    fwd_10d = (close.shift(-10) / close - 1) * 100
    fwd_20d = (close.shift(-20) / close - 1) * 100

    valid = close.index[close.notna() & fwd_5d.notna() & reg.notna()]
    bp_s = bp.reindex(valid) if bp is not None else None

    # === 卖出信号定义 ===
    sell_signals = {}

    # 基线: bp>0.80
    if bp_s is not None:
        sell_signals["bp>0.80 (基线)"] = (bp_s > 0.80) & (bp_s.shift(1) <= 0.80)

    # MACD 死叉 (三种参数)
    for name, hist in [("MACD(12,26,9)", hist_12_26),
                        ("MACD(8,17,6)", hist_8_17),
                        ("MACD(6,13,4)", hist_6_13)]:
        h = hist.reindex(valid)
        death_cross = (h < 0) & (h.shift(1) >= 0)
        sell_signals[f"{name} 死叉"] = death_cross

    # MACD 死叉 + bp>0.50 (只在区间上半部分卖)
    if bp_s is not None:
        for name, hist in [("MACD(12,26,9)+bp>0.5", hist_12_26),
                            ("MACD(8,17,6)+bp>0.5", hist_8_17)]:
            h = hist.reindex(valid)
            death_cross = (h < 0) & (h.shift(1) >= 0)
            sell_signals[name] = death_cross & (bp_s > 0.50)

    # MACD hist 由正转负 + 连续2天递减
    for name, hist in [("MACD(12,26,9) 衰竭", hist_12_26)]:
        h = hist.reindex(valid)
        weakening = (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h < 0)
        sell_signals[name] = weakening & (~weakening.shift(1, fill_value=False))

    # RSI 超买回落
    r14 = rsi_14.reindex(valid)
    r7 = rsi_7.reindex(valid)
    sell_signals["RSI14 从>70回落"] = (r14 < 70) & (r14.shift(1) >= 70)
    sell_signals["RSI7 从>80回落"] = (r7 < 80) & (r7.shift(1) >= 80)
    sell_signals["RSI14>70 + MACD死叉"] = ((r14 > 60) &
        (hist_12_26.reindex(valid) < 0) &
        (hist_12_26.reindex(valid).shift(1) >= 0))

    # === 评估: 卖出正确 = 后续没继续大涨 ===
    # 阈值: <0% (真跌), <+1% (没涨), <+2% (没大涨)
    thresholds = [0, 1, 2]
    th_labels = ["<0%", "<+1%", "<+2%"]

    print(f"\n  === 全样本卖出信号 (后续N天涨幅低于阈值的比例) ===")
    header = f"  {'Signal':30s} {'N':>5s} {'/yr':>5s} "
    for th_l in th_labels:
        header += f"  5d{th_l:>4s} 10d{th_l:>4s} 20d{th_l:>4s}"
    header += f" {'5dAvg':>7s}"
    print(header)
    print(f"  {'-' * 120}")

    all_results = []
    for name, mask in sell_signals.items():
        mask = mask.fillna(False)
        idx = valid[mask]
        n = len(idx)
        if n < 5:
            print(f"  {name:30s} {n:5d}  样本不足")
            continue
        span = (idx[-1] - idx[0]).days / 365.25 if n > 1 else 1
        py = n / span if span > 0 else 0
        avg5 = fwd_5d.reindex(idx).mean()

        row = f"  {name:30s} {n:5d} {py:5.1f} "
        result = {"name": name, "n": n, "per_year": py, "avg5": avg5}
        for th in thresholds:
            r5 = (fwd_5d.reindex(idx) < th).mean()
            r10 = (fwd_10d.reindex(idx) < th).mean()
            r20 = (fwd_20d.reindex(idx).dropna() < th).mean()
            row += f"  {r5:5.1%}  {r10:5.1%}  {r20:5.1%}"
            result[f"5d_lt{th}"] = r5
            result[f"10d_lt{th}"] = r10
            result[f"20d_lt{th}"] = r20
        row += f" {avg5:+7.2f}%"
        print(row)
        all_results.append(result)

    # === Bull Only ===
    bull_valid = valid[reg.reindex(valid) == "Bull"]
    print(f"\n  === Bull Only 卖出信号 ===")
    header = f"  {'Signal':30s} {'N':>5s} "
    for th_l in th_labels:
        header += f"  5d{th_l:>4s} 10d{th_l:>4s} 20d{th_l:>4s}"
    header += f" {'5dAvg':>7s}"
    print(header)
    print(f"  {'-' * 110}")

    bull_results = []
    for name, mask in sell_signals.items():
        mask = mask.fillna(False)
        idx = bull_valid[mask.reindex(bull_valid).fillna(False)]
        n = len(idx)
        if n < 5:
            continue
        avg5 = fwd_5d.reindex(idx).mean()

        row = f"  {name:30s} {n:5d} "
        result = {"name": name, "n": n, "avg5": avg5}
        for th in thresholds:
            r5 = (fwd_5d.reindex(idx) < th).mean()
            r10 = (fwd_10d.reindex(idx) < th).mean()
            r20 = (fwd_20d.reindex(idx).dropna() < th).mean()
            row += f"  {r5:5.1%}  {r10:5.1%}  {r20:5.1%}"
            result[f"5d_lt{th}"] = r5
            result[f"10d_lt{th}"] = r10
            result[f"20d_lt{th}"] = r20
        row += f" {avg5:+7.2f}%"
        print(row)
        bull_results.append(result)

    return all_results, bull_results


# ============================================================
# 可视化
# ============================================================

def make_plots(gld, regime, bp, dist_sup_5, e4_results, e4_bull):
    close = gld["Close"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # --- E3: 整数关口效应 ---
    ax = axes[0][0]
    # 距 $5 支撑位的距离 vs 5d胜率 (分桶)
    fwd_5d = (close.shift(-5) / close - 1) * 100
    valid = close.index[close.notna() & fwd_5d.notna()]
    ds5 = dist_sup_5.reindex(valid)
    bins = np.arange(0, 3.5, 0.25)
    bin_labels = [f"{b:.1f}" for b in bins[:-1]]
    bucket = pd.cut(ds5, bins=bins)
    means = []
    wrs = []
    centers = []
    for i in range(len(bins) - 1):
        mask = (ds5 >= bins[i]) & (ds5 < bins[i + 1])
        if mask.sum() < 30:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        wrs.append((fwd_5d.reindex(valid)[mask] > 0).mean())
        means.append(fwd_5d.reindex(valid)[mask].mean())

    ax.bar(centers, wrs, width=0.2, color="steelblue", alpha=0.7, edgecolor="gray")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Distance to $5 Support (%)")
    ax.set_ylabel("5d Win Rate")
    ax.set_title("E3: $5 Round Number Support Effect")
    ax.grid(True, alpha=0.3, axis="y")

    # --- E3: Bull中的整数关口 ---
    ax = axes[0][1]
    reg = regime.reindex(valid)
    bull_mask = reg == "Bull"
    ds5_bull = ds5[bull_mask]
    fwd_5d_bull = fwd_5d.reindex(valid)[bull_mask]

    centers_b = []
    wrs_b = []
    for i in range(len(bins) - 1):
        mask = (ds5_bull >= bins[i]) & (ds5_bull < bins[i + 1])
        if mask.sum() < 10:
            continue
        centers_b.append((bins[i] + bins[i + 1]) / 2)
        wrs_b.append((fwd_5d_bull[mask] > 0).mean())

    ax.bar(centers_b, wrs_b, width=0.2, color="#2ecc71", alpha=0.7, edgecolor="gray")
    ax.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Distance to $5 Support (%)")
    ax.set_ylabel("5d Win Rate")
    ax.set_title("E3: Bull Only — $5 Support Effect")
    ax.grid(True, alpha=0.3, axis="y")

    # --- E4: 全样本卖出信号 (用 <+1% 未大涨率) ---
    ax = axes[1][0]
    if len(e4_results) > 0:
        names = [r["name"][:20] for r in e4_results]
        nr5 = [r.get("5d_lt1", 0) * 100 for r in e4_results]
        nr10 = [r.get("10d_lt1", 0) * 100 for r in e4_results]
        nr20 = [r.get("20d_lt1", 0) * 100 for r in e4_results]
        x = np.arange(len(names))
        w = 0.25
        ax.barh(x - w, nr5, w, label="5d <+1%", color="salmon", alpha=0.7)
        ax.barh(x, nr10, w, label="10d <+1%", color="red", alpha=0.7)
        ax.barh(x + w, nr20, w, label="20d <+1%", color="darkred", alpha=0.7)
        ax.axvline(50, color="gray", linewidth=1, linestyle="--")
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Not-Rally Rate: P(fwd < +1%)")
        ax.set_title("E4: Sell Signals — All (未大涨率)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

    # --- E4: Bull卖出信号对比 ---
    ax = axes[1][1]
    if len(e4_bull) > 0:
        names_b = [r["name"][:20] for r in e4_bull]
        nr5_b = [r.get("5d_lt1", 0) * 100 for r in e4_bull]
        nr10_b = [r.get("10d_lt1", 0) * 100 for r in e4_bull]
        nr20_b = [r.get("20d_lt1", 0) * 100 for r in e4_bull]
        x = np.arange(len(names_b))
        ax.barh(x - w, nr5_b, w, label="5d <+1%", color="salmon", alpha=0.7)
        ax.barh(x, nr10_b, w, label="10d <+1%", color="red", alpha=0.7)
        ax.barh(x + w, nr20_b, w, label="20d <+1%", color="darkred", alpha=0.7)
        ax.axvline(50, color="gray", linewidth=1, linestyle="--")
        ax.set_yticks(x)
        ax.set_yticklabels(names_b, fontsize=7)
        ax.set_xlabel("Not-Rally Rate: P(fwd < +1%)")
        ax.set_title("E4: Sell Signals — Bull Only (未大涨率)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("E3: Round Number Effect + E4: Momentum Exit Signals", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "13_e3_e4_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path}")


def main():
    gld, regime, range_df, rv_20d = load_all()
    _, _, bp = build_band(range_df, gld["Close"])

    dist_sup_5 = e3_round_number(gld, regime)
    e4_results, e4_bull = e4_momentum_exit(gld, regime, bp)
    make_plots(gld, regime, bp, dist_sup_5, e4_results, e4_bull)

    print(f"\n  图表: {OUT_DIR}/13_e3_e4_analysis.png")


if __name__ == "__main__":
    main()
