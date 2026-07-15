"""
Phase 2 特征工程质量报告
生成可视化图表到 reports/phase2_features/

用法:
    conda activate gold
    python reports/phase2_features/generate_report.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- style ----------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
COLORS = {
    "tech": "#1f77b4",
    "macro": "#ff7f0e",
    "vol": "#2ca02c",
    "label": "#d62728",
    "pos": "#2ca02c",
    "neg": "#d62728",
    "neutral": "#7f7f7f",
}


def load_data():
    proc = os.path.join(PROJECT_ROOT, "data", "processed")
    features = pd.read_parquet(os.path.join(proc, "features_all.parquet"))
    labels = pd.read_parquet(os.path.join(proc, "labels.parquet"))
    gld = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "raw", "market", "gld.csv"),
        index_col=0, parse_dates=True,
    )
    return features, labels, gld


# ============================================================
# 1. Missing Rate
# ============================================================
def plot_missing_rate(features):
    miss = features.isnull().mean().sort_values(ascending=False)
    miss_nonzero = miss[miss > 0]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#d62728" if v > 0.15 else "#ff7f0e" if v > 0.05 else "#2ca02c"
              for v in miss_nonzero.values]
    ax.barh(range(len(miss_nonzero)), miss_nonzero.values, color=colors)
    ax.set_yticks(range(len(miss_nonzero)))
    ax.set_yticklabels(miss_nonzero.index, fontsize=8)
    ax.set_xlabel("Missing Rate")
    ax.set_title(f"Feature Missing Rates (showing {len(miss_nonzero)}/{len(miss)} "
                 f"features with any missing)")
    ax.invert_yaxis()
    ax.axvline(0.15, color="red", ls="--", alpha=0.5, label="15%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_missing_rate.png"), dpi=150)
    plt.close(fig)
    print("  [1/7] Missing rate chart saved")


# ============================================================
# 2. Feature-Label Correlations
# ============================================================
def plot_correlations(features, labels):
    numeric_labels = labels.select_dtypes(include=[np.number])
    label_names = list(numeric_labels.columns)

    labels_prefixed = numeric_labels.copy()
    combined = pd.concat([features, labels_prefixed], axis=1, join="inner")
    corr_matrix = combined.corr(numeric_only=True)

    # Top 10 features per label
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, label_col in enumerate(label_names):
        ax = axes[idx]
        corr_series = corr_matrix[label_col].drop(label_names, errors="ignore")
        top = corr_series.abs().sort_values(ascending=False).head(10)
        vals = corr_series.loc[top.index]

        colors = [COLORS["pos"] if v > 0 else COLORS["neg"] for v in vals]
        ax.barh(range(len(vals)), vals.values, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=7)
        ax.set_title(label_col, fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)

    fig.suptitle("Top 10 Feature Correlations per Label", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_feature_label_corr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/7] Feature-label correlation chart saved")


# ============================================================
# 3. Collinearity Heatmap (top correlated pairs)
# ============================================================
def plot_collinearity(features):
    corr = features.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr.where(mask)

    pairs = []
    for c in upper.columns:
        for i in upper.index:
            v = upper.loc[i, c]
            if pd.notna(v) and abs(v) > 0.85:
                pairs.append((i, c, v))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    pairs = pairs[:25]

    fig, ax = plt.subplots(figsize=(12, 7))
    names = [f"{a}  /  {b}" for a, b, _ in pairs]
    vals = [v for _, _, v in pairs]
    colors = ["#d62728" if abs(v) > 0.95 else "#ff7f0e" if abs(v) > 0.9
              else "#ffcc00" for v in vals]

    ax.barh(range(len(pairs)), vals, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Correlation")
    ax.set_title("Top 25 Collinear Feature Pairs (|r| > 0.85)")
    ax.invert_yaxis()
    ax.axvline(0.9, color="red", ls="--", alpha=0.5)
    ax.axvline(-0.9, color="red", ls="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_collinearity.png"), dpi=150)
    plt.close(fig)
    print("  [3/7] Collinearity chart saved")


# ============================================================
# 4. Label Distributions
# ============================================================
def plot_label_distributions(labels):
    numeric_labels = labels.select_dtypes(include=[np.number])
    cat_labels = labels.select_dtypes(include=["category"])

    n_num = len(numeric_labels.columns)
    n_cat = len(cat_labels.columns)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, max(n_num, n_cat), figure=fig)

    # Numeric labels: histograms
    for i, col in enumerate(numeric_labels.columns):
        ax = fig.add_subplot(gs[0, i])
        data = numeric_labels[col].dropna()
        ax.hist(data, bins=50, color=COLORS["label"], alpha=0.7, edgecolor="white")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        # Add stats
        ax.axvline(data.mean(), color="black", ls="--", lw=1)
        ax.text(0.95, 0.95, f"mean={data.mean():.4f}\nstd={data.std():.4f}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Categorical labels: bar charts
    for i, col in enumerate(cat_labels.columns):
        ax = fig.add_subplot(gs[1, i])
        counts = labels[col].value_counts()
        colors_list = [COLORS["pos"], COLORS["neutral"], COLORS["neg"]]
        if len(counts) > len(colors_list):
            colors_list = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        ax.bar(range(len(counts)), counts.values,
               color=colors_list[:len(counts)], edgecolor="white")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, fontsize=8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        # Percentages
        for j, (name, val) in enumerate(counts.items()):
            pct = val / counts.sum() * 100
            ax.text(j, val + counts.max() * 0.02, f"{pct:.1f}%",
                    ha="center", fontsize=7)

    fig.suptitle("Label Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_label_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/7] Label distribution chart saved")


# ============================================================
# 5. Key Features vs Gold Price (time series)
# ============================================================
def plot_key_features_ts(features, gld):
    key_features = [
        ("real_yield_10y", "10Y Real Yield (TIPS)"),
        ("gvz", "GVZ (Gold IV)"),
        ("cot_noncomm_net", "COT Non-Comm Net"),
        ("vix_level", "VIX"),
        ("close_to_sma_20", "GLD vs SMA(20)"),
        ("rsi_14", "RSI(14)"),
    ]

    fig, axes = plt.subplots(len(key_features), 1, figsize=(16, 18),
                             sharex=True)
    gold = gld["Close"]

    for i, (feat_name, title) in enumerate(key_features):
        ax = axes[i]
        if feat_name not in features.columns:
            ax.set_title(f"{title} (not found)")
            continue

        feat = features[feat_name].dropna()
        ax2 = ax.twinx()

        ax.plot(gold.index, gold.values, color="#FFD700", alpha=0.4, lw=1,
                label="GLD")
        ax2.plot(feat.index, feat.values, color=COLORS["tech"], lw=1,
                 label=feat_name)

        ax.set_ylabel("GLD", color="#FFD700", fontsize=8)
        ax2.set_ylabel(feat_name, color=COLORS["tech"], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)

    fig.suptitle("Key Features vs Gold Price", fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_key_features_ts.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/7] Key features time series chart saved")


# ============================================================
# 6. Feature Category Distribution
# ============================================================
def plot_feature_categories(features):
    # Classify features by source
    tech_kw = ["ret_", "sma", "ema", "rsi", "macd", "stoch", "roc",
               "hv_", "bb_", "atr", "daily_range", "vol_ratio", "vol_change",
               "obv", "price_vol", "gap", "gc_gld", "dxy", "vix_level",
               "vix_ret", "vix_sma", "copper", "gold_silver", "us10y",
               "crude", "close_to", "ma_alignment"]
    vol_kw = ["gvz", "vix_vix", "vix_term", "vix_back", "vix_9d", "move",
              "rv_", "vrp", "days_to", "fomc", "nfp", "cpi_window",
              "is_fomc", "is_nfp", "is_cpi"]

    cats = {"Technical": 0, "Macro": 0, "Volatility": 0}
    for col in features.columns:
        if any(col.startswith(k) or k in col for k in vol_kw):
            cats["Volatility"] += 1
        elif any(col.startswith(k) or k in col for k in tech_kw):
            cats["Technical"] += 1
        else:
            cats["Macro"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    colors = [COLORS["tech"], COLORS["macro"], COLORS["vol"]]
    ax1.pie(cats.values(), labels=cats.keys(), colors=colors,
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    ax1.set_title("Feature Count by Category")

    # Missing by category
    tech_cols = [c for c in features.columns
                 if any(c.startswith(k) or k in c for k in tech_kw)]
    vol_cols = [c for c in features.columns
                if any(c.startswith(k) or k in c for k in vol_kw)]
    macro_cols = [c for c in features.columns
                  if c not in tech_cols and c not in vol_cols]

    avg_miss = {
        "Technical": features[tech_cols].isnull().mean().mean() if tech_cols else 0,
        "Macro": features[macro_cols].isnull().mean().mean() if macro_cols else 0,
        "Volatility": features[vol_cols].isnull().mean().mean() if vol_cols else 0,
    }
    ax2.bar(avg_miss.keys(), [v * 100 for v in avg_miss.values()], color=colors)
    ax2.set_ylabel("Average Missing Rate (%)")
    ax2.set_title("Average Missing Rate by Category")
    for i, (k, v) in enumerate(avg_miss.items()):
        ax2.text(i, v * 100 + 0.3, f"{v:.1%}", ha="center", fontsize=10)

    fig.suptitle(f"Feature Overview: {features.shape[1]} features x "
                 f"{features.shape[0]} rows", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "06_feature_categories.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/7] Feature category chart saved")


# ============================================================
# 7. Feature Stability (rolling correlation with label)
# ============================================================
def plot_feature_stability(features, labels):
    target = "fwd_ret_10d"
    if target not in labels.columns:
        print("  [7/7] Skipped (no fwd_ret_10d)")
        return

    top_feats = ["real_yield_10y", "gvz", "cot_noncomm_net",
                 "close_to_sma_20", "atr_14_pct", "vix_term_slope"]
    top_feats = [f for f in top_feats if f in features.columns]

    label_s = labels[target]

    fig, ax = plt.subplots(figsize=(16, 6))
    for feat_name in top_feats:
        combined = pd.concat([features[feat_name], label_s], axis=1, join="inner")
        combined.columns = ["feat", "label"]
        roll_corr = combined["feat"].rolling(504).corr(combined["label"])
        ax.plot(roll_corr.index, roll_corr.values, lw=1.2, label=feat_name,
                alpha=0.8)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Rolling 2Y Correlation")
    ax.set_title(f"Feature Stability: Rolling 504-day Correlation with {target}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "07_feature_stability.png"), dpi=150)
    plt.close(fig)
    print("  [7/7] Feature stability chart saved")


# ============================================================
# Summary text
# ============================================================
def write_summary(features, labels):
    lines = []
    lines.append("# Phase 2 Feature Engineering Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n## Dataset Overview")
    lines.append(f"- Features: **{features.shape[1]}** columns x "
                 f"**{features.shape[0]}** rows")
    lines.append(f"- Labels: **{labels.shape[1]}** columns")
    lines.append(f"- Date range: {features.index.min().date()} ~ "
                 f"{features.index.max().date()}")

    miss = features.isnull().mean()
    lines.append(f"\n## Missing Rates")
    lines.append(f"- Features with 0% missing: {(miss == 0).sum()}")
    lines.append(f"- Features with >15% missing: {(miss > 0.15).sum()}")
    lines.append(f"- Average missing rate: {miss.mean():.2%}")

    corr = features.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper = corr.where(mask)
    n_high = sum(1 for c in upper.columns for i in upper.index
                 if pd.notna(upper.loc[i, c]) and abs(upper.loc[i, c]) > 0.9)
    lines.append(f"\n## Collinearity")
    lines.append(f"- Feature pairs with |r| > 0.9: **{n_high}**")

    lines.append(f"\n## Labels")
    for col in labels.select_dtypes(include=[np.number]).columns:
        data = labels[col].dropna()
        lines.append(f"- {col}: mean={data.mean():.4f}, "
                     f"std={data.std():.4f}, n={len(data)}")
    for col in labels.select_dtypes(include=["category"]).columns:
        counts = labels[col].value_counts()
        dist = ", ".join(f"{k}={v/counts.sum():.1%}" for k, v in counts.items())
        lines.append(f"- {col}: {dist}")

    lines.append(f"\n## Charts")
    lines.append("1. `01_missing_rate.png` — Feature missing rates")
    lines.append("2. `02_feature_label_corr.png` — Top feature-label correlations")
    lines.append("3. `03_collinearity.png` — Highly correlated feature pairs")
    lines.append("4. `04_label_distributions.png` — Label histograms & bar charts")
    lines.append("5. `05_key_features_ts.png` — Key features vs gold price")
    lines.append("6. `06_feature_categories.png` — Feature count by category")
    lines.append("7. `07_feature_stability.png` — Rolling correlation stability")

    path = os.path.join(OUTPUT_DIR, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  REPORT.md saved")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 50)
    print("  Phase 2 Feature Report Generator")
    print("=" * 50)

    features, labels, gld = load_data()
    print(f"\n  Loaded: {features.shape[1]} features, "
          f"{labels.shape[1]} labels, {len(gld)} rows\n")

    plot_missing_rate(features)
    plot_correlations(features, labels)
    plot_collinearity(features)
    plot_label_distributions(labels)
    plot_key_features_ts(features, gld)
    plot_feature_categories(features)
    plot_feature_stability(features, labels)
    write_summary(features, labels)

    print(f"\n  All charts saved to {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
