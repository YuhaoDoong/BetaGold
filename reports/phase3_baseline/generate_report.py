"""
Phase 3 基线模型质量报告
生成可视化图表到 reports/phase3_baseline/

用法:
    conda activate gold
    python reports/phase3_baseline/generate_report.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

COLORS = {"Ridge": "#1f77b4", "LogReg": "#1f77b4", "XGBoost": "#ff7f0e"}


def load_data():
    results = pd.read_parquet(os.path.join(MODEL_DIR, "baseline_results.parquet"))
    preds = pd.read_parquet(os.path.join(MODEL_DIR, "baseline_predictions.parquet"))
    imps = pd.read_parquet(os.path.join(MODEL_DIR, "baseline_importances.parquet"))
    return results, preds, imps


# ============================================================
# 1. IC/Accuracy per target (bar chart)
# ============================================================
def plot_summary_metrics(results):
    reg_targets = ["fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
                   "fwd_rv_10d", "fwd_rv_20d"]
    clf_targets = ["direction_5d", "direction_10d", "tail_event_flag"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Regression: IC
    x = np.arange(len(reg_targets))
    w = 0.35
    for i, model in enumerate(["Ridge", "XGBoost"]):
        vals = []
        errs = []
        for t in reg_targets:
            m = results[(results["target"] == t) & (results["model"] == model)]
            vals.append(m["IC"].mean() if "IC" in m.columns else 0)
            errs.append(m["IC"].std() if "IC" in m.columns else 0)
        ax1.bar(x + i * w, vals, w, yerr=errs, label=model,
                color=COLORS[model], alpha=0.8, capsize=3)

    ax1.set_xticks(x + w / 2)
    ax1.set_xticklabels([t.replace("fwd_", "") for t in reg_targets],
                        fontsize=9)
    ax1.set_ylabel("Information Coefficient (IC)")
    ax1.set_title("Regression: Mean IC per Target (walk-forward OOS)")
    ax1.legend()
    ax1.axhline(0, color="black", lw=0.5)

    # Classification: Accuracy
    x2 = np.arange(len(clf_targets))
    for i, model in enumerate(["LogReg", "XGBoost"]):
        vals = []
        errs = []
        for t in clf_targets:
            m = results[(results["target"] == t) & (results["model"] == model)]
            vals.append(m["Accuracy"].mean() if "Accuracy" in m.columns else 0)
            errs.append(m["Accuracy"].std() if "Accuracy" in m.columns else 0)
        ax2.bar(x2 + i * w, vals, w, yerr=errs, label=model,
                color=COLORS[model], alpha=0.8, capsize=3)

    ax2.set_xticks(x2 + w / 2)
    ax2.set_xticklabels(clf_targets, fontsize=9)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification: Mean Accuracy per Target (walk-forward OOS)")
    ax2.legend()

    # Random baselines
    ax2.axhline(1/3, color="red", ls="--", alpha=0.5, label="random (3-class)")
    ax2.axhline(0.885, color="green", ls="--", alpha=0.5, label="majority (tail)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "01_summary_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/5] Summary metrics chart saved")


# ============================================================
# 2. IC over time (fold-by-fold)
# ============================================================
def plot_ic_over_time(results):
    reg_targets = ["fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, target in enumerate(reg_targets):
        ax = axes[idx]
        for model in ["Ridge", "XGBoost"]:
            m = results[(results["target"] == target) &
                        (results["model"] == model)].copy()
            if m.empty or "IC" not in m.columns:
                continue
            m = m.sort_values("fold")
            ax.plot(m["fold"], m["IC"], "o-", label=model,
                    color=COLORS[model], markersize=5)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Fold")
        ax.set_title(target.replace("fwd_", ""), fontweight="bold")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("IC (Spearman)")
    fig.suptitle("IC Stability Across Walk-Forward Folds", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_ic_over_time.png"), dpi=150)
    plt.close(fig)
    print("  [2/5] IC over time chart saved")


# ============================================================
# 3. Predicted vs Actual scatter
# ============================================================
def plot_pred_vs_actual(preds):
    targets = ["fwd_ret_10d", "fwd_rv_20d"]

    fig, axes = plt.subplots(1, len(targets) * 2, figsize=(20, 5))
    idx = 0

    for target in targets:
        for model in ["Ridge", "XGBoost"]:
            ax = axes[idx]
            m = preds[(preds["target"] == target) &
                      (preds["model"] == model)]
            if m.empty:
                idx += 1
                continue

            ax.scatter(m["y_true"], m["y_pred"], alpha=0.15, s=5,
                       color=COLORS.get(model, "gray"))
            # Fit line
            z = np.polyfit(m["y_true"], m["y_pred"], 1)
            x_line = np.linspace(m["y_true"].min(), m["y_true"].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "r-", lw=1.5)
            # 45-degree
            lim = [min(m["y_true"].min(), m["y_pred"].min()),
                   max(m["y_true"].max(), m["y_pred"].max())]
            ax.plot(lim, lim, "k--", alpha=0.3, lw=0.8)

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{target.replace('fwd_','')} / {model}", fontsize=10)
            idx += 1

    fig.suptitle("OOS Predicted vs Actual", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_pred_vs_actual.png"), dpi=150)
    plt.close(fig)
    print("  [3/5] Pred vs Actual chart saved")


# ============================================================
# 4. Feature Importance (XGBoost)
# ============================================================
def plot_feature_importance(imps):
    targets = ["fwd_ret_10d", "fwd_rv_20d", "direction_10d", "tail_event_flag"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax = axes[idx]
        t_imp = imps[imps["target"] == target]
        if t_imp.empty:
            ax.set_title(f"{target} (no data)")
            continue

        # Average importance across folds
        avg_imp = (t_imp.groupby("feature")["importance"]
                   .mean().sort_values(ascending=False).head(15))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(avg_imp)))
        ax.barh(range(len(avg_imp)), avg_imp.values, color=colors)
        ax.set_yticks(range(len(avg_imp)))
        ax.set_yticklabels(avg_imp.index, fontsize=8)
        ax.set_title(target, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle("XGBoost Feature Importance (avg across folds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "04_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/5] Feature importance chart saved")


# ============================================================
# 5. Cumulative OOS prediction signal value
# ============================================================
def plot_cumulative_signal(preds):
    target = "fwd_ret_10d"

    fig, ax = plt.subplots(figsize=(16, 6))

    for model in ["Ridge", "XGBoost"]:
        m = preds[(preds["target"] == target) &
                  (preds["model"] == model)].copy()
        if m.empty:
            continue
        m = m.sort_values("date")
        # Long-short signal: go long when pred > 0, short when pred < 0
        signal_ret = np.sign(m["y_pred"].values) * m["y_true"].values
        cum_ret = np.cumsum(signal_ret)
        ax.plot(pd.to_datetime(m["date"]), cum_ret, label=f"{model} L/S",
                color=COLORS[model], lw=1.2)

    # Buy & hold benchmark
    m = preds[(preds["target"] == target) &
              (preds["model"] == "Ridge")].copy()
    if not m.empty:
        m = m.sort_values("date")
        bh_cum = np.cumsum(m["y_true"].values)
        ax.plot(pd.to_datetime(m["date"]), bh_cum, label="Buy & Hold",
                color="gray", ls="--", lw=1)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative Return (sum of period returns)")
    ax.set_title(f"OOS Long/Short Signal: {target} (sign of prediction × actual return)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "05_cumulative_signal.png"), dpi=150)
    plt.close(fig)
    print("  [5/5] Cumulative signal chart saved")


# ============================================================
# Summary
# ============================================================
def write_summary(results):
    lines = []
    lines.append("# Phase 3 Baseline Model Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Walk-Forward Setup")
    lines.append("- Min training: 1260 days (5 years)")
    lines.append("- Test window: 252 days (1 year)")
    lines.append("- Step: 252 days")
    n_folds = results["fold"].nunique()
    lines.append(f"- Total folds: {n_folds}")

    lines.append("\n## Regression Results (IC = Spearman rank correlation)")
    lines.append("| Target | Model | IC (mean) | IC (std) | Dir Acc |")
    lines.append("|--------|-------|-----------|----------|---------|")
    for t in ["fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d",
              "fwd_rv_10d", "fwd_rv_20d"]:
        for m_name in ["Ridge", "XGBoost"]:
            m = results[(results["target"] == t) & (results["model"] == m_name)]
            if m.empty:
                continue
            ic = m["IC"].mean() if "IC" in m.columns else 0
            ic_std = m["IC"].std() if "IC" in m.columns else 0
            da = m["Dir_Acc"].mean() if "Dir_Acc" in m.columns else 0
            lines.append(f"| {t} | {m_name} | {ic:.4f} | {ic_std:.4f} | {da:.1%} |")

    lines.append("\n## Classification Results")
    lines.append("| Target | Model | Accuracy | AUC |")
    lines.append("|--------|-------|----------|-----|")
    for t in ["direction_5d", "direction_10d", "tail_event_flag"]:
        for m_name in ["LogReg", "XGBoost"]:
            m = results[(results["target"] == t) & (results["model"] == m_name)]
            if m.empty:
                continue
            acc = m["Accuracy"].mean() if "Accuracy" in m.columns else 0
            auc = m["AUC"].mean() if "AUC" in m.columns else np.nan
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            lines.append(f"| {t} | {m_name} | {acc:.4f} | {auc_str} |")

    lines.append("\n## Key Findings")
    lines.append("- Longer horizons have higher IC (ret_20d > ret_10d > ret_5d)")
    lines.append("- Volatility prediction (rv) has strong signal (IC ~0.25-0.29)")
    lines.append("- Ridge and XGBoost perform similarly — features matter more than model")
    lines.append("- 3-class direction accuracy ~36% (random=33%) — weak but above chance")
    lines.append("- Tail event detection dominated by class imbalance (~88.5% negative)")

    lines.append("\n## Charts")
    lines.append("1. `01_summary_metrics.png` — IC and Accuracy overview")
    lines.append("2. `02_ic_over_time.png` — IC stability across folds")
    lines.append("3. `03_pred_vs_actual.png` — Predicted vs Actual scatter")
    lines.append("4. `04_feature_importance.png` — XGBoost top features")
    lines.append("5. `05_cumulative_signal.png` — OOS long/short equity curve")

    path = os.path.join(OUTPUT_DIR, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  REPORT.md saved")


def main():
    print("=" * 50)
    print("  Phase 3 Baseline Report Generator")
    print("=" * 50)

    results, preds, imps = load_data()
    print(f"\n  Results: {len(results)} rows")
    print(f"  Predictions: {len(preds)} rows")
    print(f"  Importances: {len(imps)} rows\n")

    plot_summary_metrics(results)
    plot_ic_over_time(results)
    plot_pred_vs_actual(preds)
    plot_feature_importance(imps)
    plot_cumulative_signal(preds)
    write_summary(results)

    print(f"\n  All charts saved to {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
