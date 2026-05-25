"""v3.7.243: DL Range OOS calibration audit.

Reads ``data/models/dl_range_v2_oos.parquet`` (GLD) or
``dl_range_slv_oos.parquet`` (SLV) and produces a per-month (and optionally
per-regime) calibration report.

**Label definition is authoritative.** This script reads the parquet's
``actual_upper_pct`` / ``actual_lower_pct`` columns directly, which were
written by ``src/models/train_dl_range.build_targets``:

  max_high_5d = high.shift(-1).rolling(5).max().shift(-4)   # 5d forward
  min_low_5d  = low.shift(-1).rolling(5).min().shift(-4)
  upper_pct   = (max_high_5d / close - 1) * 100
  lower_pct   = (min_low_5d  / close - 1) * 100

These are **5-day forward max-high / min-low vs t-day close**, NOT single-day
overnight returns. The original idea draft's "5-6x wider, 87.6% coverage"
figures used the wrong (single-day H/L vs prior Close) definition; the
correct figures per the parquet are reported here.

CLI::

    python scripts/eval/model_calibration_audit.py --asset GLD \
        --start 2025-12-01 --end 2026-05-13 \
        [--regime-col regime] [--out-dir ...]

Output: ``<out_dir>/<asset>_<start>_<end>.csv`` plus a small markdown summary.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Parent-of-parent so the script can be invoked stand-alone
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


REQUIRED_COLUMNS = ("pred_upper_pct", "pred_lower_pct",
                     "actual_upper_pct", "actual_lower_pct")

DEFAULT_OUT_DIR = "/Users/yhdong/Gold/data/backtest_history/v3.7.243_calibration_audit"

ASSET_TO_PARQUET = {
    "GLD": "/Users/yhdong/Gold/data/models/dl_range_v2_oos.parquet",
    "SLV": "/Users/yhdong/Gold/data/models/dl_range_slv_oos.parquet",
}


def _safe_ratio(num: pd.Series, denom: pd.Series) -> float:
    """Return mean(num) / mean(denom) with a tiny-denominator guard."""
    n = float(num.mean())
    d = float(denom.mean())
    if abs(d) < 1e-9:
        return float("nan")
    return n / d


def compute_calibration_metrics(df: pd.DataFrame) -> dict:
    """Compute the per-group calibration metrics from an OOS slice.

    Args:
        df: rows of the OOS parquet (already filtered to a window/group).
            Must include ``pred_upper_pct``, ``pred_lower_pct``,
            ``actual_upper_pct``, ``actual_lower_pct``.

    Returns:
        Dict with ``n``, mean predicted/realized bounds, width ratios per
        side, and three coverage rates.

    Note:
        ``width_ratio_upper`` uses **signed** means so the ratio reflects
        directional over-prediction. For coverage rates we use the standard
        ``actual_upper ≤ pred_upper`` and ``actual_lower ≥ pred_lower``
        per ``src/models/train_dl_range.eval_range``.
    """
    sub = df.dropna(subset=list(REQUIRED_COLUMNS))
    n = len(sub)
    if n == 0:
        return {"n": 0}
    pred_u = sub["pred_upper_pct"]
    pred_l = sub["pred_lower_pct"]
    actual_u = sub["actual_upper_pct"]
    actual_l = sub["actual_lower_pct"]
    upper_covered = (actual_u <= pred_u)
    lower_covered = (actual_l >= pred_l)
    both = upper_covered & lower_covered
    return {
        "n": n,
        "pred_upper_mean": round(float(pred_u.mean()), 3),
        "pred_lower_mean": round(float(pred_l.mean()), 3),
        "actual_upper_mean": round(float(actual_u.mean()), 3),
        "actual_lower_mean": round(float(actual_l.mean()), 3),
        "width_ratio_upper": round(_safe_ratio(pred_u, actual_u), 3),
        "width_ratio_lower": round(_safe_ratio(pred_l, actual_l), 3),
        "coverage_upper": round(float(upper_covered.mean()), 4),
        "coverage_lower": round(float(lower_covered.mean()), 4),
        "coverage_both": round(float(both.mean()), 4),
    }


def run_audit(asset: str,
                 start: pd.Timestamp,
                 end: pd.Timestamp,
                 regime_col: str = None,
                 parquet_path: str = None) -> pd.DataFrame:
    """Load the OOS parquet for ``asset`` and emit a per-month [×per-regime] table.

    Raises:
        ValueError if any of REQUIRED_COLUMNS is missing (we refuse to
        substitute single-day overnight returns silently — that was the
        original bug).
    """
    path = parquet_path or ASSET_TO_PARQUET.get(asset.upper())
    if path is None:
        raise ValueError(f"Unknown asset {asset!r}; pass --parquet-path explicitly")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet at {path} missing required columns {missing}. "
            f"This script refuses to recompute from OHLC because the original "
            f"v3.7.232 draft used single-day overnight returns by mistake; "
            f"the authoritative labels are 5-day forward H/L per "
            f"src/models/train_dl_range.build_targets."
        )
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    win = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if not len(win):
        return pd.DataFrame()
    win["__month"] = win.index.to_period("M").astype(str)
    if regime_col and regime_col in win.columns:
        group_cols = ["__month", regime_col]
    else:
        group_cols = ["__month"]
    rows = []
    for keys, sub in win.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(compute_calibration_metrics(sub))
        rows.append(row)
    # Also a single all-window aggregate row
    agg = {"__month": "ALL"}
    if regime_col and regime_col in win.columns:
        agg[regime_col] = "ALL"
    agg.update(compute_calibration_metrics(win))
    rows.append(agg)
    return pd.DataFrame(rows)


def _render_markdown(asset: str, start: str, end: str,
                       summary_df: pd.DataFrame) -> str:
    if not len(summary_df):
        return f"# {asset} calibration audit ({start} → {end})\n\nNo rows in window.\n"
    agg = summary_df[summary_df["__month"] == "ALL"].iloc[0] \
            if (summary_df["__month"] == "ALL").any() else None
    lines = [
        f"# {asset} DL Range OOS Calibration Audit",
        "",
        f"- Window: **{start} → {end}**",
        f"- Label definition: **5-day forward H/L vs t-day close** "
          f"(per `src/models/train_dl_range.build_targets`).",
        "",
        "## Aggregate",
        "",
    ]
    if agg is not None:
        lines.extend([
            f"- n = {int(agg['n'])}",
            f"- pred band  = [{agg['pred_lower_mean']}%, {agg['pred_upper_mean']}%]",
            f"- actual band = [{agg['actual_lower_mean']}%, {agg['actual_upper_mean']}%]",
            f"- width ratio upper = **{agg['width_ratio_upper']}×**",
            f"- width ratio lower = **{agg['width_ratio_lower']}×**",
            f"- coverage upper = {agg['coverage_upper']*100:.2f}%",
            f"- coverage lower = {agg['coverage_lower']*100:.2f}%",
            f"- coverage both  = **{agg['coverage_both']*100:.2f}%**",
            "",
        ])
    lines.append("## Per-month breakdown\n")
    cols = list(summary_df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in summary_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## Calibration goal (per the plan)")
    lines.append("")
    lines.append("Coverage repair toward training target (80%), **not** band "
                  "narrowing. Narrowing a band that already under-covers will "
                  "further degrade coverage.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, choices=["GLD", "SLV"])
    parser.add_argument("--start", required=True,
                          help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True,
                          help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--regime-col", default=None,
                          help="Optional regime column for per-regime breakdown")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--parquet-path", default=None,
                          help="Override OOS parquet path (for tests)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_audit(args.asset, args.start, args.end,
                          regime_col=args.regime_col,
                          parquet_path=args.parquet_path)

    csv_path = out_dir / f"{args.asset}_{args.start}_{args.end}.csv"
    summary.to_csv(csv_path, index=False)
    md_path = out_dir / f"{args.asset}_{args.start}_{args.end}.md"
    md_path.write_text(_render_markdown(args.asset, args.start, args.end, summary))
    print(f"[calibration-audit] wrote {csv_path}")
    print(f"[calibration-audit] wrote {md_path}")
    if len(summary):
        print(summary.to_string(index=False))
    else:
        print("[calibration-audit] WARNING: no rows in window")


if __name__ == "__main__":
    main()
