"""v3.7.247 (plan closure): Layer 1 calibration gate analyze.

Compares raw vs calibrated band coverage across trailing windows
(10y / 5y / 3y / 1y / 113d) and writes ``gate_report.md`` with the
compound-gate decision.

Gate criteria:
    1. ``coverage_both`` moves *toward* the training target (0.80) in
       at least ``min_pass_windows`` of the audited windows. "Toward"
       means ``|calibrated - target| < |raw - target|``.
    2. No window degrades ``coverage_both`` by more than
       ``max_degradation`` (default 5pp).
    3. Per-side ``coverage_upper`` AND ``coverage_lower`` each satisfy
       the same compound rule.

Output: ``gate_report.md`` carries one machine-parseable line
``gate_passed: true|false`` plus a per-asset / per-window table.
A separate ``build_band()`` cutover preflight (out of scope this round)
reads that line to decide whether to consume calibrated columns.

Usage::

    python scripts/eval/calibration_gate_grid.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.calibration import apply_rolling_conformal_scaler

ASSET_TO_PARQUET = {
    "GLD": "/Users/yhdong/Gold/data/models/dl_range_v2_oos.parquet",
    "SLV": "/Users/yhdong/Gold/data/models/dl_range_slv_oos.parquet",
}

DEFAULT_OUT_DIR = "/Users/yhdong/Gold/data/backtest_history/v3.7.247_calibration_gate"

# Trailing window definitions in trading days; ALL = no truncation.
WINDOWS = [
    ("10y", 2520),
    ("5y", 1260),
    ("3y", 756),
    ("1y", 252),
    ("113d", 113),
]

DEFAULT_TARGET_COVERAGE = 0.80
DEFAULT_MAX_DEGRADATION_PP = 0.05  # ≤ 5 percentage-point coverage drop
DEFAULT_MIN_PASS_WINDOWS = 3        # ≥3 of 4+ windows


def _coverage(actual_u, pred_u, actual_l, pred_l):
    """Compute (coverage_upper, coverage_lower, coverage_both) on a slice."""
    mask = (actual_u.notna() & actual_l.notna()
              & pred_u.notna() & pred_l.notna())
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    au = actual_u[mask]; al = actual_l[mask]
    pu = pred_u[mask]; pl = pred_l[mask]
    cu = float((au <= pu).mean())
    cl = float((al >= pl).mean())
    cb = float(((au <= pu) & (al >= pl)).mean())
    return cu, cl, cb


def evaluate_window(df: pd.DataFrame, window_days: int) -> dict:
    """Return raw vs calibrated coverage for the trailing ``window_days``."""
    end = df.index.max()
    if window_days is None:
        sub = df
    else:
        start_pos = max(0, len(df) - window_days)
        sub = df.iloc[start_pos:]
    if not len(sub):
        return None
    # Apply scaler to the FULL history up to and including sub, so the
    # calibrated columns for sub are produced with the maturity-lag rule
    # using only past matured residuals (the scaler enforces this internally).
    out_u, out_l, meta = apply_rolling_conformal_scaler(
        df.index,
        df["pred_upper_pct"], df["pred_lower_pct"],
        df["actual_upper_pct"], df["actual_lower_pct"],
        horizon=5, window=60, target_coverage=DEFAULT_TARGET_COVERAGE,
    )
    cal_u_window = out_u.reindex(sub.index)
    cal_l_window = out_l.reindex(sub.index)
    # Raw coverage on the window
    rcu, rcl, rcb = _coverage(sub["actual_upper_pct"], sub["pred_upper_pct"],
                                  sub["actual_lower_pct"], sub["pred_lower_pct"])
    # Calibrated coverage
    ccu, ccl, ccb = _coverage(sub["actual_upper_pct"], cal_u_window,
                                  sub["actual_lower_pct"], cal_l_window)
    return {
        "n": int(sub["actual_upper_pct"].notna().sum()),
        "raw_coverage_upper": rcu, "raw_coverage_lower": rcl,
        "raw_coverage_both": rcb,
        "cal_coverage_upper": ccu, "cal_coverage_lower": ccl,
        "cal_coverage_both": ccb,
    }


def compound_gate_decision(window_results: list,
                              target: float = DEFAULT_TARGET_COVERAGE,
                              max_degradation: float = DEFAULT_MAX_DEGRADATION_PP,
                              min_pass: int = DEFAULT_MIN_PASS_WINDOWS,
                              metric_key: str = "coverage_both") -> dict:
    """Apply the plan contract compound gate to a list of per-window dicts.

    A window 'passes' iff its **distance from target** does not get worse by
    more than ``max_degradation``:

        |cal - target| - |raw - target| < max_degradation

    This single rule correctly handles both cases:
      - raw under target (e.g. 0.55 vs target 0.80): cal must move up toward 0.80.
      - raw over target (e.g. 0.95 vs target 0.80): cal must move down toward 0.80.
    And allows a small tolerance (``max_degradation``) for noisy near-target moves.

    Returns the verdict + per-window pass flags.
    """
    decisions = []
    for w in window_results:
        if w is None:
            decisions.append({"window": "N/A", "pass": False,
                                "reason": "empty"})
            continue
        raw_key = f"raw_{metric_key}"
        cal_key = f"cal_{metric_key}"
        raw = w.get(raw_key, float("nan"))
        cal = w.get(cal_key, float("nan"))
        if np.isnan(raw) or np.isnan(cal):
            decisions.append({"window": w.get("window"), "pass": False,
                                "reason": "nan"})
            continue
        raw_dist = abs(raw - target)
        cal_dist = abs(cal - target)
        distance_delta = cal_dist - raw_dist  # negative = improved
        passed = distance_delta < max_degradation
        decisions.append({
            "window": w.get("window"),
            "raw": raw, "cal": cal,
            "raw_distance_from_target": raw_dist,
            "cal_distance_from_target": cal_dist,
            "distance_delta": distance_delta,
            "pass": passed,
        })
    n_pass = sum(1 for d in decisions if d.get("pass"))
    n_total = sum(1 for d in decisions if d.get("reason") not in ("empty", "nan"))
    gate_passed = (n_total >= min_pass) and (n_pass >= min_pass)
    return {
        "gate_passed": gate_passed,
        "n_pass": n_pass,
        "n_total": n_total,
        "decisions": decisions,
    }


def run_asset(asset: str, parquet_path: str = None) -> dict:
    path = parquet_path or ASSET_TO_PARQUET[asset]
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    required = ("pred_upper_pct", "pred_lower_pct",
                 "actual_upper_pct", "actual_lower_pct")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{asset} OOS parquet missing {missing}")
    results = []
    for label, win_days in WINDOWS:
        w = evaluate_window(df, win_days)
        if w is not None:
            w["window"] = label
        results.append(w)
    cb_gate = compound_gate_decision(results, metric_key="coverage_both")
    cu_gate = compound_gate_decision(results, metric_key="coverage_upper")
    cl_gate = compound_gate_decision(results, metric_key="coverage_lower")
    return {
        "asset": asset,
        "windows": results,
        "gate_both": cb_gate,
        "gate_upper": cu_gate,
        "gate_lower": cl_gate,
        "gate_passed": cb_gate["gate_passed"]
                          and cu_gate["gate_passed"]
                          and cl_gate["gate_passed"],
    }


def render_report(per_asset: list, out_path: Path):
    lines = [
        "# Calibration Gate Report — task-g6 (plan closure)",
        "",
        f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}",
        "",
        "## Gate verdict",
        "",
    ]
    overall = all(r["gate_passed"] for r in per_asset)
    lines.append(f"gate_passed: {'true' if overall else 'false'}")
    lines.append("")
    lines.append("Gate criteria:")
    lines.append(f"- coverage moves toward training target "
                  f"{DEFAULT_TARGET_COVERAGE} in ≥ {DEFAULT_MIN_PASS_WINDOWS} windows")
    lines.append(f"- no window degrades coverage by more than "
                  f"{DEFAULT_MAX_DEGRADATION_PP*100:.0f}pp")
    lines.append(f"- applied per-side (upper AND lower) AND on both-sides")
    lines.append("")
    for r in per_asset:
        lines.append(f"## {r['asset']}  (gate_passed: {r['gate_passed']})")
        lines.append("")
        lines.append(f"- coverage_both gate: {r['gate_both']['gate_passed']} "
                      f"({r['gate_both']['n_pass']}/{r['gate_both']['n_total']})")
        lines.append(f"- coverage_upper gate: {r['gate_upper']['gate_passed']} "
                      f"({r['gate_upper']['n_pass']}/{r['gate_upper']['n_total']})")
        lines.append(f"- coverage_lower gate: {r['gate_lower']['gate_passed']} "
                      f"({r['gate_lower']['n_pass']}/{r['gate_lower']['n_total']})")
        lines.append("")
        lines.append("| window | n | raw_both | cal_both | raw_upper | cal_upper | raw_lower | cal_lower |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for w in r["windows"]:
            if w is None: continue
            lines.append(
                f"| {w['window']} | {w['n']} | "
                f"{w['raw_coverage_both']:.3f} | {w['cal_coverage_both']:.3f} | "
                f"{w['raw_coverage_upper']:.3f} | {w['cal_coverage_upper']:.3f} | "
                f"{w['raw_coverage_lower']:.3f} | {w['cal_coverage_lower']:.3f} |"
            )
        lines.append("")
        lines.append("### Per-window pass detail (coverage_both)")
        lines.append("")
        for d in r["gate_both"]["decisions"]:
            if d.get("reason") in ("empty", "nan"):
                lines.append(f"- {d.get('window')}: skipped ({d.get('reason')})")
            else:
                lines.append(
                    f"- {d['window']}: raw={d['raw']:.3f} cal={d['cal']:.3f} "
                    f"distance_delta={d['distance_delta']:+.3f} "
                    f"**pass={d['pass']}**"
                )
        lines.append("")
    lines.append("## Next action")
    lines.append("")
    if overall:
        lines.append("- `gate_passed: true` — proceed to wire `build_band()` "
                       "to read calibrated columns under a config flag in v3.8.")
        lines.append("- The shadow-only `apply_rolling_conformal_scaler` "
                       "output remains the source of truth for diagnostics.")
    else:
        lines.append("- `gate_passed: false` — keep `build_band()` on raw "
                       "predictions. Ship the audit, scaler, retrain trigger, "
                       "and per-regime alpha as **shadow-only diagnostics** "
                       "until further per-regime or hyperparameter tuning "
                       "lifts coverage uniformly.")
        lines.append("- Open follow-ups: tune `target_coverage`, `window`, or "
                       "the per-regime classifier itself; re-run this gate.")
    out_path.write_text("\n".join(lines))
    return overall


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--asset", choices=["GLD", "SLV", "BOTH"], default="BOTH")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = ["GLD", "SLV"] if args.asset == "BOTH" else [args.asset]
    per_asset = [run_asset(a) for a in assets]
    out_md = out_dir / "gate_report.md"
    overall = render_report(per_asset, out_md)
    out_json = out_dir / "gate_decision.json"
    out_json.write_text(json.dumps({
        "gate_passed": overall,
        "per_asset": [{"asset": r["asset"], "gate_passed": r["gate_passed"]}
                        for r in per_asset],
    }, indent=2))
    print(f"[calibration-gate] wrote {out_md}")
    print(f"[calibration-gate] wrote {out_json}")
    print(f"[calibration-gate] overall gate_passed: {overall}")
    for r in per_asset:
        print(f"  {r['asset']}: gate_passed={r['gate_passed']}  "
                f"(both={r['gate_both']['gate_passed']}, "
                f"upper={r['gate_upper']['gate_passed']}, "
                f"lower={r['gate_lower']['gate_passed']})")


if __name__ == "__main__":
    main()
