"""v3.7.244 (AC-8): horizon-aware split-conformal scaler for DL Range bands.

Why this exists
---------------
The Round 5 calibration audit (`scripts/eval/model_calibration_audit.py`)
confirmed the GLD/SLV DL Range OOS bands under-cover the 5-day forward
realized H/L (GLD coverage_both 54.87% vs 80% training target). The audit's
per-month break-out showed the drift is **asymmetric**: some months the
model over-predicts one side and under-predicts the other simultaneously
(e.g., GLD 2026-03: width_ratio_upper 3.53 with lower 0.73). A symmetric
"shrink the band" scaler would make things worse.

What this module provides
-------------------------
``apply_rolling_conformal_scaler`` produces per-date, per-side signed
calibration deltas using the classic split-conformal idea: for each side,
pick the quantile of the historical residual that achieves the requested
empirical coverage on the past pool.

Temporal discipline
-------------------
For 5-day forward labels, ``actual_*_pct[s]`` is only fully realized at
``s + 5 trading days``. At calibration time ``t``, we must only use
residuals whose ``label_end_date(s) < t`` (strict). This is the AC-8
maturity-lag rule and is enforced via an explicit ``eligible`` mask on
business-day arithmetic, NOT a generic ``shift(1)``.

Shadow vs cutover
-----------------
This module is a pure function. The decision of *whether* to feed the
calibrated columns back into ``core.signals.build_band()`` lives in a
config preflight check elsewhere (see AC-8: ``calibration.live_cutover``
flag + ``gate_report.md``). This module never touches the filesystem or
the live pipeline directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_HORIZON = 5            # 5-day forward labels per train_dl_range.build_targets
DEFAULT_WINDOW = 60            # rolling residual pool length
DEFAULT_TARGET_COVERAGE = 0.80  # training target per AC-8 + DEC-5
MIN_POOL_SIZE = 20             # below this → no calibration applied


@dataclass(frozen=True)
class ScalerMeta:
    """Per-date diagnostics. Useful for audit and for the retrain trigger."""

    n_eligible: int
    delta_upper: float
    delta_lower: float
    realized_coverage_upper: float
    realized_coverage_lower: float
    fallback_reason: Optional[str] = None  # 'insufficient_pool' | None


def _trading_day_index(dates: pd.DatetimeIndex) -> dict:
    """Map each date → its zero-based business-day rank within ``dates``.

    Cheaper than ``pd.bdate_range`` calls inside the per-date inner loop.
    """
    return {d: i for i, d in enumerate(dates)}


def apply_rolling_conformal_scaler(
    dates: pd.DatetimeIndex,
    pred_upper: pd.Series,
    pred_lower: pd.Series,
    actual_upper: pd.Series,
    actual_lower: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    window: int = DEFAULT_WINDOW,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    min_pool: int = MIN_POOL_SIZE,
) -> tuple:
    """Per-date split-conformal calibrated bounds (horizon-aware, per-side).

    Args:
        dates: business-day-indexed DatetimeIndex aligned with the four series.
        pred_upper, pred_lower: model-predicted bounds at each ``dates[t]``.
        actual_upper, actual_lower: realized 5-day forward max-high /
            min-low (per ``src/models/train_dl_range.build_targets``).
        horizon: label window length in trading days (default 5).
        window: rolling pool length of past *matured* residuals (default 60).
        target_coverage: per-side coverage target (default 0.80).
        min_pool: below this many eligible residuals, return raw bounds.

    Returns:
        ``(pred_upper_calibrated, pred_lower_calibrated, meta_df)``:
        - Two Series indexed by ``dates``.
        - ``meta_df`` indexed by ``dates`` with one ``ScalerMeta`` per row
          serialized to columns. Useful for audit + the retrain trigger.

    Notes:
        Split-conformal direction per side:
            For upper: ``actual_upper <= pred_upper + delta_upper`` should
            hold ≥ ``target_coverage`` of the time on the eligible pool.
            ⇒ ``delta_upper = quantile(actual_upper - pred_upper, target_coverage)``.
            (Negative ⇒ model over-predicts ⇒ shrink upper bound.)
            For lower: ``actual_lower >= pred_lower - delta_lower`` ⇒
            ``delta_lower = quantile(pred_lower - actual_lower, target_coverage)``.
            (Positive ⇒ model under-predicts lower ⇒ widen lower bound.)

        Function is **pure**: no file/network I/O, no globals beyond module
        constants, deterministic in inputs. A separate caller writes
        ``meta_df`` to disk if desired.
    """
    if not (0.0 < target_coverage < 1.0):
        raise ValueError(f"target_coverage must be in (0,1), got {target_coverage}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if window < min_pool:
        raise ValueError(
            f"window ({window}) must be >= min_pool ({min_pool}); "
            f"otherwise the scaler can never have enough samples to calibrate."
        )

    dates = pd.DatetimeIndex(dates)
    pred_upper = pd.Series(pred_upper, index=dates, dtype=float)
    pred_lower = pd.Series(pred_lower, index=dates, dtype=float)
    actual_upper = pd.Series(actual_upper, index=dates, dtype=float)
    actual_lower = pd.Series(actual_lower, index=dates, dtype=float)

    # Residuals per source date s (NaN where actual is unknown)
    res_upper = (actual_upper - pred_upper).values
    res_lower = (pred_lower - actual_lower).values
    n = len(dates)

    # Map each date index → its label_end_date index (business-day shifted)
    # We use the index position because `dates` IS the bdate index.
    out_upper = pred_upper.copy()
    out_lower = pred_lower.copy()
    meta_rows = []
    for t in range(n):
        # Eligibility: source s is usable for calibration at time t iff
        # label_end_date(s) = s + horizon < t.
        # In positional terms: s + horizon < t  ⇔  s <= t - horizon - 1.
        cutoff_idx = t - horizon - 1
        if cutoff_idx < 0:
            meta_rows.append(_empty_meta("no_history"))
            continue
        pool_start = max(0, cutoff_idx - window + 1)
        pool_slice = slice(pool_start, cutoff_idx + 1)
        ru = res_upper[pool_slice]
        rl = res_lower[pool_slice]
        # Drop NaN residuals (label not materialized yet for s in pool)
        mask = np.isfinite(ru) & np.isfinite(rl)
        ru = ru[mask]; rl = rl[mask]
        n_eligible = int(mask.sum())
        if n_eligible < min_pool:
            meta_rows.append(_empty_meta("insufficient_pool", n=n_eligible))
            continue
        # Split-conformal per-side deltas
        delta_u = float(np.quantile(ru, target_coverage))
        delta_l = float(np.quantile(rl, target_coverage))
        out_upper.iat[t] = pred_upper.iat[t] + delta_u
        out_lower.iat[t] = pred_lower.iat[t] - delta_l
        # Diagnostics: realized coverage on the pool with the chosen delta
        cov_u = float((ru <= delta_u).mean())
        cov_l = float((rl <= delta_l).mean())
        meta_rows.append(ScalerMeta(
            n_eligible=n_eligible,
            delta_upper=delta_u,
            delta_lower=delta_l,
            realized_coverage_upper=cov_u,
            realized_coverage_lower=cov_l,
            fallback_reason=None,
        ))

    meta_df = pd.DataFrame(
        [{
            "n_eligible": m.n_eligible,
            "delta_upper": m.delta_upper,
            "delta_lower": m.delta_lower,
            "realized_coverage_upper": m.realized_coverage_upper,
            "realized_coverage_lower": m.realized_coverage_lower,
            "fallback_reason": m.fallback_reason,
        } for m in meta_rows],
        index=dates,
    )
    return out_upper, out_lower, meta_df


def _empty_meta(reason: str, n: int = 0) -> ScalerMeta:
    return ScalerMeta(
        n_eligible=n,
        delta_upper=float("nan"),
        delta_lower=float("nan"),
        realized_coverage_upper=float("nan"),
        realized_coverage_lower=float("nan"),
        fallback_reason=reason,
    )
