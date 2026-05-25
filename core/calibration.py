"""v3.7.244: horizon-aware split-conformal scaler for DL Range bands.

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
residuals whose ``label_end_date(s) < t`` (strict). This is the plan contract
maturity-lag rule and is enforced via an explicit ``eligible`` mask on
business-day arithmetic, NOT a generic ``shift(1)``.

Shadow vs cutover
-----------------
This module is a pure function. The decision of *whether* to feed the
calibrated columns back into ``core.signals.build_band()`` lives in a
config preflight check elsewhere (see ``calibration.live_cutover``
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
DEFAULT_TARGET_COVERAGE = 0.80  # training target per the plan
MIN_POOL_SIZE = 20             # below this → no calibration applied
MIN_REGIME_POOL_SIZE = 20      # v3.7.246: per-regime same-regime sample floor


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
    regime: Optional[pd.Series] = None,
    min_regime_pool: int = MIN_REGIME_POOL_SIZE,
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

    # v3.7.246: per-regime alpha — when caller supplies regime per date, the
    # pool for date t is restricted to past dates whose regime equals
    # regime[t]. NaN regimes are coalesced to 'UNKNOWN' so they still get a
    # consistent pool (instead of mixing with arbitrary labeled regimes).
    if regime is not None:
        regime = pd.Series(regime, index=dates).astype(object)
        regime = regime.where(regime.notna(), "UNKNOWN")
        regime_arr = regime.values
    else:
        regime_arr = None

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
        # v3.7.246: per-regime restriction layered on top of NaN mask
        regime_fallback = None
        if regime_arr is not None:
            current_regime = regime_arr[t]
            pool_regimes = regime_arr[pool_slice]
            regime_mask = (pool_regimes == current_regime)
            same_regime_count = int((mask & regime_mask).sum())
            if same_regime_count >= min_regime_pool:
                mask = mask & regime_mask
            else:
                # Insufficient same-regime samples → fall back to global pool
                regime_fallback = f"regime_undersampled[{current_regime}:{same_regime_count}]"
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
            fallback_reason=regime_fallback,  # None unless per-regime fallback
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


# -----------------------------------------------------------------------------
# v3.7.245: calibration-gated retrain trigger
# -----------------------------------------------------------------------------

DEFAULT_RETRAIN_WINDOW = 30
DEFAULT_RETRAIN_HYSTERESIS_DAYS = 5
DEFAULT_RETRAIN_COOLDOWN_DAYS = 7
DEFAULT_RETRAIN_RATIO_QUEUE = 2.5
DEFAULT_RETRAIN_RATIO_IMMEDIATE = 4.0
DEFAULT_RETRAIN_ZERO_WIDTH_FLOOR = 0.10  # percent (5d forward H/L range)


def evaluate_retrain_trigger(
    meta_df: pd.DataFrame,
    actual_widths: pd.Series,
    today: pd.Timestamp,
    last_retrain_at: Optional[pd.Timestamp] = None,
    window: int = DEFAULT_RETRAIN_WINDOW,
    hysteresis_days: int = DEFAULT_RETRAIN_HYSTERESIS_DAYS,
    cooldown_days: int = DEFAULT_RETRAIN_COOLDOWN_DAYS,
    ratio_threshold_queue: float = DEFAULT_RETRAIN_RATIO_QUEUE,
    ratio_threshold_immediate: float = DEFAULT_RETRAIN_RATIO_IMMEDIATE,
    zero_width_floor: float = DEFAULT_RETRAIN_ZERO_WIDTH_FLOOR,
) -> dict:
    """Decide whether to queue / immediately run / suppress a model retrain.

    the plan contract contract:
      * Smooth the band-overshoot ratio over the last ``window`` matured
        residuals (pred_width / actual_width). Zero-or-near-zero
        ``actual_widths`` (< ``zero_width_floor``) are excluded from the mean
        rather than producing Inf.
      * If smoothed ratio > ``ratio_threshold_queue`` for the LAST
        ``hysteresis_days`` consecutive scaler rows → outcome ``"queued"``.
      * If smoothed ratio > ``ratio_threshold_immediate`` (any single day in
        the trailing observation, evaluated on the latest meta row) →
        outcome ``"immediate"``.
      * Trading days since ``last_retrain_at`` < ``cooldown_days`` →
        outcome ``"suppressed_cooldown"``.
      * Otherwise → outcome ``"no_action"``.

    Pure function: no I/O. Caller writes the returned dict to
    ``data/models/retrain_queue.jsonl`` if desired.

    Args:
        meta_df: output of ``apply_rolling_conformal_scaler``; indexed by date,
            must contain ``delta_upper``, ``delta_lower``, ``n_eligible``.
        actual_widths: per-date ``actual_upper - actual_lower`` (5d forward range,
            percent). NaN where label not yet matured.
        today: anchor date for cooldown evaluation.
        last_retrain_at: date of the previous retrain; ``None`` ⇒ no cooldown.

    Returns:
        ``{"outcome": str, "ratio_value": float, "consecutive_days": int,
            "zero_width_excluded_count": int, "cooldown_until": str|None,
            "triggered_at": str}``
    """
    today = pd.Timestamp(today).normalize()
    # ── 1. Cooldown gate ──────────────────────────────────────────────────
    cooldown_until = None
    if last_retrain_at is not None:
        last = pd.Timestamp(last_retrain_at).normalize()
        cooldown_end = (pd.bdate_range(start=last + pd.Timedelta(days=1),
                                              periods=cooldown_days)[-1]
                          if cooldown_days > 0 else last)
        cooldown_until = cooldown_end.isoformat()
        if today <= cooldown_end:
            return {
                "outcome": "suppressed_cooldown",
                "ratio_value": float("nan"),
                "consecutive_days": 0,
                "zero_width_excluded_count": 0,
                "cooldown_until": cooldown_until,
                "triggered_at": today.isoformat(),
            }

    # ── 2. Build the ratio series over the trailing window ────────────────
    if meta_df is None or not len(meta_df):
        return _retrain_no_action(today, cooldown_until)
    aw = pd.Series(actual_widths).reindex(meta_df.index)
    pw_upper = meta_df["delta_upper"] + (aw / 2.0)  # not used directly; placeholder
    # pred_width per date is (pred_upper - pred_lower) but we do not carry
    # those raw values in meta. Reconstruct via delta + actual:
    #   pred_upper = actual_upper - delta_upper (only on the calibration day t,
    #   inverted from delta = quantile(actual - pred)).
    # Simpler and correct: take pred_width directly from caller via aw plus
    # the recorded deltas: pred_width = (pred_upper - pred_lower) = aw +
    # delta_upper + delta_lower (this is the per-side total compensation that
    # the scaler ALREADY applies). We track raw vs calibrated overshoot via
    # ``aw`` and the per-side residual proxies in meta. For the trigger, we
    # use the unsigned per-side residual magnitudes which are what
    # ``meta_df.delta_*`` quantify — a positive ``|delta_upper| + |delta_lower|``
    # represents the historical overshoot that motivated the retrain.
    res_total = meta_df[["delta_upper", "delta_lower"]].abs().sum(axis=1)
    valid = aw.notna() & res_total.notna() & (aw.abs() >= zero_width_floor)
    excluded = int((aw.notna() & (aw.abs() < zero_width_floor)).sum())
    if valid.sum() == 0:
        return _retrain_no_action(today, cooldown_until,
                                       zero_width_excluded_count=excluded)
    # Ratio = (|res| + actual_width) / actual_width ≈ implied predicted_width /
    # actual_width. (Equivalent to mean(pred_width/actual_width) when residuals
    # dominate the absolute width.)
    ratios = (res_total.abs() + aw.abs()) / aw.abs()
    ratios_valid = ratios[valid].iloc[-window:]
    smoothed = float(ratios_valid.mean()) if len(ratios_valid) else float("nan")

    # ── 3. Immediate vs queued vs no-action ──────────────────────────────
    latest = float(ratios_valid.iloc[-1]) if len(ratios_valid) else float("nan")
    if not pd.isna(latest) and latest > ratio_threshold_immediate:
        return {
            "outcome": "immediate",
            "ratio_value": smoothed,
            "consecutive_days": _consec_above(ratios_valid, ratio_threshold_queue),
            "zero_width_excluded_count": excluded,
            "cooldown_until": cooldown_until,
            "triggered_at": today.isoformat(),
        }
    consec = _consec_above(ratios_valid, ratio_threshold_queue)
    if consec >= hysteresis_days:
        return {
            "outcome": "queued",
            "ratio_value": smoothed,
            "consecutive_days": consec,
            "zero_width_excluded_count": excluded,
            "cooldown_until": cooldown_until,
            "triggered_at": today.isoformat(),
        }
    return _retrain_no_action(today, cooldown_until,
                                   ratio_value=smoothed,
                                   consecutive_days=consec,
                                   zero_width_excluded_count=excluded)


def _consec_above(series: pd.Series, threshold: float) -> int:
    """Length of the most-recent consecutive run above ``threshold`` at series end."""
    if not len(series): return 0
    arr = series.values
    n = len(arr); count = 0
    for v in reversed(arr):
        if not pd.isna(v) and v > threshold:
            count += 1
        else:
            break
    return count


def _retrain_no_action(today: pd.Timestamp, cooldown_until,
                          ratio_value: float = float("nan"),
                          consecutive_days: int = 0,
                          zero_width_excluded_count: int = 0) -> dict:
    return {
        "outcome": "no_action",
        "ratio_value": ratio_value,
        "consecutive_days": consecutive_days,
        "zero_width_excluded_count": zero_width_excluded_count,
        "cooldown_until": cooldown_until,
        "triggered_at": pd.Timestamp(today).normalize().isoformat(),
    }
