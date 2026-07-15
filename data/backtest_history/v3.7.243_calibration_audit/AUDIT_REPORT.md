# DL Range OOS Calibration Audit — task-g2 (AC-1)

Run anchor: v3.7.243. Audit window: 2025-12-01 → 2026-05-13 (113 trading days for GLD).

## Label definition (authoritative)

Per `src/models/train_dl_range.py:build_targets`:

```python
max_high_5d = high.shift(-1).rolling(5).max().shift(-4)   # 5d forward (t+1..t+5)
min_low_5d  = low.shift(-1).rolling(5).min().shift(-4)
upper_pct   = (max_high_5d / close - 1) * 100
lower_pct   = (min_low_5d  / close - 1) * 100
```

`actual_upper_pct` and `actual_lower_pct` in the OOS parquet are the **5-day forward max-high / min-low vs t-day close**, NOT single-day overnight returns.

## Correction vs the v3.7.232 idea-draft figures

| Quantity | Draft (single-day overnight) | Corrected (5d forward, parquet) | Source of error |
|---|---:|---:|---|
| Predicted band mean (113d) | "5-6× wider" | upper +5.99% / lower -4.76% | draft used `H/L vs prior Close` — single-day |
| Realized band mean | "-0.84% / +0.90%" | upper +3.08% / lower -2.86% | same — single-day H/L is ~1/5 of 5d forward |
| Width ratio upper | "5-6×" | **1.95×** | over-stated by ~3× |
| Width ratio lower | (not separated) | **1.66×** | |
| In-band coverage | "87.6%" | **54.87%** | over-stated by ~33 pp |
| Coverage target (training) | 80% | 80% | unchanged |

The correct read of the OOS data is **less alarming on band width but more alarming on coverage**: bands are ~2× too wide, but coverage is ~25 pp below the 80% training target, meaning the model misses tail moves on both sides.

## GLD Aggregate (2025-12-01 → 2026-05-13, n=113)

- Predicted band: **[-4.762%, +5.994%]**
- Realized band: **[-2.863%, +3.076%]**
- Width ratio upper: **1.948×**, lower: **1.663×**
- Coverage upper: **82.30%**, lower: **70.80%**, both sides: **54.87%**

**Coverage upper is close to target (82% vs 80%) but lower is 9pp short and both-sides coverage is dragged below 55% by the upper-or-lower joint event.**

## GLD per-month break-out

| Month | n | width_ratio_upper | width_ratio_lower | coverage_upper | coverage_lower | coverage_both |
|---|---:|---:|---:|---:|---:|---:|
| 2025-12 | 22 | 1.23 | **3.26** | 0.591 | 0.955 | 0.546 |
| 2026-01 | 20 | **0.77** | 2.12 | 0.500 | 0.800 | 0.400 |
| 2026-02 | 19 | 1.98 | 2.11 | 0.947 | 0.632 | 0.579 |
| 2026-03 | 22 | **3.53** | **0.73** | 1.000 | 0.455 | 0.455 |
| 2026-04 | 21 | 4.09 | 1.89 | 1.000 | 0.810 | 0.810 |
| 2026-05 | 9 | 4.23 | **0.85** | 1.000 | 0.444 | 0.444 |

**Asymmetric drift signature**:
- 2026-01: upper ratio 0.77 means the model **under-predicted** the upper bound (actual upper > predicted) — coverage_upper 50%.
- 2026-03 and 2026-05: lower ratio < 1 means the model under-predicted the lower bound — coverage_lower 45% in March.
- 2026-03 simultaneously has upper ratio 3.53 (massively over-predicts the upper side).

This is exactly the pattern that the original idea-draft's "narrow the bands" framing would worsen. The Phase G conformal scaler (task-g3) must shift, not just shrink — DEC-5's coverage-repair-toward-target framing is correct.

## SLV Aggregate (2025-12-01 → 2026-05-13, n=82)

- Predicted band: **[-12.532%, +12.940%]**
- Realized band: **[-6.697%, +9.078%]**
- Width ratio upper: **1.425×**, lower: **1.871×**
- Coverage upper: **67.07%**, lower: **84.15%**, both sides: **53.66%**

SLV under-covers on the **upper** side (67%, 13pp below target). 2025-12 and 2026-01 stand out with `coverage_upper` of 36% and 45% respectively.

Note: 2026-04 and 2026-05 produce `n=0` because the 5-day forward labels for SLV are not yet materialized for the most recent month-and-a-half (`actual_*_pct` is NaN until `t+5` business days have elapsed AND the OOS extension job has populated the cell). This is a **data-pipeline gap**, not a model gap.

## SLV per-month break-out

| Month | n | width_ratio_upper | width_ratio_lower | coverage_upper | coverage_lower | coverage_both |
|---|---:|---:|---:|---:|---:|---:|
| 2025-12 | 22 | **0.92** | **4.76** | 0.364 | 1.000 | 0.364 |
| 2026-01 | 20 | 0.85 | 1.72 | 0.450 | 0.750 | 0.300 |
| 2026-02 | 19 | 2.12 | 1.94 | 0.895 | 0.947 | 0.842 |
| 2026-03 | 21 | **3.09** | 1.22 | 1.000 | 0.667 | 0.667 |
| 2026-04 | 0 | — | — | — | — | — |
| 2026-05 | 0 | — | — | — | — | — |

SLV's 2025-12 is the inverse of GLD's 2026-03: upper ratio 0.92 (model under-predicts upper) AND lower ratio 4.76 (massively over-predicts lower). Coverage_upper 36% means the model missed the upside tail.

## Implications for Phase G (task-g3..g6)

1. **The scaler must support asymmetric, signed adjustments per side.** Symmetric scaling (`× s_global`) would degrade either upper or lower coverage in the months above.
2. **Per-regime conformal alpha (task-g5)** is justified empirically: the failure mode in 2025-12 SLV (over-predicted lower) and 2026-03 GLD (over-predicted upper) likely correspond to different regimes.
3. **Coverage target reframing:** AC-8 + DEC-5 already capture this — the gate criterion is coverage moves *toward* target, NOT band narrowing for its own sake.
4. **Retrain trigger (task-g4)** should monitor the per-side width ratio AND coverage delta, not just one number. The aggregate 1.95×/1.66× ratio + 54.87% coverage is a meaningful retrain signal, but the per-month breakdown shows the cause varies by month.
5. **Data-pipeline gap (SLV 2026-04/05)**: `extend_oos_predictions` writes `pred_*_pct` for new dates but `actual_*_pct` only materializes after the 5-day forward window completes AND the live inference job populates the row. This is an existing lag we should accept; the calibration audit naturally skips these rows (NaN-aware).

## Cross-check vs the draft's numbers

The draft's "5-6× wider, 87.6% coverage" was derived by recomputing realized values as single-day overnight returns:

```python
my_high = (ohlc['High'] / ohlc['Close'].shift(1) - 1) * 100   # WRONG — 1d range
my_low  = (ohlc['Low']  / ohlc['Close'].shift(1) - 1) * 100
```

Mean of those is ~±0.9% — about 1/5 of the 5-day forward realized mean (±3%). The "predicted band" was correctly read from the parquet at ±5%, so the ratio became 5/0.9 ≈ 5.5× instead of the real 5/3 ≈ 1.7×.

The high in-band coverage (87.6%) had the same cause: a band of ±5% trivially contains a single-day overnight move of ±0.9% almost always; against the 5-day forward range of ±3%, the same band only contains the joint event 55% of the time.

This is documented as the v3.7.232 NON-NORMATIVE block in the idea-draft appendix. AC-1's locked label definition and the `ValueError` on missing `actual_*_pct` columns prevent this misread from recurring.

## Artifacts

- `GLD_2025-12-01_2026-05-13.csv` — per-month + ALL aggregate
- `GLD_2025-12-01_2026-05-13.md` — markdown view
- `SLV_2025-12-01_2026-05-13.csv` / `.md`
- Code: `scripts/eval/model_calibration_audit.py`
- Tests: `tests/test_calibration_audit.py` (8 cases, all PASS)
