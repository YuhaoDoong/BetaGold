# Phase 2 Feature Engineering Report

Generated: 2026-03-07 12:43

## Dataset Overview
- Features: **116** columns x **5356** rows
- Labels: **12** columns
- Date range: 2004-11-18 ~ 2026-03-05

## Missing Rates
- Features with 0% missing: 39
- Features with >15% missing: 10
- Average missing rate: 3.47%

## Collinearity
- Feature pairs with |r| > 0.9: **25**

## Labels
- fwd_ret_5d: mean=0.0025, std=0.0249, n=5351
- fwd_ret_10d: mean=0.0050, std=0.0346, n=5346
- fwd_ret_20d: mean=0.0100, std=0.0483, n=5336
- fwd_rv_10d: mean=0.1603, std=0.0839, n=5346
- fwd_rv_20d: mean=0.1636, std=0.0764, n=5336
- iv_rv_spread: mean=0.0238, std=0.0532, n=4120
- tail_event_flag: mean=0.1150, std=0.3191, n=5356
- max_dd_10d: mean=-0.0270, std=0.0214, n=5346
- direction_5d: flat=36.3%, up=36.2%, down=27.4%
- direction_10d: up=42.5%, down=31.1%, flat=26.4%
- magnitude_10d: small=48.1%, medium=38.8%, large=13.1%
- vol_regime: low_vol=36.1%, high_vol=34.6%, normal=29.3%

## Charts
1. `01_missing_rate.png` — Feature missing rates
2. `02_feature_label_corr.png` — Top feature-label correlations
3. `03_collinearity.png` — Highly correlated feature pairs
4. `04_label_distributions.png` — Label histograms & bar charts
5. `05_key_features_ts.png` — Key features vs gold price
6. `06_feature_categories.png` — Feature count by category
7. `07_feature_stability.png` — Rolling correlation stability
