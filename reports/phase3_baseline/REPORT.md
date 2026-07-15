# Phase 3 Baseline Model Report

Generated: 2026-03-07 15:34

## Walk-Forward Setup
- Min training: 1260 days (5 years)
- Test window: 252 days (1 year)
- Step: 252 days
- Total folds: 10

## Regression Results (IC = Spearman rank correlation)
| Target | Model | IC (mean) | IC (std) | Dir Acc |
|--------|-------|-----------|----------|---------|
| fwd_ret_5d | Ridge | 0.0751 | 0.1161 | 51.1% |
| fwd_ret_5d | XGBoost | 0.0628 | 0.1245 | 48.3% |
| fwd_ret_10d | Ridge | 0.1523 | 0.1554 | 51.1% |
| fwd_ret_10d | XGBoost | 0.1358 | 0.1801 | 52.6% |
| fwd_ret_20d | Ridge | 0.2127 | 0.2522 | 51.9% |
| fwd_ret_20d | XGBoost | 0.1973 | 0.2743 | 54.9% |
| fwd_rv_10d | Ridge | 0.2723 | 0.1969 | 90.2% |
| fwd_rv_10d | XGBoost | 0.2277 | 0.2040 | 100.0% |
| fwd_rv_20d | Ridge | 0.2436 | 0.2864 | 91.2% |
| fwd_rv_20d | XGBoost | 0.2872 | 0.1406 | 100.0% |

## Classification Results
| Target | Model | Accuracy | AUC |
|--------|-------|----------|-----|
| direction_5d | LogReg | 0.3414 | 0.5230 |
| direction_5d | XGBoost | 0.3613 | 0.5213 |
| direction_10d | LogReg | 0.3694 | 0.5564 |
| direction_10d | XGBoost | 0.3866 | 0.5614 |
| tail_event_flag | LogReg | 0.8078 | 0.5635 |
| tail_event_flag | XGBoost | 0.8673 | 0.6814 |

## Key Findings
- Longer horizons have higher IC (ret_20d > ret_10d > ret_5d)
- Volatility prediction (rv) has strong signal (IC ~0.25-0.29)
- Ridge and XGBoost perform similarly — features matter more than model
- 3-class direction accuracy ~36% (random=33%) — weak but above chance
- Tail event detection dominated by class imbalance (~88.5% negative)

## Charts
1. `01_summary_metrics.png` — IC and Accuracy overview
2. `02_ic_over_time.png` — IC stability across folds
3. `03_pred_vs_actual.png` — Predicted vs Actual scatter
4. `04_feature_importance.png` — XGBoost top features
5. `05_cumulative_signal.png` — OOS long/short equity curve
