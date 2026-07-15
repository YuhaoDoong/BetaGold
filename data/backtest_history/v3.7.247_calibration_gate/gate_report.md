# Calibration Gate Report — task-g6 (AC-8 closure)

Generated: 2026-05-25T03:19:32.114140+00:00

## Gate verdict

gate_passed: false

Gate criteria (AC-8 + DEC-5):
- coverage moves toward training target 0.8 in ≥ 3 windows
- no window degrades coverage by more than 5pp
- applied per-side (upper AND lower) AND on both-sides

## GLD  (gate_passed: False)

- coverage_both gate: False (1/5)
- coverage_upper gate: True (5/5)
- coverage_lower gate: True (5/5)

| window | n | raw_both | cal_both | raw_upper | cal_upper | raw_lower | cal_lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10y | 2520 | 0.752 | 0.533 | 0.873 | 0.761 | 0.879 | 0.759 |
| 5y | 1260 | 0.751 | 0.544 | 0.848 | 0.762 | 0.901 | 0.770 |
| 3y | 756 | 0.733 | 0.552 | 0.857 | 0.767 | 0.873 | 0.774 |
| 1y | 252 | 0.690 | 0.631 | 0.829 | 0.802 | 0.853 | 0.813 |
| 113d | 113 | 0.549 | 0.593 | 0.823 | 0.832 | 0.708 | 0.735 |

### Per-window pass detail (coverage_both)

- 10y: raw=0.752 cal=0.533 distance_delta=+0.219 **pass=False**
- 5y: raw=0.751 cal=0.544 distance_delta=+0.207 **pass=False**
- 3y: raw=0.733 cal=0.552 distance_delta=+0.181 **pass=False**
- 1y: raw=0.690 cal=0.631 distance_delta=+0.060 **pass=False**
- 113d: raw=0.549 cal=0.593 distance_delta=-0.044 **pass=True**

## SLV  (gate_passed: False)

- coverage_both gate: False (1/5)
- coverage_upper gate: True (5/5)
- coverage_lower gate: True (4/5)

| window | n | raw_both | cal_both | raw_upper | cal_upper | raw_lower | cal_lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10y | 2482 | 0.738 | 0.538 | 0.859 | 0.759 | 0.878 | 0.762 |
| 5y | 1222 | 0.706 | 0.544 | 0.827 | 0.763 | 0.876 | 0.766 |
| 3y | 718 | 0.721 | 0.536 | 0.829 | 0.758 | 0.890 | 0.769 |
| 1y | 214 | 0.598 | 0.444 | 0.692 | 0.673 | 0.897 | 0.748 |
| 113d | 75 | 0.533 | 0.573 | 0.680 | 0.827 | 0.827 | 0.720 |

### Per-window pass detail (coverage_both)

- 10y: raw=0.738 cal=0.538 distance_delta=+0.199 **pass=False**
- 5y: raw=0.706 cal=0.544 distance_delta=+0.162 **pass=False**
- 3y: raw=0.721 cal=0.536 distance_delta=+0.185 **pass=False**
- 1y: raw=0.598 cal=0.444 distance_delta=+0.154 **pass=False**
- 113d: raw=0.533 cal=0.573 distance_delta=-0.040 **pass=True**

## Next action

- `gate_passed: false` — keep `build_band()` on raw predictions. Ship the audit, scaler, retrain trigger, and per-regime alpha as **shadow-only diagnostics** until further per-regime or hyperparameter tuning lifts coverage uniformly.
- Open follow-ups: tune `target_coverage`, `window`, or the per-regime classifier itself; re-run this gate.