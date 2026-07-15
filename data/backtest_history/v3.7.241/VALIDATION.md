# v3.7.241 VALIDATION

Tests: `tests/test_max_move_window.py`

```
============================= test session starts ==============================
platform darwin -- Python 3.11.14, pytest-9.0.3, pluggy-1.6.0 -- /Users/yhdong/miniconda3/envs/gold/bin/python
cachedir: .pytest_cache
rootdir: <REPO>
collecting ... collected 7 items

tests/test_max_move_window.py::test_forward_max_window_known_values PASSED [ 14%]
tests/test_max_move_window.py::test_forward_min_window_known_values PASSED [ 28%]
tests/test_max_move_window.py::test_insufficient_future_window_returns_nan PASSED [ 42%]
tests/test_max_move_window.py::test_legacy_offbyone_vs_corrected PASSED  [ 57%]
tests/test_max_move_window.py::test_op_invalid_raises PASSED             [ 71%]
tests/test_max_move_window.py::test_nan_in_window_skipped PASSED         [ 85%]
tests/test_max_move_window.py::test_anchor_offset_zero_means_inclusive_today PASSED [100%]

============================== 7 passed in <NORMALIZED>s ===============================
```
