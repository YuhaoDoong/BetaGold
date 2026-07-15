[P0] None  
[P1] None  
[P2] None  
[P3] None  
[P4] None  

[NOTES]  
Reviewed `v3.7.251..v3.7.252`. The fix correctly propagates `sim["state"]` in both missing paths:

- refresh path: [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:446)
- cross-asset path: [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:643)

It also clears stale `AWAITING_EXPIRY_CLOSE` on later close because `sim.get("state", None)` writes `None` when the simulator closes normally.

Targeted verification run:

```text
pytest -q tests/test_ledger_pending_retry.py
6 passed in 0.01s
```

I did not rerun the full 139-test suite.

REVIEW_VERDICT: APPROVE
