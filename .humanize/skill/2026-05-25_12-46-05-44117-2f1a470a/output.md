[P0]  
None.

[P1]  
None. The retry dedup fix is structurally correct: `refreshed + new_rows` then `drop_duplicates(["asset", "signal_date", "strategy"], keep="first")` preserves refreshed frozen rows.

[P2]  
`AWAITING_EXPIRY_CLOSE` is still lost on normal refresh. New option rows copy `sim.get("state")` at [scripts/build_positions_ledger.py:304](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:304), but existing open rows call `simulate_option_exit()` at [scripts/build_positions_ledger.py:427](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:427) and update fields through [scripts/build_positions_ledger.py:441](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:441) without writing `row["state"]`.

This is the common lifecycle: a position is opened before expiry, then later refresh reaches expiry while the exact ETF close is missing. I confirmed with a monkeypatch that `_refresh_open_position(...).get("state")` remains `None` even when `simulate_option_exit()` returns `{"state": "AWAITING_EXPIRY_CLOSE"}`.

Also, cross-asset new rows omit `state` after `simulate_option_exit()` at [scripts/build_positions_ledger.py:614](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:614).

Fix: after every option `simulate_option_exit()`, set/clear `state`, e.g. `row["state"] = sim.get("state")`, including refresh and cross-asset paths. Add a regression test for refresh producing `AWAITING_EXPIRY_CLOSE`, and ideally another for clearing stale state when the row later closes.

[P3]  
None blocking. The new retry tests lock the dedup/clamp invariants, but they are unit-level copies of the policy rather than end-to-end ledger-cycle tests. The v3.8 backlog correctly captures that integration gap.

[P4]  
None.

[NOTES]  
Verified actual diff `v3.7.250..v3.7.251`. `git diff --check` clean. Targeted tests passed: `22 passed`. `pytest tests -q` passed: `137 passed`. Plain `pytest -q` also collects `scripts/bc_entry_filter_test.py` and errors on a missing fixture after the 137 tests pass; that appears outside this patch.

REVIEW_VERDICT: APPROVE_WITH_FIXES
