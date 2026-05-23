# Round 0 Summary

## Status

Round 0 of a multi-round implementation of the GLD/SLV correctness + calibration plan (15 ACs, 24 tasks across Phases A–G). This round lands **Phase A correctness-floor parameter flips** (3 atomic patches) plus **pytest harness scaffold**, totalling 4 tasks. Phases B/C/D/E/F-body/G remain queued for subsequent rounds. **The full plan is NOT yet complete**; the loop should iterate.

## What Was Implemented

### v3.7.233 — RegimeClassifier no-lookahead (task-a1, AC-2)

- `core/regime.py:__init__` default `min_hold_days` flipped from `20` to `1`. The old 20-day debounce silently rewrote historical regime labels by looking forward (`_apply_min_hold` wrote `values[i:j] = current` whenever a new regime chunk was shorter than 20 days), introducing a structural look-ahead bias in any production consumer of `RegimeClassifier`.
- Belt-and-suspenders explicit `min_hold_days=1` added at 6 production-critical call sites: `app.py:112`, `scripts/build_positions_ledger.py:92`, `scripts/continuous_runner.py:81`, `scripts/backfill_intraday_signals.py:85`, `scripts/build_futures_signals.py:71`, `scripts/cross_asset_sync.py:48`.
- 24 research/grid/validation scripts left at the new default (1-day).
- `docs/regime_classifier_call_sites.md` archives the production allow-list, the research sites, and the synthetic invariant check.
- Validation: `RegimeClassifier().min_hold_days == 1` confirmed; truncated-vs-full-series equality holds on a 200-day GLD feature window.
- Pytest regression: `tests/test_regime_no_lookahead.py` — 2 PASS.
- Tag: `v3.7.233` (local).

### v3.7.234 — SELL PUT realized_pnl_pct sign correction (task-a2, AC-12)

- `core/paper_positions.py` spot-fallback `sign` logic rewritten: BUY CALL / SELL PUT / SPOT / FUTURES_LONG → `+1` (positive-delta strategies); STRADDLE / SHORT_VOL → `0` (non-directional); other → `-1`.
- Old logic placed SELL PUT in the `-1` bucket, contradicting the project's own `_strategy_pnl_formula` docstring (SP has +30% delta).
- Grep audit confirmed no downstream consumer compensated for the old sign; fix is structurally clean.
- The `price_strategy_at` change introducing `underlying_entry_price=spot_at_trigger` was inadvertently bundled into this commit; it logically belongs to v3.7.235 but is its prerequisite, so the bundling does not create a release-ordering hazard. Documented for transparency.
- Tag: `v3.7.234` (local).

### v3.7.235 — `entry_spot` schema migration (task-a3, AC-13)

- `core/positions_ledger.py:LEDGER_COLS` gains `underlying_entry_price` (semantically correct); the legacy `entry_spot` is retained as a one-release alias scheduled for removal in v3.8.
- `core/positions_ledger.py:add_entry` writes both fields; `entry_spot` falls back to the old option-close source only when the new field is missing.
- `/Users/yhdong/Gold/data/positions_ledger_meta.json` augmented with a `schema_migrations` list and a v3.7.235 entry including `entry_spot_alias_until_version: v3.8`.
- Tag: `v3.7.235` (local).

### v3.7.236-prep — pytest harness scaffold (task-f1, AC-10)

- `requirements.txt` adds `pytest>=7.0`.
- `tests/conftest.py` adds two session fixtures (`fixture_dir`, `fixed_today=2026-05-22`) to keep tests offline + deterministic.
- `tests/fixtures/.gitkeep` reserves the offline fixture directory.
- `tests/test_regime_no_lookahead.py` is the first regression test (2 cases, both PASS).
- Verified: `conda run -n gold python -m pytest tests/ -v` → 2 PASS in 0.02s.
- The tag is prefixed `-prep` because the v3.7.236 feature itself (max_fallback_days + freshness gate, AC-6) ships in a later round; this commit delivers only the underlying test infrastructure.

## Files Changed

### Modified
- `core/regime.py`
- `core/paper_positions.py`
- `core/positions_ledger.py`
- `app.py`
- `scripts/build_positions_ledger.py`
- `scripts/continuous_runner.py`
- `scripts/backfill_intraday_signals.py`
- `scripts/build_futures_signals.py`
- `scripts/cross_asset_sync.py`
- `requirements.txt`
- `/Users/yhdong/Gold/data/positions_ledger_meta.json`

### Created
- `docs/regime_classifier_call_sites.md`
- `tests/conftest.py`
- `tests/fixtures/.gitkeep`
- `tests/test_regime_no_lookahead.py`

## Validation

- pytest: `conda run -n gold python -m pytest tests/ -v` → **2 passed in 0.02s**.
- Runtime check: `RegimeClassifier().min_hold_days == 1` ✓; truncated-prefix invariant holds.
- Grep audit: no downstream consumer of `realized_pnl_pct` outside `core/paper_positions.py` itself.

## Remaining Items (Queued For Future Rounds)

| Phase | Tasks | ACs |
|---|---|---|
| B | task-b1, task-b2 | AC-6 |
| C | task-c1, task-c2 | AC-3, AC-4 |
| D | task-d1, task-d2 | AC-7, AC-15 |
| E | task-e1, task-e2, task-e3 | AC-5, AC-14 |
| F (test bodies) | task-f2, task-f3, task-f4 | AC-4, AC-3, AC-8, AC-9 |
| G | task-g1..g6 | AC-1, AC-8, AC-9 |
| Closure | task-h1, task-h2 | AC-10, AC-11 |

11 of 15 ACs still need work. **The plan is not complete.** Codex review should identify the remaining work and prompt for Round 1.

## AC Status

| AC | Status |
|---|---|
| AC-1 | not started |
| AC-2 | LANDED (v3.7.233) |
| AC-3 | not started |
| AC-4 | not started |
| AC-5 | not started |
| AC-6 | not started |
| AC-7 | not started |
| AC-8 | not started |
| AC-9 | not started |
| AC-10 | partial — pytest harness scaffolded; per-AC tests deferred |
| AC-11 | partial — no plan markers in v3.7.233-235 code; final audit deferred |
| AC-12 | LANDED (v3.7.234) |
| AC-13 | LANDED (v3.7.235) |
| AC-14 | not started |
| AC-15 | not started |

## Risks Observed

- v3.7.234 tag boundary smudged into v3.7.235's `underlying_entry_price` prerequisite — functionally clean but worth tracking; future rounds will use `git add -p` for crisper boundaries.
- 24 research scripts silently shifted to `min_hold_days=1`; documented but not individually re-validated. If any published reference number relied on the old 20-day behavior, it will need explicit `min_hold_days=20` annotation in that script.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: This round did not encounter problems requiring multi-round iteration; all four tasks landed on first attempt with passing validation. Empty bitlesson knowledge base is appropriate at this stage.
