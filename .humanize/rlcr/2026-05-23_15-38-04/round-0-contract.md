# Round 0 Contract

## Round Objective

Land **Phase A correctness-floor parameter flips** (3 atomic patches) plus **pytest infrastructure scaffold** (task-f1) so subsequent rounds can attach tests immediately. Phase A patches are chosen first because:
1. Each is a single-line or single-section change with deterministic behavior
2. Each is independently revertible (no cross-file state dependencies)
3. They unblock Phase B/C work by removing latent bugs that would otherwise contaminate downstream test fixtures

## In-Scope Tasks This Round

| Task | AC | Tag | Notes |
|------|----|----|-------|
| task-a1 | AC-2 | v3.7.233 | `core/regime.py` default `min_hold_days=20 → 1`; audit + patch every production call site; produce research-only allow-list |
| task-a2 | AC-12 | v3.7.234 | `core/paper_positions.py` SP fallback sign `-1 → +1`; grep audit consumers of `realized_pnl_pct` |
| task-a3 | AC-13 | v3.7.235 | `core/positions_ledger.py` add `underlying_entry_price` field; keep `entry_spot` alias; write migration note into `data/positions_ledger_meta.json` |
| task-f1 | AC-10 | v3.7.236-prep | Add `pytest` to `requirements.txt` (if missing); scaffold `tests/conftest.py` and `tests/fixtures/`; verify `pytest tests/` runs (collects 0 tests but exits 0) |

## Out-of-Scope This Round (Queued For Subsequent Rounds)

- task-b1, task-b2: Phase B defensive guards (next round)
- task-c1, task-c2: Phase C per-asset cfg + Straddle/ShortVol expiry (Round 2+)
- task-d1, task-d2: Phase D derived metrics (Round 2+)
- task-e1, task-e2, task-e3: Phase E cross-asset + Dashboard (Round 3+)
- task-f2, task-f3, task-f4: Phase F test bodies (each depends on its Phase B/C/G impl)
- task-g1 .. task-g6: Phase G calibration (Round 4+)
- task-h1, task-h2: closure tasks

## Verification Plan (Per Task)

Each Phase A task gets a focused validation step before commit:
- **task-a1**: Run `grep -rn "RegimeClassifier()" core/ scripts/ app.py` after patch; assert every production hit uses explicit `min_hold_days=1`; document research-only exceptions in `docs/regime_classifier_call_sites.md`.
- **task-a2**: Apply patch; construct an SP entry fixture with spot moving up 1%, compute `realized_pnl_pct` via spot fallback, assert positive sign (winner). Move spot down 1%, assert negative.
- **task-a3**: Patch `price_strategy_at` to return `underlying_entry_price`; `positions_ledger.py` writes both fields; meta file gains `entry_spot_alias_until_version` key. Verify a synthetic ledger row contains both fields.
- **task-f1**: Run `pytest tests/` after scaffold; assert exit code 0 even with no collected tests.

## Commit Discipline

Per DEC-3 (user-confirmed): each Phase A patch ships as a real `v3.7.*` git tag:
- `v3.7.233`: task-a1 (regime min_hold_days)
- `v3.7.234`: task-a2 (SP sign)
- `v3.7.235`: task-a3 (entry_spot migration)
- `v3.7.236-prep`: task-f1 (pytest scaffold; not a feature tag yet, prefixed `-prep` because Phase B = v3.7.236 hasn't shipped)

Tags are local-only this round per CLAUDE.md safety protocol; user authorizes push separately if desired.

## Bitlessons Consulted

- BitLesson knowledge base file is initialized but contains no entries (Round 0 is the first round in this lesson base).
- For each task this round, the relevant bitlesson selection is **NONE** (no precedent lessons apply).
- If novel issues are solved during Round 0, they will be added to `.humanize/bitlesson.md` per the strict template.

## Risk Assessment for Round 0 Scope

- **Asset-threading dependency**: Phase A intentionally avoids touching `simulate_*_position` signatures. Asset-threading risk is parked for task-c1.
- **Regime `min_hold_days=1` intraday jitter**: Confirmed in plan risk section; EOD-only mitigation already in place; will add 2-week shadow log comparing pre/post WR before declaring done.
- **SP sign cascading semantics**: Will audit `app.py`, `notifier.py`, `compute_strategy_stats.py` before committing task-a2. If any consumer compensates for the old `-1` sign, that consumer is fixed in the same commit.
- **entry_spot rename backward compat**: One-release alias preserves external reader (`notifier`, dashboard) compatibility; readers migrate in task-a3 same commit.
