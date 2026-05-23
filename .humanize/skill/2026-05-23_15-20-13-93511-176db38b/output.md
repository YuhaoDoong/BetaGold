AGREE:
- AC-1 / AC-8 / AC-9 now use the correct 5-day forward label semantics, horizon-aware maturity lag, and coverage-repair framing.
- AC-5 normative acceptance criteria now correctly separate pure selector from caller-owned shadow logging and evaluate GVZ freshness relative to `signal_date`.
- AC-4, AC-14, AC-15 are materially fixed: 16 expiry scenarios, IC asymmetric-wing guard, intraday exit parity with signal drift attribution, and explicit `max_move` off-by-one AC.
- OI / risk controls / yfinance bid-ask / exposure caps are now explicitly out of scope, which keeps the plan focused.

DISAGREE:
- No conceptual disagreement with the AC-level design. I do not mark this PLAN_READY because there are still stale implementation-guidance conflicts that can mislead execution.

REQUIRED_CHANGES:
1. Update `task-e1` signature at line 276. It still says `select_gld_sync_strategy(date, gld_signal_row, gvz_value)`, omitting required `gvz_asof_date`. It must match AC-5: `select_gld_sync_strategy(signal_date, gld_signal_row, gvz_value, gvz_asof_date)`.
2. Remove stale exact signal-column parity guidance at line 202. It still says the wrapper asserts identical `buy_signal` / `signal_tier` / `exit_signal` series. That conflicts with AC-14, where signal columns may drift if explained by `signal_drift_attribution.csv`. Update this to intraday StopLoss/Pullback/ACTIVE parity only, with signal drift audited.

OPTIONAL_IMPROVEMENTS:
- In DEC-4, adjust the Claude Position wording to avoid repeating exact signal-column parity. Since DEC-4 is a user decision section, this is not blocking if AC-14 remains authoritative, but cleanup would reduce ambiguity.
- AC-5 live cutover evidence could require both `first_record_at` and recent/continuous dual-branch records; `first_record_at ≥14 days prior` alone proves age, not accumulation quality.

UNRESOLVED:
- DEC-1 through DEC-7 remain valid user decisions.
- No remaining reviewer-vs-plan design opposition after the two required text fixes above.
