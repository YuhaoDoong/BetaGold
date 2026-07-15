[P0]  
无。

[P1]  
1. `P1#3` 没有真正做到 retry idempotent。`PENDING_KLINE_DATES` 会把水位线 clamp 到 earliest pending 前一交易日，但写 ledger 时只是 `refreshed + new_rows` 后直接排序写出，没有按 `(asset, signal_date, strategy)` 去重或 upsert。场景：第一次 run 在日期 D 遇到 `PENDING_KLINE`，但 D+1/D+2 已生成 rows 并落盘；水位线被 clamp 到 D-1；第二次 run 会从 D 重新生成 D+1/D+2，旧 rows 仍在 `refreshed`，新 rows 追加进去，ledger 出现重复 key。  
证据：[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:147) 的 skip 依赖水位线，[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:642) 直接合并写出，[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:660) 做 clamp，但没有 dedup。注释声称 idempotent，但实现不成立。

[P2]  
1. `AWAITING_EXPIRY_CLOSE` helper 行为本身对了，但 ledger caller 没保留 state/reason，open row 会落成 `current_value=0`, `pnl_pct=0`, `hold_days=0` 这类静默默认值。到期日 exact close 未到时，dashboard/ledger 不容易区分“等待 expiry close”与普通 open/无价。  
证据：helper 返回 state 在 [core/strategies/options_exit.py](/Users/yhdong/GoldDash/core/strategies/options_exit.py:301)，但 row 组装只读 `is_closed/current_value/pnl_pct/hold_days` 默认值：[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:309)；refresh 也同样丢 state：[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:425)。

[P3]  
1. 新增测试覆盖不完整。`expiry_intrinsic` 覆盖了 helper 的 today==expiry 和 missing exact close；`calibration_retrain` 覆盖了 `pred_widths` canonical path；`cross_asset_selector` 覆盖了 future-asof invalid。但没有测试 `P1#3` 的 ledger clamp + retry + key 去重 invariant，也没有测试 `GOLD_CROSS_LIVE_CUTOVER=true` 在 `build_positions_ledger.py` 的 env preflight 集成路径。  
2. v3.8 backlog 没完整记录用户列出的 defer 项。`P2#1`、`P3#3`、`P4#2` 有记录；`P2#3 build_band preflight` 只被比较含糊地写成 “build_band reads calibrated columns under config flag”；`P3#2 freshness 集成` 未看到明确 backlog item。见 [docs/BACKLOG_v3.8.md](/Users/yhdong/GoldDash/docs/BACKLOG_v3.8.md:32) 和 [docs/BACKLOG_v3.8.md](/Users/yhdong/GoldDash/docs/BACKLOG_v3.8.md:62)。

[P4]  
无新增阻断项。

[NOTES]  
目标新增测试通过：`51 passed in 0.06s`。  
按项目测试目录跑通过：`pytest tests -q` 得到 `133 passed in 0.51s`。  
裸跑 `pytest -q` 会额外收集 `scripts/bc_entry_filter_test.py` 并因缺 fixture `bc` 报错；这不是本 diff 引入，但会影响全仓 pytest 口径。

REVIEW_VERDICT: NEEDS_REWORK
