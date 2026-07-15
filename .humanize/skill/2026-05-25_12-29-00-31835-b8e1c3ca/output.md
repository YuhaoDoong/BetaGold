[P0] 无

[P1] Expiry-day intrinsic close 未按 AC 执行
  - File: core/strategies/options_exit.py:273
  - Issue: `today <= expiry_dt` 直接返回 `None`，导致 `today_dt == expiry_dt` 即使 ETF expiry close 已存在也不会 intrinsic close。AC-4 明确要求 expiry 当日且 close 可用时关闭；测试文件文档也写了这个要求，但断言反而期望 `None`。
  - Fix: 改为 `today < expiry_dt` 才返回；`today == expiry_dt` 时要求 exact expiry close 存在才 close，否则返回 `state="AWAITING_EXPIRY_CLOSE"`。

[P1] Expiry intrinsic 可能用过期 spot close 静默结算
  - File: core/strategies/options_exit.py:241
  - Issue: `spot_close_on_or_before()` 会在 expiry close 缺失时取更早 close。AC-4 要求按 `expiry_dt` close；缺 expiry close 应等待或 fail-safe，不应使用 stale spot 结算最终 P&L。
  - Fix: force-close 路径改成 exact-date lookup；若 `expiry_dt` 不在 ETF daily CSV，返回 awaiting state，不生成 closed P&L。

[P1] PENDING_KLINE 信号会被永久跳过
  - File: scripts/build_positions_ledger.py:276
  - Issue: `price_strategy_at()` 返回 `PENDING_KLINE` 后这里只 `continue`，没有写 pending 记录；随后 `positions_ledger_meta.json` 在 `scripts/build_positions_ledger.py:630` 用 ETF 最新日期推进 `evaluated_through`。下次 kline 恢复后，该 signal_date 已被水位线跳过，AC-6 的“next refresh retry + dedup”不成立。
  - Fix: 不要让 pending 日期推进 `evaluated_through`，或写入持久 pending queue/ledger row，恢复后按 `(asset, signal_date, strategy)` 幂等补建。

[P1] Retrain ratio 不是 AC-9 的 `pred_width / actual_width`
  - File: core/calibration.py:318
  - Issue: `evaluate_retrain_trigger()` 没有输入 raw `pred_width`，改用 `abs(delta_upper)+abs(delta_lower)` 推导 ratio。该值不是 `pred_width / actual_width`：例如模型严重 under-wide 时 raw ratio 应 <1，但这里会因 residual 大而触发 immediate retrain。
  - Fix: 函数签名接收 `pred_widths` 或 `pred_upper/pred_lower`，按 matured trailing window 直接计算 `mean(pred_width / actual_width)`，zero-width 过滤保持不变。

[P1] Cross-asset live-cutover gate 没有接入生产路径
  - File: scripts/build_positions_ledger.py:520
  - Issue: `CROSS_LIVE_CUTOVER = False` 是局部硬编码；`live_cutover_allowed()` 从未被生产调用，也没有配置入口。未来手动改 True 时不会验证 14 天 shadow accumulation gate，AC-5 的 flip gate 实际不可执行。
  - Fix: 把 `shadow_logging/live_cutover` 放到配置；启动或 ledger build 时若 `live_cutover=True` 必须调用 `live_cutover_allowed()`，失败则 fail-loud 并保持 raw/default behavior。

[P2] per-asset cfg 失败时静默回默认
  - File: core/paper_positions.py:756
  - Issue: 传了 `asset` 后，`get_option_exit_config()` 或策略 sim 异常都被吞掉并回到默认 cfg/旧 inline 逻辑。这样 per-asset threading 失效不会暴露，和 AC-3 的“不能 silent fallback”精神冲突。
  - Fix: 对 config resolver 异常 fail-loud；策略 sim 异常至少记录 structured warning/error，不要无条件 fallback。

[P2] Cross-asset selector 接受未来 GVZ asof
  - File: core/cross_asset_signal.py:96
  - Issue: `gvz_asof_date > signal_date` 被当作 fresh。生产 caller 当前用 `<= d` 规避了，但纯函数 contract 是“evaluated against signal_date”，未来 asof 应视为 invalid/future leak。
  - Fix: `asof > sig_d` 返回 `GVZ_UNAVAILABLE` 或 raise，测试覆盖未来 asof。

[P2] Calibration cutover/preflight 只写在注释里
  - File: core/signals.py:17
  - Issue: `build_band()` 完全不识别 calibrated columns 或 `calibration.live_cutover`；`scripts/eval/calibration_gate_grid.py:18` 还写明 preflight out of scope。AC-8 要求 shadow columns + gated live cutover 机制，当前只有 scaler 和 gate report。
  - Fix: 增加 config flag + startup preflight；只有 gate report `gate_passed: true` 时 `build_band()` 才读 calibrated columns，否则 raw。

[P3] Expiry tests 与 AC 自相矛盾
  - File: tests/test_expiry_intrinsic.py:84
  - Issue: docstring 写 “today == expiry with spot close available force-close”，断言却是 `res is None`。这直接掩盖了 AC-4 expiry-day bug。
  - Fix: 拆成 close-known 和 close-missing fixture；known close 断言 closed，missing close 断言 `AWAITING_EXPIRY_CLOSE`。

[P3] Freshness 测试没有覆盖真实 retry/waterline
  - File: tests/test_data_freshness.py:66
  - Issue: 测试只检查 state policy，没有驱动 `build_positions_ledger.py` 的 `PENDING_KLINE -> evaluated_through` 行为，所以漏掉永久跳过信号的问题。
  - Fix: 加集成级 fixture：第一次 stale 产生 pending 且不推进该日期，第二次 fresh 后补建且不重复。

[P3] Dashboard parity 没有真实跑 legacy vs unified
  - File: tests/test_dashboard_parity.py:39
  - Issue: 主要测试把 `_replay_one_pass()` 同时当 legacy/unified 输入，验证的是 helper 自身一致性，不是 `core.signals_v2.run_backtest` 与 `run_unified_backtest` 的历史 60 日 parity。
  - Fix: 用真实/fixture OHLC、band、regime、rv_pctile 同时跑 legacy `run_backtest()` 和 unified path，并断言 event count drift。

[P4] v3.8 backlog 文件缺失
  - File: .humanize/rlcr/2026-05-23_15-38-04/plan.md:382
  - Issue: plan 写明 deferred items tracked in `docs/BACKLOG_v3.8.md`，但仓库里没有该文件。shadow-only calibration cutover、legacy removal、typed exit context 等后续事项没有落到 repo artifact。
  - Fix: 新增 `docs/BACKLOG_v3.8.md`，列出 deferred flip/removal/architecture items 和 gate 条件。

[P4] Cross-asset shadow log 不含 dual-branch P&L estimates/manifest
  - File: core/cross_asset_signal.py:145
  - Issue: shadow record 只写 decision 和 inputs；AC-5 要求 shadow log 写 both-branch P&L estimates，并通过 manifest 证明首条记录已满 14 天。
  - Fix: record schema 加 `branches: {buy_call, sell_put}` P&L/MTM estimate，以及 manifest/summary record 或独立 metadata。

[NOTES]
  - 我读了核心 diff、plan AC、关键生产文件和测试；额外跑了 `pytest tests/test_expiry_intrinsic.py tests/test_calibration_retrain.py tests/test_cross_asset_selector.py -q`，结果 47 passed。
  - `git diff --check` 被 `.humanize/skill/.../output.md` 里的既有 trailing whitespace 拦住；这是 diff 范围内的文档噪音，不是核心代码问题。
  - `force_close_at_expiry` 的 asymmetric IC max-risk 用 `max(call_wing, put_wing) - credit`，这一点实现方向是对的。

REVIEW_VERDICT: NEEDS_REWORK
