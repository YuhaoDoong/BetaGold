AGREE:
- v2 已经吸收 Round 1 的关键纠偏：AC-8 的 horizon-aware maturity lag、coverage repair 目标、shadow/live 分离、gate preflight，以及 AC-4/5/7/12/13/14 的补强方向都是合理的。
- Goal 明确把 OI/risk/yfinance/exposure cap 移出本计划，降低了范围膨胀风险。
- Lower Bound 不再把 AC-7 降级成 report-only，这点已经一致。
- Original Draft 标为 NON-NORMATIVE，保留历史分歧但不作为实现依据，这个处理可以接受。

DISAGREE:
- AC-5 同时要求 `select_gld_sync_strategy(...)` 是 pure function，又要求它“emit structured warning into jsonl”。纯函数不应写文件；应返回 decision/reason，由调用方写 shadow log。
- AC-5 的函数签名只有 `gvz_value`，无法判断 “stale >2 trading days”。必须传 `gvz_asof_date` 或结构化 GVZ record，否则 stale policy 不可实现且回放会混用 wall-clock today。
- AC-8 仍有边界矛盾：文本同时写 “dates ≤ t − horizon” 和 `label_end_date < calibration_as_of_date`。对 5-day forward label，严格 `< as_of` 通常意味着不能包含 `t-horizon` 那条刚在 t 才结束的 label。
- Milestone 3 仍写 “shift(1) discipline”，Claude-Codex Agreements 也写 “shift(1) past-only residuals”。这和 AC-8/task-g3 的 horizon-aware maturity lag 冲突。
- task-f2 说 12 scenarios，但 AC-4 明确要求 16 scenarios，并区分 expiry-day close known / unknown。
- Dashboard AC-14 的“新 path 与 legacy `buy_signal/buy_type/signal_tier/exit_signal` exact match”可能阻止修复旧 Dashboard 没走 canonical filters 的问题。这里应明确 parity 是针对 intraday StopLoss/Pullback/ACTIVE 语义，还是要求先把 legacy wrapper 改成 canonical 后再比较。
- Goal 声称范围“exactly AC”，但 task-d1 修 `max_move_{h}d` off-by-one 没有对应 AC。要么纳入 AC-7，要么单独加 AC。

REQUIRED_CHANGES:
- 修 AC-5：把 selector 改成纯返回值，例如返回 `{strategy, reason, gvz_status}`；shadow log 写入放到 caller。
- 修 AC-5 签名：加入 `gvz_asof_date` / `gvz_record`，staleness 相对 `signal_date` 判断，不用运行当天 wall-clock。
- 修 AC-8 temporal boundary：统一为 `label_end_date < calibration_as_of_date`，并删除/改写 “≤ t − horizon” 这类可能 off-by-one 的描述；测试也按这个边界写。
- 删除所有 normative section 里的 `shift(1)` 说法，尤其 Milestone 3 和 Agreements。
- 把 task-f2 改成 16 scenarios，与 AC-4 一致。
- 明确 AC-14 parity contract：如果目标是修 Dashboard 旧逻辑，允许 canonical signal columns 出现预期差异并要求差异归因；只对 intraday exit state/event semantics 做 exact/±1 parity。
- 给 `max_move` off-by-one 添加明确 AC，或从本 plan 的 required task 中移除。

OPTIONAL_IMPROVEMENTS:
- 把 appendix / Original Draft 内旧 calibration 数字再加水印式提示，避免实现者从 Objective Evidence 误读 5-6×/87.6%。
- 统一 freshness 阈值表述：AC-6 的 2/3 trading days 与 task-b1 的 `max_fallback_days=7` 建议解释为两层 gate，避免看起来冲突。
- AC-9 补一个 `actual_width == 0` 或极小值保护，避免 overshoot ratio 除零或异常放大。
- tag 数量、task 数量、Phase H 校验任务之间建议补一张 tag mapping，减少执行时歧义。

UNRESOLVED:
- DEC-1 到 DEC-7 仍需 user 决定，但这些是设计选择，不是 reviewer 与 plan 的实质对立。
- 当前不建议标记 PLAN_READY；上面的 REQUIRED_CHANGES 修完后才可以收敛。
