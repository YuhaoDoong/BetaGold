# Ask Codex Input

## Question

你是 RLCR loop code review 第二轮 reviewer. 我吸收了你 Round 1 的所有 P1 + P2#2 + P4#1 反馈, 提交 v3.7.250.

Plan 状态: 15/15 AC LANDED + 1 review-fix tag = v3.7.232..v3.7.250 (18 tags).
Pytest: 133/133 PASS in 0.62s.

修复清单 (v3.7.250):
1. P1#1: force_close_at_expiry 改 today < expiry_dt 返回 None; today >= expiry_dt
   时检查 exact 日期 close.
2. P1#2: 新增 spot_close_on_exact_date(), expiry 强平用它; exact 缺失返回
   {is_closed: False, state: 'AWAITING_EXPIRY_CLOSE'}.
3. P3#1: test_expiry_intrinsic.py 4 个 today_equals_expiry 翻转 + 新增
   AWAITING case (18 total).
4. P1#3: build_positions_ledger.py 加 PENDING_KLINE_DATES 跟踪 + 水位线 clamp
   = min(latest, earliest_pending - 1 trading day). 下次 retry, ledger 按
   (asset, signal_date, strategy) idempotent dedup.
5. P1#4: evaluate_retrain_trigger 新增 pred_widths: Optional 参数. 传入用真
   pred_width/actual_width; 不传退化 legacy proxy. 加 2 pytest.
6. P1#5: GOLD_CROSS_LIVE_CUTOVER env var + 启动时 live_cutover_allowed()
   preflight; 14-day gate 未过 fall back shadow-only.
7. P2#2: selector asof > sig_d 改为 status='future_asof_invalid'. 加 pytest.
8. P4#1: docs/BACKLOG_v3.8.md 新建.

P2/P3 中等优先级 (P2#1 fail-loud, P2#3 build_band preflight, P3#2 freshness 集成,
P3#3 dashboard real-replay, P4#2 shadow log dual-branch) 入 v3.8 backlog
记录但未在本 round 修, 因为不影响 v3.7.* 正确性.

请实际 Read 改动文件 + 跑 git diff v3.7.249..v3.7.250 看实际改动. 评估:
- P1 5 项是否真已修对 (不只是表面修)
- 新加 test 是否真覆盖 invariant
- 没引入新的 P0/P1 regression
- 留下的 P2/P3 在 backlog 是否合理 defer

输出格式 (中文):
[P0] / [P1] / [P2] / [P3] / [P4] / [NOTES]
末尾 REVIEW_VERDICT: APPROVE / APPROVE_WITH_FIXES / NEEDS_REWORK.

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-05-25_12-40-41
- Tool: codex
