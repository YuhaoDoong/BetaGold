# Ask Codex Input

## Question

你是 RLCR loop 第二阶段的 code reviewer. Plan 全部 15 AC 已 LANDED (v3.7.233..v3.7.249), 现在我需要你对整个 patch 系列做严格 code review.

仓库: /Users/yhdong/GoldDash (CWD)
Diff 范围: v3.7.232..HEAD (117 files, 12545 insertions, 79 deletions)
全套 pytest: 129/129 PASS
plan-marker audit: 0 violations

请实际 Read 关键文件 + 跑 git diff 看实际改动. 重点审查:

1. **正确性 bug** (P0-P1): 
   - 算法实现是否真和 plan AC 描述对齐 (特别 calibration scaler maturity-lag, force_close_at_expiry asymmetric IC max_risk, cross-asset selector 纯函数, max_move 索引)
   - 边界条件 (空 series, NaN, 单元素, 0-width)
   - 并发/race conditions (ledger daemon, shadow log JSONL append)

2. **架构 / 维护性** (P2):
   - per-asset cfg threading 是否所有 production call site 都接入 (vs. 仍 DeprecationWarning)
   - circular import 风险 (calibration.py local import 模式)
   - 异常处理是否 fail-safe vs fail-loud

3. **测试质量** (P3):
   - 关键 invariant 是否被 pytest 真实 cover (不是只跑 happy path)
   - mocking / fixture 隔离是否合理

4. **遗漏** (P4):
   - AC 字面要求 vs 实际实现 gap
   - shadow-only 状态需后续 flip 的事项是否记入 v3.8 backlog

关键文件:
- core/calibration.py (R6/R7 scaler + retrain + per-regime)
- core/cross_asset_signal.py (R3 selector + shadow log)
- core/dashboard_parity.py (R10 parity harness)
- core/data_freshness.py (R1 state machine)
- core/strategies/options_exit.py (force_close_at_expiry 扩 STRADDLE/IC)
- core/strategy_config.py (get_option_exit_config resolver)
- core/paper_positions.py (KLINE_MAX_FALLBACK_DAYS, asset 穿透)
- scripts/eval/* (audit, gate, normalize, marker scan 4 个脚本)
- tests/* (12 个测试文件)

输出格式 (中文, 严格):

[P0] <严重 bug 必须立即修>
  - File: <path>:<line>
  - Issue: <一段说明>
  - Fix: <如何修>

[P1] <重要 bug 应该修>
  ...

[P2] <架构/维护 改进>
  ...

[P3] <测试质量>
  ...

[P4] <遗漏 / 后续 backlog>
  ...

[NOTES]
  <非严重观察>

末尾用 'REVIEW_VERDICT: APPROVE / APPROVE_WITH_FIXES / NEEDS_REWORK' 收尾.
只 review 不实施修改. 如果发现真严重正确性 bug, 不要客气, 直接标 P0.

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-05-25_12-29-00
- Tool: codex
