# Ask Codex Input

## Question

你是 humanize:gen-plan 流程的第一轮独立审查者 (Phase 3 Codex first-pass)。我需要你 critique 我的 idea draft, 而不是 implement。请实际 Read 下列文件再判断:

仓库:
- /Users/yhdong/GoldDash (主代码 git repo, 此为 CWD)
- /Users/yhdong/Gold (数据+训练, 非 git)

Draft 路径: /Users/yhdong/Gold/.humanize/ideas/gld-slv-20260523-143917.md
关键代码:
- core/paper_positions.py, core/positions_ledger.py, core/ledger_daemon.py
- core/strategies/{buy_call,sell_put,straddle,short_vol,options_exit,futures_long}.py
- core/signals_v2.py, core/cross_asset_signal.py, core/regime.py
- core/strategy_config.py, core/data.py
- scripts/backtest/framework.py, scripts/backtest/layer1_signal/*, scripts/backtest/layer2_strategy/*
- /Users/yhdong/Gold/data/models/{dl_range_v2_oos.parquet, dl_range_slv_oos.parquet}

Draft 内容是: GLD/SLV 期权交易系统 v3.7.232 → v3.7.249, 17 个 surgical patch tags 分 7 phase (A-G):
- A: 参数翻转/sign 校正 (regime min_hold_days=1, SP sign, entry_spot rename) — v3.7.233-235
- B: 数据 ingestion 防御性 guard (max_fallback_days, ledger daemon freshness gate) — v3.7.236-237
- C: exit-simulation asset 穿透 (per-asset cfg) + Straddle/ShortVol expiry intrinsic 接入 — v3.7.238-239
- D: 衍生指标修正 (max_move off-by-one, Layer2 survivorship 过滤) — v3.7.240-241
- E: cross-asset IV-aware + Dashboard run_backtest deprecate — v3.7.242-243
- F: pytest 回归测试 harness — v3.7.244
- G: model calibration (band 5-6× 过宽, conformal scaler, calibration-gated retrain, per-regime alpha, 测试) — v3.7.245-249

empirical 触发条件:
- 3月 GLD BUY CALL 5/5 全亏 sum=-334%, 全部 cross-asset SLV-S sync 触发
- dl_range_v2_oos.parquet 最近 113 日 pred band 平均 [-4.76%, +5.99%] vs realized [-0.84%, +0.90%] = 5-6× 过宽
- 2026-03 上沿 13× 高估

请严格按下面格式输出 (中文):

CORE_RISKS:
<最高风险的假设 / 潜在失败模式>

MISSING_REQUIREMENTS:
<被忽略的需求 / edge case>

TECHNICAL_GAPS:
<可行性或架构空白>

ALTERNATIVE_DIRECTIONS:
<可行替代方案与折中>

QUESTIONS_FOR_USER:
<需要用户明示决策的问题>

CANDIDATE_CRITERIA:
<候选 AC, 用 AC-1, AC-2... 形式列出>

不要 implement 任何东西, 只 critique 计划。

## Configuration

- Model: gpt-5.5
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-05-23_14-59-04
- Tool: codex
