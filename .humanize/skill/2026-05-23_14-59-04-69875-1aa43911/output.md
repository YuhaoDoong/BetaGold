CORE_RISKS:
- Phase G 的 empirical basis 需要重验。我实际读了两个 parquet：`dl_range_v2_oos.parquet` 最近 113 个有效样本的 actual mean 是约 `[-2.86%, +3.08%]`，不是 draft 写的 `[-0.84%, +0.90%]`；coverage 约 `54.9%`，不是 `87.6%`。如果 draft 用的是另一套“realized”定义，必须明确定义，否则 conformal shrink 可能在错误目标上过拟合。
- 现有模型训练代码已经有 conformal calibration。Phase G 再做 post-hoc conformal scaler 有“双重校准”风险，且 live 新增 OOS 行没有未来 `actual_*`，scaler 必须严格 `shift(1)` 只用过去残差，否则会引入 look-ahead。
- v3.7.238 假设 `get_config(asset)` 能返回 BC/SP/STRADDLE/SHORT_VOL exit 参数，但当前 `core/strategy_config.py` 的 `AssetConfig` 主要是信号阈值，不含 `BCConfig.profit_target_mult`、`SPConfig.profit_target_credit_pct` 等字段。直接让策略模块读 `get_config(asset)` 会类型/语义不匹配。
- “每 tag 单 root cause、≤15 LOC”低估了真实耦合。cross-asset live path 实际在 `scripts/build_positions_ledger.py` 里直接用 `CROSS_STRATEGY` 构造行；`core/cross_asset_signal.py` 没有 draft 里的 `select_gld_sync_strategy()`。
- 3 月 5/5 GLD BC 亏损样本太小，且和模型 band、cross-asset、IV、entry option pricing、exit cfg 混在一起。把它作为 Phase G 强触发会把交易执行问题误归因到模型校准。

MISSING_REQUIREMENTS:
- 需要定义“生产 call site”范围。`RegimeClassifier()` 默认调用远不止 `app.py` 和 `scripts/build_positions_ledger.py`，还包括 backfill、continuous runner、futures signals、多个 research scripts。
- `entry_spot` rename 需要 schema migration/backward compatibility。当前主 ledger JSON/parquet path 更多使用 `entry_etf/entry_open/entry_close`，`core/positions_ledger.py` 可能不是唯一消费面。
- `PENDING_KLINE` 需要完整生命周期：如何写入、何时 retry、是否进入 filtered log、是否保留原 signal date、是否防止重复补单。
- freshness gate 不应只看 kline。主链还依赖 ETF OHLC、features、OOS predictions、GVZ、sig_df snapshot、Binance futures data；每个源需要独立 stale 策略。
- pytest harness 缺少环境要求。`requirements.txt` 当前没有 `pytest`，tests 目录为空；需要先定义 test runner、fixture 数据、是否允许读 `/Users/yhdong/Gold` 真数据。
- 每个 tag revert 的数据层影响未定义。ledger schema rename、calibrated columns、retrain log、history archives 都不是简单代码 revert 能完全回滚。

TECHNICAL_GAPS:
- Phase C 的 per-asset cfg threading 应先设计配置 contract：是扩展 `AssetConfig`，还是新增 `OptionExitConfig` mapping。否则传了 `asset` 也不会正确改变 exit 行为。
- STRADDLE/SHORT_VOL 接 `force_close_at_expiry` 时要确认 `strategy_kind` 和 `max_risk`。SHORT_VOL 的 IC max risk 计算在模块内较复杂，不能简单复用 fallback 的 `credit_spread` 逻辑。
- `max_fallback_days` 不能只改 `pick_liquid_monthly_option()`。`price_strategy_at()` 当前返回 default `out`，caller 看到 no legs 就 silent skip；需要让 “stale data” 和 “no liquid contract” 可区分。
- Layer2 survivorship 修正不能只用 `signal_date <= kline_max - max_dte - hold_buffer`。实际 expiry/DTE 是每笔 leg 决定的，且 current code 有 yfinance live fallback 和 expiry intrinsic fallback。
- Dashboard `run_backtest()` 是 spot-level replay，保留 intraday log entry/exit、StopLoss、Pullback、ACTIVE trade 语义。替换成 `generate_daily_signals()` thin replay 容易丢行为 parity。
- Phase G 说 `build_band()` 自动读 `_calibrated` columns，但所有历史对比、Layer1 gate、dashboard 都需要同时能指定 raw vs calibrated，否则无法做 A/B 和 rollback。

ALTERNATIVE_DIRECTIONS:
- 先加一个 Phase 0 evidence audit：固定 label 定义、复算 GLD/SLV calibration、复算 3 月亏损 attribution，再决定 Phase G 是否进入 production。
- 对 exit 层优先做 typed `ExitContext` 或明确 `OptionExitConfig(asset,strategy)`，比到处传 `asset` 更能防止半接入。
- Calibration 先 shadow-only：追加 calibrated columns 和 audit report，但 `build_band()` 默认继续 raw；只有 Layer1/Layer2 gate 通过后再切 flag。
- Freshness 做 source-state table：`FRESH/STALE/FROZEN` per source，option entry gate 只消费 kline/source state，不影响 futures/MTM/backfill。
- Cross-asset 先把策略选择抽成纯函数，并在 ledger builder 调用；shadow-log BC vs SP 结果，不立即 live flip。

QUESTIONS_FOR_USER:
- Phase G 的 “realized [-0.84%, +0.90%]” 指的是哪种标签？parquet 里的 `actual_upper_pct/actual_lower_pct` 不匹配这个数字。
- 你希望 per-asset exit 参数放进现有 `AssetConfig`，还是新增独立的 option exit config registry？
- kline stale 时，是只禁止新期权开仓，还是连 cross-asset 期权同步也全部禁止？期货是否继续允许？
- 17 个 v3.7.* tag 是否必须真 git tag，还是 humanize patch label 即可？
- Dashboard 旧 `run_backtest()` 是保留兼容 wrapper 一版，还是允许一次性替换输出 schema？

CANDIDATE_CRITERIA:
- AC-1: 对 GLD/SLV OOS parquet 产出可复现 calibration audit，明确 label 定义、日期窗口、coverage、width ratio，数字与 draft 触发条件一致或解释差异。
- AC-2: 所有 production `RegimeClassifier()` call site 显式传 `min_hold_days`，并列出 research-only 例外。
- AC-3: `PENDING_KLINE` 与 `NO_CONTRACT` 在 ledger/log 中可区分，且 retry 不重复开仓。
- AC-4: BC/SP/STRADDLE/SHORT_VOL exit tests 覆盖 missing kline、expiry day、past expiry、per-asset cfg 差异。
- AC-5: Cross-asset strategy selector 是纯函数，有 BC/SP shadow result，live flip 由显式 flag 控制。
- AC-6: Layer2 统计同时报告 `n_signal / n_entered / n_closed / n_open / n_skipped_stale`，不再只看 closed。
- AC-7: Calibrated bands shadow-only 通过 raw-vs-calibrated Layer1 gate 后，才允许 production `build_band()` 切换默认列。
