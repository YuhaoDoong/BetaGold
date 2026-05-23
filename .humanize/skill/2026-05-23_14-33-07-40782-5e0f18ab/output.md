我实际读了你列的主链文件、`scripts/backtest` 两层、以及 `/Users/yhdong/Gold/data/positions_ledger.json`。一个现实差异先说明：当前 `all_klines.parquet` 我读到的 `max(date)=2026-05-06`，不是 2026-03-10，但相对今天 2026-05-23 仍滞后 17 天；`positions_ledger_meta.json` 已经把 GLD/SLV 水位线推进到 2026-05-21，这会放大 stale pricing 问题。

**严重 Bug**
1. [core/paper_positions.py](/Users/yhdong/GoldDash/core/paper_positions.py:689) 没有把 per-asset 策略参数传给退出模拟。`simulate_option_exit()` 调 `simulate_sp_position(..., cfg=None)`，所以 [core/strategies/sell_put.py](/Users/yhdong/GoldDash/core/strategies/sell_put.py:51) 实盘默认永远用 `SPConfig(profit_target_credit_pct=50)`，但生产配置在 [core/strategy_configs.py](/Users/yhdong/GoldDash/core/strategy_configs.py:107) 是 GLD=70、SLV=30。  
改法：给 `simulate_option_exit(entry_pricing, ..., asset=asset)` 加 `asset` 参数，内部 `from core.strategy_configs import get_config`，对 `SELL PUT/BUY CALL/STRADDLE/SHORT_VOL` 统一取 cfg 后传入；同步改 [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:276)、[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:402)、[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:543)。

2. [core/paper_positions.py](/Users/yhdong/GoldDash/core/paper_positions.py:366) 的 kline entry pricing fallback 没有最大陈旧天数。现在 2026-05-15 的 open 仓可能用 2026-05-06 期权 OHLC 插值成入场价，并被 ledger freeze。更严重的是 [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:491) 用 ETF OHLC 最新日期推进 `evaluated_through`，不看 option kline 最新日期。  
改法：`pick_liquid_monthly_option(..., max_fallback_days=2)`，超过则返回 `None` 或 `source="PENDING_KLINE"`；ledger 水位线拆成 `signal_evaluated_through` 和 `execution_priced_through`，期权仓不应在 kline 陈旧时冻结 entry。

3. v3.7.232 expiry intrinsic 兜底没有接到 STRADDLE/SHORT_VOL 独立模块。[core/strategies/buy_call.py](/Users/yhdong/GoldDash/core/strategies/buy_call.py:57) 和 [core/strategies/sell_put.py](/Users/yhdong/GoldDash/core/strategies/sell_put.py:72) 已接，但 [core/strategies/straddle.py](/Users/yhdong/GoldDash/core/strategies/straddle.py:33)、[core/strategies/short_vol.py](/Users/yhdong/GoldDash/core/strategies/short_vol.py:73) 仍先查 kline，缺数据时会继续 OPEN。  
改法：在两个模块解析 `legs` 后、查 `first_kdb` 前调用 `force_close_at_expiry()`；STRADDLE 用 `strategy_kind="long_vol"`，SHORT_VOL 用 `strategy_kind="credit_spread", max_risk=max_risk`。

4. 生产 regime 和回测 regime 不一致，并且生产历史重建有 look-ahead 风险。[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:92) 用 `RegimeClassifier()` 默认 `min_hold_days=20`；其实现 [core/regime.py](/Users/yhdong/GoldDash/core/regime.py:118) 会向后看一段 regime 是否持续，再改写历史 regime。回测框架 [scripts/backtest/framework.py](/Users/yhdong/GoldDash/scripts/backtest/framework.py:70) 改成了 `min_hold_days=1`。  
改法：生产也改 `RegimeClassifier(min_hold_days=1)`，或实现真正 online 的 debounce：只有连续观测满 20 天后从“今天起”切 regime，不回写过去。

5. Dashboard 的“真实策略回测”仍用旧 `run_backtest()`，和当前信号主链不一致。[app.py](/Users/yhdong/GoldDash/app.py:1345) 调 [core/signals_v2.py](/Users/yhdong/GoldDash/core/signals_v2.py:404)，但该回测入场只看 RV extreme 和 `bp_low`，没有应用 MA filter、IV 三档、`sp_score`、`ret_20d_max_hard`、tier 逻辑。  
改法：废弃或重写 `run_backtest()`，先调用 `generate_daily_signals()`，再只按 `buy_signal/buy_type/signal_tier` 执行，确保 Dashboard、ledger、回测同源。

**中等问题**
1. Cross-asset 现在不是 IV-aware。[core/cross_asset_signal.py](/Users/yhdong/GoldDash/core/cross_asset_signal.py:43) 固定 `CROSS_STRATEGY="BUY CALL"`，[scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:521) 直接套这个策略。  
改法：加 `select_gld_sync_strategy(d, gld_sig, gvz)`：若 `GLD bp_low <= 0.10 and GVZ >= 25`，返回 `SELL PUT`，否则 `BUY CALL`。同时把 `gvz` 写入 `sig_df` 或在 cross block 里从 `gvz_close` 取。你提的 IV-aware 规则应该上，至少作为 shadow log 后再切实盘。

2. Layer2 期权回测只统计已关闭交易，存在幸存者/数据滞后偏差。[scripts/backtest/layer2_strategy/directional_options/run_all.py](/Users/yhdong/GoldDash/scripts/backtest/layer2_strategy/directional_options/run_all.py:59)、[scripts/backtest/layer2_strategy/vol_options/run_all.py](/Users/yhdong/GoldDash/scripts/backtest/layer2_strategy/vol_options/run_all.py:35) 都 `if sim.get("is_closed")` 才计入。  
改法：要么把 open MTM 纳入 score，要么只回测 `signal_date <= kline_max - max_dte - hold_buffer` 的样本，避免近期未闭合交易被静默丢掉。

3. Layer1 `max_move` 前向窗口有 off-by-one 风险。[scripts/backtest/framework.py](/Users/yhdong/GoldDash/scripts/backtest/framework.py:120) 用 `rolling(h).max().shift(-(h+1))`，对信号日 t 实际会跳过 t+1 entry day，并包含更后的日期。  
改法：写显式 helper：`entry_i=i+1; exit_i=i+1+h; high[entry_i:exit_i+1].max()`，不要用反向 rolling 猜窗口。

4. OI 微观结构修正只在 Dashboard 展示层使用，没有进入 ledger/backtest 主链。[app.py](/Users/yhdong/GoldDash/app.py:1387) 和 [app.py](/Users/yhdong/GoldDash/app.py:5236) 有 OI-adjust band；但 ledger 在 [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:94) 直接用 raw band 生成信号。  
改法：要么把 OI-adjust band 下沉到 `generate_daily_signals()` 的输入前，要么 Dashboard 明确标注“展示阈值，不参与交易”。

5. [core/paper_positions.py](/Users/yhdong/GoldDash/core/paper_positions.py:177) 的 underlying fallback 对 `SELL PUT` 方向算反了。现在非 BUY CALL/SPOT 都 `sign=-1`，但 sell put 是 bullish，标的上涨应盈利。  
改法：`SELL PUT` fallback sign 应为 `+1`；更好是任何期权仓都不用 spot sign 兜底，统一走 entry legs + option exit/MTM。

6. [core/positions_ledger.py](/Users/yhdong/GoldDash/core/positions_ledger.py:87) 的 `entry_spot` 写成了 `entry_pricing["daily_close_price"]`，这是期权 daily close，不是 ETF spot。  
改法：`price_strategy_at()` 返回 `underlying_entry_price`，ledger 用这个字段；`entry_credit` 才用 `entry_price`。

**优化建议**
1. 加数据新鲜度硬闸：ledger daemon 每次写入前检查 `kline_db max_date`，超过 2 个交易日只刷新已有仓 MTM，不新增期权仓；Dashboard 顶部显示 stale 天数。

2. 把风控从旧回测迁到 ledger 主链：按 asset/strategy 统计最近 closed 交易，连续 2 笔亏损暂停新仓；同时加 `max_open_positions_per_asset`、`max_margin_pct_per_asset`、`same_day_strategy_cap`。当前 [scripts/build_positions_ledger.py](/Users/yhdong/GoldDash/scripts/build_positions_ledger.py:139) 逐日 append，没有全局 exposure gate。

3. 给 v3.7.232 加回归测试：BC/SP/STRADDLE/SHORT_VOL 四类，kline 缺 expiry bar、today > expiry 时都必须关闭；再加 stale kline entry、per-asset SP config、cross IV-aware 三个测试。

4. live yfinance fallback 建议用 bid/ask mid 优先，`lastPrice` 只作兜底；否则盘后/冷门合约 last 很容易 stale。相关入口在 [core/paper_positions.py](/Users/yhdong/GoldDash/core/paper_positions.py:298) 和 [core/paper_positions.py](/Users/yhdong/GoldDash/core/paper_positions.py:889)。
