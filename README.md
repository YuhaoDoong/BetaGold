# GoldDash — 贵金属交易决策系统

基于宏观因子 + 深度学习的贵金属交易决策系统。支持黄金 (GLD) 和白银 (SLV)，三层模型 + 多策略竞争 + 实证回测验证。

**GitHub**: https://github.com/YuhaoDoong/BetaGold

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构 — 三层模型, 数据流, 项目结构, Dashboard 模式 |
| [docs/MODELS.md](docs/MODELS.md) | 模型架构 — LSTM+Transformer Ensemble, Conformal 校准, Regime 7 因子分类 |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | 策略详解 — 完整策略矩阵, vega/delta 分析, 胜率定义, 工具映射 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验记录 — 所有回测结果, 阈值调优, regime 分段, 期货 vs 期权对比 |

## 快速入门

```bash
conda activate gold

# 启动 Dashboard
streamlit run app.py

# 历史回填盘中触发 (首次或规则变更后)
python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60
python scripts/backfill_intraday_signals.py --asset SLV --timeframe 60

# 月度阈值重测 (建议 cron / launchd 自动化)
python scripts/monthly_retune.py              # 全部资产
python scripts/tune_thresholds.py --asset GLD # 单个资产手动
```

### 月度自动调度 (macOS launchd)
```bash
cp scripts/com.golddash.retune.plist.example \
   ~/Library/LaunchAgents/com.golddash.retune.plist
# 修改文件中的 USERNAME 和 Python 路径
launchctl load ~/Library/LaunchAgents/com.golddash.retune.plist
```

每月 1 号 04:00 SGT 自动跑全部资产 grid search, 输出到
`data/tune_history/`, dashboard 侧边栏自动显示状态 (🟢🟡🔴).

## 核心特点

### 三层模型架构
```
日线层 (今日预测)    模型预测 → 5日区间 + bp030/bp090 阈值 ("开窗信号", 不是入场价)
        ↓
盘中层 (盘中信号)    实时盘中: 价格在阈值外侧 + Stoch RSI/MACD/KDJ 确认
        ↓ 写入 data/intraday_signal_log.parquet
固化层 (历史代表价)  每日多触发取最差 (买:max / 卖:min) → 持仓/回测全部读这里
```

### 完整策略矩阵 (v3.7.50 真实期权 18mo 实证)

| RV %tile | 方向性信号 | 推荐工具 | 波动率策略 |
|----------|-----------|----------|------------|
| < 切点 (低 IV) | BUY CALL 类 | **Buy Call** (期权便宜, delta gain) | tech-score ≥6 → Long Straddle |
| ≥ 切点 (高 IV) | SELL PUT 类 | **Sell Put** (收 premium 替代 long call) | tech-score ≥6 → Long Straddle |

**切点** (BC↔SP 切换, 单切, per-asset 在 strategy_config 月度重训):
- GLD: rv_pctile **0.45** (实证看涨 n=64, 切换合成胜率 59.4%)
- SLV: rv_pctile **0.75** (实证看涨 n=69, 切换合成胜率 66.7%)

**STRADDLE tech-score** (v3.7.49 重构, 实证 score≥6 触发):
- BBW pctile / ATR ratio / Donchian width / RV%tile / RV abs / RV momentum + 事件辅助
- 18mo 真实期权胜率: GLD 73% (n=22) / **SLV 70% (n=53, 从 baseline 33% 反转)**
- score 4-5 是噪音区 (17-33% 胜), 必须避开

**Sizing** (score 6=1× / 7=1.5× / 8+=2×): 累计收益 +125% vs 单切 +68%

### 胜率定义 (按 vega/delta 实际盈亏)

| 策略 | 胜利条件 (动态 sigma_pct = RV × √h/252) |
|------|-----------------------------------------|
| BUY CALL | `max_up > 1σ` (横盘是亏) |
| SELL PUT | `max_down < 1σ` (横盘+上涨都赢) |
| STRADDLE | `max_move > 1σ` (双向移动 > premium) |
| SHORT_VOL | `max_move < 1.6σ` (IC 短腿内) |
| 期货多头 | `ret_5d > 0` (任何正向收盘) |

详见 [docs/STRATEGIES.md](docs/STRATEGIES.md)

## 关键实证结论 (近 5 年)

1. **方向性 RV 极值过滤**: 排除中位 50-85% 后, 胜率 78% → **81%**, Sharpe 0.53 → **0.61**
2. **期货代替 BUY CALL 期权**: 胜率 73% → **96%**, Sharpe 0.23 → **1.16** (5 倍提升)
3. **Iron Condor 严格时机**: 89% 胜率 (vs Short Strangle 40%), 翼锁定最大亏损
4. **Regime 分段**: Bull 84% / Mixed 45% / Bear 30% — 现状已 regime-optimal
5. **STRADDLE 完全 regime-agnostic**: Mixed 反而是最佳战场 (90%)

详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 版本历史

| 版本 | 主题 |
|------|------|
| v1.0-v2.x | 日线 Band + 盘中触发 + 12h 止盈 + 3% 止损 |
| v3.0-v3.4 | 白银扩展 + Qlib Alpha158 + Ensemble 模型 + 多周期 Stoch RSI + 伦敦金价位 |
| v3.5 | 盘中触发模块: 参数化规则 (Stoch RSI/MACD/KDJ) + 持久 log + 历史回填 |
| v3.6.x | 完整策略矩阵: 做多/做空波动率 + 方向性 RV 过滤 + 期货独立统计 + 工具映射 |
| v3.7 | 模块化重构 (core/strategies/) + 4 文档拆分 + 持仓管理 SHORT_VOL |
| v3.7.x | 修复持仓管理时间倒序 + 活跃持仓暴露 + 完整交易历史合并 |
| v3.7.8 | 熔断 A/B 验证 + 默认关闭 (实证不提升胜率, 错杀赢面) |
| v3.7.9 | 持仓管理加退出日 + 退出原因列 |
| v3.7.10 | 修复 MIXED 优先级 bug (score≥6 单走 vol, 4-5 才 MIXED) |
| v3.7.11-13 | IV crush 模块化 (core/iv_crush.py) + 用真实 GVZ 数据校准 |
| v3.7.14 | IV crush 调整默认关闭 (GLD 实证不显著), 保留模块 + Dashboard 显示 IV/RV |
| v3.7.15-25 | UI / 主图 1h 化 / 5 子图 sharex / 期权策略实时面板 / 持仓管理实时退出判定 / FOMC 日期修正 |
| v3.7.26-28 | SLV 1h 数据兜底 / auto_refresh 补 SLV 1h / 主图横坐标 / 1-60 天 / 今日预测 sharex 合并 |
| v3.7.29 | RV 阈值精细网格 (步长 0.025) GLD 优化 |
| v3.7.30 | Per-Asset 校准: SLV 单独 grid search + 集中参数管理 core/strategy_config.py |
| v3.7.31 | 月度自动重测系统: scripts/monthly_retune.py + 状态跟踪 + Dashboard 显示 + launchd 示例 |
| v3.7.32 | STRADDLE 加 RV %tile < 0.50 过滤 (5y 实证 Sharpe +5%) |
| **v3.7.33** | **RV %tile 精度统一 0.01 (GRID_PRECISION 集中管理) + GLD/SLV STRADDLE per-asset 校准: GLD 0.42 (Sharpe 0.922), SLV 0.20 (Sharpe 1.258); SLV STRADDLE 实际比 GLD 表现更好** |
| v3.7.39-43 | 真实期权回测 (yfinance OCC) + 多 expiry DTE 适配 + 4 策略止损 + Moomoo fallback |
| v3.7.44-46 | RV 阈值反复 (跟用户原 SELL_PUT 设计意图调和) — 单切 GLD 0.45 / SLV 0.75 |
| v3.7.47-48 | 波动率技术指标重构 (vol_indicators.py BBW/ATR/Donchian/long_vol/short_vol score) |
| v3.7.49 | events.py STRADDLE/SHORT_VOL 改 tech-score 主导 (事件 30%), score≥6 触发 |
| **v3.7.50** | **真实期权 18mo 验证 STRADDLE 通过 (GLD 73%/SLV 70% 胜率), Dashboard sizing 显示, generate_signals 与 strategy_config 切点 unify** |
| v3.7.51 | kline_db (本地 EOD 累积) 作为 yfinance/Moomoo 第三 fallback |
| v3.7.52 | 数据腐败检测 (close 重复+量低 → 强制重拉); daily_eod_options.py 每日采集 cron |
| v3.7.53-83 | UI 大量优化: 时区美东对齐 / 主图 5 子图重排 / K线精度可配 / dirty 数据过滤 / 自动 dedup |
| v3.7.84-90 | 主图改 GC=F 期货源 (23h 全球夜盘) / 触发 confirm_mode 客观最优 / EXIT marker 单一来源 |
| v3.7.91 | **kline_db 真实期权 OHLC + spot 比例插值 替代 BS+假IV (单日误差 <5%)** |
| v3.7.92-95 | sig_df buy_type/vol 信号 detect 重构 / 历史未平仓+期权模拟 OPEN/CLOSED 分表 |
| v3.7.96 | **真实期权退出规则: SELL PUT +50%/-100%/expiry · BUY CALL ±50%/100% · STRADDLE 14d · SHORT_VOL 30d** |
| v3.7.97 | BUY CALL 单腿 vs 价差条件选 / SELL PUT 强制 put credit spread / 单腿价格显示 |
| v3.7.98-101 | 触发 confirm_mode any/dedupe/分桶 (premarket vs RTH 独立) / stale 改交易日判 |
| v3.7.102 | 全部 "伦敦金/伦敦银" → "纽约金/纽约银" (跟 GC=F/SI=F 数据源对齐) |
| v3.7.103-105 | ETF live spot 直拉 + ratio 2h TTL + ETF prepost 16h (premarket+aftermarket) |
| **v3.7.106** | **Range 模型重训用 GC=F 期货 23h 数据替代 GLD ETF 6.5h (tightness 持平 0.11, lower band 更紧)** |
| v3.7.107-108 | 多策略并行持仓 (5-4 SELL PUT + SHORT_VOL + FUTURES 同日) / 状态栏字体缩小 |
| **v3.7.109** | **Binance XAUUSDT 永续 20× 接入 (实时 mark/funding/liq 真值) + 双 scale ETF/GC=F + 5 项退出修** |
| v3.7.110-115 | 全历史回测三阶段路由 + 真实期权回测 only + paired BC/SP 比较 (用户方法论) + SP PnL 改 margin 分母 |
| v3.7.116-117 | IV × bp_low 深破过滤 grid + GVZ IV 三阶过滤 (低/中/高 IV 不同处理, 高 IV 强制 SP) |
| v3.7.118-121 | dashboard 回测分析重写 + 多维 paired grid (9 维) + 4 个强力 SP 选择条件实证 |
| v3.7.122 | 智能 DTE (尽量短期权代替 LEAPS, dte = max(base, days_since+30)) + 日线技术指标特征 (MACD/RSI/Stoch) |
| **v3.7.123** | **sp_score 多因子选 BC vs SP — paired 验证: GLD acc 42% → 86%, SLV +725% 累计 PnL** |
| v3.7.124 | dashboard 当日信号 metric 显示 sp_score 各因子命中明细 |
| **v3.7.125** | **修今日触发表空 bug — detect+upsert 提前到 kline 加载即跑, 不再依赖 1h chart 视图** |
| v3.7.125 retune | 月度 grid (5y, GLD/SLV 全维度) — 当前配置接近最优, 不切换. paired sp_score thr=3.5/2.5 仍最优 |
| v3.7.126 | 三实验: 期货止损 -2%→-3% (wr +6-8pp), bp_low 不需更深破, sp_score 阈值维持 |
| **v3.7.127** | **方向性入场加 ma_trend (MA20/MA50) 过滤 — 全链路累计 PnL +1400%** |

## v3.7.127 ma_trend 入场过滤 (核心方向性优化)

### 因子分析驱动 (entry_signal_factor_analysis.py)

跑 13 个候选因子的 winner-vs-loser 分析, 发现 `ma_trend = MA20/MA50` 是最强单因子分化器:

| 资产 | ma_trend < 0.99 (下行趋势) BC 胜率 |
|---|---|
| GLD | **0% (0/8 笔, 全输)** |
| SLV | **12.5% (3/24 笔)** |

### 原理

**ma_trend 三档**:

| 值 | 状态 | 价位结构 |
|---|---|---|
| `> 1.01` | 上行趋势 | 短均价 > 长均价 (多头排列) |
| `0.99 ~ 1.01` | 横盘震荡 | 两均线缠绕 |
| `< 0.99` | 下行趋势 | 死叉后 (空头排列) |

**为什么下行趋势 BC 几乎全输**:
1. **结构性下跌** — MA20 跌破 MA50 是趋势反转的滞后确认. 反弹概率 < 继续跌概率.
2. **接飞刀效应** — `bp_low<0.30` 在下行趋势中只是"价格创新低", 不是 mean-revert 信号. Range 模型下沿会跟着下移.
3. **vs 上行 pullback** — 同样的 `bp_low<0.30`, 在 `ma_trend > 1.01` 是健康回调, 80%+ 反弹; 在下行中是趋势延续, 多数继续跌.

**阈值 0.99 而非 1.0** — 横盘 (0.99-1.01) 不过滤 (mean-revert 仍有效), 仅过滤明确下行 (MA20 比 MA50 低超过 1%).

### 实施

```python
# core/signals_v2.py: generate_daily_signals 内
ma_trend = close.rolling(20).mean() / close.rolling(50).mean()

if buy_sig and ma_trend.get(d) < 0.99:
    buy_sig = False  # MA20 < MA50 时跳过方向性入场
```

### 全链路回测对比

| 资产 | 策略 | 改前 wr / 累计 | 改后 wr / 累计 | Δ累计 |
|---|---|---|---|---|
| GLD | BUY CALL | 64.0% / +2997% | **76.2% / +3421%** | +14% |
| GLD | SELL PUT | 72.0% / +34% | **83.3% / +493%** | **+1349%** |
| SLV | BUY CALL | 51.1% / +1611% | **91.3% / +2437%** | **+51%** |
| SLV | SELL PUT | 52.8% / +289% | **83.3% / +449%** | +55% |

**合计累计 PnL 提升 ~+1400%**, 单一过滤产生质变. sp_score 在 ma_trend 过滤后 GLD chosen wr **88.1%** / 累 **+3598%** (距完美上限 90.5% / +3945% 仅 2.4pp).

### 五层过滤体系 (BC 入场完整链路)

```
1. bp_low < 0.30           — Range 模型下沿 (Range-bound 触发器)
2. Bull regime             — 长期方向性环境 (年级)
3. RV 极值过滤             — 排除中位温水区 (避免噪音方向)
4. ma_trend ≥ 0.99 (新)    — 中期趋势方向 (5-10 周, 过滤接飞刀)
5. sp_score 决定 BC vs SP  — 多因子选择最优工具
```

各层互补 — Bull regime 是年级方向, ma_trend 是中期 (5-10w), 触发器是日内. ma_trend 填补了 Bull regime 与日内触发之间的中间层.

### 完整重测 2026-05-06 结果

跑 `monthly_retune.py` (5y 全维度 grid, 步长 0.025) + `full_history_backtest.py` + `paired_score_validate.py`:

**Grid search 结论**: 当前 SHORT_VOL / STRADDLE / 方向性配置 **接近最优 (改进 < 5%)**, 不切换.
- GLD SHORT_VOL: 0.45/0.80 → n=69 wr **91%** Sharpe **1.27**
- SLV SHORT_VOL: 0.25/0.775 → n=81 wr **86%** Sharpe **0.72**

**Stage 路由实证** (最有价值发现):

| Stage | GLD wr / avg | SLV wr / avg | 备注 |
|---|---|---|---|
| stage2_leaps_aux (90-365d) | **79.8% / +36.3%** | **82.3% / +32.0%** | LEAPS 主力 |
| stage1_spot_only (>1y) | 61.1% / +0.5% | 53.4% / +0.4% | 仅期货代理 |
| stage2_main_3m (<90d) | 27.6% / -24% | 36.3% / -6% | ⚠️ 近月波动大 |

→ **LEAPS (90-365d) 是收益主源**, 近月期权风险/收益比最差. 智能 DTE 系统已自动倾向 LEAPS.

**sp_score 在新 CSV (20260506) 验证仍稳定**: GLD acc 86% 累 +3279% / SLV acc 67% 累 +762%.


## v3.7.123 sp_score 系统 (核心 BC vs SP 决策)

替代单切 RV 阈值, 用 7 因子加权打分决定 BUY CALL 或 SELL PUT. **paired 同信号实证**:

| 资产 | 单切 RV 准确率 | sp_score 准确率 | 平均 PnL/笔 (RV → score) | 累计 PnL |
|---|---|---|---|---|
| GLD | 42% | **86%** (thr=3.5) | +22.2% → **+65.7%** | +1112% → +3284% |
| SLV | 38% | **68%** (thr=2.5) | +3.1% → **+22.7%** | +113% → +862% |

**因子权重** (paired_grid_multi 9 维实证后):

| 因子 | 权重 | 触发条件 | 说明 |
|---|---|---|---|
| RSI < 30 | **2.0** | 超卖 | ★最强 (paired 100% wr) |
| IV-RV gap > 0 | 1.5 | IV > 实际波动率 | 卖贵 premium |
| bp_low < 0.05 | 1.0 | 深破下沿 | 反弹概率高 |
| bp_close < 0.30 | 1.0 | close 在 band 下沿 | 区间预测 |
| GVZ ≥ 28 | 1.0 | 高 IV | 收 premium 替代 |
| Stoch %K < 40 | 0.5 | 非超买 | 辅助确认 |
| MACD hist < -0.5 | 0.5 | 空头动能 | 辅助确认 |

`sp_score >= threshold` → SELL PUT (credit spread), 否则 BUY CALL.

dashboard 信号 metric 实时显示 breakdown: `score 4.5 (RSI28*2.0 + IV-RV+5*1.5 + bp_low0.04*1.0)`.

## v3.7.109 重大变更概览

### 数据源统一 (v3.7.102-106)
- ❌ **不再用** "伦敦金"/"伦敦银" 称谓 (LBMA spot 没接, 全局换 "纽约金"/"纽约银")
- ✅ **GC=F (COMEX 黄金期货 23h)** = 价格显示主源 + Range 模型训练源
- ✅ **GLD ETF (RTH 6.5h + prepost 16h)** = 期权底层 + 触发 detect 用
- ✅ **Binance XAUUSDT 永续** = 期货策略真实 (公开 API, 无 key)
- ✅ **kline_db EOD 期权 OHLC** (本地累积 ~48k 行) = 历史期权价插值源
- ratio 自动 2h TTL 更新 (ETF premium/discount 漂移 ±0.3% 跟得上)

### 期权回测精度提升 (v3.7.91+96)
| 方法 | 误差 | 说明 |
|---|---|---|
| BS + IV=0.20 硬编码 (旧) | 权利金低估 ~20%, Greek 全错 | 不可靠 |
| **kline_db OHLC + spot 比例插值 (新)** | **单日 spot <2% 移动 误差 <5%** | 接近真实成交 |

退出规则 (simulate_option_exit):
- SELL PUT credit spread: +50% 早平 / **-50% stop (1.5×entry)** / expiry
- BUY CALL: +100% / -50% / expiry
- STRADDLE long vol: +100% / 14d 定时 / expiry
- SHORT_VOL credit: +50% / -50% / 30d / expiry
- **FUTURES 期货多头 (新): 5d 持仓 / +3% 止盈 / -2% 止损**

### 多策略并行持仓 (v3.7.107)
单日可同时持有 5 类:
1. **期货多头** (Binance XAUUSDT 20× perp, premarket 触发)
2. **跨式期权 STRADDLE** (long ATM call + put)
3. **BUY CALL** (单腿 ATM call OR bull call spread, IV 自适应)
4. **SELL PUT credit spread** (-ATM put / +-5% put)
5. **铁鹰 SHORT_VOL** (当前简化为 SELL PUT credit spread, 完整 4-leg IC 待后续)

### Binance XAUUSDT 期货模拟 (core/binance_futures.py)
- 公开 endpoint 无 API key
- 实测 5-5: mark $4541.87, funding 0.0026%/8h, taker 0.05%
- 20× long $4540 → margin $227, **liq $4335** (-4.5% 爆仓)
- 含 funding 累积 + 双边 fee 真实计算

### 持仓管理表新结构
| 列 | 内容 |
|---|---|
| 信号日 / 策略 / 合约 | OCC ticker 或 期货 USDT 标识 |
| **入场ETF / 入场GC=F** | 双 scale 触发时点 spot |
| 入场期权 | 单腿价 → 组合 net (e.g. `-P$465@$22 / +P$445@$13 → 收$8.55`) |
| 平/现ETF / 平/现GC=F | 平仓或 mark-to-market spot |
| 平/现期权 | 单腿出场价 + 出场总值 |
| P&L% / 出场原因 | 真实退出规则触发原因 (e.g. `+50% profit` / `-50% stop` / `expiry`) |

## 用户偏好

- 时区: 新加坡 SGT (UTC+8)
- 信号基于金价，不限定交易品种 (期权/期货/现货)
- 数据每天及时更新, 不用陈旧数据
- 不偷工减料, 不用模拟数据
