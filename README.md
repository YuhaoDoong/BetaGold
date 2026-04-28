# GoldDash — 贵金属交易决策系统

基于宏观因子 + 深度学习的贵金属交易决策系统。支持黄金 (GLD) 和白银 (SLV) 切换。
日线模型给"开窗阈值"，盘中层在阈值外侧由 Stoch RSI / MACD / KDJ 找真实入场点；
图表与价位以伦敦金/伦敦银 (USD/oz) 显示，跨市场（GLD ETF、COMEX、伦敦现货、沪金）一并换算。

GitHub: https://github.com/YuhaoDoong/BetaGold

---

## 系统三层模型

```
日线层 (今日预测)         模型预测 → 5日区间 + bp030 / bp090 阈值
        ↓                                    "开窗阈值", 不是入场价
盘中层 (盘中信号)         实时盘中: 价格在阈值外侧
                          + Stoch RSI / MACD / KDJ 确认
                          → 一天可触发多次, 每次记录真实价 + 时间
        ↓                 写入 data/intraday_signal_log.parquet
固化层 (历史代表价)       每日多触发取最差 (买:max / 卖:min)
                          → 持仓管理 / 近期信号 / 回测全部读这里
```

阈值线 (bp030 蓝线 / bp090 红线) 只是信号开关，从不作为入场价。
所有历史 marker、入场价、回测交易记录都来自 log 中的真实盘中触发价；
没记录的日子退到 close 兜底，并标 `收盘` 让用户看清来源。

---

## 三种模式

### 盘中信号 (主模式 — 操作)
- **顶部交易价位** (蓝色框) + **实时价格** (紫色框)：自动刷新 1/3/5/10 分钟
- **市场分析**：事件倒计时 (FOMC/OPEX/NFP/期货交割) + 宏观指标 + Straddle 预警
- **主图 (伦敦金/银 价位)**：Band + 信号 marker (落在 log 真实触发价) + 止盈 marker + 事件竖线 + Straddle ★ + 5 日预测区间 (含 OI 修正漏斗)
- **日线 Stoch RSI** (主图正下方，与主图同 x 轴范围对齐)
- **盘中 K线 + 多周期 Stoch RSI** 单图：
  - K 线 (默认 50 根，可切 5m/15m/30m/1h)
  - 1h Stoch RSI 子图
  - 15m Stoch RSI 子图
  - Squeeze Momentum
  - 散点：盘中触发记录 (绿 ▲ BUY / 红 ▼ EXIT) 用 `np.interp` 投影到 K 线索引 → 与 K 线/Stoch/Squeeze 完全对齐
- **今日盘中触发表**：当日所有触发的 (时间, 方向, 价格, 阈值, 周期, 命中规则)，含汇总 `今日已触发 N 次 | BUY 最差价 $X (M 次) | EXIT 最差价 $Y (P 次)`
- **持仓管理**：方向性 (Pullback/BandExit) + Straddle (波动/事件到期)；入场列显示 `盘中×N` 或 `收盘`
- **近期信号**：最近 10 个，入场列同上
- **统一策略回测**：方向性 + Straddle + 退出合并胜率
- **回测交易记录**：双视图 — `入场 GLD/SLV` + `入场 伦敦金/银` + `入场源`，`出场 GLD/SLV` + `出场 伦敦金/银` + `出场源`，`ETF 收益` + `期货 收益`，附汇总 (胜率/累计/平均)

### 今日预测 (分析)
- **顶部 metrics**：headline 改为伦敦金/银实时价 (m1)，ETF 价作 delta 副信息
- **主图**：伦敦金/银 价位 + Band + 5 日预测区间 + bp030/bp090 阈线 + Straddle ★
- **日线 Stoch RSI** (主图下方，单子图，无重复价格图，与主图同范围)
- **5 日区间预测表**：双列 (`ETF $` + `伦敦金/银 $/oz`) + OI 修正后区间
- **下一交易日阈值**：bp030 (买入) + bp090 (退出)，含 OI 修正
- **跨市场价位换算**：GLD/SLV ETF + 伦敦现货 + COMEX 期货 + 沪金/沪银 (实时 GC=F/SI=F + USD/CNY)
- **OI 微观结构**：Max Pain / Call Wall / Put Wall / PCR / 主导到期 / Net GEX
- **前瞻分析**：未来 10 天关键日程 + Straddle 信号
- **宏观环境**：RV / VIX / DXY / 实际利率
- **期权策略推荐**：当日最优 (根据信号 + IV 自动选)

### 回测分析
- v1.0 vs v2.2 并排净值 + 近 6 月 / 1 年 / 2 年
- 回测交易记录详表

---

## 盘中触发模块 (`core/intraday_triggers.py`)

完全参数化，所有规则、时间尺度、交易时段都可调。

### 内置规则

```python
RULES_BUY = (
  "stoch_rsi_cross_up_oversold",   # K 上穿 oversold 阈
  "stoch_rsi_in_oversold",         # K 当前 < oversold (软条件)
  "macd_bullish_cross",            # MACD line 上穿 signal line
  "macd_hist_turn_up",             # MACD 柱由减弱转加强
  "kdj_j_cross_up_oversold",       # KDJ J 上穿 oversold
  "kdj_k_cross_d_up",              # KDJ K 上穿 D
)
RULES_EXIT = (
  "stoch_rsi_cross_down_overbought",
  "stoch_rsi_in_overbought",
  "macd_bearish_cross",
  "macd_hist_turn_down",
  "kdj_j_cross_down_overbought",
  "kdj_k_cross_d_down",
)
```

### 触发判定

一条触发同时满足：
1. **价格条件**：`close < bp030` (BUY) 或 `close > bp090` (EXIT)
2. **规则确认**：`confirm_mode` 控制 — `"any"` (任一规则)、`"all"` (全部)、整数 k (k_of_n)
3. **时段过滤**：`session_utc` 控制 — `None` 全天，或 `US_OPTIONS_SESSION_UTC`，或 `FUTURES_SESSION_24H`，或自定义 `(time, time)`

### 持久化

- 文件：`data/intraday_signal_log.parquet`
- 列：`date, trigger_time, price, side, asset, timeframe, rules, bp_threshold, n_confirms`
- 去重 key：`(date, trigger_time, side, asset, timeframe, rules)`
- API：`load_log(path)` / `upsert_log(new, path)` / `worst_of_day(triggers, side)`

### 历史回填脚本

```bash
# 默认规则 (Stoch RSI 上穿 OR MACD 金叉, confirm=any, 24h 全天)
python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60 --source etf
python scripts/backfill_intraday_signals.py --asset SLV --timeframe 60 --source etf

# 自定义规则 + 时段
python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60 \
    --buy-rules stoch_rsi_cross_up_oversold kdj_k_cross_d_up \
    --exit-rules stoch_rsi_cross_down_overbought \
    --confirm 2 \
    --session us_options
```

### 期权 vs 期货时段

- **期货** (BUY/EXIT 方向性)：`FUTURES_SESSION_24H`，CME Globex 几乎 24h
- **期权** (Straddle / 美股期权策略)：`US_OPTIONS_SESSION_UTC` = 14:30–21:00 UTC (09:30–16:00 ET)
  - 期权策略推荐只在美股开盘窗口执行
  - 美股闭市时只显示期货方向性信号

---

## 信号系统

### 完整策略矩阵 (v3.6+)

| RV %tile | 方向性 | 波动率 | 推荐期权策略 |
|----------|--------|--------|--------------|
| < 50% (低/中性) | ✅ **BUY CALL** | 做多 (RV<20%) | 买 ATM Call/Put 或 Long Straddle |
| 50-85% (温水) | ❌ 屏蔽 | ✅ **做空 IC** | **Iron Condor (16Δ/5Δ)** |
| > 85% (恐慌) | ✅ **SELL PUT** | 做多 (临界) | 卖 OTM Put / Long Straddle |

**关键洞察 (近 5 年回测验证)：**
- RV 温水区 (50-85%) 是方向性信号最差区间 — 趋势不明且 IV 衰减不够极端
- 排除温水区后: BUY CALL 胜率 81% → **88%**, Sharpe 0.53 → **0.61**
- BUY CALL 数量保留合理 (5y 27 → 17 笔)，质量大幅提升

### 方向性 (盘中 H/L 触发 + RV 极值过滤)

| 信号 | 条件 | 退出优先级 |
|------|------|------|
| BUY CALL | Bull + bp(Low)<0.30 + **RV %tile < 0.50** | StopLoss(3%) > BandExit > Pullback > Timeout |
| SELL PUT | Bull + bp(Low)<0.30 + **RV %tile > 0.85** | 同上 |
| EXIT | bp(High)>0.90 ∪ Regime 退出 Bull | — |

### 做多波动率 — Long Straddle / Strangle (评分 ≥3)

| 条件 | 分数 |
|------|------|
| RV < 20% | +2 |
| RV 下降 > 30% | +1 |
| 距 FOMC ≤ 3 天 | +3 |
| 距 NFP ≤ 3 天 | +2 |
| 距 OPEX ≤ 3 天 | +1 |
| **RV > 25% → 否决** | 成本过高 |

### 做空波动率 — Iron Condor 16Δ/5Δ (评分 ≥7, 严格时机)

| 条件 | 分数 |
|------|------|
| RV %tile ∈ [0.35, 0.65] (中位窄带) | +2 |
| RV ∈ [13%, 28%] | +1 |
| RV 3 日均值 < 10 日均值 (趋势回落) | +2 |
| 距 FOMC > 15 天 | +2 |
| 距 NFP > 7 天 | +1 |
| 距 OPEX > 5 天 | +1 |
| Bull / Range regime | +1 |
| 近 5 日日均振幅 < 1.5% | +1 |

**硬门槛** (任一命中 → 不触发)：RV %tile > 0.75 / < 0.25, RV 越界, 距 FOMC ≤ 10 天, 距 NFP ≤ 7 天, Bear regime, 持仓窗口内有任何主要事件

**结构**：卖 1.6σ 短腿 (Put + Call) + 买 3σ 长翼，净 credit ≈ 1σ premium × 0.4，最大亏损锁定在翼宽 1.4σ。回测胜率 89%，Sharpe 0.94-1.21。

### 统一策略选择 (3 路竞争)

```
EXIT 优先 → 波动率与方向性都强 → MIXED (同时推荐)
         → 波动率 score ≥ 5 → 走波动率
         → 仅方向性 → 走方向性
         → 仅波动率 → 走波动率
```

vol 取做多/做空两者较强者 (RV 高 → SHORT_VOL, RV 低 → STRADDLE), 与方向性独立竞争。

### 止损
- 单笔：跌超 3% → StopLoss
- 连续：2 笔止损后暂停买入 (bp > 0.50 恢复)
- Pullback: 持仓期峰值涨幅 > 2% 且回撤 ≥ 1.5%

---

## 价位换算

主图 / 表格 / 标注一律用伦敦金/伦敦银 (USD/oz)，不用 ETF 单位。

- **比例计算**：实时 GC=F (黄金) / SI=F (白银) ÷ ETF last_close
- **兜底**：近 60 日 期货/ETF 均比 (`gc_gld_ratio`)
- **白银修复**：SLV 模式独立计算 silver_futures/SLV 比例 (~1.10)，不再错用 GC/GLD 比例
- **跨市场表**：ETF + 伦敦现货 + COMEX + 沪金 (USD/CNY) 同行展示，方便做单时直接看对应品种价格

---

## 模型

### 区间预测 (日线 LSTM+Transformer Ensemble)
- 20 年训练 (2004–2026)，Walk-Forward 20 折，Conformal 80%
- **双架构集成** (v3.2+)：LSTM + Transformer 输出平均 → 校准
  - 单 LSTM：       cov=71.3% width=6.96% tightness=0.102
  - 单 Transformer：cov=69.8% width=6.74% tightness=0.104
  - **Ensemble**：  cov=71.1% width=**6.50%** tightness=**0.109** (宽度最窄 +7%)
- **Qlib Alpha158 因子** (v3.2+)：KBAR / BETA / RSQR / QTLU / CORR / CNTP / SUMP 等 110 项
- 预测：未来 5 日 High/Low %，RV(10d) 归一化，Quantile Loss (q=0.85/0.15)，AdamW

### 训练频率
- **每周训练一次**。Dashboard 侧边栏会在模型 > 7 天未训练时显示黄色警告
- 侧边栏 **模型训练** 面板：
  - GLD / SLV 各自实时训练状态 + 已运行时长
  - **一键启动训练** 按钮 (后台 subprocess，不阻塞页面)
  - 训练日志滚动查看 + 手动停止

### Regime 分类器 (7 因子)
价格动量 25% + 联储利率 20% + 美元 15% + 央行购金 15% + 风险 10% + 通胀 10% + 实际利率 5%

### 事件日历
FOMC / OPEX (月度第三周五) / NFP (第一周五) / 期货交割日，硬编码 2025–2026

---

## 数据

启动时自动检测过期，缺天即增量下载：

1. **yfinance**：GLD / SLV / GC=F / SI=F / DXY / VIX / 原油 / 铜 / 美债 / CNY (~12 个)
2. **FRED**：实际利率 / 联邦基金 / 通胀 / M2 / 贸易加权美元 (8 个)
3. **CBOE**：GVZ
4. **特征全量重建**：64 列 (GLD) / 53 列 (SLV)，从原始数据计算 (不简化)
5. **模型在线推理**：加载权重 → 新日期预测 → 追加 OOS

---

## 项目结构

```
GoldDash/
├── app.py                              # Streamlit Dashboard
├── core/
│   ├── data.py                         # 数据加载 + 全量刷新 + 在线推理
│   ├── training_status.py              # 模型训练状态 + 后台启动 (GLD/SLV)
│   ├── dl_range.py                     # LSTM+Attention 模型
│   ├── regime.py                       # Regime 7 因子分类器
│   ├── signals.py                      # v1.0 收盘价信号 + Hybrid Band
│   ├── signals_v2.py                   # 盘中信号 + 12h 止盈 + 止损 + log 接入
│   ├── intraday_triggers.py            # 盘中触发模块 (NEW)
│   ├── events.py                       # 事件日历 + Straddle 信号
│   ├── strategy_selector.py            # 统一策略选择器
│   ├── options.py                      # 期权策略推荐 (Moomoo live + EOD)
│   ├── options_pnl.py                  # 双套盈亏 (金价 + 期权实际)
│   └── oi_factors.py                   # OI 微观结构修正
├── scripts/
│   ├── setup_data.py                   # 数据下载 + 特征构建 + 模型训练
│   └── backfill_intraday_signals.py    # 盘中触发回填 (NEW)
├── data/
│   └── intraday_signal_log.parquet     # 盘中触发持久 log (NEW)
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 关键参数 (`core/signals_v2.py`)

```python
BUY_BP           = 0.30      # 买入阈值
EXIT_BP          = 0.90      # 退出阈值
STOP_LOSS_PCT    = 3.0       # 单笔止损
PULLBACK_GAIN    = 2.0       # Pullback: 涨幅 > N%
PULLBACK_DD      = 1.5       # Pullback: 回撤 > N%
CONSECUTIVE_STOP = 2         # 连续止损熔断
MAX_HOLD_DAYS    = 30        # 安全帽

# RV 极值过滤 (v3.6.1, 默认开启)
RV_FILTER_LOW    = 0.50      # < 此值 → BUY CALL (vol 偏低/中性)
RV_FILTER_HIGH   = 0.85      # > 此值 → SELL PUT (vol 极端高位)
RV_FILTER_ENABLED = True     # 关闭恢复 v1.0 行为 (无中位屏蔽)
```

## 做空波动率参数 (`core/events.py`)

```python
SHORT_VOL_RV_PCTILE_LO    = 0.35    # RV %tile 中位窄带下限
SHORT_VOL_RV_PCTILE_HI    = 0.65    # RV %tile 中位窄带上限
SHORT_VOL_RV_ABS_MIN      = 13.0    # RV 绝对下限
SHORT_VOL_RV_ABS_MAX      = 28.0    # RV 绝对上限
SHORT_VOL_FOMC_BUFFER     = 10      # 距 FOMC 必须 > N 天
SHORT_VOL_NFP_BUFFER      = 7
SHORT_VOL_SCORE_TRIGGER   = 7       # 触发分数门槛
SHORT_VOL_STRIKE_SIGMA    = 1.6     # IC 短腿 ≈ 16Δ
SHORT_VOL_WING_SIGMA      = 3.0     # IC 长翼 ≈ 5Δ
SHORT_VOL_PREMIUM_RATIO   = 0.40    # 净 credit ≈ 1σ × 0.4
```

回测可传入：

```python
trades = run_backtest(
    ..., 
    entry_log_lookup=worst_buy_lookup,   # 入场价改读 log 代表价
    exit_log_lookup=worst_exit_lookup,   # BandExit 退出价同上
)
```

每条 trade dict 含 `entry_source` / `exit_source` (`"盘中×N"` / `"阈值"` / `"收盘"`)，便于审计入场/退出价的来源。

---

## 快速开始

```bash
conda activate gold

# 启动 Dashboard
streamlit run app.py

# 历史回填盘中触发 (首次或规则变更后)
python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60
python scripts/backfill_intraday_signals.py --asset SLV --timeframe 60
```

依赖与一键安装：见 `requirements.txt` (PyTorch / pandas / streamlit / yfinance / qlib 等)。

---

## P&L 统计 (期货 vs 期权)

- **期货 / 期货 ETF**：按价差百分比直接统计，已实装 (回测交易记录表 `期货 收益` 列)
- **期权 (历史回测)**：受隐含波动率 (IV) 影响，与期货价差不等同；当前简化为价差 % 占位
- **期权 (实时)**：等待 Moomoo API 接通后采集真实期权报价，按真实期权 P&L 统计
  - 方向对了也可能因 IV 跌而亏钱，必须用真实期权价

---

## 版本

| 版本 | 核心改进 |
|------|---------|
| v1.0 | 日线收盘信号 + OI 修正 |
| v2.0~2.1 | 1h 模型 (实验) → 正确架构: v1.0 Band + 盘中触发 |
| v2.2 | 12h 止盈 + 参数化 |
| v2.3 | 3% 止损 + 连续熔断 |
| v2.4 | 全量数据更新 + 事件日历 + Straddle 信号 + 统一策略 + 双套盈亏 |
| v2.5 | 智能去重 + 前瞻分析 + 最优策略推荐 + Straddle 持仓管理 |
| v2.6 | 15m K 线 + Stoch RSI 入场窗口 + Squeeze + 实时 GC=F |
| v2.7 | 1h "反转+BB下轨" 入场窗口 (61% WR, 全间隔回测验证) |
| v3.0 | 白银 SLV 模型 + 资产切换 (GLD/SLV) |
| v3.1 | 币安行情 + 区间修复 + 看多看空信号 + 白银增强 |
| v3.2 | Qlib Alpha158 因子 + LSTM+Transformer Ensemble + Dash 训练按钮 |
| v3.3 | 多周期 Stoch RSI (日线 / 1h / 15m, 全部 x 轴对齐, 同一公式) |
| v3.4 | SLV 价位 / 比例 / 实时期货独立修正 + 主图改伦敦金/银 + 5 日区间双视图 |
| v3.5 | 盘中触发模块: 参数化规则 (Stoch RSI / MACD / KDJ) + 持久 log + 历史回填 + 主图/持仓/近期信号/回测全部读 log 真实触发价 + Straddle ★ 标记 + ETF/期货双视图汇总 |
| v3.6 | 完整策略矩阵: 做空波动率 Iron Condor 16Δ/5Δ (89% 胜率) + 方向性 RV 极值过滤 + 状态栏 5 列 (新增波动率信号) + 今日预测 RV 走势图 + 三方策略竞争 |
| **v3.6.1** | **RV 阈值调优 (0.25/0.75 → 0.50/0.85): BUY CALL 胜率 81% → 88%, Sharpe 0.53 → 0.61, 屏蔽真正的温水区 50-85% 而非误伤低 RV 的 BUY CALL 机会; 修复 rv_filter=False 时 SELL PUT 误判 bug** |
