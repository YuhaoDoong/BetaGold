# 系统架构

## 三层模型

```
┌─────────────────────────────────────────────────────────────────┐
│ 日线层 (今日预测)                                                │
│   LSTM+Transformer Ensemble → 5 日 High/Low % 预测              │
│   Conformal 校准 → 80% 覆盖区间                                  │
│   Hybrid Band: upper(lag1) + lower(lag1,2,3 平均)                │
│   bp030 / bp090 = 阈值价位 (开窗信号, 不是入场价)                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 盘中层 (盘中信号)                                                │
│   实时盘中价格 + Stoch RSI(14,14,3,3) / MACD(12,26,9) / KDJ(9,3,3) │
│   触发条件: 价格在阈值外侧 AND 至少 N 个规则确认                 │
│   时段过滤: US 期权时段 / 24h 期货 / 自定义                      │
│   写入 data/intraday_signal_log.parquet (持久化)                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 固化层 (历史代表价)                                              │
│   每日所有触发取 worst-of-day:                                   │
│     BUY:  max(price) — 当日最坏可能买入价                        │
│     EXIT: min(price) — 当日最坏可能卖出价                        │
│   持仓管理 / 近期信号 / 回测交易记录全部读这里                   │
└─────────────────────────────────────────────────────────────────┘
```

阈值线 (bp030 蓝 / bp090 红) **只是信号开关，从不作为入场价**。所有历史 marker、入场价、回测记录都来自 log 中的真实盘中触发价；没记录的日子退到收盘 (close) 兜底，并标 `收盘` 让用户看清来源。

## 数据流

```
yfinance / FRED / CBOE
    ↓
config.yaml 启动检测
    ↓ (缺天 → 增量下载)
原始数据 → 全量特征重建 (64 列 GLD / 53 列 SLV)
    ↓
模型在线推理 (加载权重 + 新日期预测 + 追加 OOS)
    ↓
Hybrid Band → Daily 信号 (build_unified_signals)
    ↓
盘中层 (1h K 线 + Stoch RSI/MACD/KDJ 触发)
    ↓
持久 log → Dashboard 渲染
```

### 数据源 (24+ 个因子)
- **yfinance**: GLD / SLV / GC=F / SI=F / DXY / VIX / 原油 / 铜 / 美债 / CNY
- **FRED**: 实际利率 (5y/10y) / 联邦基金利率 / CPI / M2 / 联邦债务 / 贸易加权美元
- **CBOE**: GVZ (黄金 VIX)

启动时自动检测 → 缺天即增量更新 → 全量重建特征 → 在线推理。

## Dashboard 4 种模式

### 1. 盘中信号 (主操作模式)
- **状态栏 5 列**: 今日窗口 / 盘中实时 / 波动率信号 / Regime / RV %tile
- **顶部交易价位** (蓝色框) + **实时价格** (紫色框): 1/3/5/10 分钟自动刷新
- **市场分析**: 事件倒计时 (FOMC/OPEX/NFP/期货交割) + 宏观指标 + Straddle/Short_Vol 预警
- **主图**: Band + 信号 marker (落在 log 真实触发价) + 止盈 marker + 事件竖线 + 5 日预测区间
- **日线 Stoch RSI** (主图正下方, 与主图同 x 轴)
- **盘中 K 线 + 多周期 Stoch RSI** (1h/15m + Squeeze Momentum)
- **今日盘中触发表** (临时, 第二天清零, worst 沉淀到持仓管理)
- **持仓管理**: 方向性 + STRADDLE + SHORT_VOL Iron Condor (含状态/盈亏/早平/止损建议)
- **近期信号** + **统一策略回测** + **波动率交易历史**
- **回测交易记录** (双视图 ETF + 期货价位)

### 2. 今日预测 (分析)
- 顶部 metrics: 伦敦金/银实时价 + ETF 价 (delta)
- 主图: 伦敦金/银 + Band + 5 日预测 + 阈线 + Straddle ★
- 日线 Stoch RSI
- 5 日区间预测表 (双列: ETF $ + 伦敦金/银 $/oz)
- 跨市场价位换算 (ETF + 伦敦现货 + COMEX 期货 + 沪金/沪银)
- OI 微观结构 (Max Pain / Call Wall / Put Wall / PCR / Net GEX)
- **波动率走势图 (近 6 个月)**: RV + 阈值线 + 信号窗口着色
- 前瞻分析: 未来 10 天事件 + Straddle/Short_Vol 信号
- **交易工具推荐**: 按信号类型给出首选 + 备选 + 禁用项
- 期权策略详情

### 3. 历史回看
v1.0 vs v2.2 净值并排 + 近 6 月 / 1 年 / 2 年

### 4. 回测分析
ETF + 期货 双视图 + 4 种入场价模式 + 退出分布

## 项目结构

```
GoldDash/
├── app.py                              # Streamlit Dashboard
├── core/
│   ├── data.py                         # 数据加载 + 全量刷新 + 在线推理
│   ├── training_status.py              # 模型训练状态 + 后台启动
│   ├── dl_range.py                     # LSTM+Transformer Ensemble
│   ├── regime.py                       # Regime 7 因子分类器
│   ├── signals.py                      # v1.0 收盘信号 + Hybrid Band
│   ├── signals_v2.py                   # 方向性 + RV 极值过滤
│   ├── intraday_triggers.py            # 盘中触发模块 (规则可配置)
│   ├── events.py                       # 事件日历 + STRADDLE + SHORT_VOL Iron Condor
│   ├── strategy_selector.py            # 三方竞争统一策略 (vega 兼容矩阵)
│   ├── strategies/                     # 策略子模块 (v3.7 模块化)
│   │   ├── __init__.py
│   │   ├── win_metrics.py              # 各策略胜率定义 (vega/delta-aware)
│   │   └── futures_pnl.py              # 期货 P&L (与期权对比)
│   ├── options.py                      # 期权策略推荐 (Moomoo live + EOD)
│   ├── options_pnl.py                  # 双套盈亏 (金价 + 期权)
│   └── oi_factors.py                   # OI 微观结构修正
├── scripts/
│   ├── setup_data.py                   # 数据下载 + 特征构建 + 训练
│   └── backfill_intraday_signals.py    # 盘中触发回填
├── data/
│   └── intraday_signal_log.parquet     # 盘中触发持久 log
├── docs/
│   ├── ARCHITECTURE.md                 # 本文件
│   ├── MODELS.md                       # 模型架构
│   ├── STRATEGIES.md                   # 策略详解
│   └── EXPERIMENTS.md                  # 实验记录
├── config.yaml
├── requirements.txt
└── README.md                           # 顶层 summary + 链接
```

## 核心设计原则

1. **Regime-aware**: 不同宏观区间用不同工具 (Bull→做多 / Mixed→IC / Bear→空仓)
2. **Vega/delta-aware**: 胜率定义按各策略真实 P&L 模型, 不用一刀切阈值
3. **数据透明**: 所有历史价格 / 信号 marker / 回测记录都标明来源 (盘中×N / 收盘)
4. **参数化**: 触发规则、时段、阈值全部可配置, 不硬编码
5. **持久化优先**: 盘中触发写 parquet log, 历史可回填 + 跨重启恢复
6. **质量 > 数量**: 严格过滤 (RV 极值 / 事件缓冲 / regime) 减少交易频率换更高胜率
