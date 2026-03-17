# GoldDash — GLD 期权交易决策系统

基于宏观因子 + 深度学习的黄金期权交易决策系统。信号基于金价，适用于 GLD 期权、COMEX 期货、沪金等。

## 系统架构 (v2.2)

```
日线 LSTM+Attention (20年训练, 3种子集成, Conformal 80%覆盖)
    → 5日 High/Low 区间预测 → Hybrid Band (上界灵敏, 下界平滑)
        ↓
日线 Regime 分类器 (7因子加权, Bull/Bear/Mixed)
        ↓
盘中 H/L 触发: 价格触及 Band 底部 (bp<0.30) 即买入, 不等收盘
        ↓
12h K线止盈: BandExit (bp>0.90) > Pullback (峰值回撤) > MACD弱化
        ↓
跨市场换算: GLD / COMEX / XAU / 沪金, 实时价格 + 信号价位
        ↓
OI 微观结构修正: Max Pain / Call Wall / Put Wall / Net GEX
```

## 回测结果

### v2.2 vs v1.0 (入场: H/L vs 收盘, 止盈: 12h vs 日线)

| 周期 | v1.0 收盘 | v2.2 盘中+12h | 买入持有 |
|------|---------|-------------|---------|
| 近6月 | 5笔 +7.2% | **12笔 83%WR +33.6%** | +15.7% |
| 近1年 | 13笔 +32.2% | **30笔 73%WR +82.3%** | +41.2% |
| 近2年 | 29笔 +62.3% | **64笔 72%WR +163.8%** | — |

### 止盈时间尺度对比 (2025-09 ~ 2026-03, 134天)

| 尺度 | 笔数 | 胜率 | 累计 | 最大亏 | Sharpe |
|------|------|------|------|--------|--------|
| 日线 | 10 | 90% | +20.3% | -5.03% | 0.55 |
| **12h** | **13** | **85%** | **+37.5%** | **-0.96%** | **0.78** |
| 4h | 13 | 85% | +35.9% | -1.50% | 0.70 |
| 1h | 13 | 85% | +30.6% | -1.35% | 0.77 |

12h 最优: Sharpe 0.78, 最大亏仅 -0.96%, 比日线快止损, 比 1h 更有耐心。

## Dashboard 功能

### 盘中信号 v2.2 (主模式)
- **交易价位** (蓝色背景): 买入/退出价 + COMEX/沪金换算
- **实时价格** (紫色背景): COMEX/XAU/GLD/沪金, 自动刷新 (1-10分钟可配)
- **止盈预测**: 根据入场价+峰值计算 Pullback 止盈位
- **信号图表**: Band + 买入/退出信号 + 止盈标注 (淡色)
- **信号历史表**: bp(L/C/H) + 买入价/退出价
- **12h 止盈回测**: 近期交易明细
- **时区选择**: SGT/ET/UTC

### 今日预测 v1.0
- 5日区间预测 + OI 微观结构修正 (漏斗形)
- 期权策略推荐 (单腿/价差, IV推荐)
- 跨市场换算 + OI 详情

### 回测分析
- v1.0 vs v2.2 并排净值曲线 (近6月/1年/2年)
- 统计对比表

### 历史回看
- 自定义时间范围, 快速选择 (2月~全部)
- 长周期 (>120天) 自动隐藏信号, 突出 Regime

## 信号系统

### 入场 (基于金价, 不限品种)
| 信号 | 条件 |
|------|------|
| BUY CALL | Bull + 盘中Low触及bp<0.30 + RV≤85% |
| SELL PUT | Bull + 盘中Low触及bp<0.30 + RV>85% |

### 退出 (优先级, 12h K线检测)
| 类型 | 条件 |
|------|------|
| BandExit | 盘中High触及bp>0.90 |
| Pullback | 从峰值涨>2%后回撤>1.5% |
| MACD弱化 | MACD柱由正转负 + 盈利>1% |
| Timeout | 持仓>10天 |

### 参数 (core/signals_v2.py 顶部可配置)
```python
EXIT_TIMEFRAME = "12h"    # 止盈尺度 (1h/2h/4h/8h/12h)
PULLBACK_GAIN  = 2.0      # Pullback: 涨幅>N%
PULLBACK_DD    = 1.5      # Pullback: 回撤>N%
MACD_MIN_GAIN  = 1.0      # MACD: 最低盈利>N%
MAX_HOLD_DAYS  = 10       # 最大持仓天数
BUY_BP         = 0.30     # 买入bp阈值
EXIT_BP        = 0.90     # 退出bp阈值
```

## 模型

### 区间预测 (v1.0 日线 LSTM+Attention)
```
Input (seq_len=20, ~50维) → BatchNorm → BiLSTM(64) → Attention → 双头输出
  Upper: Linear→ReLU→Dropout→Softplus  (5日最高价%)
  Lower: Linear→ReLU→Dropout→Linear    (5日最低价%)
```
- 20年日线数据 (2004-2026), Walk-Forward, 3种子集成, Conformal 80%覆盖
- Quantile Loss (q_upper=0.85, q_lower=0.15), AdamW, RV(10d)归一化

### Hybrid Band
- 上界 = close(t-1) × (1 + pred_upper%(t-1)) — Lag1, 灵敏
- 下界 = avg[close(t-k) × (1 + pred_lower%(t-k))] — Lag1-3均值, 平滑
- bp = (price - lower) / (upper - lower)

### Regime 分类器 (7因子加权)
| 因子 | 权重 |
|------|------|
| 价格动量 | 25% |
| 联储利率方向 | 20% |
| 美元趋势 | 15% |
| 央行购金 | 15% |
| 风险情绪(GVZ) | 10% |
| 通胀 | 10% |
| 实际利率 | 5% |

Bull>+0.2, Bear<-0.2, EWM(60)平滑, 最小持续20天。

## OI 微观结构修正
| 因子 | 作用 |
|------|------|
| Max Pain | Pin效应, 引力拉回 (15%×到期因子) |
| Call Wall | 上方阻力, 压制上界 (30% blend) |
| Put Wall | 下方支撑, 抬升下界 (20% blend) |
| Net GEX | 正Gamma压缩, 负Gamma放大 |

DTE 从 strike_time 实时重算 (SGT), 使用 OI 最集中的主导到期日。

## 项目结构

```
GoldDash/
├── app.py                  # Streamlit Dashboard (4模式)
├── core/
│   ├── data.py             # 数据加载 + yfinance自动刷新
│   ├── dl_range.py         # v1.0 日线 RangeLSTM
│   ├── dl_range_1h.py      # v2.0 1h RangeLSTM (实验, 保留)
│   ├── features_1h.py      # 1h 多粒度特征工程
│   ├── regime.py           # Regime 7因子分类器
│   ├── signals.py          # v1.0 信号 (收盘价)
│   ├── signals_1h.py       # v2.0 1h 信号 (实验, 保留)
│   ├── signals_v2.py       # v2.2 信号 (H/L入场+12h止盈) ← 当前主力
│   ├── options.py          # 期权策略推荐
│   └── oi_factors.py       # OI 微观结构修正
├── scripts/
│   ├── setup_data.py       # 日线数据下载+特征构建+模型训练
│   └── train_1h_model.py   # 1h 模型训练
├── config.yaml
├── requirements.txt
└── README.md
```

## 快速开始

```bash
conda activate gold
streamlit run app.py
```

## 依赖

```
streamlit, streamlit-autorefresh   # Dashboard + 自动刷新
pandas, numpy, matplotlib, pyyaml  # 核心
yfinance, fredapi, requests        # 数据下载
torch, scikit-learn                # 模型训练
```

---

## 版本历史

| 版本 | 日期 | 核心改进 |
|------|------|---------|
| v1.0 | 2026-03-16 | 日线收盘信号 + OI修正 + 期权策略 Dashboard |
| v2.0 | 2026-03-17 | 1h 模型 (实验, Band校准不足, 已废弃) |
| v2.1 | 2026-03-17 | 正确架构: v1.0 Band + 盘中H/L触发 |
| **v2.2** | **2026-03-17** | **12h止盈 + 参数化 + 回测对比 + Dashboard统一** |

详细版本日志见 git log。
