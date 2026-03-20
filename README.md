# GoldDash — 黄金期权交易决策系统

基于宏观因子 + 深度学习的黄金交易决策系统。信号基于金价，适用于 GLD 期权、COMEX 期货、沪金等。

GitHub: https://github.com/YuhaoDoong/BetaGold

## 系统架构

```
日线 LSTM+Attention (20年训练, Conformal 80%覆盖)
    → 5日区间预测 → Hybrid Band
        ↓
Regime 分类器 (7因子) + 事件日历 (FOMC/OPEX/NFP)
        ↓
┌─ 方向性信号: 盘中 H/L 触及 Band → BUY CALL / SELL PUT / EXIT
├─ Straddle 信号: RV压缩 + 临近事件日 → 做多波动率
└─ 统一策略选择器: 按评分择优推荐
        ↓
12h K线止盈 (BandExit > Pullback > MACD > StopLoss)
+ 3%止损 + 连续2笔熔断
        ↓
跨市场换算 (GLD/COMEX/XAU/沪金) + 期权策略推荐 (Moomoo实时/EOD)
+ 双套盈亏统计 (金价收益 + 期权收益)
```

## 统一回测 (2025-09 ~ 2026-03)

| 策略 | 信号数 | 胜率 | 说明 |
|------|--------|------|------|
| SELL PUT | 6 | **83%** | 高IV时卖Put，最稳 |
| BUY CALL | 4 | **75%** | 正常IV买Call |
| STRADDLE | 12 | **67%** | 做多波动率 (事件前) |
| EXIT | 13 | 62% | 退出/平仓 |
| **统一** | **35** | **69%** | 择优策略 |

## Dashboard 功能

### 盘中信号 (主模式)
- **交易价位** (蓝色): 买入/退出价 + COMEX/沪金换算 + OI修正
- **实时价格** (紫色): COMEX/XAU/GLD/沪金, 自动刷新
- **市场分析**: 事件倒计时 (FOMC/OPEX/NFP) + 宏观指标 + Straddle 预警
- **图表**: Band + 信号 + 止盈 + **事件日竖线** + **Straddle星号★**
- **期权策略**: 单腿/价差/卖Put, 盈亏+止损, Moomoo实时报价
- **止盈预测**: 持仓 Pullback 止盈位
- **统一回测**: 方向性+Straddle+EXIT 合并胜率
- **时区**: SGT/ET/UTC 可选

### 今日预测
- 5日区间 + OI微观结构修正 (漏斗形)
- 期权策略推荐 + 宏观分析

### 回测分析
- v1.0 vs v2.2 并排净值曲线
- 近6月/1年/2年

## 信号系统

### 方向性 (盘中 H/L 触发)
| 信号 | 条件 | 退出 |
|------|------|------|
| BUY CALL | Bull + bp(Low)<0.30 + RV≤85% | BandExit > Pullback > MACD > StopLoss(3%) |
| SELL PUT | Bull + bp(Low)<0.30 + RV>85% | 同上 |
| EXIT | bp(High)>0.90 ∪ Regime退出 | — |

### Straddle (评分制, ≥3分触发)
| 条件 | 分数 |
|------|------|
| RV < 20% | +2 |
| RV 下降 > 30% | +1 |
| 距 FOMC ≤ 3天 | +3 |
| 距 NFP ≤ 3天 | +2 |
| 距 OPEX ≤ 3天 | +1 |
| **RV > 25% → 否决** | 成本过高 |

### 统一策略选择
重叠时优先级: **EXIT > Straddle(score≥5) > 方向性 > Straddle(score<5)**

### 止损
- 单笔: 跌超 3% → StopLoss
- 连续: 2笔止损后暂停买入 (bp>0.50 恢复)

## 数据 (每日自动更新)

启动时全量刷新:
1. **yfinance**: GLD/GC=F/DXY/VIX/原油/铜/银/美债/CNY (10个)
2. **FRED**: 实际利率/联邦基金/通胀/M2/贸易加权美元 (8个)
3. **CBOE**: GVZ
4. **特征全量重建**: 64列, 从原始数据计算 (不简化)
5. **模型在线推理**: 加载权重 → 新日期预测 → 追加 OOS

## 模型

### 区间预测 (日线 LSTM+Attention)
- 20年训练 (2004-2026), Walk-Forward, 3种子集成, Conformal 80%
- 预测: 未来5日 High/Low %, RV(10d)归一化
- Quantile Loss (q=0.85/0.15), AdamW

### Regime 分类器 (7因子)
价格动量25% + 联储利率20% + 美元15% + 央行购金15% + 风险10% + 通胀10% + 实际利率5%

### 事件日历
FOMC/OPEX(月度第三周五)/NFP(第一周五) 硬编码2025-2026

## 项目结构

```
GoldDash/
├── app.py                    # Streamlit Dashboard
├── core/
│   ├── data.py               # 数据加载 + 全量刷新 + 在线推理
│   ├── dl_range.py           # LSTM+Attention 模型 (含 save/load)
│   ├── regime.py             # Regime 7因子分类器
│   ├── signals.py            # v1.0 收盘价信号
│   ├── signals_v2.py         # v2.2 盘中信号 + 12h止盈 + 止损
│   ├── events.py             # 事件日历 + Straddle信号
│   ├── strategy_selector.py  # 统一策略选择器
│   ├── options.py            # 期权策略推荐 (Moomoo live + EOD)
│   ├── options_pnl.py        # 双套盈亏 (金价 + 期权实际)
│   └── oi_factors.py         # OI微观结构修正
├── scripts/
│   └── setup_data.py         # 数据下载 + 特征构建 + 模型训练
├── config.yaml
└── requirements.txt
```

## 参数 (core/signals_v2.py)

```python
EXIT_TIMEFRAME = "12h"     # 止盈检测尺度
PULLBACK_GAIN  = 2.0       # Pullback: 涨幅>N%
PULLBACK_DD    = 1.5       # Pullback: 回撤>N%
STOP_LOSS_PCT  = 3.0       # 单笔止损
CONSECUTIVE_STOP = 2       # 连续止损熔断
BUY_BP = 0.30              # 买入阈值
EXIT_BP = 0.90             # 退出阈值
```

## 快速开始

```bash
conda activate gold
streamlit run app.py
```

## 版本

| 版本 | 核心改进 |
|------|---------|
| v1.0 | 日线收盘信号 + OI修正 |
| v2.0~2.1 | 1h模型 (实验) → 正确架构: v1.0 Band + 盘中触发 |
| v2.2 | 12h止盈 + 参数化 |
| v2.3 | 3%止损 + 连续熔断 |
| **v2.4** | **全量数据更新 + 事件日历 + Straddle信号 + 统一策略 + 双套盈亏** |
