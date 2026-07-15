# DL Range 区间预测 + 交易信号 详细分析

## 1. 模型架构

### LSTM + Attention + Quantile Loss

```
输入: 44 特征 × 20 天序列
  → BatchNorm → LSTM(64, 2层) → Attention加权
  → upper_head (Softplus, 保证正) → 预测 upper sigma multiplier
  → lower_head → 预测 lower sigma multiplier
```

**关键设计:**
- **RV 归一化**: 目标 = actual_pct / rv_scale, 模型预测 sigma multiplier, 预测时乘回 rv
  - rv_scale = rolling(10).std() × √5 × 100 (10日窗口, 匹配5日预测周期)
  - 原用 RV(20d), 2026-03 改为 RV(10d): 波动恢复更快 (2/27 区间宽度 18%→12%)
  - RV(5d) 也测试过: 模型覆盖率 66% (vs 10d 74%), 区间更窄但信号更激进, 不够稳健
  - 没有归一化模型学不到量级 (不同波动率环境差异太大)
- **Quantile Loss (Pinball)**: 上界用 q_upper=0.85, 下界用 q_lower=0.15, 不对称惩罚
- **独立 Conformal 校准**: cal 集独立于 train/val, 找最小 margin 使覆盖率达标
  - 不能用 val 集做校准 (early stopping 已用 val, 会泄露)
- **3 种子集成**: 不同随机种子训练3个模型, 取平均, 提高稳定性

### 预测目标

```
upper = max(High[t+1], ..., High[t+5]) / Close[t] - 1  (未来5天最高涨幅%)
lower = min(Low[t+1], ..., Low[t+5]) / Close[t] - 1   (未来5天最低跌幅%)
```

## 2. Walk-Forward 评估

数据拆分 (每 fold):
- train: expanding window (≥1260天)
- val: 最后252天 (early stopping)
- cal: 126天独立集 (conformal校准)
- test: 126天 OOS (最后一个fold允许≥60天)

### 配置对比 (20 folds, 2016-2026)

| 配置 | 分位数 | 校准目标 | Coverage | Width | Ratio | Cov_std |
|------|--------|---------|----------|-------|-------|---------|
| **A: q85/15** | **85/15** | **80%** | **73%** | **6.6%** | **2.3x** | **10.6%** |
| B: q90/10 | 90/10 | 80% | 74% | 6.7% | 2.4x | 9.8% |
| C: q85/15 | 85/15 | 70% | 63% | 5.6% | 2.0x | 11.3% |
| RV-based (对照) | — | — | 99% | 16.0% | 5.7x | — |

Config A 当选 (紧凑度 × 稳定性最优)

> 上述为 RV(10d) 版本结果 (2026-03 重训)

### 按波动率分段

| 波动率分段 | Coverage | Width | 实际宽度 | Ratio |
|-----------|----------|-------|---------|-------|
| 低波动 Q1 | 89% | 5.6% | 1.4% | 3.9x |
| 中波动 Q2-3 | 77% | 6.4% | 2.5% | 2.6x |
| **高波动 Q4** | **49%** | **7.8%** | **4.8%** | **1.6x** |

高波动期覆盖率低 = 尾部突破频繁 → 这些正好是交易机会 (极端价格回归)

## 3. Hybrid Band — 上下界不同重算方式

### 设计思路

DL Range 在 t 日预测的是 "未来5天区间", 覆盖 t+1 ~ t+5。为构建 **当日可用** 的价格区间, 需要用历史预测。关键发现: **上界和下界适合不同的平滑程度**。

```python
# 上界: Daily (仅 t-1) — 灵敏, 及时捕捉顶部
upper_band = close[t-1] × (1 + pred_upper_pct[t-1] / 100)

# 下界: LagAvg (t-1, t-2, t-3) — 平滑, 过滤买入噪声
lower_band = mean(close[t-lag] × (1 + pred_lower_pct[t-lag] / 100) for lag in 1,2,3)

bp = (close - lower_band) / (upper_band - lower_band)
```

### 7种方案对比

| 方案 | 上界 | 下界 | Buy 5d WR | Buy TP+1% | Exit N | Exit 1d NR | Exit 5d NR |
|------|------|------|-----------|-----------|--------|-----------|-----------|
| Daily | t-1 | t-1 | 69% | 69% | 61 | **89%** | 66% |
| Lag2 | t-1,2 | t-1,2 | 71% | 73% | 72 | 81% | 68% |
| LagAvg | t-1,2,3 | t-1,2,3 | 71% | 73% | 94 | 82% | 64% |
| Lag4 | t-1~4 | t-1~4 | 71% | 73% | 117 | 82% | 64% |
| **Hybrid D/L3** | **t-1** | **t-1,2,3** | **72%** | **74%** | **62** | **85%** | **68%** |
| Hybrid D/L2 | t-1 | t-1,2 | 71% | 71% | 62 | 87% | 68% |
| Hybrid L2/L3 | t-1,2 | t-1,2,3 | 71% | 74% | 77 | 79% | 69% |

**Hybrid D/L3 最优**: 买入端最高WR (72%), 平仓端精准 (62信号, 85% 1d NR)

### 为什么上下界需要不同平滑?

- **下界 → 买入决策**: 需要平滑过滤噪声。单日预测波动大, 假突破会触发错误买入。LagAvg取3天均值, 降低噪声
- **上界 → 平仓决策**: 需要灵敏响应。价格触顶后延迟1天平仓可能错过最佳卖点。Daily直接用最新预测

### Band 配置系统测试 (2025-09 ~ 2026-03, Band+Pullback 退出)

固定 Upper=Daily, 对比 Lower band:

| 配置 | 交易 | Avg | WR | Hold | Sharpe |
|------|------|-----|-----|------|--------|
| **U=Daily L=LagAvg [选定]** | **8** | **+2.5%** | **88%** | **6.5d** | **0.79** |
| U=Daily L=Daily | 7 | +2.4% | 86% | 7.9d | 0.77 |

LagAvg 下界优势: 大跌后保持较高基线 → 识别超跌买入 (如 2/2 bp=-0.11 → +6.3%)。Daily 下界跟踪暴跌立即下移, 错过机会。

固定 Lower=LagAvg, 对比 Upper band:

| 配置 | 交易 | Exit信号 | 1/20-1/28 Exit | 问题 |
|------|------|---------|---------------|------|
| **U=Daily [选定]** | **8** | **7** | **3天** | — |
| U=LagAvg | 9 | **17** | **8天连续** | 大涨时信号失效 |

**Upper=LagAvg 致命缺陷**: 1月20-28日大涨期间, LagAvg上界滞后于价格 (close超出上界+2.6%), bp持续>1.0, 连续8天发退出信号 — 这是噪声不是信号。Daily上界用昨日close快速跟涨, 只在真正拐点触发。

### RV 归一化窗口对比 (U=Daily L=LagAvg)

| RV窗口 | 交易 | Avg | WR | Sharpe | 模型Coverage | 特点 |
|--------|------|-----|-----|--------|-------------|------|
| RV(20d) | — | — | — | — | ~74% | 恢复太慢, 已弃用 |
| **RV(10d) [选定]** | **8** | **+2.5%** | **88%** | **0.79** | **74%** | 平衡: 灵敏 + 稳健 |
| RV(5d) | 9 | +2.8% | 89% | 0.90 | 66% | 更多信号但覆盖率低 |

RV(5d) 在 6 个月样本上 Sharpe 略高, 但:
- 模型覆盖率 66% vs 74% — 区间预测精度下降
- 区间更窄 → bp 更容易触发 → 信号更多但边缘质量差 (如 9/17 bp=0.264 刚好卡线)
- RV(5d) 百分位波动更大 → Buy Call/Sell Put 分类不稳定 (如 11/14 同一天不同判断)

综合考虑模型稳健性, 选择 RV(10d)。

### Band Position (bp)

```
bp = (close - lower_band) / (upper_band - lower_band)
```
- bp ≈ 0: 价格在区间下沿 (买入区)
- bp ≈ 1: 价格在区间上沿 (平仓区)
- bp < 0: 突破下沿 (超跌)
- bp > 1: 突破上沿 (超涨)

## 4. 区间校准验证 (E1)

按 Regime / RV / GVZ 分层检验覆盖率:

| 维度 | 分层 | Coverage | PredWidth | 发现 |
|------|------|----------|-----------|------|
| Regime | Bull | 64.5% | 5.90% | 上界突破18.5% (趋势超预期) |
| | Mixed | 60.9% | 4.83% | 平衡 |
| | **Bear** | **49.0%** | 5.21% | **下界突破38.9%** (跌势超预期) |
| RV分位 | Q1低 | **49.6%** | 3.88% | 低波动区间过窄 |
| | Q4高 | **75.3%** | 6.76% | 高波动覆盖好 |

关键发现: Bull上界突破多 + Bear下界突破多 → **Regime有效区分市场方向**

## 5. bp 分桶单调性 (E2)

### Bull Only: Spearman ρ=-0.893, p=0.007 — bp越低收益越高

| bp桶 | N | 5d胜率 | 10d胜率 | 5d MAE |
|------|---|--------|---------|--------|
| <0.10 | 65 | 70.8% | 69.2% | -1.24% |
| 0.10-0.20 | 50 | 70.0% | **78.0%** | -0.82% |
| 0.20-0.30 | 88 | 64.8% | 69.3% | -1.15% |
| 0.30-0.40 | 136 | 68.4% | 61.8% | -1.07% |
| 0.60-0.80 | 174 | **48.9%** | 53.4% | -1.51% |

### Mixed 完全反向: bp低位是陷阱

| bp桶 | N | 5d胜率 | 5d均值 |
|------|---|--------|--------|
| <0.10 | 116 | **44.8%** | **-0.25%** |
| 0.60-0.80 | 278 | **60.4%** | +0.37% |

→ Bull-only过滤不可妥协

## 6. 信号定义 (期权视角)

### 水平触发 + 期权类型区分

```python
rv_high = rv_pctile > 0.85
buy_zone = is_bull & (bp < 0.30)

buy_call = buy_zone & (~rv_high)   # 正常RV → buy call
sell_put = buy_zone & rv_high      # 高RV → sell put (高IV收premium)

bull_exit = is_bull.shift(1) & (~is_bull)  # Regime退出Bull
exit = (bp > 0.90) | bull_exit             # 平仓
```

**水平触发**: 每天判定条件, 满足即为信号。期权交易每天都是潜在入场点。
**RV>85% 不作为sell信号**: 改为区分买入类型 (call→put)。高IV时卖put双重获利。

### 三类信号效果

| 信号 | N | 次/年 | 5d WR | 10d WR | 5d TP+2% | 逻辑 |
|------|---|------|-------|--------|---------|------|
| Buy Call | 123 | 12.4 | 72% | 75% | 37% | Bull + bp<0.30 + RV≤85% |
| Sell Put | 66 | 6.6 | 73% | 79% | **47%** | Bull + bp<0.30 + RV>85% |
| Exit | 62 | 6.2 | — | — | — | bp>0.90 ∪ Regime退出 |

### 止盈命中率 (用 High 价格)

| 窗口 | Buy Call ≥+2% | Sell Put ≥+2% | 合计 ≥+2% |
|------|--------------|--------------|-----------|
| 3d | 23% | 36% | 28% |
| 5d | 37% | **47%** | 40% |
| 10d | 54% | **76%** | 61% |

Sell Put 显著优于 Buy Call — 高RV = 价格已超跌 + IV膨胀, 双重利好

### Exit 信号效果

| 窗口 | NR (<+1%) | NR (<+2%) | Avg Fwd |
|------|-----------|-----------|---------|
| 1d | **85%** | 94% | -0.00% |
| 5d | **68%** | 79% | +0.16% |
| 10d | 68% | 76% | -0.27% |

## 7. 止盈拐点检测 (Smart TP)

### 方法

买入后10天内, 同时检查5种退出条件, 最早触发即止盈:

| 类型 | 条件 | 含义 |
|------|------|------|
| **MACD** | MACD(8,17,6) hist 由正转负, 已盈利>0.3% | 动量拐点 |
| **MACDweak** | MACD hist 连续2天缩小 (且>0), 涨幅>1% | 动量衰减 |
| **RSI** | RSI(7) 从 >70 回落到 <60, 已盈利 | 超买冷却 |
| **Pullback** | Peak涨>2%后回落≥1.5% | 大幅获利保护 |
| Timeout | 10天未触发 | 兜底退出 |

### 全量结果 (189 trades)

| 类型 | N | Avg Gain | WR | Avg Hold |
|------|---|----------|-----|---------|
| MACDweak | 23 | **+3.12%** | 100% | 7.3d |
| Pullback | 29 | +2.08% | 76% | 5.3d |
| RSI | 48 | +1.74% | 100% | 7.1d |
| MACD | 8 | +1.09% | 100% | 5.9d |
| Timeout | 81 | +0.39% | 57% | 10.0d |
| **TOTAL** | **189** | **+1.35%** | **78%** | **8.1d** |

### Smart TP vs 固定阈值

| 方法 | Avg Gain |
|------|----------|
| Fixed +1% (5d) | +0.59% |
| Fixed +2% (5d) | +0.70% |
| **Smart TP** | **+1.35%** |

Smart TP 收益近固定阈值的 **2倍** — 拐点检测优于简单阈值

**注**: 止盈拐点作为提示信号, 待 Phase 4B 期权回测中验证实际效果

### 退出策略优化 (Band Exit 基础上)

测试 Band Exit (bp>0.90) 叠加各种止盈条件, 10天 timeout:

| 策略 | N | Avg | WR | Hold | Avg/Std | 备注 |
|------|---|-----|-----|------|---------|------|
| Band only | 189 | +1.42% | 66% | 9.2d | 0.504 | 基线 |
| **Band + Pullback** | **189** | **+1.35%** | **67%** | **8.3d** | **0.508** | **风险调整最优** |
| Band + MACDweak | 189 | +1.27% | 66% | 9.0d | 0.485 | 触发率低, ≈Band only |
| Band + MACDweak + PB | 189 | +1.28% | 66% | 8.2d | 0.499 | PB先触发, MW无额外贡献 |
| Band + SmartTP(all) | 189 | +1.23% | 66% | 7.4d | 0.485 | 过度止盈损失尾部收益 |

**Band + Pullback 选定**: Peak涨>2%后回落≥1.5%即止盈, Avg/Std最优, 保护大幅获利不回吐。

OOS 验证 (2025-09~2026-03, 8笔交易):
- Avg +2.5%, WR 88%, Hold 6.5d
- Pullback 触发 4次 (平均+3.2%), BandExit 0次, Timeout 4次 (平均+1.9%)
- MACDweak 在 6 个月内仅触发 1 次 — 实战中几乎无贡献

## 8. 已验证排除的方法

| 实验 | 结果 | 结论 |
|------|------|------|
| E3 整数关口 | 效应弱 (~6pp), 非单调 | 不纳入 |
| E4 MACD死叉卖出 | Bull中不如bp基线 | 不纳入 (但MACD用于止盈拐点有效) |
| E4 RSI超买卖出 | Bull中不如bp基线 | 不纳入 (但RSI回落用于止盈有效) |
| RV过滤买入 | bp低时RV偏高, 过滤反效果 | 用并集不用过滤 |
| LagAvg统一band | Exit信号过多 (94), NR低 | Hybrid D/L3 更优 |
| **LagAvg上界** | 大涨时连续8天退出信号 (1/20-1/28) | 上界必须Daily |
| **RV(5d)归一化** | 模型覆盖率66% (vs 10d 74%), 信号边缘 | RV(10d)更稳健 |
| **MACDweak退出** | 6个月仅触发1次, 实战无贡献 | Band+Pullback更可靠 |

注: MACD/RSI 作为 **独立卖出信号** 效果差 (Bull中动量退出不可靠), 但作为 **买入后止盈拐点** 有效 (已建仓, 检测涨势衰减)

## 9. 最终策略总结

```
价格指导模块 (Phase 4A) — 最终版:

模型: LSTM+Attention, RV(10d)归一化, q85/15 cal80%, 20-fold walk-forward
Band:  Upper=Daily(t-1), Lower=LagAvg(t-1,2,3)

信号 (水平触发, Hybrid D/L3 band):
  Buy Call: Bull + bp<0.30 + RV≤85%  → 123信号, 12.4/年, 5d WR 72%
  Sell Put: Bull + bp<0.30 + RV>85%  → 66信号, 6.6/年, 5d WR 73%, TP+2% 47%
  Exit:     bp>0.90 ∪ Regime退出Bull → 62信号, 6.2/年, 1d NR 85%, 5d NR 68%

退出策略: Band Exit (bp>0.90) + Pullback (peak>2%, dd≥1.5%) + 10d Timeout
  OOS (2025-09~2026-03): 8笔, avg +2.5%, WR 88%, hold 6.5d

止盈提示 (Phase 4B 验证):
  MACD hist转负 / RSI超买回落 / Peak回落
  Smart TP avg +1.35%, WR 78%

已验证排除:
  LagAvg上界, RV(5d/20d), MACDweak退出, E3整数关口, E4独立卖出, RV过滤
```

## 10. 文件说明

| 文件 | 功能 |
|------|------|
| `dl_range_predictor.py` | LSTM + Quantile Loss + RV归一化 + Conformal + 集成 |
| `train_dl_range.py` | Walk-Forward 多配置评估 (20 folds) |
| `train_dl_range_backtest.py` | 区间交易回测 (逐笔跟踪) |
| `analysis_e1_e2.py` | E1区间校准 + E2 bp分桶单调性 |
| `analysis_signal_v2.py` | V2并集信号优化对比 |
| `analysis_e3_e4.py` | E3整数关口 + E4动量退出 |
| `analysis_method_compare.py` | Hybrid band + 信号类型 + 止盈拐点分析 |
| `analysis_regime_visual.py` | Regime可视化 (6张图) |
| `analysis_regime_rv_recent.py` | 近几年Regime + RV均值回归 |

数据:
- `data/models/dl_range_v2_oos.parquet` — OOS 预测结果 (2016-2026), 当前为 RV(10d) 版本
- `data/models/dl_range_v2_oos_rv10d.parquet` — RV(10d) 版本备份
- `data/models/dl_range_v2_oos_rv5d.parquet` — RV(5d) 版本 (测试用, 未采用)
- `data/models/dl_range_v2_oos_rv20d_backup.parquet` — 旧 RV(20d) 版本备份

可视化:
- `reports/regime_analysis/15` — V2 Hybrid 全时段
- `reports/regime_analysis/17` — V2 Hybrid 近一年
- `reports/regime_analysis/19` — V2 Hybrid 2026 Q1
- `reports/regime_analysis/20` — Hybrid vs LagAvg 对比
- `reports/regime_analysis/21~22` — 止盈拐点分析
- `outputs/rv_lower_band_4way.pdf` — RV(10d/5d) × Lower(LagAvg/Daily) 四配置对比
- `outputs/exit_strategy_comparison.pdf` — 退出策略三方案对比 (Band+PB, Band+MW, Band+MW+PB)
- `outputs/band_config_comparison.pdf` — Band 配置四方案对比 (Upper/Lower × Daily/LagAvg)
- `outputs/trade_comparison_20d_vs_10d.pdf` — RV(20d) vs RV(10d) 交易对比
