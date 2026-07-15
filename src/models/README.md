# GLD 期权交易决策系统 — Phase 4A 价格指导模块

## 系统架构总览

```
数据层 (data/processed/dataset.parquet — 5356行, 2004-2026)
  │
  ├── 123 特征 (宏观+技术+波动率+持仓+跨市场)
  │
  ├── Regime 分类器 ──→ Bull / Non-Bull 状态判定
  │     (7因子规则打分, 非ML)
  │
  ├── DL Range 区间预测 ──→ 未来5天上下界
  │     (LSTM+Attention, Quantile Loss, RV归一化, Conformal校准, 3种子集成)
  │     Hybrid Band: 上界=Daily(t-1), 下界=LagAvg(t-1,t-2,t-3)
  │
  └── RV 波动率信号 ──→ 波动率极值均值回归
        (20d RV 的 252d 滚动分位数)

信号层 (Phase 4A 最终版 — 水平触发, 期权类型区分):
  Buy Call  = Bull + bp<0.30 + RV≤85%         → 123信号, 12.4/年
  Sell Put  = Bull + bp<0.30 + RV>85%          → 66信号, 6.6/年 (高IV收premium)
  Exit      = bp>0.90 ∪ Regime退出Bull         → 62信号, 6.2/年
  波动率    = RV<15% 做多波动 / RV>85% 做空波动 → Phase 4B 独立通道

止盈提示 (Phase 4B 待验证):
  MACD histogram 转负 / 连续缩小 / RSI超买回落 / Peak回落
```

## 1. Regime 分类器

详见 [README_regime.md](README_regime.md)

### 7因子打分体系

| 因子 | 权重 | 数据来源 | Bull条件 |
|------|------|---------|---------|
| 价格动量 (ret_60d) | 25% | GLD 60日收益 | 60d涨幅>0 |
| 利率方向 (fed_funds_rate) | 20% | 联邦基金利率60d变化 | 降息/持平 |
| 美元趋势 (tw_usd) | 15% | 贸易加权美元指数20d | 美元走弱 |
| 央行购金 (cb_global_12m) | 15% | 全球央行12月滚动购金 | 净买入 |
| 风险情绪 (gvz) | 10% | 黄金波动率指数252d分位 | GVZ偏低 |
| 通胀趋势 (breakeven_10y) | 10% | 10年盈亏平衡通胀率60d | 通胀上行 |
| 实际利率水平 (real_yield_10y) | 5% | 10年实际利率 | 利率偏低 |

- 每因子打分 -1 ~ +1, 加权汇总
- EMA(60) 平滑 + min_hold_days=20 防频繁切换
- Bull/Non-Bull 二分 (Bear分类偏弱, 不单独使用)

### Regime 效果

| Regime | 占比 | 5d胜率 | 10d胜率 | 20d均值 |
|--------|------|--------|--------|--------|
| **Bull** | 29% | **59.8%** | **61.5%** | **+1.87%** |
| Non-Bull | 71% | 53.9% | 54.2% | +0.65% |

差异统计显著 (p<0.0001)。Bull退出后20天70%概率下跌 (avg -1.04%)。

## 2. DL Range 区间预测

详见 [README_dl_range.md](README_dl_range.md)

### 模型

```
输入: 44 特征 × 20 天序列
  → BatchNorm → LSTM(64, 2层) → Attention加权
  → upper_head (Softplus) → 预测 upper sigma multiplier
  → lower_head → 预测 lower sigma multiplier
```

### 关键设计

- **预测目标**: 未来5天最高涨幅 / 最低跌幅 (% from close)
- **RV归一化**: target = actual_pct / rv_scale, 预测乘回 (不同波动率环境量级统一)
- **Quantile Loss**: q85/q15 pinball loss (非对称惩罚)
- **独立 Conformal 校准**: 126天cal集 (不用val集, 防泄露), 找最小margin使覆盖率≥80%
- **3种子集成**: 不同随机种子, 取均值, 提高稳定性
- **Walk-Forward**: expanding window, 20 folds, 无未来信息泄露

### Hybrid Band — 上下界不同重算方式

```python
# 上界: Daily (仅 t-1) — 灵敏, 及时捕捉顶部
upper_band = close[t-1] × (1 + pred_upper_pct[t-1] / 100)

# 下界: LagAvg (t-1, t-2, t-3) — 平滑, 过滤买入噪声
lower_band = mean(close[t-lag] × (1 + pred_lower_pct[t-lag] / 100) for lag in 1,2,3)

bp = (close - lower_band) / (upper_band - lower_band)
```

**为什么上下界用不同方式?**

| 指标 | Daily | LagAvg | **Hybrid D/L3** |
|------|-------|--------|-----------------|
| Buy 5d WR | 69% | 71% | **72%** |
| Buy 5d TP+1% | 69% | 73% | **74%** |
| Exit信号数 | 61 | 94 | **62** |
| Exit 1d NR | 89% | 82% | **85%** |
| Exit 5d NR | 66% | 64% | **68%** |

- **下界需要平滑** — 买入决策过滤噪声, 避免假突破
- **上界需要灵敏** — 价格触顶及时平仓, 延迟一天可能错过卖点

## 3. RV 波动率信号

```python
rv_20d = 20日对数收益标准差 × sqrt(252) × 100   # 年化已实现波动率
rv_pctile = rv_20d.rolling(252).rank(pct=True)   # 252天滚动分位数
```

RV 均值回归极强 (Spearman ρ=-0.345, p<1e-147):
- RV < 15th percentile → 73%概率扩大 (avg +26.6%)
- RV > 85th percentile → 69%概率收缩 (avg -9.8%)

## 4. 最终信号定义

### 触发方式: 水平触发

每天判定条件是否成立, 满足即为信号日 (非穿越触发)。期权交易中每天都是潜在入场点。

### 三类信号

```
Buy Call  = Bull + bp<0.30 + RV_pctile≤85%   → 正常IV, 买call做多
Sell Put  = Bull + bp<0.30 + RV_pctile>85%    → 高IV, 卖put收premium
Exit      = bp>0.90 ∪ Regime退出Bull          → 平仓/止盈
```

| 信号 | N | 次/年 | 5d WR | 5d TP+2% | 逻辑 |
|------|---|------|-------|---------|------|
| **Buy Call** | 123 | 12.4 | **72%** | 37% | 正常RV, 买call做多 |
| **Sell Put** | 66 | 6.6 | **73%** | **47%** | 高IV = call贵 + put premium厚 |
| **Exit** | 62 | 6.2 | — | — | bp>0.90 或 Regime退出Bull |

**为什么区分 Buy Call / Sell Put?**
- RV>85% 时: call 贵 (高IV), 但价格已超跌 → 卖put收premium双重获利
- Sell Put 5d TP+2% = 47% vs Buy Call 37%, 高IV环境卖put显著优于买call

### Exit 信号效果

评估标准: **未大涨率** P(fwd < +1%) — 期权theta衰减, 不涨即正确

| 窗口 | NR (<+1%) | NR (<+2%) | Avg Fwd |
|------|-----------|-----------|---------|
| 1d | **85%** | 94% | -0.00% |
| 3d | 63% | 87% | -0.01% |
| 5d | **68%** | 79% | +0.16% |
| 10d | 68% | 76% | -0.27% |

### 波动率独立信号 (Phase 4B)

| 信号 | 触发 | 正确方向率 | 用途 |
|------|------|----------|------|
| 做多波动 | RV<15% entering | 73% | 买straddle/strangle |
| 做空波动 | RV>85% entering | 69% | 卖iron condor |

## 5. 止盈拐点检测 (Phase 4B 待验证)

买入后10天内, 同时检测5种退出信号, 最早触发即止盈:

| 类型 | 条件 | N | Avg Gain | WR | Avg Hold |
|------|------|---|----------|-----|---------|
| **MACDweak** | MACD hist连续2天缩小, 涨>1% | 23 | **+3.12%** | 100% | 7.3d |
| **Pullback** | Peak涨>2%后回落≥1.5% | 29 | +2.08% | 76% | 5.3d |
| **RSI** | RSI(7) 从>70回落到<60 | 48 | +1.74% | 100% | 7.1d |
| **MACD** | MACD hist由正转负, 已盈利 | 8 | +1.09% | 100% | 5.9d |
| Timeout | 10天未触发任何条件 | 81 | +0.39% | 57% | 10.0d |

**Smart TP avg +1.35%, WR 78%** vs 固定+2%止盈 avg +0.70% — 拐点检测收益近2倍

## 6. 已验证排除的方法

| 实验 | 结果 | 结论 |
|------|------|------|
| E3: $5/$10整数关口支撑 | 效应~6pp, 非单调 | 不纳入 |
| E4: MACD死叉 (3组参数) | Bull中33-43%跌率 < 基线47% | 不纳入 |
| E4: RSI超买回落 | Bull中40%跌率 < 基线47% | 不纳入 |
| RV过滤买入 | bp低时RV本来偏高, 过滤反效果 | 用并集不用过滤 |
| Bear独立分类 | 20d准确率44% < 50% | 简化为Bull/Non-Bull |
| LagAvg统一上下界 | Exit信号94个过多, NR低 | Hybrid D/L3更优 |

## 7. 文件结构

```
src/models/
  ├── data_utils.py                    # 数据加载, WalkForwardSplitter
  ├── evaluation.py                    # 评估指标
  ├── regime_classifier.py             # 7因子Regime打分 → README_regime.md
  ├── dl_fair_value.py                 # LSTM/Transformer 模型 + 44特征定义
  ├── dl_range_predictor.py            # 区间LSTM (RV归一化+conformal+集成)
  ├── train_dl_range.py                # 区间预测 walk-forward → dl_range_v2_oos.parquet
  ├── analysis_e1_e2.py                # E1覆盖率校准 + E2 bp单调性
  ├── analysis_signal_v2.py            # V2并集信号优化
  ├── analysis_e3_e4.py                # E3整数关口 + E4动量退出
  ├── analysis_method_compare.py       # Hybrid band + 信号类型 + 止盈分析
  ├── analysis_regime_visual.py        # Regime可视化 (6张图)
  ├── analysis_regime_rv_recent.py     # 近几年Regime + RV均值回归
  ├── README.md                        # ← 本文件 (系统架构总览)
  ├── README_regime.md                 # Regime详细分析
  └── README_dl_range.md               # DL Range + Band + 信号详细分析

data/models/
  └── dl_range_v2_oos.parquet          # OOS预测结果 (walk-forward, 2016-2026)

reports/regime_analysis/
  ├── 01~08                            # Regime + RV 分析图
  ├── 11~13                            # V2信号 + E3/E4实验
  ├── 15_method_V2_optimized.png       # V2 Hybrid 全时段
  ├── 17_method_V2_recent.png          # V2 Hybrid 近一年
  ├── 19_method_V2_2026Q1.png          # V2 Hybrid 2026 Q1
  ├── 20_hybrid_vs_lagavg.png          # Hybrid vs LagAvg 对比
  ├── 21_tp_exit_analysis.png          # 止盈拐点 2025.10~2026.02
  └── 22_tp_exit_analysis_full.png     # 止盈拐点 全量
```

## 8. 期权策略建议 (已集成)

信号触发时, `scripts/predict_today.py` 自动输出三档风险的期权策略:

```
信号触发
  │
  ├── BUY CALL →
  │     A. 稳健: 买 ITM Call (Delta≈0.70, DTE 30-45d) — 跟踪紧密, 杠杆低
  │     B. 中性: 买 ATM Call (Delta≈0.50, DTE 21-35d) — 平衡杠杆与成本
  │     C. 激进: 买 OTM Call (Delta≈0.25, DTE 14-28d) — 高杠杆, 归零风险高
  │
  ├── SELL PUT →
  │     A. 稳健: 卖深OTM Put (Delta≈-0.10, DTE 30-45d) — 被行权概率极低
  │     B. 中性: 卖 OTM Put (Delta≈-0.25, DTE 21-35d) — 权利金适中
  │     C. 激进: 卖近ATM Put (Delta≈-0.40, DTE 14-28d) — premium丰厚
  │
  └── EXIT → 平仓操作建议
```

每档策略包含:
- 具体合约推荐 (来自最新 EOD 快照)
- 中间价 / Delta / OI / Spread
- 若达到模型预测上界的预估 ROI
- 仓位建议: 稳健 2-5% / 中性 5-10% / 激进 ≤5%
- 风控: bp>0.90 平仓, Pullback 止盈, 10d timeout

同时输出 **下一交易日信号阈值预览** (买入/平仓价位)。

## 9. Phase 4B 期权回测

详见 [../backtest/README.md](../backtest/README.md)

```
价格方向模块 (Phase 4A) ─→ 期权策略 (Phase 4B)

Buy Call 信号  → 买 Call (方向: 看涨, 正常IV)
Sell Put 信号  → 卖 Put (方向: 看涨, 高IV收premium)
Exit 信号      → 平仓 (止盈/止损)
DL Range 宽度  → 选择行权价 (ATM/OTM)
RV 信号        → 波动率策略 (straddle / iron condor)
止盈拐点       → MACD/RSI/Pullback 提示平仓时机
```

当前瓶颈: ATM K线数据缺失, 等额度恢复后下载 $430-$520 strikes。
