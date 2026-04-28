# 策略详解

## 总览

系统支持 **5 类策略 + MIXED 组合**，按 vega/delta 维度分类，按 RV regime 分配工具。

```
┌────────────────────────────────────────────────────────┐
│   方向性 (Directional)                                  │
│   ├── BUY CALL  (long delta + long vega)                │
│   └── SELL PUT  (long delta + short vega)               │
├────────────────────────────────────────────────────────┤
│   线性 (Linear)                                         │
│   └── 期货多头 (linear delta, vega=0, theta=0)          │
├────────────────────────────────────────────────────────┤
│   做多波动率 (Long Vol)                                 │
│   └── STRADDLE  (neutral delta + long vega)             │
├────────────────────────────────────────────────────────┤
│   做空波动率 (Short Vol)                                │
│   └── SHORT_VOL Iron Condor 16Δ/5Δ (short vega)         │
├────────────────────────────────────────────────────────┤
│   退出 (Exit)                                           │
│   └── EXIT (bp_high > 0.90 ∪ Regime 退出 Bull)          │
└────────────────────────────────────────────────────────┘
```

## 完整策略矩阵

| RV %tile | 方向性 | 推荐工具 (实证) | 备选波动率 | 不交易区 |
|----------|--------|-----------------|------------|----------|
| < 25% (压缩) | BUY CALL | **期货多头 + 3% 止损** | Long Straddle (临 FOMC) | — |
| 25-50% (低/中性) | BUY CALL | **期货多头 + 3% 止损** | — | — |
| **50-85% (温水)** | ❌ 屏蔽 | ❌ | **Iron Condor 16Δ/5Δ** | 不做方向 |
| > 85% (恐慌) | SELL PUT | **期权 Sell Put** | Long Straddle (临界) | — |

## 方向性策略

### BUY CALL (long vega)
- **入场**: Bull regime + bp_low < 0.30 + RV %tile < 0.50
- **推荐工具**: **期货多头 + 3% 止损** (实证 96% wr vs 期权 73%)
- **原因**: 低 RV 时期权 IV 已压缩但仍要付 premium ~2-2.5%, 实际 max_up 平均 +2.35%, 期权刚够 breakeven; 期货线性 P&L 无 theta/vega 损耗
- **退出优先级**:
  1. StopLoss 3%
  2. BandExit (bp_high > 0.90)
  3. Pullback (gain_peak > 2% 且回撤 ≥ 1.5%)
  4. Timeout (≥ 30 天)

### SELL PUT (short vega)
- **入场**: Bull regime + bp_low < 0.30 + RV %tile > 0.85
- **推荐工具**: **期权 Sell Put** (实证 100% wr vs 期货 68%)
- **原因**: 高 RV 入场, 期权收 IV premium, 横盘+上涨都赢; 期货需要真涨, 在震荡 regime 易被震出
- **退出**: 同上

### EXIT
- bp_high > 0.90 触发 (上轨突破)
- 或 Regime 由 Bull 切到 非 Bull
- 用于平掉持仓, 不是反向做空

## 期货多头 (Linear Delta)

### 特点
- **Vega = 0, Gamma = 0, Theta = 0**: 没有时间价值损耗
- **Linear P&L**: 价格涨 1% 赚 1%
- **风险**: 双向无限 (止损管理关键)

### 应用边界
- ✅ BUY CALL 信号下: 96% 胜率, Sharpe 1.16
- ❌ SELL PUT 信号下: 仅 68% 胜率 (高 RV 震荡不利)
- ❌ Bear regime: 30% 胜率

### 止损建议
- 3% 硬止损 (实证最优)
- 5 天持仓上限
- 连续熔断 (默认关闭, 详见 EXPERIMENTS.md "熔断 A/B")

## 做多波动率 — Long Straddle

### 入场评分 (≥ 3 分触发)

| 条件 | 分数 |
|------|------|
| RV < 20% | +2 |
| RV 下降 > 30% | +1 |
| 距 FOMC ≤ 3 天 | +3 |
| 距 NFP ≤ 3 天 | +2 |
| 距 OPEX ≤ 3 天 | +1 |

**硬门槛**: RV > 25% → 否决 (成本过高)

### Regime
**完全 regime-agnostic** — 实证 Mixed regime (90%) 反胜 Bull (71%)，Bear 因 RV 通常高位自动过滤。

### P&L 模型
- **Cost** = `RV × √(5/252)` ≈ 1σ premium
- **Win**: `max_move > 1σ`
- **PnL**: `max_move - cost`

### 实战建议
- ATM Long Call + Long Put
- Hold 至事件日翌日 (FOMC 释放后)
- 50% profit lock 早平
- 实证胜率 79% (近 5y)

## 做空波动率 — Iron Condor 16Δ / 5Δ

### 结构
```
卖 Put @ 1.6σ 下方  (16Δ ≈ 16% probability ITM)
+ 买 Put @ 3.0σ 下方  (5Δ ≈ 5% 长翼保护)
+ 卖 Call @ 1.6σ 上方
+ 买 Call @ 3.0σ 上方
```
- **净 Credit** ≈ 1σ premium × 0.4
- **最大盈利**: 全部 credit (如果 |move| < 1.6σ)
- **最大亏损**: (3σ - 1.6σ) - credit ≈ 1.4σ - credit (翼锁定)

### 入场评分 (≥ 7 分触发, 严格)

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

### 硬门槛 (任一命中即屏蔽)
- RV %tile > 0.75 (高位反弹风险)
- RV %tile < 0.25 (premium 太薄)
- RV 越界 (< 13% 或 > 28%)
- 距 FOMC ≤ 10 天 (IV 还会涨)
- 距 NFP ≤ 7 天
- **Bear regime** (尾部下跌风险)
- 持仓窗口 (5d) 内有任何主要事件

### Regime
**保留 Bear 屏蔽** — 取消屏蔽仅多 3 笔, 整体胜率不变 (91%), 但暴露 panic gap 尾部风险, 性价比不高。

### P&L 模型
- **Win**: `max_move < 1.6σ` (留全部 credit)
- **Loss range**: `1.6σ ≤ max_move < 3σ` → credit - (max_move - 1.6σ)
- **Max Loss**: `max_move ≥ 3σ` → credit - 1.4σ (翼锁定)

### 实战建议
- 16Δ short strikes, 5Δ wings
- DTE < 21 天 (theta 衰减最快)
- 50% credit 早平 (锁利润)
- 200% credit 强制止损 (突破短腿前)
- 单笔仓位 ≤ 总资金 5%

### 实证 (近 5y)
- 91 笔, **89% 胜率**
- Sharpe **0.94 - 1.21**
- 总收益 +54%
- Max 单笔亏损 -1.92% (翼锁定)

## MIXED 组合 (Vega 兼容矩阵)

只有 vega 同向才允许 MIXED:

| 组合 | 兼容? | 原因 |
|------|-------|------|
| BUY CALL + STRADDLE | ✅ | 都 long vega |
| **BUY CALL + SHORT_VOL** | ❌ | long vs short vega 矛盾 |
| **SELL PUT + STRADDLE** | ❌ | short vs long vega 矛盾 |
| SELL PUT + SHORT_VOL | ✅ | 都 short vega |

不兼容时方向性优先, 仅当 vol score 极强 (≥ priority + 2) 才覆盖。

## 胜率定义 (按 vega/delta 实际盈亏)

所有阈值用 **动态 sigma_pct = RV × √(hold_days/252)**, 自动适应当时波动环境。

| 策略 | 胜利条件 | Vega | Delta |
|------|----------|------|-------|
| BUY CALL | `max_up > 1σ` | + | + |
| SELL PUT | `max_down < 1σ` | − | + |
| STRADDLE | `max_move > 1σ` | + | 0 |
| SHORT_VOL | `max_move < 1.6σ` | − | 0 |
| 期货多头 | `ret_5d > 0` | 0 | linear + |
| EXIT | `ret_5d < 3%` | — | — |
| MIXED | 任一对就赢 | — | — |

### 胜率示例 (近 5y)

| 策略 | 笔数 | 胜率 |
|------|------|------|
| BUY CALL (期权) | 23 | 70% |
| **BUY CALL → 期货多头** | 26 | **96%** |
| SELL PUT (期权) | 22 | 100% |
| SELL PUT → 期货多头 | 22 | 68% |
| STRADDLE | 46 | 76% |
| SHORT_VOL Iron Condor | 76 | 83% |

详见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 退出机制 (方向性持仓)

按优先级:

1. **StopLoss**: 日内 low 跌破入场 -3% → 立即止损
2. **BandExit**: bp_high > 0.90 → 优先 log EXIT 代表价, 兜底 bp090 阈值
3. **Pullback**: 持仓期峰值涨幅 > 2% 且回撤 ≥ 1.5% (持仓管理 "止盈位" 列)
4. **Timeout**: 持仓 ≥ 30 天 (安全帽)

实证近 5y 退出分布: Pullback ~50% / StopLoss ~20% / BandExit ~15% / Timeout ~5%。

## 持仓管理 (Dashboard)

支持 3 类持仓的实时状态展示:
- **方向性** (BUY CALL / SELL PUT): 状态 + 当前 P&L + 止盈位 + BandExit 价位
- **STRADDLE**: 状态 (待移动 / 盈利中 / 可早平 / 已到期) + 实时 P&L
- **SHORT_VOL**: 状态 (theta 衰减中 / 可早平 / 突破短腿 / 翼锁定) + 实时 P&L

MIXED 组合 (BUY CALL + STRADDLE 等) 同时出现在方向性和波动率两个区间, 双向监控。
