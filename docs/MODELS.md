# 模型架构

## 1. 区间预测模型 (Daily Range Prediction)

### 输出
预测未来 5 日的 High%, Low% (相对当前收盘价的百分比)

### 架构: LSTM + Transformer Ensemble (v3.2+)

```
Input: 60 timesteps × 88 features
              ↓
   ┌──────────────────────┐
   │  LSTM Branch         │     │  Transformer Branch  │
   │  3 layers × 128      │     │  6 heads × 128 d_model│
   │  dropout 0.2         │     │  Positional Encoding  │
   └──────────────────────┘     └──────────────────────┘
              ↓                              ↓
              └──────────── Average ─────────┘
                            ↓
              Quantile Loss (q=0.85, q=0.15)
                            ↓
                  Output: (High_85%, Low_15%)
```

### 训练细节
- **数据**: 20 年日线 (2004 - 2026)
- **Walk-Forward**: 20 折滚动训练
- **优化器**: AdamW, lr=1e-3, weight_decay=1e-4
- **Batch**: 64
- **Loss**: Quantile Loss
  - `L = max(q × diff, (q-1) × diff)` (asymmetric)
- **Early stopping**: 10 epochs patience
- **每周训练一次** (Dashboard 训练状态 + 一键启动)

### Conformal 校准 (校准至 80% 覆盖)
- OOS 预测残差排序
- 取 80% 分位数作为校准 offset
- 最终区间 = 模型预测 ± offset

### 性能对比
| 模型 | Coverage | Width | Tightness |
|------|----------|-------|-----------|
| 单 LSTM | 71.3% | 6.96% | 0.102 |
| 单 Transformer | 69.8% | 6.74% | 0.104 |
| **Ensemble** | 71.1% | **6.50%** | **0.109** |

Ensemble 相对单 LSTM 区间宽度 -7%，tightness +7%。

## 2. Regime 分类器 (7 因子)

### 因子权重
| 因子 | 权重 | 含义 |
|------|------|------|
| 价格动量 (50/200 SMA, ROC) | 25% | 中长期趋势 |
| 联储利率 (fed_funds + change_60d) | 20% | 货币政策 |
| 美元 (DXY ret_5d + zscore) | 15% | 反向相关 |
| 央行购金 (proxy via Bloomberg/wgc) | 15% | 长期需求 |
| 风险偏好 (VIX + VIX term slope) | 10% | 避险情绪 |
| 通胀 (CPI YoY + M2 YoY) | 10% | 名义需求 |
| 实际利率 (real_yield_10y) | 5% | 持有成本 |

### 输出: Bull / Mixed / Bear

```python
score = composite_factor_weighted_sum
smoothed = ewma(score, alpha=0.2)
regime = "Bull"  if smoothed > +0.3
         "Bear"  if smoothed < -0.3
         "Mixed" otherwise

# Min-hold: 切换需连续 3 天确认
```

### 历史分布 (近 8 年)
| Regime | 占比 |
|--------|------|
| Mixed | 47% |
| Bull | 49% |
| Bear | 4% |

## 3. Qlib Alpha158 因子库 (v3.2+)

110 项技术因子, 来自 Qlib Alpha158:
- **KBAR**: K线形态 (open/high/low/close 关系)
- **BETA**: 与基准 (gold spot) 协动
- **RSQR**: 滚动 R-square
- **QTLU**: 5/10/20/30/60 日分位数
- **CORR**: 跨资产相关性
- **CNTP**: 价格穿越统计
- **SUMP**: 累积动量

## 4. 事件日历

硬编码 2025-2026, 含:
- **FOMC** (FOMC 议息会议)
- **OPEX** (月度第三周五期权到期日)
- **NFP** (Non-Farm Payrolls 第一周五)
- **FUT_EXP** (期货交割日)

提供 `days_to_next_event(date, type)` 接口给信号模块用。

## 5. RV 与 RV %tile

### Realized Volatility
- 10 日窗口
- `RV = std(log_returns) × √252 × 100` (年化 %)

### RV %tile (滚动百分位)
- 252 日 rolling rank, pct=True
- 用于信号过滤 (而非绝对 RV 值)

### 在系统中的作用
- 方向性 BUY CALL: RV %tile < 0.50 (低 vol, 期权便宜)
- 方向性 SELL PUT: RV %tile > 0.85 (高 vol, 收 IV)
- STRADDLE: 看绝对 RV (< 20%) + 事件距离
- SHORT_VOL: RV %tile ∈ [0.35, 0.65] (中位窄带) + RV 趋势回落

## 6. OI 微观结构修正

每日修正 bp030 / bp090 阈值:
- **Max Pain**: 期权到期日总损失最小价
- **Call Wall**: 最大 Call OI strike (通常压力位)
- **Put Wall**: 最大 Put OI strike (通常支撑位)
- **PCR**: Put/Call 比例
- **Net GEX**: 做市商 Gamma Exposure
- **DTE-weighted**: 越临近到期日权重越高

## 7. Hybrid Band 构建

```python
upper_band = close.shift(1) × (1 + pred_upper_pct.shift(1) / 100)
lower_band = avg([close.shift(k) × (1 + pred_lower_pct.shift(k) / 100)
                   for k in [1, 2, 3]])
bp = (close - lower_band) / (upper_band - lower_band)
```

- **Upper**: 仅 lag1 (灵敏, 及时捕捉顶部)
- **Lower**: lag1, 2, 3 三日平均 (平滑, 过滤买入噪声)
- **bp**: Band Position [0, 1], > 1 突破上轨, < 0 跌破下轨

## 8. 训练频率

- **每周一次**: 数据漂移监控
- 模型 > 7 天未训练 → 侧边栏黄色警告
- 一键启动: 后台 subprocess 不阻塞 dashboard
- 训练时长: 单配置 + n_ensemble=2 ≈ 40-60 分钟 (M1 Mac MPS)
