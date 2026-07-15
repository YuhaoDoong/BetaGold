# Phase 2: 特征工程层

将原始数据转化为模型可用的特征矩阵 + 预测标签。

---

## 一、总览

| 指标 | 数值 |
|------|------|
| 特征数 | **116** (技术52 + 宏观33 + 波动率31) |
| 标签数 | **12** (3回归收益 + 2回归波动 + 4分类 + 3其他) |
| 样本数 | 5,356 行 (2004-11 ~ 2026-03) |
| 平均缺失率 | 3.47% |
| >50%缺失特征 | 0 个 |
| 高相关对(r>0.9) | 25 对 (清理前75对) |

输出文件 (在 `data/processed/`):
- `features_all.parquet` — 全部特征矩阵
- `labels.parquet` — 预测标签
- `dataset.parquet` — 合并数据集 (128列)
- `feature_list.csv` / `label_list.csv` — 名单 + 缺失率

质量报告: `reports/phase2_features/` (7张图 + REPORT.md)

---

## 二、特征分类

### 2.1 快通道: 技术面 (52个, 日频)

**收益率** (7个):
- `ret_1d/2d/3d/5d/10d/20d/60d` — 历史累计收益率

**均线偏离度** (7个):
- `close_to_sma_5/10/20/60/120` — 价格相对SMA的偏离度 (非原始SMA价位)
- `sma_20_slope`, `sma_60_slope` — 均线斜率 (5日变化率)
- `ma_alignment` — 均线排列 (短期在长期之上的数量, 0-3)

**动量** (7个):
- `rsi_7`, `rsi_14` — RSI
- `macd`, `macd_signal`, `macd_hist` — MACD三件套
- `stoch_k_14`, `stoch_d_14` — 随机指标

**波动** (7个):
- `hv_5d`, `hv_60d` — 历史波动率 (年化, 10d/20d由vol模块提供避免重复)
- `bb_width`, `bb_position` — Bollinger带宽度和位置 (非原始上下轨)
- `atr_14`, `atr_14_pct` — ATR及其占价格比例
- `hv_5d_change` — 波动率变化率
- `daily_range_pct` — 日内高低幅

**成交量** (5个):
- `vol_ratio_5d`, `vol_ratio_20d` — 量比
- `vol_change_1d` — 成交量变化率
- `obv_direction` — OBV方向
- `price_vol_confirm` — 量价配合

**跳空** (3个):
- `gap_pct`, `gap_up`, `gap_down`

**关联品种** (16个):
- 黄金期货: `gc_gld_ratio`, `gc_gld_ratio_zscore`
- 美元: `dxy_ret_1d/5d`, `dxy_sma20_dev`
- VIX: `vix_level`, `vix_ret_1d`, `vix_sma20_dev`
- 铜金比: `copper_gold_ratio`, `copper_gold_ratio_change`
- 金银比: `gold_silver_ratio`
- 债券: `us10y_level`, `us10y_change_5d`
- 原油: `crude_ret_5d`

### 2.2 慢通道: 宏观面 (33个, 月/季频 -> ffill到日频)

**实际利率** (4个): `real_yield_10y`, `real_yield_5y`, `real_yield_10y_change_20d`, `real_yield_10y_zscore`

**利率曲线** (1个): `real_yield_curve` (10Y-5Y)

**通胀** (3个): `breakeven_10y`, `breakeven_10y_change_20d`, `cpi_yoy`

**政策** (3个): `fed_funds_rate`, `fed_funds_rate_change_60d`, `real_fed_funds`

**财政** (4个): `federal_debt`, `federal_debt_yoy`, `fiscal_deficit`, `fiscal_deficit_12m_sum`

**货币** (1个): `m2_yoy`

**美元** (3个): `tw_usd`, `tw_usd_ret_20d`, `tw_usd_zscore`

**COT持仓** (7个): `cot_noncomm_net`, `cot_noncomm_net_change`, `cot_noncomm_net_pctile`, `cot_comm_net`, `cot_comm_net_change`, `cot_open_interest`, `cot_oi_change_pct`

**央行购金** (7个): `cb_global_net_tonnes`, `cb_global_3m/6m/12m_rolling`, `cb_global_yoy_change`, `cb_key_banks_net_tonnes`, `cb_china_net_tonnes`

### 2.3 波动率通道 (31个, 日频)

**GVZ** (7个): `gvz`, `gvz_ret_1d/5d`, `gvz_pctile_252d`, `gvz_sma20_dev`, `gvz_high`, `gvz_low`

**VIX期限结构** (7个): `vix_vix9d/vix3m/vix6m`, `vix_term_slope`, `vix_backwardation`, `vix_term_slope_6m`, `vix_9d_vs_30d`

**MOVE** (3个): `move`, `move_ret_5d`, `move_pctile_252d`

**已实现波动率 & VRP** (5个): `rv_10d`, `rv_20d` (×100对齐GVZ), `vrp_10d`, `vrp_20d`, `vrp_20d_pctile`

**事件窗口** (最多9个): `days_to_fomc/nfp/cpi`, `fomc/nfp/cpi_window_3d`, `is_fomc/nfp/cpi_day`

---

## 三、预测标签 (12个)

### 3.1 收益率标签 (回归)

| 标签 | 定义 | 用途 |
|------|------|------|
| `fwd_ret_5d` | 未来5日 GLD 收益率 | 短期方向 |
| `fwd_ret_10d` | 未来10日 GLD 收益率 | 中期方向 |
| `fwd_ret_20d` | 未来20日 GLD 收益率 | 波段预测 |

注：标签统一使用 `fwd_` 前缀（forward），避免与特征中的历史收益率 `ret_Xd` 列名冲突。

### 3.2 波动率标签 (回归)

| 标签 | 定义 | 用途 |
|------|------|------|
| `fwd_rv_10d` | 未来10日实现波动率 (年化) | 波动率预测 |
| `fwd_rv_20d` | 未来20日实现波动率 (年化) | 波动率预测 |
| `iv_rv_spread` | GVZ/100 - 未来实现RV | IV偏离度 |

### 3.3 分类标签

| 标签 | 定义 | 用途 |
|------|------|------|
| `direction_5d` | 涨(>+1%) / 跌(<-1%) / 震荡 | 策略方向 |
| `direction_10d` | 同上, 10日窗口 | 策略方向 |
| `magnitude_10d` | 小幅(<2%) / 中幅(2-5%) / 大幅(>5%) | 策略类型 |
| `vol_regime` | 低波 / 正常 / 高波 (基于252d分位数) | 仓位调整 |
| `tail_event_flag` | 未来10D内是否出现>3σ日移动 | 风险预警 |
| `max_dd_10d` | 未来10日最大回撤 | 风控参考 |

### 3.4 标签分布

| 标签 | 分布 |
|------|------|
| direction_5d | flat 36.3% / up 36.2% / down 27.4% |
| direction_10d | up 42.5% / down 31.1% / flat 26.4% |
| magnitude_10d | small 48.1% / medium 38.8% / large 13.1% |
| vol_regime | low 36.1% / high 34.6% / normal 29.3% |
| tail_event_flag | 0: 88.5% / 1: 11.5% |
| fwd_ret_5d | mean=0.0025, std=0.0249 |
| fwd_rv_20d | mean=0.1636, std=0.0764 |

---

## 四、共线性清理

原始 132 个特征 → 清理后 116 个，高相关对从 75 → 25。

### 已删除的冗余特征 (16个)

| 删除 | 原因 | 保留替代 |
|------|------|----------|
| `roc_5/10/20` | 与 `ret_5d/10d/20d` 完全相同 (r=1.0) | ret_Xd |
| `log_ret_1d` | 与 `ret_1d` 完全相同 (r=0.9999) | ret_1d |
| `sma_5/10/20/60/120` | 原始价位互相 r>0.999 | close_to_sma_X (偏离度) |
| `ema_12/26` | 同上 | 已在MACD中体现 |
| `bb_upper/bb_lower` | 与价格共线 | bb_width, bb_position |
| `hv_10d/hv_20d` | 与vol模块 rv_10d/20d 重复 (r=1.0) | rv_10d/20d (×100) |
| `vix_vix` | 与 vix_level 重复 | vix_level |

### 剩余高相关对 (r>0.9, 前10)

| 对 | r | 原因 |
|----|---|------|
| cot_noncomm_net / cot_comm_net | -0.99 | 结构性镜像 (正常) |
| vix_vix3m / vix_vix6m | +0.99 | VIX期限结构内在相关 |
| fed_funds_rate / real_fed_funds | +0.98 | 名义利率与实际利率 |
| macd / macd_signal | +0.96 | 信号线是MACD的EMA |
| real_yield_10y / real_yield_5y | +0.96 | 同类资产 |

这些属于结构性相关，不是冗余，保留合理。

---

## 五、特征-标签相关性 (关键发现)

### 收益率标签的最佳预测因子

| 排名 | fwd_ret_5d | fwd_ret_10d | fwd_ret_20d |
|------|-----------|-------------|-------------|
| 1 | real_yield_10y (+0.10) | real_yield_10y (+0.14) | real_yield_10y (+0.21) |
| 2 | real_yield_5y (+0.09) | real_yield_5y (+0.14) | real_yield_5y (+0.21) |
| 3 | real_fed_funds (+0.09) | real_fed_funds (+0.13) | real_fed_funds (+0.18) |

**结论**: 宏观因子（实际利率）是收益率最强预测因子，且长周期信号更强。

### 波动率标签的最佳预测因子

| 排名 | fwd_rv_10d | fwd_rv_20d |
|------|-----------|-------------|
| 1 | atr_14_pct (+0.63) | atr_14_pct (+0.65) |
| 2 | gvz (+0.60) | rv_20d (+0.60) |
| 3 | rv_20d (+0.59) | gvz (+0.59) |

**结论**: 波动率有强持续性 (当前高波 → 未来高波)，GVZ是优秀的前瞻指标。

### 尾部事件最佳预测因子

| 排名 | 特征 | r |
|------|------|---|
| 1 | gvz_sma20_dev | +0.31 |
| 2 | gvz_ret_5d | +0.22 |
| 3 | vrp_20d | +0.21 |
| 4 | vix_9d_vs_30d | +0.21 |

**结论**: GVZ短期异动是尾部事件的最佳预警信号。

---

## 六、特征稳定性

Rolling 504天 (2年) 相关性分析显示:
- `real_yield_10y` 对 fwd_ret_10d 的相关性最稳定 (持续正相关, 2006-2026)
- `cot_noncomm_net` 最不稳定 (符号多次翻转)
- `gvz` 对波动率标签的预测力在 2020 COVID 后显著增强

详见 `reports/phase2_features/07_feature_stability.png`

---

## 七、关键设计决策

1. **只输出相对指标，不输出原始价位** — SMA/EMA/BB用偏离度替代，避免模型学到价格水平而非模式
2. **波动率统一单位** — vol_features的rv_10d/20d乘以100对齐GVZ百分比格式，方便VRP计算
3. **标签 fwd_ 前缀** — 特征 `ret_5d` (过去5日) vs 标签 `fwd_ret_5d` (未来5日) 含义不同，必须区分
4. **宏观数据暂用 ffill** — 后续 Phase 需升级为严格 release-date 对齐
5. **分位数用 rolling 252天** — 1年窗口，避免全样本前视偏差
