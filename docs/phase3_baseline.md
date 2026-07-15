# Phase 3: 基线模型 (Preliminary Baseline)

Walk-forward 时间序列评估，验证特征信息含量和基线预测能力。

> **注意**：当前基线使用 ffill 对齐宏观数据，尚未严格按 release-date 对齐。结果为 preliminary baseline，待 release-date 重建后需复核。IC 等指标可能因前视偏差被轻微高估。

---

## 一、评估框架

### Walk-Forward 设置

| 参数 | 值 | 说明 |
|------|---|------|
| 最小训练集 | 1,260 天 (5年) | 足够学习多种市场环境 |
| 测试窗口 | 252 天 (1年) | 每个fold评估一整年 |
| 步进 | 252 天 | 测试窗口不重叠 |
| 总 folds | 10 | OOS覆盖 2016-01 ~ 2026-02 |
| 训练模式 | Expanding window | 累积所有历史数据 |

```
Fold 0:  Train [... ~ 2016-01], Test [2016-01 ~ 2017-01]
Fold 1:  Train [... ~ 2017-01], Test [2017-01 ~ 2018-01]
Fold 2:  Train [... ~ 2018-02], Test [2018-02 ~ 2019-02]
Fold 3:  Train [... ~ 2019-02], Test [2019-02 ~ 2020-02]
Fold 4:  Train [... ~ 2020-02], Test [2020-02 ~ 2021-02]
Fold 5:  Train [... ~ 2021-02], Test [2021-02 ~ 2022-02]
Fold 6:  Train [... ~ 2022-02], Test [2022-02 ~ 2023-02]
Fold 7:  Train [... ~ 2023-02], Test [2023-02 ~ 2024-02]
Fold 8:  Train [... ~ 2024-02], Test [2024-02 ~ 2025-02]
Fold 9:  Train [... ~ 2025-03], Test [2025-03 ~ 2026-02]
```

注：有效样本从 2009-09 开始 (GVZ数据起始点)，共 3,762 行。

### 模型

| 模型 | 回归任务 | 分类任务 | 说明 |
|------|---------|---------|------|
| **Ridge / LogReg** | Ridge(alpha=1.0) + StandardScaler | LogisticRegression(C=1.0) | 线性基线，含标准化 |
| **XGBoost** | XGBRegressor | XGBClassifier | n_estimators=300, max_depth=5, lr=0.05 |

XGBoost 参数: subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0

### 评估指标

**回归**: R², MAE, RMSE, **IC** (Spearman秩相关), 方向准确率 (Dir_Acc)
**分类**: Accuracy, Precision, Recall, AUC

IC (Information Coefficient) 是量化投资中最核心的指标 — 它衡量预测排序与实际排序的一致性，对金融时间序列比 R² 更有意义。

---

## 二、预测目标

共 8 个目标，覆盖三大预测模块:

| # | 目标 | 类型 | 对应模块 |
|---|------|------|---------|
| 1 | `fwd_ret_5d` | 回归 | 短中期预测 |
| 2 | `fwd_ret_10d` | 回归 | 短中期预测 |
| 3 | `fwd_ret_20d` | 回归 | 短中期预测 |
| 4 | `fwd_rv_10d` | 回归 | 波动率预测 |
| 5 | `fwd_rv_20d` | 回归 | 波动率预测 |
| 6 | `direction_5d` | 3分类 | 短中期预测 |
| 7 | `direction_10d` | 3分类 | 短中期预测 |
| 8 | `tail_event_flag` | 2分类 | 风险预警 |

---

## 三、核心结果

### 3.1 回归: IC (Spearman秩相关)

| 目标 | Ridge IC | Ridge std | XGB IC | XGB std | 胜者 |
|------|---------|-----------|--------|---------|------|
| fwd_ret_5d | **0.075** | 0.116 | 0.063 | 0.125 | Ridge |
| fwd_ret_10d | **0.152** | 0.155 | 0.136 | 0.180 | Ridge |
| fwd_ret_20d | **0.213** | 0.252 | 0.197 | 0.274 | Ridge |
| fwd_rv_10d | **0.272** | 0.197 | 0.228 | 0.204 | Ridge |
| fwd_rv_20d | 0.244 | 0.286 | **0.287** | 0.141 | XGB |

### 3.2 回归: 方向准确率

| 目标 | Ridge Dir_Acc | XGB Dir_Acc |
|------|-------------|-------------|
| fwd_ret_5d | **51.1%** ± 7.0% | 48.3% ± 3.4% |
| fwd_ret_10d | 51.1% ± 9.8% | **52.6%** ± 6.9% |
| fwd_ret_20d | 51.9% ± 11.7% | **54.9%** ± 9.4% |

### 3.3 分类

| 目标 | LogReg Acc | XGB Acc | 随机基线 |
|------|-----------|---------|---------|
| direction_5d | 34.1% ± 5.7% | **36.1%** ± 5.4% | 33.3% |
| direction_10d | 36.9% ± 8.3% | **38.7%** ± 5.9% | 33.3% |
| tail_event_flag | 80.8% ± 13.3% | **86.7%** ± 3.5% | 88.5% (majority) |

---

## 四、关键发现

### 4.1 波动率比收益率更可预测

rv 系列的 IC (0.23-0.29) 远高于 ret 系列 (0.06-0.21)。这符合金融理论：波动率具有强聚集效应 (volatility clustering)，当前高波环境大概率延续到未来。

**对期权交易的意义**：波动率预测信号可直接指导 IV 交易 (买低卖高)，这是期权策略的核心优势之一。

### 4.2 长周期信号更强

ret_20d IC (0.21) 是 ret_5d IC (0.08) 的近 3 倍。原因：宏观因子 (实际利率、联邦债务) 是驱动黄金的主力，而宏观因子变化缓慢，对长周期的解释力更强。

**对策略的意义**：偏好 20-30 DTE 的期权 (匹配20日预测窗口)，而非短期期权。

### 4.3 Ridge ≈ XGBoost

4/5 回归任务中 Ridge IC 更高，说明在当前特征集上线性关系是主导的。XGBoost 的非线性优势未体现，可能因为：
- 特征已经做了非线性变换 (z-score, 分位数, 偏离度)
- 样本量 ~3700 对 XGBoost 的 116 特征来说偏小
- XGBoost 在 fwd_rv_20d 上胜出，暗示波动率有非线性结构

**结论**：特征质量比模型复杂度更重要。验证了 "先打好基线" 的设计原则。

### 4.4 方向分类信号微弱

direction_5d/10d 准确率仅 34-39%，虽高于随机 (33.3%)，但增量不大。3分类 (涨/跌/震荡) 本身就难。

**改进方向**：
- 可转为 2 分类 (涨 vs 不涨) 提升信号纯度
- 或用回归预测的符号代替分类模型

### 4.5 尾部事件检测受类别不平衡影响

tail_event_flag 的 XGBoost 准确率 86.7%，但 majority baseline (全预测0) 就有 88.5%。说明模型并未真正学到有效的尾部信号。

**改进方向**：
- 使用 AUC / F1 代替 Accuracy
- 对正样本 (尾部事件) 上采样
- 或重新定义阈值 (当前 3σ 可能太严)

### 4.6 2024-2025 Ridge 信号失效

累积 L/S 信号图显示：Ridge 在 2016-2023 表现稳定，但 2024 后急剧下降。原因是 2024-2025 金价暴涨 (从 $190 到 $280+)，线性模型无法适应这种 regime shift。

**XGBoost 相对更稳健**，因为树模型天然能处理分段关系。这支持了后续加入 Regime 切换模型的计划。

---

## 五、XGBoost 特征重要性 (Top 15)

### fwd_ret_10d (收益率预测)
1. gvz_high (0.054) — GVZ处于高位
2. vix_backwardation (0.037) — VIX期限结构倒挂
3. m2_yoy (0.032) — M2同比增速
4. macd (0.029) — MACD
5. federal_debt (0.027) — 联邦债务水平
6. copper_gold_ratio (0.025) — 铜金比

### fwd_rv_20d (波动率预测)
1. gvz (0.206) — GVZ水平 (压倒性重要)
2. federal_debt (0.081) — 联邦债务
3. gvz_pctile_252d (0.065) — GVZ分位数
4. fed_funds_rate (0.063) — 联邦基金利率
5. tw_usd (0.048) — 贸易加权美元

### tail_event_flag (尾部事件)
1. gvz_sma20_dev (0.055) — GVZ短期偏离
2. us10y_level (0.028) — 10Y收益率
3. gvz_pctile_252d (0.024) — GVZ分位数
4. federal_debt_yoy (0.023) — 债务增速
5. real_yield_10y_zscore (0.022) — 实际利率Z-score

**跨任务共性**: GVZ 在所有任务中都是 top 特征，验证了 "GVZ 是关键变量" 的设计假设。

---

## 六、输出文件

保存到 `data/models/`:

| 文件 | 内容 | 行数 |
|------|------|------|
| `baseline_results.parquet` | 逐 fold 评估指标 | 160 |
| `baseline_predictions.parquet` | 逐日 OOS 预测 | 40,016 |
| `baseline_importances.parquet` | XGBoost 特征重要性 | 2,400 |

质量报告: `reports/phase3_baseline/` (5张图 + REPORT.md)

图表:
1. `01_summary_metrics.png` — IC 和 Accuracy 总览
2. `02_ic_over_time.png` — IC 在各 fold 的稳定性
3. `03_pred_vs_actual.png` — 预测 vs 实际散点图
4. `04_feature_importance.png` — XGBoost 特征重要性 (4个目标)
5. `05_cumulative_signal.png` — OOS L/S 信号累积收益

---

## 七、下一步改进方向

| 方向 | 优先级 | 预期收益 |
|------|--------|---------|
| Regime 切换模型 | 高 | 解决 2024+ Ridge 失效问题 |
| 方向标签转 2 分类 | 中 | 提升方向信号纯度 |
| 尾部事件类别平衡 | 中 | 改善尾部检测召回率 |
| 特征选择 (LASSO/mutual info) | 中 | 减少噪声特征，提升稳定性 |
| GARCH 波动率基线 | 高 | 与 ML 基线对比 |
| 宏观 release-date 严格对齐 | 高 | 消除潜在前视偏差 |
