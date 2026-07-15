# Phase 2: 特征工程层

将原始数据转化为模型可用的特征矩阵 + 预测标签。

## 模块结构

| 模块 | 内容 | 特征数 |
|------|------|--------|
| technical_features.py | 价格/成交量技术指标 + 关联品种 | 52 |
| macro_features.py | FRED 宏观因子 + COT 持仓 + 央行购金 | 33 |
| vol_features.py | GVZ/VIX/MOVE 波动率 + 事件窗口 | 31 |
| label_builder.py | 预测标签 (收益率/方向/波动率/尾部) | 12 |
| build_features.py | 主入口: 合并全部特征 + 保存 | - |

## 运行

```bash
conda activate gold
python src/features/build_features.py
```

## 输出

保存到 `data/processed/`:
- `features_all.parquet` — 全部特征矩阵
- `labels.parquet` — 预测标签
- `dataset.parquet` — 合并数据集 (特征+标签)
- `feature_list.csv` — 特征名单 + 缺失率
- `label_list.csv` — 标签名单

## 特征分类

### 快通道 (日频, 52个)
- 收益率: 1d/2d/3d/5d/10d/20d/60d
- 均线偏离度: close_to_sma(5/10/20/60/120), 均线排列, 斜率
- 动量: RSI(7/14), MACD(line/signal/hist), 随机指标
- 波动: HV(5/60d), Bollinger(width/position), ATR, 日内幅度
- 成交量: 量比, OBV方向, 量价配合
- 跳空: 缺口幅度/方向
- 关联品种: DXY, VIX, 铜金比, 金银比, 10Y收益率, 原油

已清理: 删除 log_ret_1d(=ret_1d), ROC(=ret), 原始SMA/EMA价位, BB上下轨, hv_10d/20d(→vol模块)

### 慢通道 (月/季 -> ffill 到日频, 33个)
- 实际利率: 10Y/5Y TIPS, 利率曲线, Z-score
- 通胀: 盈亏平衡通胀率, CPI 同比
- 政策: 联邦基金利率, 实际联邦基金利率
- 财政: 联邦债务增速, 赤字12月滚动
- 货币: M2 同比
- 美元: 贸易加权美元, Z-score
- COT: 非商业/商业净持仓, 变化, 分位数
- 央行: 全球购金量, 滚动, 中国购金

### 波动率通道 (31个)
- GVZ: 水平, 变化, 分位数, SMA偏离
- VIX: VIX9D/VIX3M/VIX6M, 期限结构斜率, backwardation, 9D vs 30D
- MOVE: 水平, 变化, 分位数
- RV: 10d/20d已实现波动率(×100对齐GVZ), VRP, VRP分位数
- 事件: 距FOMC/NFP/CPI天数, 事件窗口标记

已清理: 删除 vix_vix(=tech vix_level)

## 关键规则

- 宏观数据按 release date 对齐 (目前先用 ffill，Phase 后续加入严格对齐)
- 标签列名以 `fwd_` 前缀标记 (如 `fwd_ret_5d`)，避免与特征同名
- 标签为未来值 (shift(-N))，回测中不可作为特征使用
- 分位数计算使用 rolling 252 天窗口 (1年)
- 可视化报告: `reports/phase2_features/` (7张图 + REPORT.md)
