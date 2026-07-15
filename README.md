# GLD Options Trading System - 黄金期权量化交易系统

基于宏观因子 + 深度学习，构建 GLD ETF 期权交易决策系统。

---

## 一、项目背景与核心思路

### 1.1 起点：中金公司黄金定价模型

中金公司公开发布的 4 因子黄金定价模型（2024年1月首发）：

```
黄金价格 = beta0 + beta1 * 美债实际利率 + beta2 * 美元指数 + beta3 * 央行购金 + beta4 * 美国债务规模 + epsilon
```

- 基于 2003-2023 年数据构建，近 20 年解释误差控制在 200 美元/盎司以内
- 2025 年更新版：延长至 1971-2024 年，剔除美债利率（长期负相关不稳定），精简为 3 因子
- 实盘改进版将误差压缩至 50 美元以内，可进行中频交易

**该模型的局限性**：
- 线性假设，无法捕捉非线性关系
- 低频宏观锚定模型，适合中长期定价，不适合直接做短周期期权择时
- 只预测方向，不预测波动率（期权定价的核心）

### 1.2 我们要解决的问题

"预测 GLD 会涨" 不等于 "买 call 就对"。期权交易需要同时判断三件事：

1. **方向**：GLD 接下来涨还是跌，幅度多大
2. **节奏**：几天、几周、还是几个月内发生
3. **波动率**：即使方向对了，IV（隐含波动率）塌了也可能亏钱

因此，系统不是做 "黄金因子 -> 买 call/put" 的一层模型，而是一个多层管线架构。

### 1.3 核心设计原则

1. **预测模型与策略模型严格分离** — 预测模型学市场运动，策略模型学在给定期权定价下怎么赚钱
2. **先强基线，再深度学习** — 用 XGBoost/线性回归打底，深度学习吃非线性残差
3. **分布预测优于点预测** — 期权 payoff 天生非线性，尾部比均值重要得多
4. **上级时间框架约束下级** — 月线定方向，周线定波段，日线定时机
5. **不造假数据，不偷工减料** — 数据拿不到就停下来讨论，不用模拟数据糊弄
6. **分步推进，稳扎稳打** — 每一步验证通过后再进入下一步

---

## 二、系统总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     A. 数据采集层 (Data Pipeline)                     │
│  8个采集模块覆盖26类数据项: 市场行情/宏观因子/波动率/COT/央行/事件/期权  │
│  状态: ✅ 已完成                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                B. 特征工程层 (Feature Engineering)                    │
│  123个特征 + 12个标签, 5362行 (2004-2026)                             │
│  状态: ✅ 已完成                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 C. 预测引擎层 (Prediction Engine)                     │
│  LSTM+Attention 区间预测 (RV归一化+Conformal+集成)                     │
│  Hybrid Band (U=Daily/L=LagAvg) + Regime + RV → 交易信号               │
│  状态: ✅ 已完成 (Phase 4A)                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               D. 期权策略层 (Options Strategy Engine)                 │
│  信号 → 三档风险期权策略 (稳健/中性/激进) + 风控规则                     │
│  回测引擎 (真实期权价格) + EOD快照自动采集                              │
│  状态: 🔧 Phase 4B 进行中 (数据积累)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、开发路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 数据采集 — 8模块26类数据 | **已完成** |
| Phase 1.5 | 历史期权链存档 — 日终全链+盘中横截面 | **已完成** |
| Phase 2 | 特征工程 — 123特征+12标签, 共线性清理 | **已完成** |
| Phase 3 | 基线模型 — Ridge+XGBoost walk-forward | **已完成** |
| Phase 4A | DL区间预测 + Regime + Hybrid Band → 交易信号 | **已完成** |
| Phase 4B | 期权策略回测 — 真实期权价格验证 | **进行中** |
| Phase 5 | Qlib 集成 — Alpha158 因子 + LSTM/Transformer/ALSTM + Ensemble | **已完成** |

### Phase 5 成果总结 (2026-04)

从 Microsoft Qlib 引入两类资产到单资产时序区间预测:

1. **Alpha158 特征工程** — `src/features/technical_features.py::_add_alpha158_factors`
   - 新增 110 个技术因子 (KBAR / BETA / RSQR / RESI / QTLU / QTLD / IMAX / IMIN /
     CORR / CORD / CNTP / CNTN / SUMP / SUMN / VMA / VSTD)
   - IC/ICIR 筛选后 18 个强因子进入 `SELECTED_FEATURES`
     (`qtld_60d` ICIR=1.41, `sumd_60d` 1.21 等)

2. **模型架构** — `src/models/dl_range_predictor.py`
   - `RangeTransformer`: positional encoding + norm_first encoder
   - `RangeALSTM`: 双路 attention (context + last_hidden)
   - `DLRangePredictor(model_type="ensemble")` **默认** 同时训练 LSTM+Transformer
     (每架构 `n_ensemble` 种子), 预测取平均后做 Conformal 校准

3. **Walk-Forward 对比 (20 折, 3909 样本)**
   | 模型 | Coverage | Width | Tightness | Time |
   |------|---------|-------|----------|------|
   | LSTM | 71.3% | 6.96% | 0.102 | 415s |
   | Transformer | 69.8% | 6.74% | 0.104 | 682s |
   | ALSTM | 69.4% | 6.93% | 0.100 | 461s |
   | **Ensemble (LSTM+Trans)** | **71.1%** | **6.50%** | **0.109** | 2576s |

   Ensemble 宽度最窄 (-7% vs LSTM), 紧凑度最高 (+7%), 匹配 LSTM 覆盖率.

### Phase 4A 成果总结

- **DL Range 模型**: LSTM+Attention, RV(10d)归一化, 独立Conformal校准, 3种子集成
  - OOS 覆盖率 74%, 宽度 6.4%, 紧凑度 0.12 (RV-based 的 2倍)
- **Hybrid Band**: 上界=Daily(灵敏→平仓), 下界=LagAvg(平滑→买入)
- **信号系统**:
  - **BUY CALL**: Bull + bp<0.30 + RV≤85% → 123信号, WR 72%
  - **SELL PUT**: Bull + bp<0.30 + RV>85% → 66信号, WR 73%
  - **EXIT**: bp>0.90 ∪ Regime退出Bull
- **退出策略**: Band Exit + Pullback (peak>2%, dd≥1.5%) + 10d Timeout
- **策略建议**: 信号触发时自动输出三档期权策略 (ITM/ATM/OTM)

详见: [src/models/README.md](src/models/README.md), [src/models/README_dl_range.md](src/models/README_dl_range.md)

### Phase 4B 当前状态

- 回测引擎已完成 (`src/backtest/options_backtest.py`)
- EOD快照自动采集 (3天), K线数据库 74合约 (strikes $200-$320)
- **瓶颈**: ATM K线数据缺失 (GLD $460+, K线额度用完)
- 详见: [src/backtest/README.md](src/backtest/README.md)

---

## 四、项目结构

```
Gold/
├── README.md                              # 本文件 (项目概览)
│
├── scripts/                               # 运行脚本
│   ├── predict_today.py                   # 🔑 每日预测 + 信号 + 期权策略建议
│   ├── build_kline_db.py                  # 期权K线数据库构建
│   ├── smart_kline_download.py            # 智能K线下载 (额度管理)
│   ├── daily_options_collect.sh           # Crontab 每日EOD快照采集
│   └── archive/                           # 历史实验脚本
│       ├── band_compare.py                # Band配置对比 (U/L × Daily/LagAvg)
│       ├── rv_band_compare.py             # RV窗口+Band综合对比
│       ├── exit_viz.py / exit_optimization.py  # 退出策略对比
│       ├── trade_viz.py                   # 交易可视化
│       └── train_rv5d.py                  # RV 5d训练实验
│
├── src/
│   ├── data/                              # 数据采集层
│   │   ├── market/ macro/ volatility/     # 市场/宏观/波动率
│   │   ├── positioning/ events/           # 持仓/事件
│   │   └── options/                       # 期权数据 (Moomoo API)
│   │       ├── moomoo_data.py             # API连接器
│   │       ├── options_archive.py         # EOD快照采集
│   │       └── options_history_builder.py # K线数据库构建
│   │
│   ├── features/                          # 特征工程层
│   │   └── build_features.py             # 123特征+12标签
│   │
│   ├── models/                            # 预测引擎层 (Phase 4A)
│   │   ├── dl_range_predictor.py          # LSTM+Attention 区间预测
│   │   ├── train_dl_range.py              # Walk-forward 20折训练
│   │   ├── regime_classifier.py           # Regime 7因子打分
│   │   ├── analysis_method_compare.py     # Hybrid Band + 信号生成 (核心)
│   │   ├── dl_fair_value.py               # LSTM/Transformer 模型定义
│   │   └── data_utils.py / evaluation.py  # 通用工具
│   │
│   └── backtest/                          # 期权策略层 (Phase 4B)
│       ├── options_backtest.py            # 回测引擎 (真实价格)
│       ├── bs_pricer.py                   # Black-Scholes 参考
│       └── underlying_backtest.py         # 标的回测基线
│
├── data/
│   ├── raw/                               # 原始数据
│   │   ├── market/gld.csv                 # GLD OHLCV ($100~$496)
│   │   └── options_history/               # 期权快照+K线
│   ├── processed/dataset.parquet          # 特征矩阵 (123特征, 5362行)
│   └── models/dl_range_v2_oos.parquet     # DL Range OOS预测
│
├── outputs/                               # 可视化输出
├── reports/                               # 分析报告
└── logs/                                  # 运行日志
```

---

## 五、日常使用

### 5.1 每日预测 (最常用)

```bash
conda activate gold

# 1. 更新数据 (如有新交易日)
python src/data/collect_all.py
python src/features/build_features.py

# 2. 预测 + 信号 + 期权策略建议
python scripts/predict_today.py
```

输出包含:
- 当日 GLD 价格/Regime/RV/Band位置
- 交易信号 (BUY CALL / SELL PUT / EXIT / 无)
- 5天区间预测 (上界/下界)
- **期权策略建议** (稳健/中性/激进三档, 含具体合约+ROI估算)
- **下一交易日信号阈值** (买入/平仓价位预览)

### 5.2 模型训练

```bash
# DL Range 全量 walk-forward 训练 (LSTM+Transformer Ensemble, 20折, ~40-60分钟)
python src/models/train_dl_range.py

# 单架构对比实验 (LSTM vs Transformer vs ALSTM)
python src/models/train_dl_range_compare.py

# Regime 分类
python src/models/train_regime.py
```

> **训练频率**: 建议每周一次. 通过 GoldDash 的 Dashboard 侧边栏可查看上次训练时间,
> 过期时自动提示, 并提供一键启动按钮 (后台运行, 不阻塞页面).

### 5.3 期权数据采集

```bash
# 启动 OpenD
nohup /Users/yhdong/Software/moomoo_OpenD_9.6.5618_Mac/.../OpenD &

# 手动采集 EOD 快照
python -m src.data.options.options_archive

# K线数据库构建 (受额度限制: 100次/30天)
python scripts/build_kline_db.py
```

---

## 六、技术栈

| 类别 | 工具 |
|------|------|
| **环境** | Conda env: `gold`, Python 3.11, Apple Silicon MPS |
| **深度学习** | PyTorch 2.10 (LSTM+Attention, Quantile Loss) |
| **传统ML** | scikit-learn, XGBoost |
| **数据获取** | yfinance, fredapi, moomoo-api |
| **数据处理** | pandas 3.x, numpy, pyarrow |
| **可视化** | matplotlib |
| **回测** | 自建框架 (真实期权价格) |

---

## 七、关键技术备忘

| 问题 | 解决方案 |
|------|----------|
| LSTM vs Transformer | 单模型 LSTM 略胜; **Ensemble (LSTM+Transformer) 最佳** (width -7%) |
| Qlib Alpha158 | 借 KBAR/BETA/RSQR/QTLU/CORR/CNTP/SUMP 因子, ICIR 筛选 18 个进入生产 |
| Ensemble vs 单模型 | 双架构 +2 种子 = 4 模型, 训练 ~6× 但周度训练成本可接受 |
| 区间预测归一化 | 目标除以 RV×sqrt(5), 模型预测 sigma multiplier |
| Conformal calibration | 必须用独立cal集 (不能用val集), 需过滤NaN |
| Hybrid Band 上界 | Daily(lag1), 不用LagAvg (大涨时连续假Exit信号) |
| Hybrid Band 下界 | LagAvg(lag1,2,3), 平滑 → 捕捉急跌底部买点 |
| RV归一化窗口 | RV(10d), 不用5d (覆盖率66%→74%) 或 20d |
| 退出策略 | Band+Pullback (peak>2%, dd≥1.5%) + 10d timeout |
| RV>85% | 不作sell信号, 改为区分买入类型 (Call→Put) |
| Moomoo K线额度 | 100次/30天, 滚动释放 |
| pandas 3.x | resample 用 'YE'; fillna(method=...) 改用 .ffill() |
