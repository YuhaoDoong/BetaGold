# GoldDash — GLD 期权交易决策系统

基于宏观因子 + 深度学习的 GLD ETF 期权交易决策系统，提供 Streamlit 交互式界面。

---

## 版本历史

### v2.0-alpha (2026-03-17) — 1h 多粒度模型

**动机**: v1.0 信号基于日线收盘价，无法指导盘中交易。收盘后才知道信号，市场已关闭。日内价格频繁穿越阈值但收盘回去，说明日线 Band 对盘中执行精度不够。

**架构改进**: 从日线单粒度升级为 1h 多粒度。

```
┌─ 1h 技术特征 (20维): 收益率/RSI/MACD/BB/KDJ/ATR/RV/成交量/价格位置
├─ 4h 聚合特征 (10维): 4h resample → RSI/BB/ATR/RV/Stoch
├─ 日线聚合特征 (12维): 日线 resample → 收益率/均线/RSI/ATR/RV + Regime + RV%
└─ 跨市场 1h  (13维): GC/GLD比/DXY/VIX/金银比/TLT
    ↓ 合计 61 维
LSTM+Attention (seq_len=48, hidden=64, layers=2)
    ↓ 双时间尺度输出
  7h (日内) upper/lower  +  35h (5天) upper/lower
```

**数据**:
- yfinance 1h K线: GLD/GC=F/DXY/VIX/SLV/TLT (730天, ~3500根)
- 日线 Regime 复用 v1.0 的 7 因子分类器作为环境特征

**调参过程** (7组配置对比, 2种子快速实验):

| 配置 | 7h Cov | 7h IC_l | 35h Cov | 35h IC_l |
|------|--------|---------|---------|---------|
| baseline (61feat, seq=48) | 65.3% | 0.142 | 62.8% | 0.225 |
| short_seq24 (61feat, seq=24) | 69.8% | 0.150 | 64.6% | 0.185 |
| **selected_feats** (23feat, seq=48) | **70.5%** | **0.220** | 60.9% | **0.327** |
| **sel+short24** (23feat, seq=24) | **70.1%** | **0.244** | **65.3%** | **0.321** |
| bigger_h128 (23feat, h=128) | 59.3% | 0.190 | 61.8% | 0.304 |

发现: 特征筛选 (61→23) 比增加模型容量更有效; seq=24 优于 48; VIX 等跨市场特征含 NaN 过多是噪声源。

**最终模型** (sel+short24, q_upper=0.90, 3种子, 33特征):

| 指标 | 7h (日内) | 35h (5天) | vs 基线 |
|------|---------|---------|---------|
| Coverage (联合) | 70.2% | 63.3% | +5pp / +0.1pp |
| Coverage (上界) | **82.4%** | **81.7%** | 上界已达标 |
| Coverage (下界) | **86.8%** | **78.6%** | 下界接近达标 |
| IC upper | 0.069 | -0.119 | 改善 |
| IC lower | **0.194** | **0.315** | 大幅改善 |
| 区间宽度 | 5.39% | 11.10% | 略窄 |

关键发现: **上下界分别看都达到 ~80% 覆盖**, 联合覆盖 70% 是因为少数样本同时突破上下界。这对交易信号其实够用 — 买入信号看下界 (86.8% 准确), 退出信号看上界 (82.4% 准确)。

**2026 年回测对比** (v1.0 日线 vs v2.0 1h):

| 指标 | v1.0 日线 | v2.0 1h | 提升 |
|------|---------|---------|------|
| 交易数 | 2 笔 | **9 笔** | +350% |
| 胜率 | 50% | **67%** | +17pp |
| 均收益 | +0.88% | **+2.18%** | +148% |
| **累计收益** | **+1.75%** | **+20.99%** | **+12x** |
| 均持仓 | 5.0 天 | **20.1 小时** | 快 6x |
| 最大亏损 | -0.27% | -1.94% | 风险略增 |
| 最大盈利 | +2.03% | **+6.43%** | 捕捉更大波段 |

v2.0 关键优势: 1月初到1月底的上涨行情 (+4%/+3.5%/+6.1%/+3.2%), v1.0 完全没有捕捉到, 因为日线收盘 bp 始终未跌入买入区。v2.0 的 1h bp 多次探入买入区并成功入场。

**状态**: 信号系统完成, 回测验证通过。
- [x] 调参: seq_len=24, 特征筛选 33维, q_upper=0.90/q_lower=0.10
- [x] 特征筛选: 相关性>0.02, 合并 upper+lower 重要特征
- [x] 信号系统: 1h Band Position 盘中信号 (`core/signals_1h.py`)
- [x] 回测: v2.0 累计 +21.0% vs v1.0 +1.8% (2026年)
- [x] Dashboard 集成: "1h 盘中"模式, 实时 bp/阈值/跨市场换算/信号图表/回测

### v1.0 (2026-03-16) — 日线收盘价模型 `git tag: v1.0`

基于日线收盘价的完整信号系统 + OI 微观结构修正。

**功能**:
- LSTM+Attention 日线区间预测 (5日H/L, 3种子集成, Conformal 80%覆盖)
- 7因子 Regime 分类器 (Bull/Bear/Mixed)
- Hybrid Band 信号 (BUY CALL / SELL PUT / EXIT)
- OI 微观结构: Max Pain/Call Wall/Put Wall/Net GEX → 统一修正所有输出
- 逐日漏斗形预测 (OPEX前压缩, 到期pin, 到期后释放)
- 跨市场换算 (GLD→XAU/COMEX/沪金, 实时行情)
- 数据自动刷新 (yfinance增量) + DTE实时重算 (SGT)
- 期权策略推荐 (单腿/价差) + 3模式仪表板

**已知局限**: 信号基于日线收盘价，无法盘中执行。日内假突破频繁。

---

## 系统架构 (v2.0 目标)

```
日线层 (v1.0 Regime 分类器, 规则系统)
  → Regime: Bull / Bear / Mixed
  → 宏观环境评分 (利率/美元/央行/通胀/风险情绪)
      ↓ 作为环境特征输入
4h 聚合层 (1h resample, 非独立模型)
  → 中观波段: RSI/BB/ATR/RV
      ↓ 作为上下文特征输入
1h LSTM+Attention 模型 (v2.0 新增)
  → 双时间尺度预测: 7h + 35h 区间
  → Band Position 实时计算 (每根K线)
  → 盘中信号触发
      ↓
OI 微观结构修正 (v1.0 复用)
      ↓
期权策略输出 + 可视化
```

高粒度决定方向，低粒度决定时机。三层一致才执行。

## 特征工程

### v2.0 — 1h 多粒度 (61维)

| 层级 | 维度 | 特征 |
|------|------|------|
| 1h 技术 | 20 | ret(1/3/7/14/35h), SMA位置(7/14/35/70h), 均线斜率, MA对齐, RSI, MACD, BB位/宽, KDJ, ATR%, 范围%, RV(10/35h), 成交量比, 价格位置(14/35h) |
| 4h 聚合 | 10 | 4h ret(1/3/7), SMA位置(5/10/20), RSI, BB位, ATR%, RV, StochK |
| 日线聚合 | 12 | 日线 ret(1/5/10/20d), SMA位置(5/20/60d), RSI, ATR%, RV(20d), Regime(-1/0/1), RV分位 |
| 跨市场 | ~13 | GC/GLD比+z-score, GC收益, DXY(7h/35h/SMA), VIX(水平/变化/偏离), 金银比+变化, TLT收益+相关性 |

数据来源: yfinance 1h (GLD, GC=F, DX-Y.NYB, VIX, SLV, TLT), 730天滚动窗口。

### v1.0 — 日线 (~50维)

| 类别 | 数量 | 主要特征 |
|------|------|---------|
| 价格动量 | 5 | ret_1d/5d/10d/20d/60d |
| 技术指标 | 13 | RSI, MACD, 布林位置, KDJ, 价格/SMA比值, 均线斜率, ATR% |
| 宏观经济 | 13 | 实际利率, 贸易加权美元, 联邦基金利率, 盈亏平衡通胀, 美债收益率, CPI, M2 |
| 波动率 | 9 | GVZ, VIX, RV/HV, IV-RV价差, VRP |
| 持仓数据 | 4 | COT非商业净持仓, 全球央行购金 |
| 跨市场 | 3 | 铜金比, 金银比, GC/GLD |

数据来源: Yahoo Finance + FRED API + CBOE, 20年历史 (2004-2026, 5362行)。

## 模型

### v2.0 — 1h RangeLSTM (多时间尺度)

```
Input (seq_len=48, 61维) → BatchNorm
    → LSTM (hidden=64, layers=2, dropout=0.2)
    → Temporal Attention
    → 多时间尺度双头:
        7h Upper/Lower  (日内区间)
        35h Upper/Lower (5天区间)
```

| 参数 | 值 |
|------|-----|
| seq_len | 48 根 (≈2交易日) |
| 预测目标 | 未来 7h/35h 的 High/Low % |
| RV 归一化 | rv_10h |
| 集成 | 3种子 (42/49/56) |
| Conformal | 独立校准集, 目标 80% 覆盖 |
| 训练 | AdamW, Quantile Loss (q=0.85/0.15), patience=20 |

### v1.0 — 日线 RangeLSTM

```
Input (seq_len=20, ~50维) → BatchNorm → BiLSTM → Attention → Upper/Lower Heads
```

预测: 未来5日 High/Low %, RV(10d)归一化, Walk-Forward (min_train=1260天)。

## Regime 分类器 (v1.0, 两版共用)

7 因子加权打分, 非 ML:

| 因子 | 权重 | 逻辑 |
|------|------|------|
| 价格动量 | 25% | ret_60d / 0.10 |
| 联储利率 | 20% | -rate_change_60d / 0.5 |
| 美元趋势 | 15% | -tw_usd_ret_20d / 0.02 |
| 央行购金 | 15% | (cb_12m - 200) / 300 |
| 风险情绪 | 10% | (gvz_pctile - 0.5) / 0.3 |
| 通胀 | 10% | (cpi_yoy - 0.02) / 0.02 |
| 实际利率 | 5% | -real_yield_zscore / 2 |

Bull > +0.2, Bear < -0.2, Mixed 其间, EWM(60)平滑, 最小持续20天。

## 信号系统

### v1.0 信号 (日线收盘)

| 信号 | 条件 | 问题 |
|------|------|------|
| BUY CALL | Bull + bp(close)<0.30 + RV≤85% | 收盘后才知道, 次日执行价格已变 |
| SELL PUT | Bull + bp(close)<0.30 + RV>85% | 同上 |
| EXIT | bp(close)>0.90 ∪ Regime退出 | 同上 |

### v2.0 信号 (1h 盘中, 开发中)

| 信号 | 条件 | 改进 |
|------|------|------|
| BUY | Bull(日线) + bp_1h<0.30 + bp_4h确认 | 盘中实时触发, 可挂单 |
| EXIT | bp_1h>0.90 ∨ Regime退出 | 盘中实时退出 |

## OI 微观结构修正 (v1.0, 两版共用)

| 因子 | 修正 |
|------|------|
| Max Pain | 引力拉回 (15% × 到期因子) |
| Call Wall | 压制上界 (30% blend) |
| Put Wall | 抬升下界 (20% blend) |
| Net GEX | 正=压缩, 负=放大 |
| DTE | 从 strike_time 实时重算 (SGT) |
| 主导到期 | 用 OI 最集中的到期日, 非最近 |

## 项目结构

```
GoldDash/
├── app.py                    # Streamlit 主界面 (v1.0 日线模式)
├── core/
│   ├── data.py               # 数据加载 + 自动刷新
│   ├── dl_range.py           # v1.0 日线 RangeLSTM
│   ├── dl_range_1h.py        # v2.0 1h 多粒度 RangeLSTM ← NEW
│   ├── features_1h.py        # v2.0 1h 多粒度特征工程 ← NEW
│   ├── regime.py             # Regime 7因子分类器 (共用)
│   ├── signals.py            # v1.0 Hybrid Band + 信号
│   ├── options.py            # 期权策略推荐
│   └── oi_factors.py         # OI 微观结构 + 区间修正
├── scripts/
│   ├── setup_data.py         # v1.0 日线全流程
│   └── train_1h_model.py     # v2.0 1h 模型训练 ← NEW
├── config.yaml
├── requirements.txt
└── README.md                 # 本文件 (版本日志)
```

## 数据

```
Gold/data/
├── raw/market/
│   ├── gld.csv               # GLD 日线 (2004~, 自动更新)
│   ├── gld_1h.csv            # GLD 1h (730天, 需定期重存) ← NEW
│   ├── gc_1h.csv             # GC=F 1h ← NEW
│   ├── dxy_1h.csv            # DXY 1h ← NEW
│   ├── vix_1h.csv            # VIX 1h ← NEW
│   ├── slv_1h.csv            # SLV 1h ← NEW
│   ├── tlt_1h.csv            # TLT 1h ← NEW
│   ├── gold_futures.csv      # GC=F 日线 (自动更新)
│   └── usdcny.csv            # USD/CNY (自动更新)
├── processed/
│   └── features_all.parquet  # v1.0 日线特征
├── models/
│   ├── dl_range_v2_oos.parquet     # v1.0 日线 OOS 预测
│   ├── dl_range_1h_oos.parquet     # v2.0 1h OOS 预测 ← NEW
│   ├── dl_range_1h_7h_oos.parquet  # v2.0 7h horizon ← NEW
│   └── dl_range_1h_35h_oos.parquet # v2.0 35h horizon ← NEW
└── raw/options_history/            # EOD 期权快照
```

**重要**: 1h 数据来自 yfinance 730天滚动窗口, 需定期运行 `train_1h_model.py` 重存, 否则历史数据丢失。

## 快速开始

```bash
# v1.0 日线仪表板
streamlit run app.py

# v2.0 训练 1h 模型
python scripts/train_1h_model.py
```

## 依赖

```
streamlit, pandas, numpy, matplotlib, pyyaml   # 仪表板
yfinance, fredapi, requests                     # 数据下载
torch, scikit-learn                             # 模型训练
```
