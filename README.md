# GoldDash — GLD 期权交易仪表板

基于宏观因子 + 深度学习的 GLD ETF 期权交易决策系统，提供 Streamlit 交互式界面。

## 系统总览

```
特征工程 (7类 ~50维)
        ↓
LSTM+Attention 区间预测 (5日 upper/lower, 3种子集成 + Conformal校准)
        ↓
Hybrid Band → Band Position → 交易信号 (BUY CALL / SELL PUT / EXIT)
        ↓                                    ↓
Regime 7因子加权              EOD 期权快照 → OI 微观结构修正
        ↓                                    ↓
              ┌────────── 统一输出 ──────────┐
              │ 预测区间 · 买卖阈值 · 跨市场换算 │
              │ 期权策略 · 历史Band · 可视化图表 │
              └─────────────────────────────┘
```

## 功能

### 今日预测模式
- GLD 价格 + Hybrid Band + 交易信号 (近 N 天)
- 5日区间预测 (LSTM+Attention 模型输出)
- **OI 微观结构修正**: Max Pain / Call Wall / Put Wall / Net Gamma → 统一修正预测区间、买卖阈值、跨市场换算、期权策略目标价
- **逐日漏斗形预测**: OPEX 前压缩、到期日 pin、到期后释放 (非平矩形)
- 下一交易日买入/平仓价位阈值 (OI 修正后与原始值并排对比)
- 跨市场价位换算 (GLD → XAU/COMEX/沪金, 实时行情)
- 期权策略推荐 (退出目标价使用 OI 修正值):
  - BUY CALL → 单腿 Call + Bull Call Spread (并排对比 + IV推荐)
  - SELL PUT → Bull Put Spread (牛市看跌价差)
- OI 微观结构详情: 概念解释、方向分析、到期日分布、关键事件
- 近期信号历史 + 交易记录

### 数据自动刷新
- 每次启动自动检测 GLD / 黄金期货 / USD/CNY 数据是否过期
- 过期则通过 yfinance 下载增量数据并追加到 CSV
- EOD 期权快照的 DTE 从今日 (SGT/UTC+8) 实时重算, 不依赖快照时的陈旧值
- 侧边栏显示各数据源状态

### 历史回看模式
- 自定义起止日期, 查看任意时段的信号和交易轨迹
- 历史 Band 经 OI 快照修正 (所有可用快照)

### 回测分析模式
- 近1年/2年/3年信号策略收益曲线
- 对比买入持有基准 (含最大回撤)
- 交易统计摘要

## 特征工程

~50 维特征，覆盖 7 个大类，时间跨度 2004–2026 (5362 行):

| 类别 | 数量 | 主要特征 |
|------|------|---------|
| 价格动量 | 5 | ret_1d/5d/10d/20d/60d |
| 技术指标 | 13 | RSI, MACD, 布林位置, KDJ, 价格/SMA比值(5/20/60/120), 均线斜率, ATR% |
| 宏观经济 | 13 | 实际利率(10Y), 贸易加权美元, 联邦基金利率, 盈亏平衡通胀, 美债收益率, CPI, M2 |
| 波动率 | 9 | GVZ 及其分位数, VIX 水平/期限结构, RV/HV, IV-RV价差, VRP |
| 持仓数据 | 4 | COT非商业净持仓(变化/分位数/OI变化), 全球央行购金(12m滚动) |
| 跨市场 | 3 | 铜金比变化, 金银比, GC/GLD比值z-score |
| 辅助技术 | 3+ | 布林宽度, 成交量比率, 缺口% |

**数据来源**: Yahoo Finance (市场), FRED API (宏观), CBOE (GVZ), COT 报告, 央行数据

## 模型: LSTM+Attention 区间预测

### 架构

```
Input (seq_len=20, n_features) → BatchNorm
    → Bidirectional LSTM (hidden=64, layers=2, dropout=0.2)
    → Temporal Attention (Linear→Softmax, 加权聚合时间步)
    → 双头输出:
        Upper Head: Linear(64→32) → ReLU → Dropout → Linear(32→1) → Softplus
        Lower Head: Linear(64→32) → ReLU → Dropout → Linear(32→1)
```

### 预测目标

5 日价格区间 (百分比):
- **Upper**: `(未来5日最高价 / 当前收盘价 - 1) × 100`
- **Lower**: `(未来5日最低价 / 当前收盘价 - 1) × 100`

目标经 **RV(10d) 归一化**: `target_norm = target / rv_scale`，其中 `rv_scale = log收益率.rolling(10).std() × √5 × 100`

### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 损失函数 | Quantile (Pinball) Loss | q_upper=0.85, q_lower=0.15 |
| 优化器 | AdamW | lr=1e-3, weight_decay=1e-4, grad_clip=1.0 |
| 调度器 | ReduceLROnPlateau | factor=0.5, patience=5 |
| 训练轮数 | 150 (max) | early stopping patience=20 |
| 批大小 | 64 | |
| 集成 | 3 模型 | seeds: 42, 49, 56, 取均值 |
| 特征缩放 | RobustScaler | 仅在训练集上 fit |

### Walk-Forward 验证

滚动窗口 OOS 评估，避免未来信息泄露:

```
├─ 训练集: ≥1260天 (~5年)
├─ 验证集: 最后252天 (early stopping)
├─ 校准集: 126天 (独立于训练/验证, 用于 Conformal)
├─ 测试集: 252天/fold
└─ 步长: 126天 (半年滚动一次)
```

### Conformal 校准

在独立校准集上计算残差分位数，保证预测区间达到 **80% 覆盖率**:
- `margin = percentile(|actual - pred|, √0.80 × 100)`
- 最终预测 = 集成均值 ± 校准 margin

## Regime 分类器

基于规则的 7 因子加权打分 (非 ML)，每因子评分 [-1, +1]:

| 因子 | 权重 | 计算逻辑 |
|------|------|---------|
| 价格动量 | 25% | ret_60d / 0.10 |
| 联储利率方向 | 20% | -fed_funds_rate_change_60d / 0.5 (降息利好) |
| 美元趋势 | 15% | -tw_usd_ret_20d / 0.02 (美元弱利好) |
| 央行购金 | 15% | (cb_global_12m_rolling - 200) / 300 |
| 风险情绪 | 10% | (gvz_pctile_252d - 0.5) / 0.3 (高波动利好) |
| 通胀水平 | 10% | (cpi_yoy - 0.02) / 0.02 |
| 实际利率 | 5% | -real_yield_10y_zscore / 2 |

**Regime 划分**: 加权总分经 EWM(span=60) 平滑后:
- **Bull** > +0.2 | **Bear** < -0.2 | **Mixed** 其间
- 最小持续期: 20 天 (防止频繁切换)

## 信号系统

| 信号 | 条件 | 含义 |
|------|------|------|
| **BUY CALL** | Bull + bp<0.30 + RV≤85% | 正常IV，买入看涨期权 |
| **SELL PUT** | Bull + bp<0.30 + RV>85% | 高IV，卖出看跌期权收premium |
| **EXIT** | bp>0.90 ∪ Regime退出Bull | 平仓/止盈 |

### Hybrid Band

不对称设计 — 上界灵敏捕捉退出时机，下界平滑过滤噪音:

- **上界** (Lag1, 灵敏): `close(t-1) × (1 + pred_upper%(t-1) / 100)`
- **下界** (Lag1-3均值, 平滑): `avg[close(t-k) × (1 + pred_lower%(t-k) / 100)]`, k∈{1,2,3}
- **Band Position**: `bp = (close - lower) / (upper - lower)`

### 退出机制
- **BandExit**: bp > 0.90 触发
- **Pullback**: 涨幅>2% 后回撤≥1.5%
- **Timeout**: 最长持仓 10 天

## OI 微观结构修正

基于 EOD 期权快照，提取 OI 微观结构因子，**统一修正**以下所有输出：

| 修正目标 | 说明 |
|----------|------|
| 5日预测区间 | 逐日漏斗形 (OPEX前压缩, 到期pin, 到期后释放) |
| 买入/平仓阈值 | bp=0.30/0.90 价位经 OI 修正, 图表+文字同步 |
| 跨市场换算 | XAU/COMEX/沪金价位使用修正后阈值 |
| 期权策略目标价 | 策略推荐的退出目标使用修正后的 bp=0.90 |
| 历史 Band | 用所有可用快照对历史 Band 上下界施加 OI 修正 |

### OI 因子

| 因子 | 作用 | 修正逻辑 |
|------|------|---------|
| **Max Pain** | Pin 效应 (到期吸引) | 价格偏离 Max Pain → 引力拉回 (15% × 到期因子) |
| **Call Wall** | 上方阻力 (做市商对冲) | 预测上界高于 Call Wall → 压制上界 (30% blend) |
| **Put Wall** | 下方支撑 | 预测下界低于 Put Wall → 抬升下界 (20% blend) |
| **Net GEX** | Gamma 压缩/放大 | Long gamma → 压缩区间; Short gamma → 扩大区间 |

到期因子 `expiry_factor = clip(1 - (DTE - 7) / 30, 0.1, 1.0)`: 临近到期效应增强。

### DTE 实时重算

EOD 快照中的 `dte` 列是采集时相对于快照日期的值 (如 3/14 快照中 dte=6)。每次运行时，从 `strike_time` (实际到期日) 和今日日期 (SGT) 重新计算 DTE (如今日 3/16 → dte=4)，确保 OI 修正反映当前时间状态。

### 主导到期日

使用 OI 最集中的到期日 (`dominant_dte`) 而非最近到期日驱动 pin 效应。月度 OPEX 通常占总 OI 的 40-80%，是真正驱动做市商对冲行为的力量，周度到期的 OI 通常不足 5%。

## 快速开始

### 方式一：独立使用 (推荐)

从零开始，自动下载数据并构建特征：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据 + 构建特征 + 训练模型
#    首次需要 FRED API Key (免费: https://fred.stlouisfed.org/docs/api/api_key.html)
python scripts/setup_data.py --fred-key YOUR_API_KEY

# 3. 修改 config.yaml
#    将 data_root 改为 "data" (使用本地下载的数据)

# 4. 启动仪表板
streamlit run app.py
```

后续日常更新数据：
```bash
python scripts/setup_data.py          # 更新数据 + 重训模型
python scripts/setup_data.py --no-train  # 仅更新数据
```

> 日常使用无需手动运行 `setup_data.py` — 仪表板启动时会自动检测并下载最新的 GLD/期货/汇率数据。`setup_data.py` 仅在需要重建特征或重训模型时使用。

### 方式二：配合 Gold 项目使用

如果已有 [Gold](https://github.com/your-repo/Gold) 项目的数据：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编辑 config.yaml，将 data_root 指向 Gold 项目的 data/ 目录
#    data_root: "/path/to/Gold/data"

# 3. 启动仪表板
streamlit run app.py
```

## 配置

编辑 `config.yaml`：

```yaml
# 方式1: 使用本地下载的数据
data_root: "data"

# 方式2: 指向 Gold 项目
data_root: "/path/to/Gold/data"
```

需要的数据文件：
```
data/
├── processed/features_all.parquet   # 特征矩阵 (~50维 × 5362行, 2004-2026)
├── raw/market/gld.csv               # GLD OHLCV (启动时自动更新)
├── raw/market/gold_futures.csv      # 黄金期货 (启动时自动更新)
├── raw/market/usdcny.csv            # USD/CNY汇率 (启动时自动更新)
├── models/dl_range_v2_oos.parquet   # DL Range OOS 预测结果
└── raw/options_history/             # EOD 期权快照 (可选，用于OI修正+策略推荐)
    └── 2026-03-14/eod_full.parquet
```

## 项目结构

```
GoldDash/
├── app.py                # Streamlit 主界面 (今日预测/历史回看/回测分析)
├── core/
│   ├── data.py           # 数据加载 + 自动刷新 (yfinance增量下载)
│   ├── dl_range.py       # RangeLSTM 模型定义 + 集成推理
│   ├── regime.py         # Regime 7因子加权打分分类器
│   ├── signals.py        # Hybrid Band + 信号生成 (V2)
│   ├── options.py        # 期权策略推荐 (单腿/价差/Bull Put Spread)
│   └── oi_factors.py     # OI 微观结构因子 + 区间修正 + DTE实时重算
├── scripts/
│   └── setup_data.py     # 全流程: 数据下载 → 特征构建 → Walk-Forward训练
├── config.yaml           # 数据路径配置
├── requirements.txt
└── README.md
```

## 数据源

| 来源 | 数据 | 用途 |
|------|------|------|
| Yahoo Finance | GLD, GC=F, DXY, VIX, 原油, 铜, 白银, 美债收益率, CNY=X | 市场特征 + 实时换算 |
| FRED | 实际利率, 盈亏平衡通胀率, 联邦基金利率, CPI, M2, 贸易加权美元 | 宏观特征 + Regime因子 |
| CBOE | GVZ (黄金波动率指数) | 波动率特征 + 风险情绪 |
| Moomoo OpenD | EOD 期权快照 (OI/Greeks/IV) | OI微观结构修正 |

启动时自动增量更新: GLD, 黄金期货 (GC=F), USD/CNY (CNY=X) — 通过 yfinance。

## 依赖

核心逻辑自包含在 `core/` 中，不依赖 Gold 项目的代码。

```
streamlit, pandas, numpy, matplotlib, pyyaml   # 仪表板
yfinance, fredapi, requests                     # 数据下载
torch, scikit-learn                             # 模型训练 (可选)
```
