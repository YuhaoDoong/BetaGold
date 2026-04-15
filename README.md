# GoldDash — 贵金属交易决策系统

基于宏观因子 + 深度学习的贵金属交易决策系统。支持黄金 (GLD) 和白银 (SLV) 切换。
期权 + 期货结合，信号以看多/看空表示。

GitHub: https://github.com/YuhaoDoong/BetaGold

## 系统架构

```
日线 LSTM+Attention (20年训练, Conformal 80%覆盖)
    → 5日区间预测 → Hybrid Band
        ↓
Regime 分类器 (7因子) + 事件日历 (FOMC/OPEX/NFP)
        ↓
┌─ 方向性信号: 盘中 H/L 触及 Band → BUY CALL / SELL PUT / EXIT
├─ Straddle 信号: RV压缩 + 临近事件日 → 做多波动率
└─ 统一策略选择器: 按评分择优推荐
        ↓
12h K线止盈 (BandExit > Pullback > MACD > StopLoss)
+ 3%止损 + 连续2笔熔断
        ↓
跨市场换算 (GLD/COMEX/XAU/沪金) + 期权策略推荐 (Moomoo实时/EOD)
+ 双套盈亏统计 (金价收益 + 期权收益)
```

## 统一回测 (2025-09 ~ 2026-03)

| 策略 | 信号数 | 胜率 | 说明 |
|------|--------|------|------|
| SELL PUT | 6 | **83%** | 高IV时卖Put，最稳 |
| BUY CALL | 4 | **75%** | 正常IV买Call |
| STRADDLE | 12 | **67%** | 做多波动率 (事件前) |
| EXIT | 13 | 62% | 退出/平仓 |
| **统一** | **35** | **69%** | 择优策略 |

## Dashboard 功能

### 盘中信号 (主模式 — 侧重操作)
- **交易价位** (蓝色) + **实时价格** (紫色): 自动刷新
- **市场分析**: 事件倒计时 + 宏观指标 + Straddle 预警
- **日线图表**: Band + 信号(择优) + 止盈(淡色) + 事件竖线 + Straddle★
- **15m K线**: COMEX Gold 实时蜡烛图 + Stoch RSI + Squeeze (可切换 5m/15m/30m/1h)
  - 日线买入信号活跃时, Stoch RSI < 30 区域高亮为"入场窗口"
  - 超买 > 80 区域标为"止盈窗口"
  - 模型说"今天可以买" → Stoch RSI 说"现在可以下手"
- **期权策略**: 只推荐当日最优 (根据信号+IV自动选择)
- **持仓管理**: 方向性 (Pullback/BandExit) + Straddle (波动/事件到期)
- **统一回测**: 全部策略合并胜率

### 今日预测 (侧重分析)
- **5日区间** + OI微观结构修正 (漏斗形, Max Pain/Call Wall/Gamma)
- **前瞻分析**: 未来10天关键日程 (FOMC/OPEX/NFP) + Straddle 信号
- **宏观环境**: RV/VIX/DXY/实际利率 + 解读
- **期权策略**: 同上, 只推荐最优

### 回测分析
- v1.0 vs v2.2 并排净值曲线
- 近6月/1年/2年

## 信号系统

### 方向性 (盘中 H/L 触发)
| 信号 | 条件 | 退出 |
|------|------|------|
| BUY CALL | Bull + bp(Low)<0.30 + RV≤85% | BandExit > Pullback > MACD > StopLoss(3%) |
| SELL PUT | Bull + bp(Low)<0.30 + RV>85% | 同上 |
| EXIT | bp(High)>0.90 ∪ Regime退出 | — |

### Straddle (评分制, ≥3分触发)
| 条件 | 分数 |
|------|------|
| RV < 20% | +2 |
| RV 下降 > 30% | +1 |
| 距 FOMC ≤ 3天 | +3 |
| 距 NFP ≤ 3天 | +2 |
| 距 OPEX ≤ 3天 | +1 |
| **RV > 25% → 否决** | 成本过高 |

### 统一策略选择
重叠时优先级: **EXIT > Straddle(score≥5) > 方向性 > Straddle(score<5)**

### 止损
- 单笔: 跌超 3% → StopLoss
- 连续: 2笔止损后暂停买入 (bp>0.50 恢复)

## 数据 (每日自动更新)

启动时全量刷新:
1. **yfinance**: GLD/GC=F/DXY/VIX/原油/铜/银/美债/CNY (10个)
2. **FRED**: 实际利率/联邦基金/通胀/M2/贸易加权美元 (8个)
3. **CBOE**: GVZ
4. **特征全量重建**: 64列, 从原始数据计算 (不简化)
5. **模型在线推理**: 加载权重 → 新日期预测 → 追加 OOS

## 模型

### 区间预测 (日线 LSTM+Transformer Ensemble)
- 20年训练 (2004-2026), Walk-Forward 20 折, Conformal 80%
- **双架构集成** (v3.2+): LSTM + Transformer 输出平均 → 校准
  - 单 LSTM:       cov=71.3% width=6.96% tightness=0.102
  - 单 Transformer: cov=69.8% width=6.74% tightness=0.104
  - **Ensemble**:   cov=71.1% width=**6.50%** tightness=**0.109** (宽度最窄 +7%)
- **Qlib Alpha158 因子** (v3.2+): KBAR / BETA / RSQR / QTLU / CORR / CNTP / SUMP 等 110 项
- 预测: 未来5日 High/Low %, RV(10d) 归一化, Quantile Loss (q=0.85/0.15), AdamW

### 训练频率
- **建议每周训练一次**. Dashboard 侧边栏会在模型 > 7 天未训练时显示黄色警告
- 侧边栏 "模型训练" 面板提供:
  - 实时训练状态 + 已运行时长
  - **一键启动训练**按钮 (后台 subprocess, 不阻塞页面)
  - 训练日志滚动查看 + 手动停止

### Regime 分类器 (7因子)
价格动量25% + 联储利率20% + 美元15% + 央行购金15% + 风险10% + 通胀10% + 实际利率5%

### 事件日历
FOMC/OPEX(月度第三周五)/NFP(第一周五) 硬编码2025-2026

## 项目结构

```
GoldDash/
├── app.py                    # Streamlit Dashboard
├── core/
│   ├── data.py               # 数据加载 + 全量刷新 + 在线推理
│   ├── training_status.py    # 模型训练状态 + 后台启动 (v3.2)
│   ├── dl_range.py           # LSTM+Attention 模型 (含 save/load)
│   ├── regime.py             # Regime 7因子分类器
│   ├── signals.py            # v1.0 收盘价信号
│   ├── signals_v2.py         # v2.2 盘中信号 + 12h止盈 + 止损
│   ├── events.py             # 事件日历 + Straddle信号
│   ├── strategy_selector.py  # 统一策略选择器
│   ├── options.py            # 期权策略推荐 (Moomoo live + EOD)
│   ├── options_pnl.py        # 双套盈亏 (金价 + 期权实际)
│   └── oi_factors.py         # OI微观结构修正
├── scripts/
│   └── setup_data.py         # 数据下载 + 特征构建 + 模型训练
├── config.yaml
└── requirements.txt
```

## 参数 (core/signals_v2.py)

```python
EXIT_TIMEFRAME = "12h"     # 止盈检测尺度
PULLBACK_GAIN  = 2.0       # Pullback: 涨幅>N%
PULLBACK_DD    = 1.5       # Pullback: 回撤>N%
STOP_LOSS_PCT  = 3.0       # 单笔止损
CONSECUTIVE_STOP = 2       # 连续止损熔断
BUY_BP = 0.30              # 买入阈值
EXIT_BP = 0.90             # 退出阈值
```

## 快速开始

```bash
conda activate gold
streamlit run app.py
```

## 版本

| 版本 | 核心改进 |
|------|---------|
| v1.0 | 日线收盘信号 + OI修正 |
| v2.0~2.1 | 1h模型 (实验) → 正确架构: v1.0 Band + 盘中触发 |
| v2.2 | 12h止盈 + 参数化 |
| v2.3 | 3%止损 + 连续熔断 |
| v2.4 | 全量数据更新 + 事件日历 + Straddle信号 + 统一策略 + 双套盈亏 |
| v2.5 | 智能去重 + 前瞻分析 + 最优策略推荐 + Straddle持仓管理 |
| v2.6 | 15m K线 + Stoch RSI入场窗口 + Squeeze + 实时GC=F |
| v2.7 | 1h "反转+BB下轨" 入场窗口 (61% WR, 全间隔回测验证) |
| v3.0 | 白银 SLV 模型 + 资产切换 (GLD/SLV) |
| v3.1 | 币安行情 + 区间修复 + 看多看空信号 + 白银增强(59特征) |
| **v3.2** | **Qlib Alpha158 因子 + LSTM+Transformer Ensemble + Dash 训练按钮 (每周训练提示)** |
