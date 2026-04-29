# GoldDash — 贵金属交易决策系统

基于宏观因子 + 深度学习的贵金属交易决策系统。支持黄金 (GLD) 和白银 (SLV)，三层模型 + 多策略竞争 + 实证回测验证。

**GitHub**: https://github.com/YuhaoDoong/BetaGold

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构 — 三层模型, 数据流, 项目结构, Dashboard 模式 |
| [docs/MODELS.md](docs/MODELS.md) | 模型架构 — LSTM+Transformer Ensemble, Conformal 校准, Regime 7 因子分类 |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | 策略详解 — 完整策略矩阵, vega/delta 分析, 胜率定义, 工具映射 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验记录 — 所有回测结果, 阈值调优, regime 分段, 期货 vs 期权对比 |

## 快速入门

```bash
conda activate gold

# 启动 Dashboard
streamlit run app.py

# 历史回填盘中触发 (首次或规则变更后)
python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60
python scripts/backfill_intraday_signals.py --asset SLV --timeframe 60
```

## 核心特点

### 三层模型架构
```
日线层 (今日预测)    模型预测 → 5日区间 + bp030/bp090 阈值 ("开窗信号", 不是入场价)
        ↓
盘中层 (盘中信号)    实时盘中: 价格在阈值外侧 + Stoch RSI/MACD/KDJ 确认
        ↓ 写入 data/intraday_signal_log.parquet
固化层 (历史代表价)  每日多触发取最差 (买:max / 卖:min) → 持仓/回测全部读这里
```

### 完整策略矩阵 (v3.6.x 实证最优)

| RV %tile | 方向性信号 | 推荐工具 | 备选波动率策略 |
|----------|-----------|----------|----------------|
| < 50% (低/中性) | BUY CALL 类 | **期货多头 + 3% 止损** (96% wr) | Long Straddle (低 RV+临 FOMC) |
| 50-85% (温水) | ❌ 屏蔽 | ❌ 不做方向 | **Iron Condor 16Δ/5Δ** (83% wr) |
| > 85% (恐慌) | SELL PUT 类 | **期权 Sell Put** (100% wr) | Long Straddle (临界) |

### 胜率定义 (按 vega/delta 实际盈亏)

| 策略 | 胜利条件 (动态 sigma_pct = RV × √h/252) |
|------|-----------------------------------------|
| BUY CALL | `max_up > 1σ` (横盘是亏) |
| SELL PUT | `max_down < 1σ` (横盘+上涨都赢) |
| STRADDLE | `max_move > 1σ` (双向移动 > premium) |
| SHORT_VOL | `max_move < 1.6σ` (IC 短腿内) |
| 期货多头 | `ret_5d > 0` (任何正向收盘) |

详见 [docs/STRATEGIES.md](docs/STRATEGIES.md)

## 关键实证结论 (近 5 年)

1. **方向性 RV 极值过滤**: 排除中位 50-85% 后, 胜率 78% → **81%**, Sharpe 0.53 → **0.61**
2. **期货代替 BUY CALL 期权**: 胜率 73% → **96%**, Sharpe 0.23 → **1.16** (5 倍提升)
3. **Iron Condor 严格时机**: 89% 胜率 (vs Short Strangle 40%), 翼锁定最大亏损
4. **Regime 分段**: Bull 84% / Mixed 45% / Bear 30% — 现状已 regime-optimal
5. **STRADDLE 完全 regime-agnostic**: Mixed 反而是最佳战场 (90%)

详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

## 版本历史

| 版本 | 主题 |
|------|------|
| v1.0-v2.x | 日线 Band + 盘中触发 + 12h 止盈 + 3% 止损 |
| v3.0-v3.4 | 白银扩展 + Qlib Alpha158 + Ensemble 模型 + 多周期 Stoch RSI + 伦敦金价位 |
| v3.5 | 盘中触发模块: 参数化规则 (Stoch RSI/MACD/KDJ) + 持久 log + 历史回填 |
| v3.6.x | 完整策略矩阵: 做多/做空波动率 + 方向性 RV 过滤 + 期货独立统计 + 工具映射 |
| v3.7 | 模块化重构 (core/strategies/) + 4 文档拆分 + 持仓管理 SHORT_VOL |
| v3.7.x | 修复持仓管理时间倒序 + 活跃持仓暴露 + 完整交易历史合并 |
| v3.7.8 | 熔断 A/B 验证 + 默认关闭 (实证不提升胜率, 错杀赢面) |
| v3.7.9 | 持仓管理加退出日 + 退出原因列 |
| v3.7.10 | 修复 MIXED 优先级 bug (score≥6 单走 vol, 4-5 才 MIXED) |
| v3.7.11-13 | IV crush 模块化 (core/iv_crush.py) + 用真实 GVZ 数据校准 |
| v3.7.14 | IV crush 调整默认关闭 (GLD 实证不显著), 保留模块 + Dashboard 显示 IV/RV |
| v3.7.15-25 | UI / 主图 1h 化 / 5 子图 sharex / 期权策略实时面板 / 持仓管理实时退出判定 / FOMC 日期修正 |
| v3.7.26-28 | SLV 1h 数据兜底 / auto_refresh 补 SLV 1h / 主图横坐标 / 1-60 天 / 今日预测 sharex 合并 |
| v3.7.29 | RV 阈值精细网格 (步长 0.025) GLD 优化 |
| **v3.7.30** | **Per-Asset 校准: SLV 单独 grid search, 与 GLD 显著不同 (SLV 笔数翻倍, vol 大). 集中参数管理 core/strategy_config.py + scripts/tune_thresholds.py 定期重测脚本** |

## 用户偏好

- 时区: 新加坡 SGT (UTC+8)
- 信号基于金价，不限定交易品种 (期权/期货/现货)
- 数据每天及时更新, 不用陈旧数据
- 不偷工减料, 不用模拟数据
