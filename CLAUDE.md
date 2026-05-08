# GoldDash 项目准则 (项目级 — 永久生效)

## 第一原则: 客观理性最优, 不可主观乱改

**用户明令**:
> "你不可以因为我的任何主观想法, 或者心理因素, 而胡乱修改模型.
>  我们是为了指导交易获取最大利润, 要追求最实际最理性的最优解来获取利益.
>  一切要用事实说话."

### 行为约束

1. **用户提任何"感觉"/"intuition"/case-by-case 担忧时, 不要立刻改 cfg/code**
   - 反例 v3.7.183: 用户说 "FUT 没平 SP 平了 少吃利润" → 我立刻把 FUT 早平改激进 → 5y grid 实测总收益 -50% (反向最优). 撤回 v3.7.184.
   - 正例 v3.7.184: 用户说 "5/6 大涨 SP 平太早" → 我先跑客观 grid → 数据反而显示 SLV pt=30 比 pt=50 强 90% (更早平更优, 跟用户 intuition 反向) → 据此 per-asset 拆分.

2. **每次调参前必须先跑 grid 用数据验证**
   - 至少跨 1y / 3m 两窗口
   - 至少 GLD + SLV 两 asset
   - 看 WR / sum / sB / max_loss 多维度
   - sample 太小 (n<10) 不做决策

3. **回应用户 intuition 的标准流程**:
   - a. **复现** — 用 ledger / backtest CSV 找具体 case 数据
   - b. **诊断** — 是 bug 还是设计 trade-off? 哪种更可能?
   - c. **跑 grid** — 替代方案在历史数据上是否真更优?
   - d. **报告** — 客观数据回答, 哪怕跟 intuition 反向
   - e. **改动** — 只有 grid 验证支持时才改

4. **永远不要**:
   - 因为用户 "5/6 这一笔" 而调全局 cfg
   - 因为用户语气强烈而跳过 grid 验证
   - 因为单笔 max_loss=-100% 加 panic SL (v3.7.179 反例: grid 显示 panic 让总收益 -238%)
   - 同步两个独立策略的退出 (期权/期货 数学上本就该不同)

## 决策框架: 评分指标

```python
scoreA = WR × n × avg                  # 直观线性
scoreB = WR² × log(1+n) × avg          # ★ 高杠杆首要 (WR 平方放大)
scoreC = Kelly_f × √n × avg            # Kelly 数学最优
profit_factor = sum_wins / |sum_losses|  # 纯金额比
```

**默认决策规则**: WR ≥ 75% 内挑 max scoreB.

## 数据源优先级

1. **真实期权 (近 1y kline_db EOD)**: 主权重 0.7
2. **真实期货 (Binance perp 5个月)**: 主权重 0.7
3. **模拟期货 (yfinance GC=F/SI=F 5y COMEX)**: 参考 0.3
4. **模拟期权 (LEAPS BS proxy)**: 参考 0.3 (full_history_backtest.py)

## 项目结构 (v3.7.184)

详见 `docs/ARCHITECTURE.md`. 关键路径:
- `core/strategy_configs.py` — 所有 TP/SL/lev 单点修改
- `scripts/backtest_pipeline.py` — 多级回测主流程
- `scripts/exit_grid_v2.py` — TP/SL/lev grid search
- `scripts/kelly_analysis.py` — Kelly 仓位计算
- `data/backtest_pipeline/versions/<commit>_<date>/` — 版本归档

## 当前最优配置 (grid 实测验证)

```python
FUTURES_GLD: lev=5×, TP=200%, SL=100% margin, hold 20d
FUTURES_SLV: lev=3×, TP=200%, SL=100% margin, hold 20d
SELL_PUT_GLD: pt=50% credit, SL=100% margin, DTE=30
SELL_PUT_SLV: pt=30% credit, SL=100% margin, DTE=30  ★ per-asset
BUY_CALL: pt=1.5x, sl=0.5x, DTE=30
STRADDLE: pt=2x, hold=21d, DTE=30
SHORT_VOL: DISABLED (实战 6% WR 失效)
```

## Workflow

1. 改 cfg → 跑 `exit_grid_v2.py` 验证
2. 跑 `backtest_pipeline.py all` → 自动归档版本
3. 跑 `kelly_analysis.py` 更新仓位推荐
4. 跑 `compute_strategy_stats.py` 刷新 dashboard stats
5. `build_positions_ledger.py` 重建 ledger.json
6. Commit + push (commit 信息含 grid 数据对比表)

## 历史教训

- **v3.7.179 panic SL 反例**: 用户担心单笔 -100%, 我加 panic SL (21d/70%) → grid 反证总收益 -238% → 撤回, 改用仓位管理 (½K 15%/笔, -100%笔仅 -1.5% 总账户)
- **v3.7.183 FUT 早平反例**: 用户感觉 FUT 没平 → 改激进早平 → 5y grid sum -50% → 撤回 v3.7.184
- **永远先看 grid, 再改 cfg**
