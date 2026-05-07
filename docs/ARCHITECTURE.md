# GoldDash 系统模块架构 (v3.7.177)

## 顶层结构

```
GoldDash/
├── app.py                      # Streamlit dashboard 入口
├── core/                       # 核心业务模块
├── scripts/                    # CLI 工具 (回测/grid/数据/部署)
├── docs/                       # 文档 (ARCHITECTURE / STRATEGIES / MODELS / EXPERIMENTS)
└── /Users/yhdong/Gold/data/    # 数据 (跨项目共享, Gold/GoldDash 同时读)
    ├── raw/market/             ETF + 期货 daily OHLC (gld.csv, slv.csv)
    ├── processed/              模型 features / labels parquet
    ├── kline_db/               EOD 期权 OHLC (Moomoo daily)
    ├── backtest_pipeline/      回测多级输出 + 版本归档
    │   ├── stage0_raw_signals/  无过滤 detect 输出
    │   ├── stage1_filtered/     IV/regime/sp_score 过滤后
    │   ├── stage2_simulated/    分源 PnL parquet (real_klinedb / sim_comex)
    │   ├── stage3_summary/      加权 scoreB 汇总
    │   └── versions/            按 git commit 归档 (v3.7.177+)
    ├── intraday_signal_log.parquet
    └── positions_ledger.json   单一持仓真理 (dashboard 读)
```

## core/ 模块映射

### 信号层 (signals)
| 文件 | 职责 |
|---|---|
| `signals_v2.py` | 日级 directional (BC/SP) 信号 + sp_score 智能选 |
| `signals.py` | Band 计算 + RV pctile (旧版底层) |
| `events.py` | STRADDLE / SHORT_VOL / 事件日历 detect |
| `regime.py` | 市场状态 (Bull/Bear/Neutral) 分类 |
| `intraday_triggers.py` | 实时盘中信号 log + dedupe |

### 数据接入层 (data)
| 文件 | 职责 |
|---|---|
| `binance_futures.py` | Binance USDT-M perp API (XAUUSDT/XAGUSDT) |
| `data.py` | features/labels parquet 加载 |
| `paper_positions.py` | kline_db loader + price_strategy_at + simulate dispatcher |

### 策略层 (strategies/)
| 文件 | 职责 |
|---|---|
| `futures_long.py` | 期货多头 sim (lev/TP/SL/早平/爆仓) |
| `buy_call.py` | BC long call 模拟 (BCConfig) |
| `sell_put.py` | SP credit spread 模拟 (SPConfig) |
| `short_vol.py` | Iron Condor 模拟 (4-leg max_risk) |
| `straddle.py` | Long ATM call+put 模拟 |
| `options_exit.py` | 现代化 exit (DTE-cliff + signal-reversal) |
| `win_metrics.py` | 各策略胜率 (vega/delta) 定义 |
| `futures_pnl.py` | 期货 P&L 计算工具 |

### 配置中心 ★
| 文件 | 职责 |
|---|---|
| **`strategy_configs.py`** | **所有 TP/SL/lev/DTE 集中管理** (单点改) |

### 部署 / 守护
| 文件 | 职责 |
|---|---|
| `ledger_daemon.py` | 后台守护进程 (5min 重建 ledger.json) |

## scripts/ 关键工具

| 脚本 | 用途 |
|---|---|
| `build_positions_ledger.py` | 一键生成 `positions_ledger.json` (Dashboard 单一数据源) |
| **`backtest_pipeline.py`** ★ | **多级回测主流程** (stage0-3 + 版本归档) |
| `exit_grid_v2.py` | TP/SL/lev/DTE 4D grid search (scoreB 评分) |
| `full_history_backtest.py` | 5y/10y 全历史回测 (LEAPS BS proxy) |
| `daily_eod_options.py` | Moomoo 期权 OHLC daily download → kline_db |
| `monthly_retune.py` | 月度模型重训 (LSTM + Transformer) |
| `paired_grid_multi.py` | BC vs SP 配对 grid search |

## 评分指标 (v3.7.170)

```python
scoreA = WR × n × avg                      # 直观线性
scoreB = WR² × log(1+n) × avg              # ★ 高杠杆首要 (WR 平方放大)
scoreC = Kelly_f × √n × avg                # 含 win/loss 比, 数学最优
profit_factor = sum_wins / |sum_losses|    # 纯金额比
```

**决策规则**: WR ≥ 75% 内挑 max scoreB.

## 数据流 (信号 → 持仓 → Dashboard)

```
features.parquet (Gold/data/processed)
         ↓
signals_v2.generate_daily_signals  → BC/SP/sp_score
events.detect_*                    → STRADDLE/SHORT_VOL/事件日
         ↓                                          ↓
         └────── stage0 raw_signals ───────────────┘
                          ↓
                  IV/regime 过滤
                          ↓
                stage1 filtered_signals
                          ↓
              ┌───────────────────────┐
              ↓                       ↓
     strategies/* simulate_*    Binance/COMEX kline
              └───── stage2 ──────────┘
                   pnl parquet
                          ↓
              scoreB 加权 (real 0.7 + sim 0.3)
                          ↓
                stage3 summary parquet
                          ↓
       ⇒ versions/<git_commit>_<date>/  归档

build_positions_ledger.py (5min daemon, 实时)
         ↓
positions_ledger.json (单一真理)
         ↓
app.py (Dashboard 读 JSON 渲染)
```

## 当前推荐配置 (v3.7.177)

基于 5y COMEX grid + 1y kline_db + Binance 实测:

| 策略 | Asset | cfg | 5y WR | 备注 |
|---|---|---|---|---|
| FUTURES_LONG | GLD | lev=5× TP200/SL100/h20 | 86% | sharpe=0.56, 1.3% 爆仓 |
| FUTURES_LONG | SLV | lev=3× TP200/SL100/h20 | 77% | sharpe=0.28, 1% 爆仓 |
| BUY CALL | both | pt=1.5x sl=0.5x DTE=30 | 100%* | *上涨期触发 (sp_score 智能筛) |
| SELL PUT | both | pt=50% sl=100% DTE=30 | 75-80% | 跨期最稳 ★ |
| STRADDLE | both | pt=2x hold=21d DTE=30 | 68% | 月度 expiry |
| SHORT_VOL | both | pt=50% sl=50% hold=30d | **6%** ⚠ | 当前波动期失效, 建议停用 |

## 信号频率 (1y / 5y)

| Asset | 策略 | 1y | 5y/yr 平均 |
|---|---|---|---|
| GLD | BC | 29 | 26 |
| GLD | SP | 14 | 3 (5y 偏少) |
| GLD | FUT | 15 (Binance 窗口内) | 30 (5y sim) |
| SLV | BC | 27 | 24 |
| SLV | SP | 46 | 22 |
| SLV | FUT | 30 | 45 (5y sim) |

## 期货杠杆 grid (5y, lev vs WR vs sB)

GLD (n=151):
- lev=2-3×: WR 86% sharpe 0.67 ★ 最优 risk-adjusted
- **lev=5×** ✓ 当前: WR 86% sharpe 0.56 sB +35
- lev=10×: WR 85% avg+18% sB +63 (5% 爆仓)
- lev=20×: WR 82% avg+32% sB +107 (14% 爆仓 ⚠)

SLV (n=234):
- lev=2-3×: WR 77% sharpe 0.30 ★
- **lev=3×** ✓ 当前: WR 77% sharpe 0.28 sB +18
- lev=10×: WR 73% sB +36 (19% 爆仓 ⚠)
- lev=20×: WR **54%** sB +17 (45% 爆仓 ⚠⚠)

## 开发规范

1. **修参数**: 只在 `core/strategy_configs.py` 改, 不动各策略 Config dataclass 默认
2. **加策略**: 新建 `core/strategies/<name>.py` + Config dataclass + simulate function, strategy_configs 注册
3. **改信号**: 改 `signals_v2.py` 或 `events.py`, 注意 stage0 输出列保持向后兼容
4. **回测**: 跑 `scripts/backtest_pipeline.py all`, 自动归档版本可对比
5. **commit**: 改 strategy 后跑 grid 验证 + 重建 ledger 再 commit

## 关键版本演进

- v3.7.166-167: 集中配置 + 模块化重构 (FuturesConfig 统一)
- v3.7.168-170: WR-first scoreB 评分, lev=5×/3× wick-safe
- v3.7.171: 多级 backtest_pipeline.py (stage0-3)
- v3.7.172: STRADDLE detect 调用修 + dte 14→30
- v3.7.174-176: lev/filter 平衡, 频次 + WR 双优
- **v3.7.177**: ★ 版本归档 + 模块文档化
