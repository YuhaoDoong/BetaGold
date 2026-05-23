# 回测两层 + 子模块架构 (v3.7.230)

## 核心原则

1. **严格无 look-ahead**: 入场=信号日+1 Open, regime min_hold_days=1, rv_pctile rolling-252, OOS 模型 walk-forward 训练
2. **多 trailing window 验证**: 每窗包含最新数据 (10y/5y/3y/1y 或 1y/6m/3m); 跨窗一致才算 robust
3. **判定指标分层**: Layer 1 看信号 vs 价格预测; Layer 2 看实际策略 P&L
4. **vol 信号专属指标**: STRADDLE 用 `abs_r > BE`, SHORT_VOL 用 `abs_r < short_strike` (不用 RV vs IV)

## 架构总图

```
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 1: Signal Validation (spot 历史 10y+)                          │
│  ┌────────────────────────┐  ┌────────────────────────┐              │
│  │ directional/           │  │ vol/                   │              │
│  │ BUY CALL / SELL PUT    │  │ STRADDLE / SHORT_VOL   │              │
│  │ 指标: spot 前向回报    │  │ 指标: abs_r vs BE      │              │
│  └────────────────────────┘  └────────────────────────┘              │
│  Windows: 10y / 5y / 3y / 1y (trailing, 含最新数据)                  │
└──────────────────────────────────────────────────────────────────────┘
                                ↓ 通过 Layer 1 的信号
┌──────────────────────────────────────────────────────────────────────┐
│ Layer 2: Strategy Execution (真实 P&L)                               │
│  ┌────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │
│  │ futures/       │  │ directional_options│  │ vol_options/     │  │
│  │ GC=F 21y       │  │ BC + SP (kline 1y) │  │ STRADDLE         │  │
│  │ leverage grid  │  │ pt/sl grid         │  │ + SHORT_VOL      │  │
│  └────────────────┘  └────────────────────┘  └──────────────────┘  │
│  Windows: 5y / 3y / 1y / 6m / 3m (短窗看近期 regime)                 │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
                  Per-asset / per-tier prod cfg
```

## 目录

```
scripts/backtest/
├── README.md
├── framework.py                          # raw_universe + filters + scoring
│                                           - score (方向性 5/10/20d 前向回报)
│                                           - score_straddle (abs_r > BE)
│                                           - score_short_vol (abs_r < short_strike)
│                                           - trailing_slice / multi_window_filter
│                                           - cross_asset_pivot
├── layer1_signal/
│   ├── directional/run_all.py            # buy_bp / rv_pctile / iv_filter / ret_20d / ma_trend
│   └── vol/run_all.py                    # STRADDLE / SHORT_VOL 信号 (abs_r vs BE/strike)
└── layer2_strategy/
    ├── futures/run_all.py                # GC=F 21y leverage grid (3/5/8/10/15/20)
    ├── directional_options/run_all.py    # BC pt/sl, SP pt grid (1y kline_db)
    └── vol_options/run_all.py            # STRADDLE/SHORT_VOL 实际 P&L
```

## 数据范围

| 数据 | 起 | 止 | 范围 |
|---|---|---|---|
| GLD ETF daily | 2015-06-17 | 2026-05-15 | 10.9y |
| SLV ETF daily | 2021-05-07 | 2026-05-15 | 5.0y |
| GC=F daily (期货) | 2004-11-18 | 2026-05-18 | 21.5y |
| SI=F daily (期货) | 2006-04-28 | 2026-04-02 | 19.9y |
| 期权 kline_db | 2025-04-29 | 2026-05-06 | 1.0y |

## 验证指标定义

**Layer 1 方向性**: 信号日+1 Open 入, +N 日 Close 出. WR = (r > 0), scoreB = WR² × log(1+n) × mean

**Layer 1 STRADDLE**:
- BE = IV_entry × √(h/252)   ← entry premium 距离
- WR_abs > BE  ← 主胜率 (突破盈亏平衡)
- WR_max > BE ← intraday H/L 触及 BE
- 辅: iv_change > 0 (vega 同向 favor)

**Layer 1 SHORT_VOL**:
- short_strike = 1.6 × BE   ← ATM IC 短腿距离
- WR_abs < strike  ← 主胜率 (停在短腿内)
- 辅: iv_change < 0 (IV crush 收 premium)

**Layer 2 P&L**:
- 期货: 含 leverage + SL/TP/expiry + 爆仓
- 期权: 真实 kline_db OHLC 历史定价

## 输出归档

```
data/backtest_history/
├── v3.7.229_layer1/           # 方向性多窗
├── v3.7.230_layer1_vol/       # vol 信号 (abs_r vs BE 修正指标)
├── v3.7.229_layer2_futures/
├── v3.7.229_layer2_directional/
├── v3.7.229_layer2_vol_options/
└── v3.7.229_trailing_windows/RESULTS.md   # 综合汇总
```

## 最终 robust 结论 (跨多窗一致)

### Layer 1 方向性信号 (跨 10y/5y/3y/1y)

| Filter | 多窗 best | 一致性 | 应用 |
|---|---|---|---|
| **GLD `buy_bp`** | **0.20** | 3/4 (1y 异常) | ✅ 应用 (生产 0.30 → 0.20) |
| GLD `iv_filter_high_min` | 25 | **4/4** | ✓ 已在生产 |
| GLD `ret_20d_min` | -1.0 (不限) | **4/4** | ✓ 已在生产 |
| GLD `rv_pctile_max_hard` | 1.0 (关) | 5/8 | ⚠️ 生产 0.75, 撤回有争议 (Phase 1 uplift -1.0 但跨窗不强) |
| **SLV `iv_filter_high_min`** | **25** | **3/3** | ✅ 应用 (生产 28 → 25, 跟 GLD 统一) |
| **SLV `ma_trend_threshold`** | **0.99** | **3/3** | ✅ 应用 (生产 0.0 → 0.99) |
| **SLV `ret_20d_max_hard`** | **0.03** | **3/3** | ✅ 应用 (生产不限 → 0.03) |

### Layer 1 波动率信号 (修正指标 abs_r vs BE)

| 信号 | 10y uplift | 5y | 3y | 1y | 结论 |
|---|---|---|---|---|---|
| **GLD STRADDLE** | **+10.7pp** | +12.1pp | +9.5pp | **+21.9pp** | ✅ 跨窗 robust |
| GLD SHORT_VOL | +5.9pp | +4.1pp | +3.8pp | +7.2pp | ✅ 信号有 alpha 但 P&L 亏(tail) |
| **SLV STRADDLE** | **+7.2pp** | +7.2pp | +9.3pp | **+23.5pp** | ✅ 跨窗 robust |
| SLV SHORT_VOL | +1.6pp | +1.6pp | +0.2pp | +12.2pp | ⚠️ 弱, fragile |

### Cross-asset (10d spot, 跨多窗)

| 触发 | 10y | 1y | 一致性 |
|---|---|---|---|
| **SLV-S → GLD** | WR 82.6% | 77.8% | ✅ 4/4 一致 |
| **SLV-S+A → GLD** | 72.7% | 77.8% | ✅ |
| **GLD-B → SLV** | 73.5% | 80% | ✅ 4/4 一致 |
| GLD-S → SLV | 57.1% | 33.3% | ❌ 反向 |

### Layer 2 期货 (GC=F 21y)

| 资产-tier | 多窗一致 lev | 爆仓率 | 评估 |
|---|---|---|---|
| **GLD-A** | 10-20x | **0%** | ★ 最稳 |
| GLD-S+A | 20x | 0-8% | 稳 |
| GLD-B/ALL | 20x | 12-16% | 高收高风险 |
| **SLV-A** | **10x** | **0%** | ★ 最稳 |
| SLV-S+A | 5y/3y=15x, 1y/6m/3m=5-10x | 11-22% | 近窗保守 |
| SLV-ALL | 5y/3y=15x, 6m/3m=3x | 14-17% | 近窗大幅降 |

**含义**: 长窗高 lev 看上去好但 2020 Covid + 2026 Q1-Q2 期间 57-60% 爆仓; 真正 robust = A tier + 10x.

### Layer 2 方向性期权

| 信号 | BC pt | SP pt% |
|---|---|---|
| GLD ALL | **3.0** (1y sum +20%) | 50 (跟 prod 70 接近, 保留 prod) |
| SLV ALL | **4.0** (1y sum 5644%) | 30 ✓ prod 一致 |

### Layer 2 波动率期权 (1y)

| 信号-策略 | 表现 | 应用 |
|---|---|---|
| **GLD STRADDLE (自家信号)** | n=25 WR 64% sum +530 | ✅ 启用 |
| SHORT_VOL | sum -1547 ~ -1177 | ❌ 保持 DISABLED |

## 已知限制

- SLV ETF 5y, walk-forward fold 数受限
- 期权 kline_db 仅 1y, 6m/3m 信号 ≤5 sample 不可信
- Layer 2 期权数据偏 2025 H2 大反弹 + 2026 Q1 修正, regime 偏置
- v3.7.230 vol metric 修正后 STRADDLE 重新评估为有 alpha, 之前用 RV>IV 错误指标导致误判
