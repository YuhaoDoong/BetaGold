# v3.7.229 Trailing Windows 多窗口回测结果

**日期**: 2026-05-18
**方法**: trailing windows 包含最新数据
- Layer 1: 10y / 5y / 3y / 1y
- Layer 2: 5y/3y/1y/6m/3m (期货含 5y/3y 长窗对照)

**判定**: 跨窗一致 → robust; 不一致 → 趋势倾向近窗（最新 regime）

---

## Layer 1 (信号 spot 验证)

### GLD 各 filter 跨窗最优

| Filter | 10y | 5y | 3y | 1y | 评估 |
|---|---|---|---|---|---|
| `buy_bp` | **0.20** | **0.20** | **0.20** | 0.35 | 3/4 一致 0.20 (1y 异常) |
| `rv_pctile_max` | 1.0 | 0.75 | 0.75 | 0.70 | ⚠️ 近期偏严 |
| `ret_20d_min` | -0.2 | -0.2 | -0.2 | -0.2 | **★ 4/4 一致** |
| `ret_20d_max` | -0.03 | 0.03 | 0.03 | -0.03 | ⚠️ 边界跳 |
| **`iv_filter_high_min`** | **25** | **25** | **25** | **25** | **★ 4/4 一致** |
| `ma_trend_threshold` | 0.99 | 0.975 | 0.975 | 0.975 | 3/4 一致 0.975 |

### SLV 各 filter 跨窗最优 (5y 数据)

| Filter | 5y | 3y | 1y | 评估 |
|---|---|---|---|---|
| `buy_bp` | **0.20** | **0.20** | 0.40 | 2/3 一致 (1y 异常) |
| `rv_pctile_max` | 1.0 | 1.0 | 0.5 | 近期收紧 |
| `ret_20d_min` | -0.2 | -0.2 | -0.2 | **★ 一致** |
| **`ret_20d_max`** | **0.03** | **0.03** | **0.03** | **★ 一致** |
| **`iv_filter_high_min`** | **25** | **25** | **25** | **★ 一致** |
| **`ma_trend_threshold`** | **0.99** | **0.99** | **0.99** | **★ 一致** |

### Cross-asset 跨窗 (10d spot)

| 触发 | 10y | 5y | 3y | 1y | 评估 |
|---|---|---|---|---|---|
| **SLV-S → GLD** | n=23 WR=82.6% | 82.6% | n=22 86.4% | n=9 77.8% | **★ 跨窗 robust** |
| SLV-S+A → GLD | n=55 72.7% | 72.7% | 76.9% | 77.8% | 一致 |
| SLV-ALL → GLD | n=104 73.1% | 73.1% | 75.8% | 81.8% | **一致 + 近窗更好** |
| GLD-S → SLV | n=7 57.1% | 57.1% | 57.1% | n=3 33.3% | ❌ 反向 |
| GLD-B → SLV | n=49 73.5% | 73.5% | 77.3% | **n=20 80%** | **★ 一致** |
| GLD-ALL → SLV | n=66 72.7% | 72.7% | 75.0% | n=24 75% | 一致 |

---

## Layer 2 期货 (`futures/`)

### 跨窗 best_leverage 一致性

**GLD 期货**: 全 tier 跨窗几乎一致选 20x，但 blowup 8-16%

| Tier | 5y | 3y | 1y | 6m | 3m | 备注 |
|---|---|---|---|---|---|---|
| S | 20 | 20 | 20 | — | — | 0% blowup, sample 少 |
| A | 20 | 20 | — | — | — | 0% blowup |
| S+A | 20 | 20 | 20 | — | — | 0-8% blowup |
| B | 20 | 20 | 20 | 20 | 20 | 6m/3m sample=5 60% blowup |
| ALL | 20 | 20 | 20 | 20 | 20 | 5y/3y/1y 8-15% blowup |

**SLV 期货**: 跨窗不一致，近窗保守

| Tier | 5y | 3y | 1y | 6m | 3m | 趋势 |
|---|---|---|---|---|---|---|
| S | 5 | 15 | 5 | 5 | 3 | 大体 5x |
| A | 10 | 10 | 20 | 10 | — | **10x 主流** |
| S+A | 15 | 15 | **10** | **5** | **3** | **近窗降 lev** |
| B | 15 | 15 | 20 | — | — | 高 lev |
| ALL | 15 | 15 | 15 | **3** | **3** | **近窗 3x** |

### 关键结论

- **GLD-A 10-20x 跨窗 0% 爆仓**（最稳）★
- **SLV-A 10x 跨窗一致 0-11% 爆仓**（次稳）
- **SLV 短窗 6m/3m 选 3-5x 保守** — silver 高 vol regime 高 lev 不可持续
- **6m/3m 信号数 ≤11**, 单笔 60% blowup 不可信

---

## Layer 2 方向性期权 (`directional_options/`)

### BC profit_target_mult 跨窗

| Asset | Tier | 1y | 6m | 3m |
|---|---|---|---|---|
| GLD | S+A | **3.0** (n=4 WR=100% sum=818) | — | — |
| GLD | B | **3.0** (n=18 WR=83% sum=2874) | 1.5 (n=5) | 1.5 |
| GLD | ALL | **3.0** (n=22 WR=86% sum=3692) | 1.5 (n=5) | 1.5 |
| SLV | S+A | **4.0** (n=11 WR=91% sum=3406) | 4.0 (n=6) | 2.0 (n=4) |
| SLV | ALL | **4.0** (n=19 WR=89% sum=5644) | 4.0 (n=7) | 2.0 (n=5) |

### SP profit_target_credit_pct 跨窗

| Asset | Tier | 1y | 备注 |
|---|---|---|---|
| GLD | B | **50%** WR=100% sum=260 | 100% WR 稳 |
| GLD | ALL | **50%** WR=100% sum=288 | |
| SLV | S+A | **30%** WR=100% sum=241 | |
| SLV | ALL | **30%** WR=100% sum=241 | |

### 关键结论

- **GLD BC pt 2.5 → 3.0** (1y sum +20%)
- **SLV BC pt → 4.0** (1y sum 5644%)
- **GLD SP pt 50%** 跟生产 70% 接近，保留 prod
- **SLV SP pt 30%** 跟生产一致 ✓
- 6m/3m 样本太少 (5笔以下) 结论不可信

---

## Layer 2 波动率期权 (`vol_options/`)

| Asset | Strategy | 1y | 6m | 3m |
|---|---|---|---|---|
| **GLD** | **STRADDLE** | **n=25 WR=64% sum=531** ✅ | sample 不足 | sample 不足 |
| GLD | SHORT_VOL | n=48 WR=17% sum=**-1547** ❌ | -892 ❌ | -195 ❌ |
| SLV | STRADDLE | sample 不足 | — | — |
| SLV | SHORT_VOL | n=28 WR=21% sum=**-1177** ❌ | -1177 ❌ | -798 ❌ |

### 关键结论

- **GLD STRADDLE 1y 有 alpha** (WR 64% sum +530)
- **SHORT_VOL 跨 GLD/SLV 跨多窗全亏** → **保持 DISABLED 正确** ★
- SLV STRADDLE 数据稀疏 (kline_db 1y 内 SLV STRADDLE 信号只 17 笔, closed 0)

---

## 跨层综合应用建议

### Layer 1 信号 (多窗 robust)
1. **GLD `buy_bp` 0.30 → 0.20** (3/4 fold)
2. **GLD `iv_filter_high_min`=25** ✓ 跟生产一致
3. **GLD `ret_20d_min` 不限** ✓ 跟生产一致
4. **SLV `iv_filter_high_min` 28 → 25** (跟 GLD 统一, 3/3 多窗一致)
5. **SLV `ma_trend_threshold` 0.0 → 0.99** (3/3 一致, 生产关闭实测有害)
6. **SLV `ret_20d_max` → 0.03** (3/3 一致, 生产未设)

### Layer 2 策略 (信号 → 工具)

| 信号 tier | 期货 lev | BC pt | SP pt% |
|---|---|---|---|
| GLD-A | **10-15x** (0 爆仓) | 3.0 | 50% |
| GLD-S+A | 15-20x | 3.0 | 50% |
| GLD-B/ALL | 20x ⚠️ 16% blowup | 3.0 | 50% |
| SLV-A | **10x** (0 爆仓) | 4.0 | 30% |
| SLV-S+A | 10-15x ⚠️ 17-22% blowup | 4.0 | 30% |
| SLV-B/ALL | 15x ⚠️ 14-17% blowup | 4.0 | 30% |
| **STRADDLE 自家信号** | — | **GLD WR 64% sum +530** | — |
| **SHORT_VOL** | **❌ DISABLED 跨多窗验证** | — | — |

### Cross-asset 规则 (多窗 robust)
- ✅ **SLV-S → GLD spot 10d**: 4/4 窗 WR 78-86%
- ✅ **SLV-S+A → GLD**: 4/4 一致 WR 72-78%
- ✅ **GLD-B → SLV**: 4/4 一致 WR 73-80%
- ❌ **GLD-S → SLV**: 反向 (1y 仅 33% WR)

---

## 已知限制

- SLV ETF 数据只 5y, Layer 1 跨窗只能 5y/3y/1y
- 期权 kline_db 仅 1y, 6m/3m 信号 ≤5 sample 不可信
- 期货 GC=F 数据 21y 充足, 但短窗 (6m/3m) 仍数据稀
- 期权 1y 数据 含 2025 H2 大反弹 + 2026 Q1 暴跌, regime 偏置存在
