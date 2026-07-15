# Phase 1 + 1.5: 数据采集层

8个采集模块覆盖26类数据项，全部验证通过。Phase 1.5 增加历史期权链存档。

---

## 一、数据源总览

| # | 数据类别 | 来源 | 模块 | 记录数 | 频率 | 时间范围 |
|---|----------|------|------|--------|------|----------|
| 1 | GLD ETF | Yahoo Finance | `market_data.py` | 5,356 | 日频 | 2004-11 ~ 2026-03 |
| 2 | 黄金期货 GC=F | Yahoo Finance | `market_data.py` | 5,349 | 日频 | 2004-11 ~ 2026-03 |
| 3 | 美元指数 DXY | Yahoo Finance | `market_data.py` | 5,364 | 日频 | 2004-11 ~ 2026-03 |
| 4 | VIX 恐慌指数 | Yahoo Finance | `market_data.py` | 5,356 | 日频 | 2004-11 ~ 2026-03 |
| 5 | 原油 CL=F | Yahoo Finance | `market_data.py` | 5,353 | 日频 | 2004-11 ~ 2026-03 |
| 6 | 铜 HG=F | Yahoo Finance | `market_data.py` | 5,353 | 日频 | 2004-11 ~ 2026-03 |
| 7 | 白银 SI=F | Yahoo Finance | `market_data.py` | 5,350 | 日频 | 2004-11 ~ 2026-03 |
| 8 | 10Y 国债收益率 | Yahoo Finance | `market_data.py` | 5,351 | 日频 | 2004-11 ~ 2026-03 |
| 9 | 13W 国债利率 | Yahoo Finance | `market_data.py` | 5,351 | 日频 | 2004-11 ~ 2026-03 |
| 10 | 10Y TIPS 实际收益率 | FRED (DFII10) | `macro_data.py` | 5,555 | 日频 | 2004-11 ~ 2026-03 |
| 11 | 10Y 通胀预期 | FRED (T10YIE) | `macro_data.py` | 5,556 | 日频 | 2004-11 ~ 2026-03 |
| 12 | 5Y TIPS 实际收益率 | FRED (DFII5) | `macro_data.py` | 5,555 | 日频 | 2004-11 ~ 2026-03 |
| 13 | 联邦基金利率 | FRED (DFF) | `macro_data.py` | 7,777 | 日频 | 2004-11 ~ 2026-03 |
| 14 | 联邦债务总额 | FRED (GFDEBTN) | `macro_data.py` | 85 | 季频 | 2004-10 ~ 2025-10 |
| 15 | 财政赤字 | FRED (MTSDS133FMS) | `macro_data.py` | 255 | 月频 | 2004-11 ~ 2026-01 |
| 16 | CPI | FRED (CPIAUCSL) | `macro_data.py` | 255 | 月频 | 2004-11 ~ 2026-01 |
| 17 | M2 货币供应 | FRED (M2SL) | `macro_data.py` | 255 | 月频 | 2004-11 ~ 2026-01 |
| 18 | 贸易加权美元 | FRED (DTWEXBGS) | `macro_data.py` | 5,260 | 日频 | 2006-01 ~ 2026-02 |
| 19 | GVZ 黄金波动率 | CBOE | `vol_data.py` | 4,138 | 日频 | 2009-09 ~ 2026-03 |
| 20 | VIX 期限结构 | Yahoo (4期限) | `vol_data.py` | 3,815~5,356 | 日频 | 2004-11 ~ 2026-03 |
| 21 | MOVE 债券波动率 | Yahoo (^MOVE) | `vol_data.py` | 5,265 | 日频 | 2004-11 ~ 2026-03 |
| 22 | 黄金 COT 持仓 | CFTC | `cot_data.py` | 1,052 | 周频 | 2006-01 ~ 2026-02 |
| 23 | 央行购金 | WGC Excel | `central_bank_gold.py` | 288 月 | 月频 | 2002-01 ~ 2025-12 |
| 24 | 经济事件日历 | FRED + 硬编码 | `economic_events.py` | 4,281 | 事件驱动 | 1947 ~ 2027 |
| 25 | GLD 期权链快照 | Yahoo Finance | `market_data.py` | 6 到期日 | 快照 | 当天 |
| 26 | GLD 期权 (完整) | Moomoo API | `moomoo_data.py` | 1,256 合约 | 实时 | 当天 |

---

## 二、数据精度与单位

### 市场行情 (日频 OHLCV)

| 数据 | 单位 | 最新值示例 | 精度 |
|------|------|-----------|------|
| GLD | 美元/股 | 466.13 | float64, 2位小数 |
| 黄金期货 GC=F | 美元/盎司 | 5065.30 | float64 (已审计：GLD×10.87≈GC=F)* |

*注：GC=F 为 Yahoo Finance 连续黄金期货代理变量，不直接等同于某一官方现货口径。本系统中主要用于相对比值 (gc_gld_ratio) 和联动特征构造，不作为对外报价解释。
| 美元指数 DXY | 指数点 | 99.32 | float64 |
| VIX | 百分比点 | 23.75 | float64 |
| 10Y 收益率 | % | 4.15 | float64 |

### 宏观数据 (FRED)

| 数据 | 单位 | 频率 | 最新值 | 历史范围 |
|------|------|------|--------|---------|
| DFII10 (10Y实际收益率) | % | 日频 | 1.80% | -1.19% ~ 3.15% |
| T10YIE (通胀预期) | % | 日频 | 2.31% | 0.04% ~ 3.02% |
| DFF (联邦基金利率) | % | 日频 | 3.64% | 0.04% ~ 5.41% |
| GFDEBTN (联邦债务) | 百万美元 | 季频 | $38.5万亿 | $7.6万亿 ~ $38.5万亿 |
| CPIAUCSL (CPI) | 指数 | 月频 | 326.59 | 191.60 ~ 326.59 |
| M2SL (M2) | 十亿美元 | 月频 | $22.4万亿 | $6.4万亿 ~ $22.4万亿 |

### 波动率数据

| 数据 | 最新值 | 历史范围 | 说明 |
|------|--------|---------|------|
| GVZ | 35.31 | 8.88 ~ 48.98 | 基于 GLD 期权的黄金 IV 指数 |
| MOVE | 74.53 | 36.62 ~ 264.60 | ICE BofA 债券波动率 |
| VIX | 23.75 | 9.14 ~ 82.69 | 标普500 期权隐含波动率 |

### Moomoo 期权数据 (实时快照)

| 字段 | 说明 | 示例 |
|------|------|------|
| option_implied_volatility | 隐含波动率 (%) | median=56.61% |
| option_delta | Delta | -0.9986 ~ 0.9999 |
| option_gamma | Gamma | 0 ~ 0.03 |
| option_vega | Vega | 0 ~ 1.5 |
| option_theta | Theta (每日时间衰减) | -5.0 ~ 0 |
| option_rho | Rho | -0.5 ~ 0.5 |
| option_open_interest | 未平仓合约数 | 0 ~ 5918 |

---

## 三、数据对齐与质量

- 市场数据与 GLD 交易日对齐：最大缺失 11 天（期货节假日错位），可忽略
- 宏观面板 (ffill后)：在 GLD 交易日上 100% 覆盖 (DTWEXBGS 94.7%，2006年才开始)
- GLD 自身：**零缺失**，最大日期间隔 5 天（正常周末）
- 低频数据（季频债务、月频CPI等）通过前向填充对齐到日频

### Release-Date 对齐规则（关键）

**硬规则**：宏观特征统一按 **release timestamp（发布时间）** 生效，不按 observation period（统计所属期）生效。

- CPI、GDP、非农等数据，市场交易的是**发布时刻**，不是统计所属月份
- 典型坑：把 2026-01 的 CPI 值填到 2026-01 全月，但该数据可能 2 月中旬才公布 → **未来函数泄漏**
- 经济事件日历模块已通过 FRED release/dates API 获取各数据的历史发布日期
- 特征工程阶段将严格按 release date 对齐所有宏观因子

**实现方式**：
- 每个宏观序列关联一张 release_dates 表
- 特征构造时，每个交易日只能使用在该日之前（含）已发布的数据
- 回测中如使用 ffill，必须基于 release date 而非 observation date

---

## 四、央行购金数据

数据来源：世界黄金协会 (WGC) Excel 文件，需手动下载。

- 下载地址: https://china.gold.org/goldhub/data/gold-reserves-by-country
- 两个文件放入 `data/raw/others/`
- 建议更新频率: 每 1-2 个月

已构建 7 个模型特征：
- `cb_global_net_tonnes` — 全球月度净购金量(吨)
- `cb_global_3m/6m/12m_rolling` — 滚动购金量
- `cb_global_yoy_change` — 同比变化
- `cb_key_banks_net_tonnes` — 主要购金央行合计
- `cb_china_net_tonnes` — 中国购金（单独列出，影响力大）

---

## 五、经济事件日历

| 事件类型 | 数据量 | 来源 | 说明 |
|----------|--------|------|------|
| FOMC | 185 个 (2005-2027) | 硬编码 | 每年手动更新一次 |
| NFP (非农) | 856 个 | FRED release dates | 自动获取 |
| CPI | 942 个 | FRED release dates | 自动获取 |
| PPI | 593 个 | FRED release dates | 自动获取 |
| GDP | 849 个 | FRED release dates | 自动获取 |
| ISM 制造业 | 856 个 | FRED release dates | 自动获取 |

---

## 六、Moomoo API 期权数据

通过 Moomoo OpenD 获取实时期权链完整数据：

- **到期日**: 30 个 (最远到 2028-12-15)
- **合约**: 1,256 个 (含周/月/季到期)
- **字段**: 价格 + 完整 Greeks + IV + OI + 溢价

**限制**：
- `get_option_chain`: ≤30天跨度/次, ≤10次/30秒
- `get_market_snapshot`: ≤400代码/次

---

## 七、标准单位规范

所有模块间传递的数据统一使用以下格式：

| 字段 | 内部标准 | 说明 |
|------|----------|------|
| `iv_decimal` | 小数 (0.5661) | 百分比仅展示 |
| `theta_per_day` | 每日衰减金额 | 不按年化 |
| `vega_per_1pct_iv` | 每1%IV变化 | 注意区分 per 1% 和 per 100% |
| `price_mid` | (bid + ask) / 2 | 非 last price |
| `spread_abs` | ask - bid (美元) | 绝对点差 |
| `spread_pct` | (ask - bid) / mid * 100 | 相对点差 (%) |
| 时间戳 | UTC 存储, US/Eastern 展示 | |
| EOD 数据生效 | 当日 16:00 ET 之后 | |

---

## 八、历史期权链存档 (Phase 1.5)

支撑期权策略回测的关键基础设施。

### 存档方案

1. **日终全链快照** — 每日收盘后存一份完整期权链
   - 所有到期日 × 所有 strike × bid/ask/mid/IV/Greeks/OI/volume/DTE
   - 路径: `data/raw/options_history/YYYY-MM-DD/`
   - 格式: Parquet

2. **盘中关键横截面** — 盘中每 1-3 小时存关键期限 + 关键 delta
   - 最近 3 个月到期, delta 0.25/0.50/0.75
   - 用于 IV surface 动态监控

### 实现

- `OptionsArchiver` 自动按 30 天窗口分批请求 (绕 API 限制)
- 请求间隔 3.5s (不触发频率限制)
- 自动标准化: `price_mid`, `spread_abs`, `iv_decimal`, `dte`, `snapshot_time_utc`
- 首次运行: 8,528 contracts, 27 expirations, 35 fields, ~1.1MB Parquet

---

## 九、数据字典

### 时间戳规范

| 规则 | 说明 |
|------|------|
| 存储时区 | UTC |
| 展示时区 | US/Eastern (ET) |
| 日期字段 | `date` (YYYY-MM-DD) |
| EOD 数据生效 | 当日 16:00 ET 收盘后 |
| 事件数据生效 | release_timestamp 之后 |
| 周末/节假日 | 不生成记录，使用最近交易日 |

### 核心数据表主键

| 数据表 | 主键 | 时间字段 |
|--------|------|----------|
| market_*.csv | date (交易日) | date |
| macro_panel.csv | date (日历日) | date |
| gvz.csv | date (交易日) | date |
| gold_cot.csv | Date (报告周二) | Date |
| cb_features.csv | date (月末) | date (index) |
| economic_calendar.csv | Date + Event | Date |
| options_history/ | date + expiry + strike + type | date |
