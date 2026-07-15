# 宏观经济模块

通过 FRED API 采集 9 个宏观经济序列，构建统一的日频宏观面板。

## 数据序列

| 序列 | ID | 频率 | 说明 |
|------|-----|------|------|
| 10Y TIPS 实际收益率 | DFII10 | 日频 | 中金模型候选因子 |
| 10Y 通胀预期 | T10YIE | 日频 | 盈亏平衡通胀率 |
| 5Y TIPS 实际收益率 | DFII5 | 日频 | 短端实际利率 |
| 联邦基金利率 | DFF | 日频 | 政策利率 |
| 联邦债务总额 | GFDEBTN | 季频 | 中金模型核心因子 |
| 财政赤字 | MTSDS133FMS | 月频 | 财政扩张指标 |
| CPI | CPIAUCSL | 月频 | 通胀水平 |
| M2 | M2SL | 月频 | 货币供应 |
| 贸易加权美元 | DTWEXBGS | 日频 | 广义美元指数 |

## 输出

- `data/raw/macro/{series_id}.csv` — 各序列原始数据
- `data/raw/macro/macro_panel.csv` — 合并的日频面板 (ffill 对齐)

## 关键规则

**宏观数据必须按 release date 对齐，不按 observation date 对齐。**
否则会产生未来函数泄漏（如 1 月的 CPI 数据 2 月中旬才发布）。
详见 `src/data/events/` 模块中的 release dates 数据。
