# 持仓数据模块

采集黄金市场持仓结构数据：CFTC COT 报告 + WGC 央行购金。

## 子模块

### cot_data.py — CFTC COT 持仓

- 来源: CFTC 年度 ZIP 文件
- 频率: 周频 (每周二报告日)
- 内容: COMEX 黄金期货非商业/商业净持仓
- 匹配: `"GOLD - COMMODITY EXCHANGE INC."` (列名用空格非下划线)
- 输出: `data/raw/cot/gold_cot.csv`

### central_bank_gold.py — 央行购金

- 来源: WGC Excel 文件 (需手动下载)
- 频率: 月频
- 下载: https://china.gold.org/goldhub/data/gold-reserves-by-country
- 文件放入: `data/raw/others/`
- 输出: `data/raw/central_bank/` (monthly, annual, features)
- Excel 解析: Monthly sheet header_row=7, data_start=8, date_col=3+

已构建 7 个模型特征:
- cb_global_net_tonnes, cb_global_3m/6m/12m_rolling
- cb_global_yoy_change, cb_key_banks_net_tonnes, cb_china_net_tonnes
