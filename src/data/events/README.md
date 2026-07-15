# 经济事件日历模块

采集重要经济事件发布日期，用于事件驱动特征和交易窗口管理。

## 事件类型

| 事件 | 数据量 | 来源 |
|------|--------|------|
| FOMC | 185 个 (2005-2027) | 硬编码 (每年手动更新) |
| NFP | 856 个 | FRED release/dates API |
| CPI | 942 个 | FRED release/dates API |
| PPI | 593 个 | FRED release/dates API |
| GDP | 849 个 | FRED release/dates API |
| ISM | 856 个 | FRED release/dates API |

## 输出

- `data/raw/events/economic_calendar.csv` — 事件日历 (Date + Event)
- `data/raw/events/event_features.csv` — 事件特征矩阵 (可选)

## 生成的特征

- `days_to_next_{event}` — 距下一个事件天数
- `is_{event}_day` — 当天是否事件日
- `{event}_window_3d` — 事件前 3 天窗口标记

## 重要用途

Release dates 同时用于宏观数据的 release-date 对齐，
确保回测中不使用未发布的宏观数据（防止未来函数泄漏）。
