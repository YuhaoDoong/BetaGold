# Phase 1 + 1.5: 数据采集层

统一调度所有数据源的采集，覆盖 8 个采集模块、26 类数据项。

## 模块结构

```
data/
├── collect_all.py           # 全量采集主入口 (python src/data/collect_all.py)
├── market/                  # 市场行情 (Yahoo Finance, 9个品种)
├── macro/                   # 宏观经济 (FRED, 9个序列)
├── volatility/              # 波动率 (GVZ/VIX/MOVE)
├── positioning/             # 持仓数据 (CFTC COT + WGC 央行购金)
├── events/                  # 经济事件日历 (FOMC/NFP/CPI等)
└── options/                 # 期权数据 (Moomoo API + 历史存档)
```

## 运行

```bash
# 全量采集
conda activate gold
python src/data/collect_all.py

# 单独模块
python -m src.data.market.market_data
python -m src.data.macro.macro_data
python -m src.data.volatility.vol_data
python -m src.data.positioning.cot_data
python -m src.data.positioning.central_bank_gold
python -m src.data.events.economic_events
python -m src.data.options.moomoo_data
python -m src.data.options.options_archive
```

## 数据存储

所有原始数据保存在 `data/raw/` 对应子目录下，格式为 CSV。
期权历史存档使用 Parquet 格式 (`data/raw/options_history/`)。
