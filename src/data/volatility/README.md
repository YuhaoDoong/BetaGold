# 波动率模块

采集 GVZ、VIX 期限结构、MOVE 三类波动率数据。

## 数据源

| 数据 | 来源 | 说明 |
|------|------|------|
| GVZ | CBOE CSV | 黄金 IV 指数 (基于 GLD 期权)，期权交易核心变量 |
| VIX 期限结构 | Yahoo Finance | VIX / VIX9D / VIX3M / VIX6M 四个期限 |
| MOVE | Yahoo Finance (^MOVE) | ICE BofA 债券波动率指数 |

## 输出

- `data/raw/volatility/gvz.csv`
- `data/raw/volatility/vix_term_structure.csv`
- `data/raw/volatility/move.csv`

## 技术要点

- GVZ CSV 非标准格式：列名是 DATE/GVZ（非 OHLCV），用位置匹配
- FRED 不提供 MOVE 指数，改用 Yahoo `^MOVE`
- GVZ 数据从 2009-09 才开始，比其他数据晚 ~5 年
