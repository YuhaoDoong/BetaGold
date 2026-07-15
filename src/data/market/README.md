# 市场行情模块

通过 Yahoo Finance 采集 9 个品种的日频 OHLCV 数据 + GLD 期权链快照。

## 数据品种

| 品种 | Ticker | 说明 |
|------|--------|------|
| GLD | GLD | SPDR Gold Shares ETF |
| 黄金期货 | GC=F | COMEX Gold Futures |
| 美元指数 | DX-Y.NYB | US Dollar Index |
| VIX | ^VIX | CBOE Volatility Index |
| 原油 | CL=F | WTI Crude Oil Futures |
| 铜 | HG=F | Copper Futures |
| 白银 | SI=F | Silver Futures |
| 10Y国债 | ^TNX | 10-Year Treasury Yield |
| 13W国债 | ^IRX | 13-Week Treasury Bill Rate |

## 输出

- `data/raw/market/{ticker}.csv` — 各品种日频 OHLCV
- `data/raw/options/` — GLD 期权链快照 (6 个最近到期日)

## 技术要点

- yfinance 单 ticker 返回 MultiIndex columns，需 `get_level_values(0)` 展平
- 数据范围: 2004-11-18 (GLD 上市日) ~ 今天
