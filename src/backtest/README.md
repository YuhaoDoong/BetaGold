# Phase 4B: GLD 期权策略回测

## 当前状态

### 数据基础设施 (已就位)

```
数据采集 (Moomoo OpenD API):
  ├── EOD 全链快照: 每日收盘后自动采集 (crontab 已设置)
  │     9118 合约, 30 到期日, bid/ask/IV/Greeks/OI
  │     路径: data/raw/options_history/YYYY-MM-DD/eod_full.parquet
  │
  ├── 历史K线数据库: data/raw/options_history/kline_db/all_klines.parquet
  │     74 合约 (strike $200-$320), 2025-04 ~ 2026-03
  │     ⚠️ K线额度限制: 100次/30天, 已用完
  │
  └── 已有快照:
        2026-03-06: EOD + Intraday
        2026-03-11: EOD
```

### 限制与挑战

1. **Moomoo API 无法获取已到期合约的历史数据**
   - 2025年到期的期权合约已不可获取
   - 只能获取当前上市合约的K线

2. **历史K线额度: 100次/30天**
   - 已用完 (首批74合约 + 错误脚本消耗)
   - 额度滚动恢复, 约3-4次/天

3. **Strike 覆盖不足**
   - 已下载: $200-$320
   - GLD 2025: $243-$417, GLD 2026: $398-$496
   - ATM合约需要 $380-$520 strikes → 等额度恢复后下载

### 前进方案

**短期 (本周)**:
- 每日自动采集 EOD 快照 (crontab 已设)
- K线额度恢复后优先下载 ATM 合约 ($380-$520)
- 用 2 天快照 (3/6, 3/11) 做结构验证

**中期 (2-4周)**:
- 积累 15-20 天 EOD 快照
- K线额度陆续恢复, 补充关键合约历史
- 用真实数据做小规模回测

**长期 (1-3月)**:
- 积累足够历史用于完整回测
- 前瞻性验证: 信号触发 → 记录真实期权价格 → 验证 P&L

## 文件结构

```
src/backtest/
  ├── options_backtest.py          # 回测引擎 (真实价格版)
  ├── bs_pricer.py                 # Black-Scholes 定价 (参考用)
  └── README.md                    # ← 本文件

src/data/options/
  ├── moomoo_data.py               # Moomoo API 数据采集
  ├── options_archive.py           # EOD/盘中快照存档
  └── options_history_builder.py   # 历史K线数据库构建

scripts/
  ├── daily_options_collect.sh     # 每日自动采集脚本 (crontab)
  ├── build_kline_db.py            # K线数据库批量构建
  └── test_kline_request.py        # API 测试

data/raw/options_history/
  ├── YYYY-MM-DD/                  # 日终快照
  │     ├── eod_full.parquet
  │     └── eod_full.csv
  └── kline_db/                    # 历史K线数据库
        ├── all_klines.parquet
        └── all_klines.csv
```

## 回测引擎设计

```python
OptionPriceDB    # 期权价格查询 (K线 + EOD快照)
  → find_contract(date, gld_price, option_type, target_dte)
  → get_price(code, date)

OptionsBacktest  # 回测循环
  参数: target_dte=30, max_hold=10, max_positions=3
  入场: Buy Call → ATM call; Sell Put → OTM put
  出场: Exit 信号 / TP 拐点 / Max hold / 到期
  P&L: 真实价格 + 滑点 2% + 佣金 $0.65/合约
```

## Phase 4A 旧方案参考

旧点预测方案 (IC=0.06-0.09) 不足以跑赢强势行情:
- Ridge fwd_ret_20d: CAGR=4.1%, Sharpe=0.42
- Buy & Hold: CAGR=13.8%, Sharpe=0.89

新方案 (Hybrid Band + Regime + RV) 信号质量更好:
- Buy Call: 5d WR 72%, Sell Put: 5d WR 73%
- 详见 `src/models/README.md`
