# 期权数据模块

通过 Moomoo API 采集 GLD 完整期权链 + 历史存档系统。

## 子模块

### moomoo_data.py — Moomoo API 实时期权数据

- 连接: OpenD (127.0.0.1:11111)，需独立启动
- 内容: 期权链 + 完整 Greeks (Delta/Gamma/Vega/Theta/Rho) + IV + OI
- 覆盖: 30 个到期日，8000+ 合约

### options_archive.py — 历史期权链存档 (Phase 1.5)

期权策略回测的关键基础设施。

**日终全链快照** (默认):
```bash
python -m src.data.options.options_archive
```
- 所有到期日 x 所有 strike x bid/ask/mid/IV/Greeks/OI/volume/DTE
- 保存: `data/raw/options_history/YYYY-MM-DD/eod_full.parquet`

**盘中关键横截面**:
```bash
python -m src.data.options.options_archive --intraday
```
- 最近 3 个月到期 x delta=0.25/0.50/0.75 的 call+put
- 保存: `data/raw/options_history/YYYY-MM-DD/intraday_HHMMSS.parquet`

**定时任务 (crontab)**:
```
5 16 * * 1-5  conda run -n gold python -m src.data.options.options_archive
30 11,13,15 * * 1-5  conda run -n gold python -m src.data.options.options_archive --intraday
```

## Moomoo API 限制

- `get_option_chain`: 每次 ≤30 天跨度，每 30 秒 ≤10 次
- `get_market_snapshot`: 每次 ≤400 个代码
- OpenD 启动: `nohup .../OpenD.app/Contents/MacOS/OpenD &` (非 gateway 模式)
