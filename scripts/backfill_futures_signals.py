"""v3.7.191 期货 intraday detect (GC=F/SI=F 24h)

独立于 ETF intraday backfill, 用 GC/SI scale.

输入:
  gc_1h.csv / si_1h.csv (24h 1h kline)
  sig_df_gc.parquet / sig_df_si.parquet (GC scale threshold)

输出:
  data/futures_signal_log.parquet
  字段同 intraday_signal_log 但 price 在 GC scale, asset='GC'/'SI'

用法:
  python scripts/backfill_futures_signals.py --asset GLD
  python scripts/backfill_futures_signals.py --asset SLV
"""
from __future__ import annotations
import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from core.data import load_config
from core.intraday_triggers import (
    detect_triggers, TriggerConfig, save_log, upsert_log,
    DEFAULT_BUY_RULES, DEFAULT_EXIT_RULES, dedupe_intraday,
)


_ASSET_CFG = {
    "GLD": {"kline_1h": "gc_1h.csv", "sig": "sig_df_gc.parquet", "tag": "GC"},
    "SLV": {"kline_1h": "si_1h.csv", "sig": "sig_df_si.parquet", "tag": "SI"},
}

LOG_PATH = "/Users/yhdong/Gold/data/futures_signal_log.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=list(_ASSET_CFG.keys()), default="GLD")
    ap.add_argument("--timeframe", type=int, default=60)
    ap.add_argument("--confirm", default="any")
    args = ap.parse_args()

    cfg = load_config()
    cfg_a = _ASSET_CFG[args.asset]
    market_dir = os.path.dirname(cfg["resolved"]["gld_csv"])
    kline_path = os.path.join(market_dir, cfg_a["kline_1h"])
    sig_path = os.path.join(cfg["data_root"], "processed", cfg_a["sig"])

    if not os.path.exists(kline_path):
        raise SystemExit(f"{kline_path} 不存在")
    if not os.path.exists(sig_path):
        raise SystemExit(f"{sig_path} 不存在 (先跑 build_futures_signals.py)")

    kline = pd.read_csv(kline_path, index_col=0, parse_dates=True)
    sig_df = pd.read_parquet(sig_path)
    thresholds = sig_df[["bp030_price", "bp090_price"]]
    # v3.7.191: ffill 周末 threshold (GC=F 周一 daily bar 含 Sun-Mon, 周五 → Sat/Sun 用周五)
    all_days = pd.date_range(thresholds.index.min(),
                                kline.index.max().normalize(), freq="D")
    thresholds = thresholds.reindex(all_days).ffill()
    print(f"[fut-backfill] {args.asset} ({cfg_a['tag']}): "
          f"kline {kline.index[0]} → {kline.index[-1]} ({len(kline)} bars)")
    print(f"  threshold (ffill 周末): 截止 {thresholds.index.max()}")

    confirm = "any" if args.confirm == "any" else int(args.confirm)

    cfg_buy = TriggerConfig(timeframe_minutes=args.timeframe, side="BUY",
                              rule_set=DEFAULT_BUY_RULES, confirm_mode=confirm)
    cfg_exit = TriggerConfig(timeframe_minutes=args.timeframe, side="EXIT",
                               rule_set=DEFAULT_EXIT_RULES, confirm_mode=confirm)

    buys = detect_triggers(kline, thresholds, cfg_buy, asset=cfg_a["tag"])
    exits = detect_triggers(kline, thresholds, cfg_exit, asset=cfg_a["tag"])

    # v3.7.195: 按日 dedupe — 跨天的 trigger 不应该因为前一天价格低而被吃掉
    # 之前是全历史 dedupe, 导致 5/11 02:00 ($4656) 把 5/12 ($4692) 全过滤
    def _dedupe_per_day(df, side):
        if df is None or len(df) == 0: return df
        df = df.copy()
        df["trigger_time"] = pd.to_datetime(df["trigger_time"])
        out = []
        for _, grp in df.groupby(df["trigger_time"].dt.date):
            d = dedupe_intraday(grp, side=side, min_drop_pct=0.3)
            if len(d): out.append(d)
        return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]

    if len(buys): buys = _dedupe_per_day(buys, "BUY")
    if len(exits): exits = _dedupe_per_day(exits, "EXIT")

    print(f"  触发: BUY={len(buys)} EXIT={len(exits)}")
    if len(buys):
        latest = buys.sort_values("trigger_time").iloc[-1]
        print(f"  最新 BUY: {latest['trigger_time']} @${latest['price']:.2f}")

    # upsert 到独立期货 log
    if len(buys): upsert_log(buys, LOG_PATH)
    if len(exits): upsert_log(exits, LOG_PATH)

    log = pd.read_parquet(LOG_PATH)
    print(f"  futures_signal_log: 累计 {len(log)} 行")


if __name__ == "__main__":
    main()
