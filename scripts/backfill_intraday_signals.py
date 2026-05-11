"""历史回填盘中触发 log.

用法 (在 GoldDash 根下):
    conda activate gold
    python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60
    python scripts/backfill_intraday_signals.py --asset SLV --timeframe 60
    python scripts/backfill_intraday_signals.py --asset GLD --timeframe 60 \
        --buy-rules stoch_rsi_cross_up_oversold \
        --exit-rules stoch_rsi_cross_down_overbought \
        --confirm any

输出: data/intraday_signal_log.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_config, load_features, load_oos_predictions
from core.regime import RegimeClassifier
from core.signals import build_band, compute_rv_pctile
from core.signals_v2 import generate_daily_signals
from core.intraday_triggers import (
    DEFAULT_BUY_RULES, DEFAULT_EXIT_RULES,
    RULES_BUY, RULES_EXIT, US_OPTIONS_SESSION_UTC, FUTURES_SESSION_24H,
    backfill, load_log,
)


# 资产 → (1h kline 文件, 日 close 文件, OOS 预测路径键)
_ASSET_CFG = {
    "GLD": {
        "kline_1h": "gld_1h.csv",
        "kline_15m": None,  # 暂无 15m 数据
        "daily_close": "gld.csv",
        "oos_key": "oos_predictions",
        "futures_1h": "gc_1h.csv",
    },
    "SLV": {
        "kline_1h": "slv_1h.csv",
        "kline_15m": None,
        "daily_close": "slv.csv",
        "oos_key": None,  # 见下: 直接读 dl_range_slv_oos.parquet
        "futures_1h": "si_1h.csv",  # v3.7.189 SLV: 加 SI=F 1h (24h 数据)
    },
}


def _load_kline(market_dir: str, fname: str) -> pd.DataFrame:
    path = os.path.join(market_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _build_daily_thresholds(asset: str, cfg: dict) -> pd.DataFrame:
    """构造每日 bp030_price / bp090_price (历史). 用 sig_df 输出."""
    market_dir = os.path.dirname(cfg["resolved"]["gld_csv"])
    if asset == "GLD":
        gld = pd.read_csv(cfg["resolved"]["gld_csv"],
                          index_col=0, parse_dates=True)
        range_df = load_oos_predictions(cfg)
    else:
        gld = pd.read_csv(os.path.join(market_dir, "slv.csv"),
                          index_col=0, parse_dates=True)
        slv_oos = os.path.join(cfg["data_root"], "models",
                               "dl_range_slv_oos.parquet")
        range_df = pd.read_parquet(slv_oos)

    # Band
    common = gld.index.intersection(range_df.index)
    upper_band, lower_band, _ = build_band(
        range_df.loc[common], gld.loc[common, "Close"])

    # Regime + RV
    features = load_features(cfg)
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]
    rv_10d = features.get("rv_10d")
    rv_pctile = compute_rv_pctile(rv_10d) if rv_10d is not None \
        else pd.Series(0.5, index=features.index)

    sig_df = generate_daily_signals(
        gld["Close"], gld["High"], gld["Low"],
        upper_band, lower_band, regime, rv_pctile)
    return sig_df[["bp030_price", "bp090_price"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=list(_ASSET_CFG.keys()), default="GLD")
    ap.add_argument("--timeframe", type=int, default=60,
                    help="盘中 K 线分钟 (60=1h, 15=15m). 仅 1h 数据当前可用.")
    ap.add_argument("--source", choices=["etf", "futures"], default="etf",
                    help="价格源: etf=GLD/SLV ETF, futures=GC=F / SI=F")
    ap.add_argument("--buy-rules", nargs="+",
                    default=list(DEFAULT_BUY_RULES),
                    choices=list(RULES_BUY))
    ap.add_argument("--exit-rules", nargs="+",
                    default=list(DEFAULT_EXIT_RULES),
                    choices=list(RULES_EXIT))
    ap.add_argument("--confirm", default="any",
                    help='"any" / "all" / 整数 k_of_n')
    ap.add_argument("--session", choices=["24h", "us_options"], default="24h")
    ap.add_argument("--log-path", default=None)
    args = ap.parse_args()

    cfg = load_config()
    market_dir = os.path.dirname(cfg["resolved"]["gld_csv"])
    asset_cfg = _ASSET_CFG[args.asset]

    # 选择 K线源
    if args.source == "etf":
        kline_file = asset_cfg["kline_1h"]
        asset_tag = args.asset
    else:
        kline_file = asset_cfg["futures_1h"]
        asset_tag = "GC" if args.asset == "GLD" else "SI"
    if kline_file is None:
        raise SystemExit(f"{args.asset} {args.source} 1h 数据未配置")

    print(f"加载 {kline_file}...")
    kline = _load_kline(market_dir, kline_file)
    print(f"  范围 {kline.index[0]} → {kline.index[-1]} ({len(kline)} bars)")

    # v3.7.189 撤销 ratio hack — 期货/期权不同模块, 不能混 backfill 一个 log

    print(f"构造 {args.asset} 每日阈值...")
    thresholds = _build_daily_thresholds(args.asset, cfg)
    print(f"  阈值天数: {len(thresholds)}")

    # confirm
    try:
        confirm = int(args.confirm)
    except ValueError:
        confirm = args.confirm

    session_map = {
        "24h": None,  # 全天
        "us_options": US_OPTIONS_SESSION_UTC,
    }
    session_utc = session_map[args.session]

    # log_path
    log_path = args.log_path or os.path.join(
        cfg["data_root"], "intraday_signal_log.parquet")

    print(f"扫描触发 (规则={args.buy_rules}/{args.exit_rules},"
          f" confirm={confirm}, session={args.session}, tf={args.timeframe}m)")
    out = backfill(
        kline=kline,
        daily_thresholds=thresholds,
        asset=asset_tag,
        timeframe_minutes=args.timeframe,
        buy_rules=tuple(args.buy_rules),
        exit_rules=tuple(args.exit_rules),
        confirm_mode=confirm,
        session_utc=session_utc,
        log_path=log_path,
    )

    n_buy = (out["side"] == "BUY").sum() if len(out) else 0
    n_exit = (out["side"] == "EXIT").sum() if len(out) else 0
    print(f"\n触发数量: BUY={n_buy} | EXIT={n_exit}")
    if len(out) > 0:
        print(f"  覆盖天数: {out['date'].nunique()}")
        print(f"  最近触发: {out.iloc[-1].to_dict()}")
    print(f"\nlog 已写入: {log_path}")
    print(f"  累计行数: {len(load_log(log_path))}")


if __name__ == "__main__":
    main()
