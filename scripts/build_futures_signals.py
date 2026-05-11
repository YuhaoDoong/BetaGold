"""v3.7.190 期货信号链 (独立 GC=F 24h pipeline)

期货 vs 期权 严格分离:
  期货链:
    daily csv:    gc.csv / si.csv (24h, 周末跨日 = 周一 bar)
    OOS model:    dl_range_gc_oos.parquet (GC=F trained)
    sig_df_gc:    GC scale threshold (bp030_price ~$4621)
    intraday:     gc_1h.csv / si_1h.csv (24h)
    detect log:   futures_signal_log.parquet (GC scale)
    realtime:     Binance XAUUSDT / XAGUSDT (实时 mark price)

  期权链 (保留):
    daily csv:    gld.csv / slv.csv (RTH)
    sig_df:       ETF scale threshold
    intraday:     gld_1h.csv / slv_1h.csv
    detect log:   intraday_signal_log.parquet (ETF scale)

输出:
  - data/processed/sig_df_gc.parquet
  - data/processed/sig_df_si.parquet
  - data/futures_signal_log.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from core.data import load_config, load_features, load_oos_predictions
from core.signals import build_band, compute_rv_pctile
from core.signals_v2 import generate_daily_signals
from core.regime import RegimeClassifier


CFG = load_config()
SIG_DIR = Path(CFG["data_root"]) / "processed"
SIG_DIR.mkdir(parents=True, exist_ok=True)


def build_sig_df(asset: str) -> pd.DataFrame:
    """构造期货 daily signal (GC/SI scale).

    asset: 'GLD' (→GC=F) or 'SLV' (→SI=F).
    """
    if asset == "GLD":
        fut_path = Path(CFG["data_root"]) / "raw/market/gc.csv"
        oos = load_oos_predictions(CFG)  # dl_range_gc_oos
    else:
        fut_path = Path(CFG["data_root"]) / "raw/market/si.csv"
        slv_oos = Path(CFG["data_root"]) / "models/dl_range_slv_oos.parquet"
        if not slv_oos.exists():
            print(f"  [warn] {slv_oos.name} 不存在, 跳过 SLV")
            return None
        oos = pd.read_parquet(slv_oos)
    if not fut_path.exists():
        print(f"  [warn] {fut_path.name} 不存在")
        return None

    fut = pd.read_csv(fut_path, index_col=0, parse_dates=True)
    common = fut.index.intersection(oos.index)
    if len(common) < 100:
        print(f"  [warn] {asset} fut/oos 重叠仅 {len(common)} 天")
        return None
    upper, lower, _ = build_band(oos.loc[common], fut.loc[common, "Close"])

    feat = load_features(CFG)
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(feat[feat_cols])["regime"]
    rv_pct = compute_rv_pctile(feat["rv_10d"])

    common2 = common.intersection(regime.index)
    sig = generate_daily_signals(
        fut.loc[common2, "Close"],
        fut.loc[common2, "High"],
        fut.loc[common2, "Low"],
        upper.reindex(common2), lower.reindex(common2),
        regime.reindex(common2), rv_pct.reindex(common2),
        asset=asset)
    return sig


def main():
    print(f"v3.7.190 build futures signals @ {pd.Timestamp.now()}")
    for asset, sym in [("GLD", "gc"), ("SLV", "si")]:
        sig = build_sig_df(asset)
        if sig is None: continue
        out = SIG_DIR / f"sig_df_{sym}.parquet"
        sig.to_parquet(out)
        print(f"  {asset} ({sym}): 截止 {sig.index.max()} | n={len(sig)} | "
              f"latest bp030=${sig['bp030_price'].iloc[-1]:.2f} "
              f"bp090=${sig['bp090_price'].iloc[-1]:.2f}")
        recent_buys = sig[sig["buy_signal"]==True].tail(5)
        print(f"  最近 5 个 buy 信号:")
        print(recent_buys[["bp030_price","buy_type"]].to_string())
    print("\n→ futures_signal_log 单独由 backfill_futures_signals.py 写入")


if __name__ == "__main__":
    main()
