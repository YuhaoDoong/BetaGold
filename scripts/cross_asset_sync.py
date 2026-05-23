"""v3.7.224: 金银 S 信号交叉应用统计 — 验证"金银同步但幅度不同"假设.

假设:
  GLD 出 S 信号当日, SLV 同日 spot 也应该有正收益 (即使 SLV 自己没信号).
  反过来同样.

意义:
  若假设成立: S 信号可跨品种共享 (扩大入场机会), 仓位可在金银两边分散
  若不成立 (非同步 / 反向): 两个 asset 必须各自独立信号

统计:
  - GLD S 信号 → GLD forward return + SLV forward return (同日)
  - SLV S 信号 → SLV forward return + GLD forward return (同日)
  - 横轴 5d / 10d / 20d 前向窗口
  - 看 WR / mean / corr
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategy_config import get_config


def build_signals_for(asset: str) -> tuple:
    """Returns (sig_df with buy + tier, ohlc DataFrame)."""
    cfg = load_config()
    if asset == "GLD":
        oos = load_oos_predictions(cfg)
    else:
        oos = pd.read_parquet(Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet")
    feat_path = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
                  if asset == "GLD" else
                  "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    feat = pd.read_parquet(feat_path)
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]; low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier(min_hold_days=1).classify(
        feat.loc[common, feat_cols])["regime"]  # v3.7.233 explicit no-lookahead
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    sig = generate_daily_signals(close, high, low, upper, lower, regime, rv_p,
                                       asset=asset, gvz_series=gvz["Close"])
    return sig, ohlc


def add_fwd_returns(df: pd.DataFrame, ohlc: pd.DataFrame, horizons: list) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        col = f"r{h}d"
        for d in out.index:
            if d not in ohlc.index: continue
            i = ohlc.index.get_loc(d)
            ent = float(ohlc.iloc[i]["Open"])
            if i + h < len(ohlc):
                ext = float(ohlc.iloc[i + h]["Close"])
                out.loc[d, col] = (ext / ent - 1) * 100
            else:
                out.loc[d, col] = np.nan
    return out


def stats(returns: pd.Series) -> dict:
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "WR": None, "mean": None, "sum": None,
                 "std": None, "max_loss": None}
    return {
        "n": n,
        "WR": round((s > 0).mean() * 100, 1),
        "mean": round(s.mean(), 2),
        "sum": round(s.sum(), 1),
        "std": round(s.std(), 2),
        "max_loss": round(s.min(), 1),
        "max_gain": round(s.max(), 1),
    }


def main():
    horizons = [5, 10, 20]
    print("加载 GLD + SLV signals ...")
    gld_sig, gld_ohlc = build_signals_for("GLD")
    slv_sig, slv_ohlc = build_signals_for("SLV")

    gld_sig = add_fwd_returns(gld_sig, gld_ohlc, horizons)
    slv_sig = add_fwd_returns(slv_sig, slv_ohlc, horizons)
    # 同日 SLV 前向回报 (即使 SLV 没信号)
    slv_dates_fwd = add_fwd_returns(
        pd.DataFrame(index=slv_ohlc.index), slv_ohlc, horizons)
    gld_dates_fwd = add_fwd_returns(
        pd.DataFrame(index=gld_ohlc.index), gld_ohlc, horizons)

    # GLD 信号 + 同日 SLV 表现
    print()
    print("=" * 100)
    print("【1】GLD BUY 信号 → 同日 SLV spot 前向回报")
    print("=" * 100)
    gld_buy = gld_sig[gld_sig["buy_signal"]].copy()
    print(f"GLD BUY 信号 (全 tier) 共 {len(gld_buy)} 笔, 时间: "
          f"{gld_buy.index.min().date()} → {gld_buy.index.max().date()}")
    for tier in ["S", "A", "B", "ALL"]:
        if tier == "ALL":
            sub = gld_buy
        else:
            sub = gld_buy[gld_buy["signal_tier"] == tier]
        if not len(sub): continue
        print(f"\n  GLD tier={tier} (n={len(sub)}):")
        for h in horizons:
            # GLD 信号当日 GLD 回报
            gld_r = sub[f"r{h}d"]
            # 同日 SLV 回报 (从 slv ohlc lookup)
            common_idx = sub.index.intersection(slv_dates_fwd.index)
            slv_r = slv_dates_fwd.loc[common_idx, f"r{h}d"]
            sg = stats(gld_r); ss = stats(slv_r)
            # 同步率: GLD>0 时 SLV 是否也>0
            both = pd.concat([gld_r, slv_r], axis=1).dropna()
            both.columns = ["gld", "slv"]
            sync = ((both["gld"] > 0) == (both["slv"] > 0)).mean() * 100 if len(both) else None
            corr = both["gld"].corr(both["slv"]) if len(both) >= 5 else None
            print(f"    {h}d: GLD WR={sg['WR']}% mean={sg['mean']}%  |  "
                  f"SLV WR={ss['WR']}% mean={ss['mean']}%  |  "
                  f"sync={sync:.1f}% corr={corr:.2f}" if corr is not None else
                  f"    {h}d: GLD WR={sg['WR']}% mean={sg['mean']}%  |  "
                  f"SLV WR={ss['WR']}% mean={ss['mean']}%  |  sync={sync}")

    # SLV 信号 + 同日 GLD 表现
    print()
    print("=" * 100)
    print("【2】SLV BUY 信号 → 同日 GLD spot 前向回报")
    print("=" * 100)
    slv_buy = slv_sig[slv_sig["buy_signal"]].copy()
    print(f"SLV BUY 信号 (全 tier) 共 {len(slv_buy)} 笔, 时间: "
          f"{slv_buy.index.min().date()} → {slv_buy.index.max().date()}")
    for tier in ["S", "A", "B", "ALL"]:
        if tier == "ALL":
            sub = slv_buy
        else:
            sub = slv_buy[slv_buy["signal_tier"] == tier]
        if not len(sub): continue
        print(f"\n  SLV tier={tier} (n={len(sub)}):")
        for h in horizons:
            slv_r = sub[f"r{h}d"]
            common_idx = sub.index.intersection(gld_dates_fwd.index)
            gld_r = gld_dates_fwd.loc[common_idx, f"r{h}d"]
            ss = stats(slv_r); sg = stats(gld_r)
            both = pd.concat([slv_r, gld_r], axis=1).dropna()
            both.columns = ["slv", "gld"]
            sync = ((both["slv"] > 0) == (both["gld"] > 0)).mean() * 100 if len(both) else None
            corr = both["slv"].corr(both["gld"]) if len(both) >= 5 else None
            print(f"    {h}d: SLV WR={ss['WR']}% mean={ss['mean']}%  |  "
                  f"GLD WR={sg['WR']}% mean={sg['mean']}%  |  "
                  f"sync={sync:.1f}% corr={corr:.2f}" if corr is not None else
                  f"    {h}d: SLV WR={ss['WR']}% mean={ss['mean']}%  |  "
                  f"GLD WR={sg['WR']}% mean={sg['mean']}%  |  sync={sync}")

    # 全期日线 GLD/SLV daily return correlation
    print()
    print("=" * 100)
    print("【3】GLD vs SLV 日线 daily return 基准相关性 (全期)")
    print("=" * 100)
    common = gld_ohlc.index.intersection(slv_ohlc.index)
    g_r = gld_ohlc.loc[common, "Close"].pct_change()
    s_r = slv_ohlc.loc[common, "Close"].pct_change()
    df = pd.concat([g_r, s_r], axis=1).dropna()
    df.columns = ["gld", "slv"]
    print(f"  GLD vs SLV daily return corr (全 {len(df)} 天): {df['gld'].corr(df['slv']):.3f}")
    print(f"  GLD vs SLV 日线方向同步率: {((df['gld']>0)==(df['slv']>0)).mean()*100:.1f}%")
    # 按 5y/3y/1y
    for yrs, label in [(5, '近 5y'), (3, '近 3y'), (1, '近 1y')]:
        cut = pd.Timestamp.today() - pd.Timedelta(days=yrs*365)
        sub = df[df.index >= cut]
        if len(sub) > 30:
            print(f"  {label}: corr={sub['gld'].corr(sub['slv']):.3f}, "
                  f"sync={((sub['gld']>0)==(sub['slv']>0)).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
