"""ma_trend 阈值精细 grid (0.94 - 1.01 步长 0.005) — 找最优 cutoff.

之前用 0.99 拍脑袋, 但 5-4 信号 ma_trend 0.9764 被滤后实际反弹获利.
用 backtest CSV (含 ma_trend 已写) 跑 paired 比较, 看哪个阈值期望值最高.

期望值 = wr × avg_pnl × n_kept (累计 PnL)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

CSV = Path("/Users/yhdong/Gold/data/backtest_history")


def add_ma_trend(df, asset):
    """给 BC/SP 信号补 ma_trend 列 (从 raw close 算)."""
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                        index_col=0, parse_dates=True)
    ohlc["ma_trend"] = ohlc["Close"].rolling(20).mean() / ohlc["Close"].rolling(50).mean()
    df = df.copy()
    df["ma_trend"] = df["signal_date"].map(
        lambda d: ohlc["ma_trend"].get(pd.Timestamp(d).normalize(), np.nan))
    return df


for asset in ["GLD", "SLV"]:
    print(f"\n{'='*78}")
    print(f"{asset} ma_trend 阈值 grid (paired BC + SP 全策略累计 PnL)")
    print(f"{'='*78}")
    df = pd.read_csv(CSV / f"backtest_{asset.lower()}_20260506.csv",
                      parse_dates=["signal_date"])
    df = add_ma_trend(df, asset)
    df = df.dropna(subset=["ma_trend"])

    # 仅保留有 paired BC + SP 的 signal_dates (确保 paired 比较)
    bc = df[df["strategy"]=="BUY CALL"][["signal_date","pnl_pct","ma_trend",
                                            "iv_rv_gap_pct","bp_low","bp_close",
                                            "gvz_iv_pct","rsi_14","stoch_k","macd_hist"]].set_index("signal_date")
    sp = df[df["strategy"]=="SELL PUT"][["signal_date","pnl_pct"]].set_index("signal_date")
    sp.columns = ["sp_pnl"]
    paired = bc.rename(columns={"pnl_pct":"bc_pnl"}).join(sp, how="inner").reset_index()
    print(f"全集 paired: {len(paired)}\n")

    # sp_score (跟生产一致)
    from core.strategy_config import get_config
    cfg = get_config(asset)
    def score(r):
        s=0.0
        s += cfg.sp_score_w_iv_rv_gap   * (r["iv_rv_gap_pct"] > 0)
        s += cfg.sp_score_w_bp_low_deep * (r["bp_low"] < 0.05)
        s += cfg.sp_score_w_bp_close_low * (r["bp_close"] < 0.30)
        s += cfg.sp_score_w_gvz_high    * (r["gvz_iv_pct"] >= 28)
        s += cfg.sp_score_w_rsi_oversold * (r["rsi_14"] < 30)
        s += cfg.sp_score_w_stoch_low   * (r["stoch_k"] < 40)
        s += cfg.sp_score_w_macd_bear   * (r["macd_hist"] < -0.5)
        return s
    paired["score"] = paired.apply(score, axis=1)
    paired["chosen"] = np.where(paired["score"] >= cfg.sp_score_threshold,
                                   paired["sp_pnl"], paired["bc_pnl"])

    print(f"  {'thr':>6}{'n_kept':>8}{'n_skip':>8}{'wr':>7}"
          f"{'avg':>9}{'sum':>11}{'EV/月':>10}")
    # 5y ≈ 60 月
    months = 60
    for thr in [0.93, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.00]:
        sub = paired[paired["ma_trend"] >= thr]
        n = len(sub)
        skip = len(paired) - n
        if not n: continue
        wr = (sub["chosen"]>0).mean()*100
        avg = sub["chosen"].mean()
        s = sub["chosen"].sum()
        ev_month = s / months
        marker = ""
        if thr == 0.99: marker = " ← 现行"
        print(f"  ≥{thr:>5.3f}{n:>8}{skip:>8}{wr:>6.1f}%{avg:>+8.2f}%"
              f"{s:>+10.1f}%{ev_month:>+9.2f}%{marker}")

    # 单独看 BC + SP 各自的视角
    print(f"\n  BC 子集 (chosen 时走 BC) 的 wr/sum 在不同 thr:")
    print(f"  {'thr':>6}{'BC_n':>6}{'BC_wr':>8}{'BC_sum':>10}")
    for thr in [0.94, 0.96, 0.97, 0.98, 0.99, 1.00]:
        sub = paired[(paired["ma_trend"] >= thr) & (paired["score"] < cfg.sp_score_threshold)]
        if not len(sub): continue
        wr = (sub["bc_pnl"]>0).mean()*100
        s = sub["bc_pnl"].sum()
        print(f"  ≥{thr:>5.3f}{len(sub):>6}{wr:>7.1f}%{s:>+9.1f}%")
