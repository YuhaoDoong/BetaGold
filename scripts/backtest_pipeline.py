"""v3.7.171 多级回测 pipeline — 真实 0.7 + 模拟 0.3 加权.

逐级备份 (data/backtest_pipeline/stageN/):
  stage0_raw_signals/    所有 detect 输出, 完全未过滤
  stage1_filtered/       经 IV 三阶 / regime / sp_score 等过滤
  stage2_simulated/      分源模拟 PnL (real_options, bs_options, binance_perp, comex_perp)
  stage3_summary/        scoreB 汇总 + 加权报告

每级 parquet 独立可复现, 修改 cfg 后只重跑下游 stage.

用户决策权重 (v3.7.171):
  真实 (近 1y kline_db EOD options + Binance perp): 0.7
  模拟 (BS LEAPS + GC=F COMEX 5y):                  0.3

用法:
  python scripts/backtest_pipeline.py [stage0|stage1|stage2|stage3|all]
"""
from __future__ import annotations
import sys, os, argparse, json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf

from core.data import load_features, load_config, load_oos_predictions
from core.signals import build_band, compute_rv_pctile
from core.signals_v2 import generate_daily_signals
from core.regime import RegimeClassifier
from core.events import detect_straddle_signal, detect_short_vol_signal
from core.binance_futures import (fetch_perp_klines, ASSET_SYMBOL)
from core.strategies.futures_long import simulate_long_position
from core.strategies.short_vol import simulate_short_vol_position
from core.strategies.buy_call import simulate_bc_position
from core.strategies.sell_put import simulate_sp_position
from core.strategies.straddle import simulate_straddle_position
from core.strategy_configs import (get_futures_config, SHORT_VOL_DEFAULT,
                                       SELL_PUT_DEFAULT, BUY_CALL_DEFAULT,
                                       STRADDLE_DEFAULT)
try:
    from core.strategy_configs import SHORT_VOL_DISABLED
except ImportError:
    SHORT_VOL_DISABLED = False
from core.paper_positions import price_strategy_at, _load_kline_db


PIPE = Path("/Users/yhdong/Gold/data/backtest_pipeline")
STAGES = {
    "stage0": PIPE / "stage0_raw_signals",
    "stage1": PIPE / "stage1_filtered",
    "stage2": PIPE / "stage2_simulated",
    "stage3": PIPE / "stage3_summary",
}
VERSIONS = PIPE / "versions"  # v3.7.177: 版本归档 (per git commit)
for p in list(STAGES.values()) + [VERSIONS]:
    p.mkdir(parents=True, exist_ok=True)

WEIGHT_REAL = 0.7
WEIGHT_SIM = 0.3
# v3.7.173: window 可 CLI 传入 (--real-days / --sim-days)
REAL_WINDOW_DAYS = int(os.environ.get("REAL_WINDOW_DAYS", 365))
SIM_WINDOW_DAYS = int(os.environ.get("SIM_WINDOW_DAYS", 1825))


# ─────────────────────── STAGE 0: 原始信号 ───────────────────────
def stage0_raw_signals():
    """完全未过滤 — 所有 detect 函数输出 (regime/IV/sp_score 之前)."""
    print("\n" + "="*78)
    print("STAGE 0: 原始信号提取 (no filter)")
    print("="*78)
    cfg = load_config()
    features = load_features(cfg)
    today = pd.Timestamp.now().normalize()

    for asset, csv_name in [("GLD", "gld.csv"), ("SLV", "slv.csv")]:
        daily = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{csv_name}",
                              index_col=0, parse_dates=True)
        common = features.index.intersection(daily.index)
        close_d = daily["Close"][common]
        high_d = daily["High"][common]
        low_d = daily["Low"][common]
        oos = load_oos_predictions(cfg)
        upper, lower, _ = build_band(oos, close_d)
        rv_pct = compute_rv_pctile(features.loc[common, "rv_10d"])

        feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
        regime = RegimeClassifier().classify(features[feat_cols])["regime"]

        # 原始 directional 信号 (无 IV / regime / sp_score 过滤)
        sig_df = generate_daily_signals(close_d, high_d, low_d, upper, lower,
                                           regime, rv_pct, asset=asset,
                                           gvz_series=None)  # stage0 不带 GVZ → 跳过 IV 过滤
        # 加 STRADDLE / SHORT_VOL detect (规则信号, 不过滤)
        rv_s = features.loc[common, "rv_10d"]
        # detect_straddle_signal(rv_series, dates_index, rv_pctile=, close=, high=, low=, asset=)
        strad = detect_straddle_signal(rv_s, common,
                                          rv_pctile=rv_pct,
                                          close=close_d, high=high_d, low=low_d,
                                          asset=asset)
        sv = detect_short_vol_signal(rv_s, rv_pct, common, regime=regime)

        # Merge: 一行 = 一日, 每信号 boolean
        all_sigs = pd.DataFrame(index=common)
        all_sigs["close"] = close_d
        all_sigs["regime"] = regime.reindex(common).astype(str)
        all_sigs["rv_10d"] = rv_s
        all_sigs["rv_pctile"] = rv_pct.reindex(common)
        all_sigs["upper"] = upper.reindex(common)
        all_sigs["lower"] = lower.reindex(common)
        for col in ["buy_signal", "buy_type", "sell_signal"]:
            if col in sig_df.columns:
                all_sigs[col] = sig_df[col].reindex(common)
        all_sigs["straddle_signal"] = strad["straddle_signal"].reindex(common).fillna(False) \
                                          if "straddle_signal" in strad.columns \
                                          else False
        all_sigs["short_vol_signal"] = sv["short_vol_signal"].reindex(common).fillna(False) \
                                            if "short_vol_signal" in sv.columns \
                                            else False
        # 全部 dump
        out = STAGES["stage0"] / f"raw_signals_{asset.lower()}.parquet"
        all_sigs.to_parquet(out)
        n_buy = int(all_sigs.get("buy_signal", pd.Series(False)).sum())
        n_sv = int(all_sigs["short_vol_signal"].sum())
        n_strad = int(all_sigs["straddle_signal"].sum())
        print(f"  {asset}: {len(all_sigs)} 行 → {out.name}")
        print(f"    buy={n_buy} short_vol={n_sv} straddle={n_strad}")


# ─────────────────────── STAGE 1: 过滤后 ───────────────────────
def stage1_filtered():
    """应用 IV 三阶 / regime / sp_score 等过滤."""
    print("\n" + "="*78)
    print("STAGE 1: 过滤 (IV + regime + sp_score)")
    print("="*78)
    # 拉 GVZ for IV 过滤
    try:
        gvz = yf.Ticker("^GVZ").history(period="5y")
        gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
        gvz_close = gvz["Close"]
    except Exception:
        gvz_close = None

    for asset in ["GLD", "SLV"]:
        raw = pd.read_parquet(STAGES["stage0"] / f"raw_signals_{asset.lower()}.parquet")
        # 应用 IV 三阶过滤 (复用 generate_daily_signals 全套)
        cfg = load_config()
        features = load_features(cfg)
        common = raw.index
        close_d = features.reindex(common, method=None)["close"] \
                    if "close" in features.columns else raw["close"]
        # 从 stage0 的 close 推回 daily
        daily = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                              index_col=0, parse_dates=True)
        common = common.intersection(daily.index)
        close_d = daily["Close"][common]
        high_d = daily["High"][common]
        low_d = daily["Low"][common]
        oos = load_oos_predictions(cfg)
        upper, lower, _ = build_band(oos, close_d)
        rv_pct = compute_rv_pctile(features.loc[common, "rv_10d"])
        feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
        regime = RegimeClassifier().classify(features[feat_cols])["regime"]

        sig_df = generate_daily_signals(close_d, high_d, low_d, upper, lower,
                                           regime, rv_pct, asset=asset,
                                           gvz_series=gvz_close)
        # SHORT_VOL / STRADDLE 保留 raw (event-based, no further filter)
        out_df = pd.DataFrame(index=common)
        for col in ["buy_signal", "buy_type", "sell_signal"]:
            if col in sig_df.columns:
                out_df[col] = sig_df[col].reindex(common)
        out_df["straddle_signal"] = raw["straddle_signal"].reindex(common).fillna(False)
        out_df["short_vol_signal"] = raw["short_vol_signal"].reindex(common).fillna(False)
        out_df["close"] = close_d
        out_df["regime"] = raw["regime"].reindex(common)
        out = STAGES["stage1"] / f"filtered_signals_{asset.lower()}.parquet"
        out_df.to_parquet(out)
        n_buy = int(out_df.get("buy_signal", pd.Series(False)).sum())
        n_buy_raw = int(raw.get("buy_signal", pd.Series(False)).sum())
        print(f"  {asset}: buy {n_buy_raw}→{n_buy} (drop {n_buy_raw-n_buy}) "
              f"sv={int(out_df['short_vol_signal'].sum())} "
              f"strad={int(out_df['straddle_signal'].sum())} → {out.name}")


# ─────────────────────── STAGE 2: 分源模拟 PnL ───────────────────────
def _save_pnls(asset, strat, source, rows, today_dt):
    """rows: list of dict per signal. 空 rows 也覆盖 (写空 parquet) 防残留."""
    out = STAGES["stage2"] / f"pnl_{asset.lower()}_{strat}_{source}_{today_dt.strftime('%Y%m%d')}.parquet"
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["signal_date", "asset", "strategy", "source", "pnl_pct", "hold_days", "exit_reason"])
    df.to_parquet(out)
    return out


def stage2_simulate_real(asset: str):
    """真实数据模拟 — 近 1y kline_db EOD options + Binance perp."""
    print(f"\n[STAGE 2 REAL] {asset} (近 {REAL_WINDOW_DAYS}d real options + Binance)")
    sigs_df = pd.read_parquet(STAGES["stage1"] / f"filtered_signals_{asset.lower()}.parquet")
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=REAL_WINDOW_DAYS)
    sigs_df = sigs_df[sigs_df.index >= cutoff]
    daily = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
    db = _load_kline_db()
    if db is not None and "asset" in db.columns:
        db = db[db["asset"] == asset]

    # FUTURES — Binance perp (recent only)
    sym = ASSET_SYMBOL[asset]
    fut_cfg = get_futures_config(asset)
    start_ms = int(cutoff.timestamp() * 1000)
    end_ms = int(today.timestamp() * 1000)
    klines = fetch_perp_klines(sym, start_ms, end_ms, "1d")
    df_perp = pd.DataFrame([{
        "Date": pd.Timestamp(int(k[0]), unit="ms").normalize(),
        "Open": float(k[1]), "High": float(k[2]),
        "Low": float(k[3]), "Close": float(k[4])} for k in klines]
        ).set_index("Date").sort_index()
    rows_fut = []
    # v3.7.176: 撤销 BC filter, 所有 buy 信号都打期货 (lev wick-safe 保护)
    for d in sigs_df.index[sigs_df.get("buy_signal", False) == True]:
        if d not in df_perp.index: continue
        entry = float(df_perp.loc[d, "Open"])
        res = simulate_long_position(d, entry, df_perp, today, fut_cfg)
        if res.get("closed"):
            rows_fut.append({
                "signal_date": d, "asset": asset, "strategy": "FUTURES_LONG",
                "source": "binance_perp",
                "pnl_pct": max(-100, float(res.get("ret_levered_pct", 0))),
                "hold_days": int(res.get("hold_days", 0)),
                "exit_reason": str(res.get("reason", "")),
            })
    _save_pnls(asset, "FUTURES_LONG", "real_binance", rows_fut, today)
    print(f"  FUTURES Binance: {len(rows_fut)} 笔已平")

    # OPTIONS — kline_db
    rows_bc = []; rows_sp = []; rows_sv = []; rows_strad = []
    for d in sigs_df.index:
        eO = float(daily.loc[d, "Open"]); eC = float(daily.loc[d, "Close"])
        eH = float(daily.loc[d, "High"]); eL = float(daily.loc[d, "Low"])
        bt = sigs_df.loc[d, "buy_type"] if "buy_type" in sigs_df.columns else ""
        if isinstance(bt, str) and bt == "BUY CALL":
            ent = price_strategy_at(asset, "BUY CALL", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=BUY_CALL_DEFAULT.base_dte)
            if ent.get("legs") and db is not None:
                res = simulate_bc_position(ent, d, today, db, BUY_CALL_DEFAULT)
                if res.get("is_closed"):
                    rows_bc.append({"signal_date": d, "asset": asset,
                                      "strategy": "BUY CALL", "source": "real_klinedb",
                                      "pnl_pct": max(-100, min(500, float(res.get("pnl_pct", 0)))),
                                      "hold_days": int(res.get("hold_days", 0)),
                                      "exit_reason": str(res.get("exit_reason", ""))})
        elif isinstance(bt, str) and bt == "SELL PUT":
            ent = price_strategy_at(asset, "SELL PUT", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=SELL_PUT_DEFAULT.base_dte)
            if ent.get("legs") and db is not None:
                res = simulate_sp_position(ent, d, today, db, SELL_PUT_DEFAULT)
                if res.get("is_closed"):
                    rows_sp.append({"signal_date": d, "asset": asset,
                                      "strategy": "SELL PUT", "source": "real_klinedb",
                                      "pnl_pct": max(-100, min(150, float(res.get("pnl_pct", 0)))),
                                      "hold_days": int(res.get("hold_days", 0)),
                                      "exit_reason": str(res.get("exit_reason", ""))})
        if sigs_df.loc[d, "short_vol_signal"] and not SHORT_VOL_DISABLED:
            ent = price_strategy_at(asset, "SHORT_VOL", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=SHORT_VOL_DEFAULT.base_dte)
            if ent.get("legs") and db is not None:
                res = simulate_short_vol_position(ent, d, today, db, SHORT_VOL_DEFAULT)
                if res.get("is_closed"):
                    rows_sv.append({"signal_date": d, "asset": asset,
                                      "strategy": "SHORT_VOL", "source": "real_klinedb",
                                      "pnl_pct": max(-100, min(100, float(res.get("pnl_pct", 0)))),
                                      "hold_days": int(res.get("hold_days", 0)),
                                      "exit_reason": str(res.get("exit_reason", ""))})
        if sigs_df.loc[d, "straddle_signal"]:
            ent = price_strategy_at(asset, "STRADDLE", d,
                                       d + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=STRADDLE_DEFAULT.base_dte)
            if ent.get("legs") and db is not None:
                res = simulate_straddle_position(ent, d, today, db, STRADDLE_DEFAULT)
                if res.get("is_closed"):
                    rows_strad.append({"signal_date": d, "asset": asset,
                                        "strategy": "STRADDLE", "source": "real_klinedb",
                                        "pnl_pct": max(-100, min(500, float(res.get("pnl_pct", 0)))),
                                        "hold_days": int(res.get("hold_days", 0)),
                                        "exit_reason": str(res.get("exit_reason", ""))})

    _save_pnls(asset, "BUY CALL", "real_klinedb", rows_bc, today)
    _save_pnls(asset, "SELL PUT", "real_klinedb", rows_sp, today)
    _save_pnls(asset, "SHORT_VOL", "real_klinedb", rows_sv, today)
    _save_pnls(asset, "STRADDLE", "real_klinedb", rows_strad, today)
    print(f"  Options real: BC={len(rows_bc)} SP={len(rows_sp)} SV={len(rows_sv)} STRAD={len(rows_strad)}")


def stage2_simulate_sim(asset: str):
    """模拟数据 — GC=F/SI=F COMEX 5y futures."""
    print(f"\n[STAGE 2 SIM] {asset} (5y COMEX futures + BS LEAPS proxy)")
    sigs_df = pd.read_parquet(STAGES["stage1"] / f"filtered_signals_{asset.lower()}.parquet")
    today = pd.Timestamp.now().normalize()
    cutoff = today - pd.Timedelta(days=SIM_WINDOW_DAYS)
    sigs_df = sigs_df[sigs_df.index >= cutoff]

    # COMEX futures
    comex_sym = "GC=F" if asset == "GLD" else "SI=F"
    df_perp = yf.Ticker(comex_sym).history(period="5y")
    df_perp.index = pd.to_datetime(df_perp.index).tz_localize(None).normalize()
    df_perp = df_perp[["Open", "High", "Low", "Close"]].dropna()
    fut_cfg = get_futures_config(asset)
    rows_fut = []
    # v3.7.176: 撤销 BC filter, 所有 buy 信号都打期货 (lev wick-safe 保护)
    for d in sigs_df.index[sigs_df.get("buy_signal", False) == True]:
        if d not in df_perp.index: continue
        entry = float(df_perp.loc[d, "Open"])
        res = simulate_long_position(d, entry, df_perp, today, fut_cfg)
        if res.get("closed"):
            rows_fut.append({
                "signal_date": d, "asset": asset, "strategy": "FUTURES_LONG",
                "source": "comex_proxy",
                "pnl_pct": max(-100, float(res.get("ret_levered_pct", 0))),
                "hold_days": int(res.get("hold_days", 0)),
                "exit_reason": str(res.get("reason", "")),
            })
    _save_pnls(asset, "FUTURES_LONG", "sim_comex", rows_fut, today)
    print(f"  FUTURES COMEX: {len(rows_fut)} 笔已平")
    # NOTE: Options BS LEAPS proxy 跑量大耗时, 这里用 full_history_backtest 的 csv 兼用


# ─────────────────────── STAGE 3: 加权汇总 ───────────────────────
def _scoreB(pnls: pd.Series):
    if len(pnls) == 0: return None
    wr = (pnls > 0).mean()
    return {
        "n": len(pnls),
        "wr_pct": wr * 100,
        "avg": pnls.mean(),
        "sum": pnls.sum(),
        "scoreB": (wr ** 2) * np.log(1 + len(pnls)) * pnls.mean(),
    }


def stage3_summary():
    print("\n" + "="*78)
    print(f"STAGE 3: 加权 scoreB (real {WEIGHT_REAL:.1f} + sim {WEIGHT_SIM:.1f})")
    print("="*78)
    today_str = pd.Timestamp.now().normalize().strftime('%Y%m%d')
    rows_summary = []

    for asset in ["GLD", "SLV"]:
        for strat in ["FUTURES_LONG", "BUY CALL", "SELL PUT", "SHORT_VOL", "STRADDLE"]:
            real_files = list(STAGES["stage2"].glob(
                f"pnl_{asset.lower()}_{strat}_real_*_{today_str}.parquet"))
            sim_files = list(STAGES["stage2"].glob(
                f"pnl_{asset.lower()}_{strat}_sim_*_{today_str}.parquet"))
            real_df = (pd.concat([pd.read_parquet(f) for f in real_files])
                         if real_files else pd.DataFrame())
            sim_df = (pd.concat([pd.read_parquet(f) for f in sim_files])
                        if sim_files else pd.DataFrame())
            real_m = _scoreB(real_df["pnl_pct"]) if len(real_df) else None
            sim_m = _scoreB(sim_df["pnl_pct"]) if len(sim_df) else None
            # 加权 scoreB (None 视为 0 权重)
            w_score = 0.0; w_total = 0.0
            if real_m: w_score += real_m["scoreB"] * WEIGHT_REAL; w_total += WEIGHT_REAL
            if sim_m:  w_score += sim_m["scoreB"] * WEIGHT_SIM; w_total += WEIGHT_SIM
            wgt = w_score / w_total if w_total > 0 else None
            rows_summary.append({
                "asset": asset, "strategy": strat,
                "real_n": real_m["n"] if real_m else 0,
                "real_wr": real_m["wr_pct"] if real_m else None,
                "real_avg": real_m["avg"] if real_m else None,
                "real_scoreB": real_m["scoreB"] if real_m else None,
                "sim_n": sim_m["n"] if sim_m else 0,
                "sim_wr": sim_m["wr_pct"] if sim_m else None,
                "sim_avg": sim_m["avg"] if sim_m else None,
                "sim_scoreB": sim_m["scoreB"] if sim_m else None,
                "weighted_scoreB": wgt,
            })
    rep = pd.DataFrame(rows_summary)
    out = STAGES["stage3"] / f"summary_{today_str}.parquet"
    rep.to_parquet(out)
    print(f"\n保存: {out}\n")
    # 打印
    for asset in ["GLD", "SLV"]:
        sub = rep[rep["asset"] == asset]
        print(f"\n{asset}:")
        print(f"  {'strategy':<14}{'real(n,wr,avg,sB)':<32}{'sim(n,wr,avg,sB)':<32}{'wgt sB':>8}")
        for _, r in sub.iterrows():
            real_s = (f"({int(r['real_n'])},{r['real_wr']:.0f}%,{r['real_avg']:+.1f},{r['real_scoreB']:+.1f})"
                      if r["real_n"] > 0 else "(0)")
            sim_s = (f"({int(r['sim_n'])},{r['sim_wr']:.0f}%,{r['sim_avg']:+.1f},{r['sim_scoreB']:+.1f})"
                     if r["sim_n"] > 0 else "(0)")
            wgt_s = f"{r['weighted_scoreB']:+.2f}" if r["weighted_scoreB"] is not None else "—"
            print(f"  {r['strategy']:<14}{real_s:<32}{sim_s:<32}{wgt_s:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                      choices=["stage0", "stage1", "stage2", "stage3", "all"])
    ap.add_argument("--real-days", type=int, default=None,
                      help="真实数据窗口 (默认 365)")
    ap.add_argument("--sim-days", type=int, default=None,
                      help="模拟数据窗口 (默认 1825)")
    args = ap.parse_args()
    global REAL_WINDOW_DAYS, SIM_WINDOW_DAYS
    if args.real_days is not None:
        REAL_WINDOW_DAYS = args.real_days
    if args.sim_days is not None:
        SIM_WINDOW_DAYS = args.sim_days
    print(f"v3.7.171 backtest pipeline @ {datetime.now()}")
    print(f"  PIPE = {PIPE}")
    print(f"  weights: real={WEIGHT_REAL} sim={WEIGHT_SIM}")
    if args.which in ("stage0", "all"):
        stage0_raw_signals()
    if args.which in ("stage1", "all"):
        stage1_filtered()
    if args.which in ("stage2", "all"):
        for asset in ["GLD", "SLV"]:
            stage2_simulate_real(asset)
            stage2_simulate_sim(asset)
    if args.which in ("stage3", "all"):
        stage3_summary()
    if args.which == "all":
        archive_version()


def archive_version():
    """v3.7.177: 归档当前 stage0-3 输出到 versions/<git_commit>_<date>/."""
    import subprocess, shutil
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent.parent), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"
    today_str = pd.Timestamp.now().strftime('%Y%m%d')
    archive_dir = VERSIONS / f"{commit}_{today_str}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, src in STAGES.items():
        dst = archive_dir / stage_name
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
    # cfg snapshot
    try:
        from core.strategy_configs import summary as cfg_summary
        with open(archive_dir / "cfg_snapshot.txt", "w") as f:
            f.write(f"# git commit: {commit}\n")
            f.write(f"# date: {pd.Timestamp.now().isoformat()}\n")
            f.write(f"# real window: {REAL_WINDOW_DAYS}d, sim window: {SIM_WINDOW_DAYS}d\n\n")
            f.write(cfg_summary())
    except Exception as e:
        pass
    print(f"\n[ARCHIVE] {archive_dir}")


if __name__ == "__main__":
    main()
