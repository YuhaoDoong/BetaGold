"""一键生成 positions ledger — single source of truth.

每个 (signal_date, asset, strategy) 一行, 含完整 entry pricing snapshot.
Dashboard 读这个文件就行, 不重算.

输出:
  /Users/yhdong/Gold/data/positions_ledger.parquet
  /Users/yhdong/Gold/data/positions_ledger.json (人类可读)

用法:
  python scripts/build_positions_ledger.py [--days 90]
"""
import sys, os, json, argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

# 强制重载 paper_positions cache
import core.paper_positions as _pp
_pp._KLINE_DB_CACHE = None
_pp._KLINE_DB_MTIME = None

from core.signals_v2 import generate_daily_signals
from core.signals import build_band
from core.data import load_oos_predictions, load_config
from core.events import detect_short_vol_signal, detect_straddle_signal
from core.regime import RegimeClassifier
from core.paper_positions import (price_strategy_at, simulate_option_exit)
from core.binance_futures import (fetch_perp_price_at_date, fetch_perp_klines,
                                       fetch_realtime_for_asset, ASSET_SYMBOL)
from core.strategies.futures_long import simulate_long_position
from core.strategy_configs import get_futures_config
try:
    from core.strategy_configs import SHORT_VOL_DISABLED
except ImportError:
    SHORT_VOL_DISABLED = False


LEDGER_PARQUET = "/Users/yhdong/Gold/data/positions_ledger.parquet"
LEDGER_JSON = "/Users/yhdong/Gold/data/positions_ledger.json"
# v3.7.219: 评估水位线 — 每个 asset 最后一次"已评估到"的日期 (>=该日的决策才允许新写入)
LEDGER_META = "/Users/yhdong/Gold/data/positions_ledger_meta.json"


def build_for_asset(asset: str, days_back: int, today_dt: pd.Timestamp,
                       gvz: pd.Series, evaluated_through: pd.Timestamp = None,
                       config_version: str = "") -> list:
    """对 asset 跑 last N days 信号 + 定价, 生成 ledger rows."""
    asset_lc = asset.lower()
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset_lc}.csv",
                        index_col=0, parse_dates=True)
    feat_path = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
                  if asset == "GLD" else
                  "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    features = pd.read_parquet(feat_path)
    # v3.7.193: 用 OOS 模型 band, 跟 dashboard / build_futures_signals 一致
    # (旧版用 BB SMA±2σ, bp_low 算错, 5/12 等信号被吞)
    cfg = load_config()
    if asset == "GLD":
        oos = load_oos_predictions(cfg)  # dl_range_gc_oos
    else:
        slv_oos_path = Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet"
        oos = pd.read_parquet(slv_oos_path) if slv_oos_path.exists() else None
    if oos is not None:
        common = ohlc.index.intersection(features.index).intersection(oos.index)
        close_d = ohlc.loc[common, "Close"]
        high_d = ohlc.loc[common, "High"]
        low_d = ohlc.loc[common, "Low"]
        upper, lower, _ = build_band(oos.loc[common], close_d)
    else:
        common = ohlc.index.intersection(features.index)
        close_d = ohlc.loc[common, "Close"]
        high_d = ohlc.loc[common, "High"]
        low_d = ohlc.loc[common, "Low"]
        sma = close_d.rolling(20).mean()
        std = close_d.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        print(f"[ledger] {asset} OOS 缺失, fallback BB")

    rv = features.loc[common, "rv_10d"]
    # v3.7.220: 用 rolling-252 (跟 dashboard / 实际 model 一致, 不 look-ahead)
    # 旧 v3.7.193: rv.rank(pct=True) 全样本排名是 look-ahead bug
    from core.signals import compute_rv_pctile
    rv_pctile = compute_rv_pctile(features["rv_10d"]).reindex(common)
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier(min_hold_days=1).classify(
        features.loc[common, feat_cols])["regime"]  # v3.7.233 explicit no-lookahead

    sig_df = generate_daily_signals(close_d, high_d, low_d, upper, lower,
                                       regime, rv_pctile, asset=asset, gvz_series=gvz)

    # v3.7.220: 写 sig_df 历史快照 — 冻结当日决策 (跟 ledger 同源不可变)
    try:
        from core.sig_df_history import append_snapshot
        _n_snap = append_snapshot(asset, sig_df, evaluated_through)
        if _n_snap:
            print(f"[ledger] {asset} sig_df snapshot 写入 {_n_snap} 新行")
    except Exception as _e:
        print(f"[ledger] sig_df snapshot 失败: {_e}")

    # v3.7.190: 双 pipeline — 期权用 ETF sig_df (上面), 期货另读 GC/SI scale sig_df
    sig_fut_path = (f"/Users/yhdong/Gold/data/processed/sig_df_"
                     f"{'gc' if asset == 'GLD' else 'si'}.parquet")
    try:
        sig_df_fut = pd.read_parquet(sig_fut_path)
    except Exception:
        sig_df_fut = sig_df  # fallback ETF
        print(f"[ledger] {asset} 期货 sig_df 缺失, fallback ETF scale")

    window_start = today_dt - timedelta(days=days_back)
    u_dates = sig_df.index[sig_df.index >= window_start]

    strad_df = detect_straddle_signal(rv, u_dates, rv_pctile=rv_pctile,
                                          close=close_d, high=high_d, low=low_d,
                                          asset=asset)
    sv_df = detect_short_vol_signal(rv, rv_pctile, u_dates, regime=regime,
                                          close=close_d, high=high_d, low=low_d,
                                          asset=asset)

    # v3.7.164: 期货模块独立 — 用 Binance 历史 OHLC, 不再用 ETF × ratio
    binance_sym = ASSET_SYMBOL.get(asset)  # XAUUSDT / XAGUSDT
    binance_live = fetch_realtime_for_asset(asset)
    binance_live_mark = binance_live.get("mark_price") if binance_live else None
    print(f"[ledger] {asset} → Binance {binance_sym} live mark = "
          f"${binance_live_mark:.2f}" if binance_live_mark else
          f"[ledger] {asset} → Binance {binance_sym} live: 拿不到")

    # v3.7.187: 被过滤信号 log writer
    from core.filtered_signal_log import append_log as _append_filt
    _filt_rows = []

    rows = []
    _now_iso = datetime.now(ZoneInfo("Asia/Singapore")).isoformat()
    for _du, _ru in sig_df.loc[u_dates].iterrows():
        # v3.7.219: 日期级冻结 — 已评估过的 trading day 跳过 (不管当时有没有信号)
        # 新 config 不允许追溯改写历史决策
        if evaluated_through is not None and _du.normalize() <= evaluated_through:
            continue
        is_strad = (_du in strad_df.index
                     and bool(strad_df.loc[_du, "straddle_signal"]))
        is_sv = (_du in sv_df.index
                  and bool(sv_df.loc[_du, "short_vol_signal"]))
        strats = []
        if is_strad: strats.append("STRADDLE")
        # v3.7.178: SHORT_VOL_DISABLED 真生效 (实战 24% WR, IC 大波动期失效)
        if is_sv and not SHORT_VOL_DISABLED:
            strats.append("SHORT_VOL")
        elif is_sv and SHORT_VOL_DISABLED:
            _filt_rows.append({
                "date": _du, "asset": asset, "candidate_strategy": "SHORT_VOL",
                "filter_reason": "SHORT_VOL_DISABLED (v3.7.177 起停用, 实战 6% WR)",
                "raw_trigger_price": float(ohlc.loc[_du, "Close"]) if _du in ohlc.index else 0,
                "raw_trigger_time": _du.isoformat(),
                "detect_source": "daily",
            })
        # v3.7.190: 期权用 ETF sig_df, 期货用 GC/SI sig_df (双 pipeline)
        _opt_buy = bool(_ru.get("buy_signal", False))
        _opt_bt = _ru.get("buy_type") or ""
        _fut_buy = False
        if _du in sig_df_fut.index:
            _fut_buy = bool(sig_df_fut.loc[_du].get("buy_signal", False))
        if _opt_buy:
            if _opt_bt:
                strats.append(_opt_bt)
            else:
                _filt_rows.append({
                    "date": _du, "asset": asset, "candidate_strategy": "BC/SP (方向性)",
                    "filter_reason": "IV 三阶过滤 / sp_score 未通过",
                    "raw_trigger_price": float(ohlc.loc[_du, "Close"]) if _du in ohlc.index else 0,
                    "raw_trigger_time": _du.isoformat(),
                    "detect_source": "daily (ETF)",
                })
        # 期货独立决策: GC=F daily signal (24h)
        if _fut_buy:
            strats.append("FUTURES_LONG")
        if not strats: continue

        if _du not in ohlc.index: continue
        eO = float(ohlc.loc[_du, "Open"])
        eC = float(ohlc.loc[_du, "Close"])
        eH = float(ohlc.loc[_du, "High"])
        eL = float(ohlc.loc[_du, "Low"])

        for strat in strats:
            # v3.7.167: 期货 delegate 到 simulate_long_position (futures_long 模块)
            # 用 Binance kline → DataFrame → cfg from strategy_configs
            # 不再 inline 重复 SL/TP/Liq/早平 逻辑
            if strat == "FUTURES_LONG":
                _bin_entry = fetch_perp_price_at_date(binance_sym, _du)
                if not _bin_entry:
                    continue
                _entry_perp = _bin_entry["open"]
                _start_ms = int((_du + pd.Timedelta(days=1)).timestamp() * 1000)
                _end_ms = int(today_dt.timestamp() * 1000)
                _bin_klines = fetch_perp_klines(binance_sym, _start_ms, _end_ms, "1d")
                # Binance kline → OHLC DataFrame (signal_date 之后用)
                _ohlc_records = []
                for k in _bin_klines:
                    _ohlc_records.append({
                        "Date": pd.Timestamp(k[0], unit="ms").normalize(),
                        "Open": float(k[1]), "High": float(k[2]),
                        "Low": float(k[3]), "Close": float(k[4]),
                    })
                if _ohlc_records:
                    _df_perp = pd.DataFrame(_ohlc_records).set_index("Date")
                    # entry day row 必须存在 (simulate_long_position 用 ohlc.index > entry_d)
                    if _du not in _df_perp.index:
                        _df_perp.loc[_du] = {
                            "Open": _entry_perp, "High": _bin_entry["high"],
                            "Low": _bin_entry["low"], "Close": _bin_entry["close"],
                        }
                        _df_perp = _df_perp.sort_index()
                else:
                    _df_perp = pd.DataFrame(
                        [{"Open": _entry_perp, "High": _bin_entry["high"],
                          "Low": _bin_entry["low"], "Close": _bin_entry["close"]}],
                        index=[_du])
                # Cfg: per-asset leverage + SL/TP/Liq + early locks
                _cfg = get_futures_config(asset)
                # Live mark (今日盘中)
                _live = binance_live_mark or _df_perp.iloc[-1]["Close"]
                # v3.7.204: 传 signal_tier 启用 per-tier leverage
                # 期货读 sig_df_fut (GC/SI scale), 不是 ETF sig_df
                _tier_val = sig_df_fut.loc[_du].get("signal_tier", "") if _du in sig_df_fut.index else ""
                _sim_res = simulate_long_position(
                    entry_d=_du, entry_spot=_entry_perp,
                    ohlc=_df_perp, today=today_dt, cfg=_cfg,
                    live_spot=_live, signal_tier=_tier_val)
                _is_closed = _sim_res.get("closed", False)
                _exit_d_obj = _sim_res.get("exit_date")
                _exit_d_iso = (_exit_d_obj.isoformat()
                                if isinstance(_exit_d_obj, pd.Timestamp) else None)
                _exit_v = float(_sim_res.get("exit_price", _entry_perp) or _entry_perp)
                _ret_lev = max(-100.0, float(
                    _sim_res.get("ret_levered_pct", 0) or 0))
                _hold_d = int(_sim_res.get("hold_days", 0) or 0)
                # v3.7.204: 实际生效 leverage (per-tier 已覆盖)
                _eff_lev = int(_sim_res.get("leverage", _cfg.leverage))
                row = {
                    "asset": asset, "signal_date": _du.isoformat(),
                    "strategy": "FUTURES_LONG",
                    "entry_etf": eO, "entry_perp": _entry_perp,
                    "binance_symbol": binance_sym,
                    "leverage": _eff_lev,
                    "signal_tier": _tier_val,
                    "source": f"{binance_sym} 多头 @ ${_entry_perp:.2f} ({_eff_lev}× tier={_tier_val})",
                    "entry_credit_or_premium": _entry_perp,
                    "legs": [], "entry_leg_prices": [], "exit_legs": [],
                    "is_closed": _is_closed,
                    "exit_date": _exit_d_iso if _is_closed else None,
                    "exit_value": _exit_v if _is_closed else 0.0,
                    "exit_reason": _sim_res.get("reason", "") if _is_closed else "",
                    "current_value": _exit_v if _is_closed else _live,
                    "pnl_pct": _ret_lev,
                    "hold_days": _hold_d if _is_closed
                                  else max(0, (today_dt - _du).days),
                    # v3.7.219: 冻结元数据
                    "created_at": _now_iso,
                    "config_version": config_version,
                }
                rows.append(row)
                continue
            # 期权模块 — 仅 ETF 价 + kline_db (跟期货模块完全独立)
            ent = price_strategy_at(asset, strat, _du,
                                       _du + pd.Timedelta(hours=9, minutes=30),
                                       eO, eO, eC, eH, eL,
                                       dte_target=(14 if strat == "STRADDLE" else 30))
            legs = ent.get("legs", [])
            if not legs:
                # v3.7.237: distinguish freshness-gated skips from no-contract skips
                _src = ent.get("source", "")
                if _src.startswith("PENDING_KLINE") or _src == "NO_KLINE_DB":
                    print(f"[freshness] skip {asset} {strat} @ {_du.date()}: {_src}",
                            flush=True)
                continue
            sim = simulate_option_exit(ent, _du, strat, today_dt,
                                            live_spot=eC, live_high=eH, live_low=eL)
            is_closed = sim.get("is_closed", False)
            # v3.7.206: 期权也写 signal_tier (从 ETF sig_df)
            _opt_tier = sig_df.loc[_du].get("signal_tier", "") if _du in sig_df.index else ""
            row = {
                "asset": asset, "signal_date": _du.isoformat(),
                "strategy": strat,
                "signal_tier": _opt_tier,
                "entry_etf": eO, "entry_open": eO, "entry_close": eC,
                "source": ent.get("source", "—"),
                "entry_credit_or_premium": float(ent.get("entry_price", 0) or 0),
                "legs": [list(l) for l in legs],
                "entry_leg_prices": [list(p) for p in ent.get("leg_prices", [])],
                "exit_legs": [list(p) for p in sim.get("leg_prices", [])],
                "is_closed": is_closed,
                "exit_date": (sim.get("exit_date").isoformat()
                                if isinstance(sim.get("exit_date"), pd.Timestamp)
                                else None),
                "exit_value": float(sim.get("exit_value", 0) or 0),
                "exit_reason": sim.get("exit_reason", ""),
                "current_value": float(sim.get("current_value", 0) or 0),
                "pnl_pct": float(sim.get("pnl_pct", 0) or 0),
                "hold_days": int(sim.get("hold_days", 0) or 0),
                # v3.7.219: 冻结元数据
                "created_at": _now_iso,
                "config_version": config_version,
            }
            rows.append(row)
    # v3.7.187: flush 被过滤信号 log
    if _filt_rows:
        n = _append_filt(_filt_rows)
        print(f"[ledger] {asset} 写入 {n} 笔被过滤信号 → filtered_signal_log.parquet")
    return rows


def _get_config_version() -> str:
    """Return short git SHA + version tag for freeze metadata."""
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent.parent),
             "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except Exception:
        return "unknown"


def _refresh_open_position(row: dict, today_dt: pd.Timestamp,
                              binance_live_mark: dict) -> dict:
    """v3.7.219: 刷新 open 行的当前状态 (current_value/pnl/is_closed/exit_*),
    入场字段 (legs/entry_*/strategy/signal_tier) 锁住不动."""
    if row.get("is_closed"):
        return row  # 已关，不动
    asset = row["asset"]
    strat = row["strategy"]
    sig_d = pd.Timestamp(row["signal_date"])
    if strat == "FUTURES_LONG":
        binance_sym = row.get("binance_symbol") or ASSET_SYMBOL.get(asset)
        _entry_perp = float(row.get("entry_perp", row.get(
            "entry_credit_or_premium", 0)))
        _start_ms = int((sig_d + pd.Timedelta(days=1)).timestamp() * 1000)
        _end_ms = int(today_dt.timestamp() * 1000)
        try:
            _bin_klines = fetch_perp_klines(binance_sym, _start_ms, _end_ms, "1d")
        except Exception:
            _bin_klines = []
        _ohlc_records = [{
            "Date": pd.Timestamp(k[0], unit="ms").normalize(),
            "Open": float(k[1]), "High": float(k[2]),
            "Low": float(k[3]), "Close": float(k[4]),
        } for k in _bin_klines]
        if _ohlc_records:
            _df_perp = pd.DataFrame(_ohlc_records).set_index("Date")
            if sig_d not in _df_perp.index:
                _df_perp.loc[sig_d] = {
                    "Open": _entry_perp, "High": _entry_perp,
                    "Low": _entry_perp, "Close": _entry_perp,
                }
                _df_perp = _df_perp.sort_index()
        else:
            _df_perp = pd.DataFrame(
                [{"Open": _entry_perp, "High": _entry_perp,
                  "Low": _entry_perp, "Close": _entry_perp}],
                index=[sig_d])
        _cfg = get_futures_config(asset)
        _live = binance_live_mark.get(asset) or _df_perp.iloc[-1]["Close"]
        _tier_val = row.get("signal_tier", "")
        _sim_res = simulate_long_position(
            entry_d=sig_d, entry_spot=_entry_perp,
            ohlc=_df_perp, today=today_dt, cfg=_cfg,
            live_spot=_live, signal_tier=_tier_val)
        _is_closed = _sim_res.get("closed", False)
        _exit_d_obj = _sim_res.get("exit_date")
        _exit_d_iso = (_exit_d_obj.isoformat()
                        if isinstance(_exit_d_obj, pd.Timestamp) else None)
        _exit_v = float(_sim_res.get("exit_price", _entry_perp) or _entry_perp)
        _ret_lev = max(-100.0, float(_sim_res.get("ret_levered_pct", 0) or 0))
        _hold_d = int(_sim_res.get("hold_days", 0) or 0)
        row["is_closed"] = _is_closed
        row["exit_date"] = _exit_d_iso if _is_closed else None
        row["exit_value"] = _exit_v if _is_closed else 0.0
        row["exit_reason"] = _sim_res.get("reason", "") if _is_closed else ""
        row["current_value"] = _exit_v if _is_closed else _live
        row["pnl_pct"] = _ret_lev
        row["hold_days"] = _hold_d if _is_closed else max(0,
                                                          (today_dt - sig_d).days)
        return row
    # 期权：用 locked legs + simulate_option_exit
    legs = row.get("legs", [])
    if not legs:
        return row
    entry_pricing = {
        "legs": [tuple(l) for l in legs],
        "leg_prices": [tuple(p) for p in row.get("entry_leg_prices", [])],
        "entry_price": float(row.get("entry_credit_or_premium", 0)),
    }
    # 取今日 spot/H/L (yfinance live)
    try:
        ohlc_t = yf.Ticker(asset).history(period="5d")
        live_spot = float(ohlc_t["Close"].iloc[-1])
        live_high = float(ohlc_t["High"].iloc[-1])
        live_low = float(ohlc_t["Low"].iloc[-1])
    except Exception:
        live_spot = row.get("entry_etf", 0); live_high = live_spot; live_low = live_spot
    sim = simulate_option_exit(entry_pricing, sig_d, strat, today_dt,
                                  live_spot=live_spot, live_high=live_high,
                                  live_low=live_low)
    is_closed = sim.get("is_closed", False)
    row["is_closed"] = is_closed
    row["exit_legs"] = [list(p) for p in sim.get("leg_prices", [])]
    row["exit_date"] = (sim.get("exit_date").isoformat()
                          if isinstance(sim.get("exit_date"), pd.Timestamp)
                          else None)
    row["exit_value"] = float(sim.get("exit_value", 0) or 0)
    row["exit_reason"] = sim.get("exit_reason", "")
    row["current_value"] = float(sim.get("current_value", 0) or 0)
    row["pnl_pct"] = float(sim.get("pnl_pct", 0) or 0)
    row["hold_days"] = int(sim.get("hold_days", 0) or 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--rebuild-all", action="store_true",
                     help="v3.7.219: 核武器 — 全量重算 (默认冻结历史决策)")
    args = ap.parse_args()

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    today_dt = pd.Timestamp(today_et)
    config_version = _get_config_version()
    print(f"[ledger] today_et = {today_et}, days_back = {args.days}")
    print(f"[ledger] config_version = {config_version}, "
          f"rebuild_all = {args.rebuild_all}")

    gvz = yf.Ticker("^GVZ").history(period="5y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    gvz_close = gvz["Close"]

    # v3.7.219: 日期级冻结 — 每个 asset 维护 evaluated_through 水位线
    # 旧水位线之前的日期一律 skip (新 config 不许追溯)
    existing_rows = []
    evaluated_through = {}  # {asset: pd.Timestamp}
    if not args.rebuild_all and Path(LEDGER_JSON).exists():
        try:
            with open(LEDGER_JSON) as f:
                existing_rows = json.load(f)
        except Exception as e:
            print(f"[ledger] 加载 ledger 失败: {e}")
            existing_rows = []
    if not args.rebuild_all and Path(LEDGER_META).exists():
        try:
            with open(LEDGER_META) as f:
                _meta = json.load(f)
            for k, v in _meta.get("evaluated_through", {}).items():
                evaluated_through[k] = pd.Timestamp(v).normalize()
            print(f"[ledger] 冻结模式: evaluated_through = "
                  f"{ {k: str(v.date()) for k,v in evaluated_through.items()} }")
        except Exception as e:
            print(f"[ledger] 加载 meta 失败: {e}, 走全量")
            evaluated_through = {}

    # 刷新所有 open 行的当前状态
    binance_live_mark = {}
    for asset in ["GLD", "SLV"]:
        _live = fetch_realtime_for_asset(asset)
        if _live and _live.get("mark_price"):
            binance_live_mark[asset] = _live["mark_price"]
    refreshed = []
    for r in existing_rows:
        try:
            refreshed.append(_refresh_open_position(dict(r), today_dt,
                                                      binance_live_mark))
        except Exception as e:
            print(f"[ledger] refresh 失败 {r.get('asset')} "
                  f"{r.get('signal_date')} {r.get('strategy')}: {e}")
            refreshed.append(r)
    if existing_rows:
        n_open = sum(1 for r in refreshed if not r.get("is_closed"))
        n_closed = len(refreshed) - n_open
        print(f"[ledger] 刷新完成: open={n_open}, closed={n_closed}")

    # 对 evaluated_through 之后的日期跑新信号
    new_rows = []
    latest_data_dates = {}
    for asset in ["GLD", "SLV"]:
        _et = evaluated_through.get(asset)
        rows = build_for_asset(asset, args.days, today_dt, gvz_close,
                                  evaluated_through=_et,
                                  config_version=config_version)
        print(f"[ledger] {asset}: {len(rows)} 新 positions "
              f"(evaluated_through={_et.date() if _et is not None else 'None'})")
        new_rows.extend(rows)
        # 更新水位线 = 这次跑的 sig_df 最大日期 (实际数据日期, 不一定 == today)
        _asset_lc = asset.lower()
        try:
            _ohlc = pd.read_csv(
                f"/Users/yhdong/Gold/data/raw/market/{_asset_lc}.csv",
                index_col=0, parse_dates=True)
            latest_data_dates[asset] = _ohlc.index.max().normalize()
        except Exception:
            latest_data_dates[asset] = today_dt

    # v3.7.224: Cross-asset SLV-S → GLD 同步开仓 (历史 n=23 SLV-S)
    # 默认策略 BC, 也可双开 BC+SP
    try:
        from core.cross_asset_signal import (find_cross_entries,
                                                   should_add_gld_sync,
                                                   CROSS_ENABLED,
                                                   CROSS_STRATEGY)
        from core.sig_df_history import load_history
        if CROSS_ENABLED:
            sig_hist = load_history()
            slv_sig = sig_hist[sig_hist["asset"] == "SLV"].copy()
            slv_sig["date"] = pd.to_datetime(slv_sig["date"])
            slv_sig = slv_sig.set_index("date").sort_index()
            window_start = today_dt - timedelta(days=args.days)
            slv_s_dates = find_cross_entries(slv_sig, window_start, today_dt)
            # GLD sig_df (从 snapshot 拿, 跟 sig_df 同步)
            gld_sig_hist = sig_hist[sig_hist["asset"] == "GLD"].copy()
            gld_sig_hist["date"] = pd.to_datetime(gld_sig_hist["date"])
            gld_sig_for_check = gld_sig_hist.set_index("date").sort_index()
            cross_rows = []
            _cross_strats = (CROSS_STRATEGY if isinstance(CROSS_STRATEGY, (list, tuple))
                              else [CROSS_STRATEGY])
            _g_ohlc = pd.read_csv(
                "/Users/yhdong/Gold/data/raw/market/gld.csv",
                index_col=0, parse_dates=True)
            for d in slv_s_dates:
                if not should_add_gld_sync(refreshed + new_rows, d,
                                               gld_sig_for_check):
                    continue
                if d not in _g_ohlc.index:
                    continue
                eO = float(_g_ohlc.loc[d, "Open"])
                eC = float(_g_ohlc.loc[d, "Close"])
                eH = float(_g_ohlc.loc[d, "High"])
                eL = float(_g_ohlc.loc[d, "Low"])
                for _cstrat in _cross_strats:
                    ent = price_strategy_at("GLD", _cstrat, d,
                                                  d + pd.Timedelta(hours=9, minutes=30),
                                                  eO, eO, eC, eH, eL, dte_target=30)
                    legs = ent.get("legs", [])
                    if not legs:
                        _src = ent.get("source", "")
                        if _src.startswith("PENDING_KLINE") or _src == "NO_KLINE_DB":
                            print(f"[freshness] skip cross-asset GLD {_cstrat} "
                                    f"@ {d.date()}: {_src}", flush=True)
                        continue
                    sim = simulate_option_exit(ent, d, _cstrat, today_dt,
                                                      live_spot=eC, live_high=eH,
                                                      live_low=eL)
                    row = {
                        "asset": "GLD", "signal_date": d.isoformat(),
                        "strategy": _cstrat,
                        "signal_tier": "S-sync",
                        "entry_etf": eO, "entry_open": eO, "entry_close": eC,
                        "source": f"SLV-S sync → {ent.get('source', f'GLD {_cstrat}')}",
                        "entry_credit_or_premium": float(ent.get("entry_price", 0) or 0),
                        "legs": [list(l) for l in legs],
                        "entry_leg_prices": [list(p) for p in ent.get("leg_prices", [])],
                        "exit_legs": [list(p) for p in sim.get("leg_prices", [])],
                        "is_closed": sim.get("is_closed", False),
                        "exit_date": (sim.get("exit_date").isoformat()
                                        if isinstance(sim.get("exit_date"), pd.Timestamp)
                                        else None),
                        "exit_value": float(sim.get("exit_value", 0) or 0),
                        "exit_reason": sim.get("exit_reason", ""),
                        "current_value": float(sim.get("current_value", 0) or 0),
                        "pnl_pct": float(sim.get("pnl_pct", 0) or 0),
                        "hold_days": int(sim.get("hold_days", 0) or 0),
                        "created_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                        "config_version": config_version,
                        "cross_asset_origin": "SLV-S",
                    }
                    cross_rows.append(row)
            if cross_rows:
                print(f"[ledger] Cross-asset SLV-S→GLD sync: 加 {len(cross_rows)} 笔")
                new_rows.extend(cross_rows)
    except Exception as _ca_e:
        print(f"[ledger] cross-asset 失败: {_ca_e}")

    all_rows = refreshed + new_rows
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[ledger] 无信号, 不写入"); return

    os.makedirs(os.path.dirname(LEDGER_JSON), exist_ok=True)
    df_sorted = df.sort_values(["asset", "signal_date", "strategy"])
    with open(LEDGER_JSON, "w") as f:
        json.dump(df_sorted.to_dict(orient="records"), f,
                    indent=2, default=str, ensure_ascii=False)

    # 写水位线 meta
    new_evaluated_through = dict(evaluated_through)
    for asset, _d in latest_data_dates.items():
        new_evaluated_through[asset] = _d
    with open(LEDGER_META, "w") as f:
        json.dump({
            "evaluated_through": {k: v.isoformat()
                                    for k, v in new_evaluated_through.items()},
            "last_run_at": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
            "last_run_config_version": config_version,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[ledger] saved {len(df)} rows ({len(refreshed)} 冻结 + "
          f"{len(new_rows)} 新):")
    print(f"  json:    {LEDGER_JSON}")
    print(f"  meta:    {LEDGER_META}")
    print(f"  evaluated_through: " + ", ".join(
        f"{k}={v.date()}" for k, v in new_evaluated_through.items()))
    print(f"\n按 asset / strategy 汇总:")
    print(df.groupby(["asset", "strategy"]).size().to_string())


if __name__ == "__main__":
    main()
