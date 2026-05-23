"""v3.7.226: 统一回测 framework — no look-ahead, raw signals 起步.

设计:
  build_raw_universe(asset)
    → 输出所有 Bull regime 日 (无任何过滤), 含 bp_low / rv_pctile / ret_20d /
      sp_score / gvz / ma_trend / signal_tier / 5/10/20d forward returns.
    输入端确认无 look-ahead:
      ✓ rv_pctile = rolling 252
      ✓ OOS predictions = walk-forward trained
      ✓ build_band = shift(1,2,3)
      ✓ regime min_hold_days=1 (消 forward look-ahead)

  apply_filter(raw, name, value)
    → 单 filter 作为 boolean mask 应用 (快, 无需重跑 generate_daily_signals)

  walk_forward_filter(raw, filter_name, grid, train_years, horizon)
    → 滚动 train/test, 每 fold 选 train 最优 value, 测 test OOS scoreB

  cross_asset_test(asset_src, asset_dst, tier_filter, target_strategy)
    → src 信号触发 dst 策略入场, 计算 dst 策略 P&L
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategy_config import get_config


def _load_features(asset: str) -> pd.DataFrame:
    p = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
         if asset == "GLD" else
         "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    return pd.read_parquet(p)


def _load_ohlc(asset: str) -> pd.DataFrame:
    return pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)


def _load_oos(asset: str) -> pd.DataFrame:
    cfg = load_config()
    if asset == "GLD":
        return load_oos_predictions(cfg)
    return pd.read_parquet(Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet")


def build_raw_universe(asset: str) -> tuple:
    """raw universe = (regime, all features, fwd returns) per date.
    NOT filtered by any signal logic (除 regime Bull mask).
    Returns (df, ohlc).
    """
    oos = _load_oos(asset)
    feat = _load_features(asset)
    ohlc = _load_ohlc(asset)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]
    high = ohlc.loc[common, "High"]
    low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv = feat.loc[common, "rv_10d"]
    rv_pctile = compute_rv_pctile(feat["rv_10d"]).reindex(common)
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    # min_hold_days=1: 消 forward look-ahead
    regime = RegimeClassifier(min_hold_days=1).classify(
        feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    gvz_close = gvz["Close"]

    # MA trend (跟 signals_v2 一致)
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma_trend = ma20 / ma50

    # 完整 sig_df 也算一遍 (拿到 sp_score, signal_tier — 这俩跟 buy_signal/buy_type 独立)
    sig_full = generate_daily_signals(close, high, low, upper, lower, regime, rv_pctile,
                                            asset=asset, gvz_series=gvz_close)
    # 取我们需要的列
    cols_from_sig = ["bp_low", "bp_close", "sp_score", "buy_type", "signal_tier",
                       "buy_signal"]
    sig_pick = sig_full[cols_from_sig].copy() if all(c in sig_full.columns for c in cols_from_sig) else sig_full

    # 组装 raw universe
    raw = pd.DataFrame(index=common)
    raw["close"] = close
    raw["high"] = high
    raw["low"] = low
    raw["upper"] = upper
    raw["lower"] = lower
    raw["rv_pctile"] = rv_pctile
    raw["regime"] = regime
    raw["ma_trend"] = ma_trend
    raw["ret_20d"] = close.pct_change(20)
    raw["gvz"] = gvz_close.reindex(common)
    # merge sig 派生
    for c in ["bp_low", "bp_close", "sp_score", "buy_type", "signal_tier",
               "buy_signal"]:
        if c in sig_full.columns:
            raw[c] = sig_full[c]

    # v3.7.227: 修复 look-ahead — entry 改成"下一日 Open"
    next_open = ohlc["Open"].reindex(common).shift(-1)
    next_close = ohlc["Close"].reindex(common).shift(-1)
    daily_ret = close.pct_change()
    for h in (5, 10, 20):
        # 方向性: r{h}d 信号日+1 Open 入, +h 日后 Close 出
        ext_close = ohlc["Close"].reindex(common).shift(-(h + 1))
        raw[f"r{h}d"] = (ext_close / next_open - 1) * 100
        # 波动率验证指标:
        # 1) abs_r{h}d: |signed return|, STRADDLE 看大, SHORT_VOL 看小
        raw[f"abs_r{h}d"] = raw[f"r{h}d"].abs()
        # 2) max_move_{h}d: 入场后 h 日内 intraday H/L 最大偏离 (实际触及)
        # h 日窗口 max High / min Low, 从 next_open 起算
        high_max = ohlc["High"].reindex(common).rolling(h).max().shift(-(h+1))
        low_min = ohlc["Low"].reindex(common).rolling(h).min().shift(-(h+1))
        max_up = (high_max - next_open) / next_open * 100
        max_down = (next_open - low_min) / next_open * 100
        raw[f"max_up_{h}d"] = max_up
        raw[f"max_down_{h}d"] = max_down
        raw[f"max_move_{h}d"] = pd.concat([max_up, max_down], axis=1).max(axis=1)
        # 3) rv_fwd_{h}d: 入场后 h 日实现波动率 (年化 %)
        rv_fwd = daily_ret.rolling(h).std().shift(-h) * (252**0.5) * 100
        raw[f"rv_fwd_{h}d"] = rv_fwd
        # 4) iv_change_{h}d: GVZ 入场 → h 日后变化 (vega 方向证据)
        gvz_now = gvz_close.reindex(common)
        gvz_fwd = gvz_close.reindex(common).shift(-h)
        raw[f"iv_change_{h}d"] = gvz_fwd - gvz_now

    # 入场 IV 参考 (用 GVZ 当代理)
    raw["iv_entry"] = gvz_close.reindex(common)

    return raw, ohlc


# -------- vol scoring (STRADDLE / SHORT_VOL 专属) --------

def score_straddle(sub: pd.DataFrame, horizon: int = 10) -> dict:
    """STRADDLE 成功 = |spot return| > entry premium 距离 (BE).
    BE = IV_entry * sqrt(horizon/252) (近似 ATM straddle BS).
    主胜率: abs_r > BE (触达盈亏平衡距离).
    辅: iv_change > 0 (vega 同向赚), rv_fwd > iv_entry."""
    cols = ["iv_entry", f"rv_fwd_{horizon}d", f"abs_r{horizon}d",
              f"iv_change_{horizon}d", f"max_move_{horizon}d"]
    if not len(sub) or not all(c in sub.columns for c in cols):
        return {"n": 0}
    valid = sub[cols].dropna()
    if not len(valid): return {"n": 0}
    iv = valid["iv_entry"]; rv = valid[f"rv_fwd_{horizon}d"]
    abs_r = valid[f"abs_r{horizon}d"]; iv_chg = valid[f"iv_change_{horizon}d"]
    max_move = valid[f"max_move_{horizon}d"]
    BE = iv * (horizon / 252) ** 0.5   # entry premium 距离 (%)
    n = len(valid)
    win_be = (abs_r > BE).mean() * 100             # close-vs-BE 主胜率
    win_max = (max_move > BE).mean() * 100         # intraday 触及 BE (更宽)
    win_rv = (rv > iv).mean() * 100                # vol regime 辅证
    return {
        "n": n,
        "WR_abs_gt_BE": round(win_be, 1),
        "WR_max_gt_BE": round(win_max, 1),
        "WR_rv_gt_iv": round(win_rv, 1),
        "mean_abs_r": round(abs_r.mean(), 2),
        "mean_BE": round(BE.mean(), 2),
        "mean_iv_entry": round(iv.mean(), 2),
        "mean_iv_change": round(iv_chg.mean(), 2),
        "mean_rv_fwd": round(rv.mean(), 2),
    }


def score_short_vol(sub: pd.DataFrame, horizon: int = 10) -> dict:
    """SHORT_VOL 成功 = |spot return| < short strike 距离 (≈ IV * sigma).
    短腿距离 1.6σ (ATM IC 配置), 价格在内 → 赚 premium.
    主胜率: abs_r < 1.6 * BE (短腿距离)."""
    cols = ["iv_entry", f"rv_fwd_{horizon}d", f"abs_r{horizon}d",
              f"iv_change_{horizon}d", f"max_move_{horizon}d"]
    if not len(sub) or not all(c in sub.columns for c in cols):
        return {"n": 0}
    valid = sub[cols].dropna()
    if not len(valid): return {"n": 0}
    iv = valid["iv_entry"]; rv = valid[f"rv_fwd_{horizon}d"]
    abs_r = valid[f"abs_r{horizon}d"]; iv_chg = valid[f"iv_change_{horizon}d"]
    max_move = valid[f"max_move_{horizon}d"]
    BE = iv * (horizon / 252) ** 0.5
    short_strike_dist = 1.6 * BE   # IC 1.6σ 短腿
    n = len(valid)
    win_inside = (abs_r < short_strike_dist).mean() * 100
    win_max_inside = (max_move < short_strike_dist).mean() * 100
    win_rv = (rv < iv).mean() * 100
    return {
        "n": n,
        "WR_abs_lt_strike": round(win_inside, 1),
        "WR_max_lt_strike": round(win_max_inside, 1),
        "WR_rv_lt_iv": round(win_rv, 1),
        "mean_abs_r": round(abs_r.mean(), 2),
        "mean_strike_dist": round(short_strike_dist.mean(), 2),
        "mean_iv_entry": round(iv.mean(), 2),
        "mean_iv_change": round(iv_chg.mean(), 2),
        "mean_rv_fwd": round(rv.mean(), 2),
    }


# -------- filter library --------
# 每个 filter 输入 raw DataFrame + value, 返回 boolean mask (True=保留)

def filter_buy_bp(raw, value):
    return raw["bp_low"] < value


def filter_rv_pctile_max(raw, value):
    return raw["rv_pctile"] < value


def filter_ret_20d_min(raw, value):
    return raw["ret_20d"] > value


def filter_ret_20d_max(raw, value):
    return raw["ret_20d"] < value


def filter_iv_high_min(raw, value):
    """高 IV (GVZ >= value) 要求 bp_low <= 0.10. 不够深破 → 拒."""
    high_iv = raw["gvz"] >= value
    deep_break = raw["bp_low"] <= 0.10
    return (~high_iv) | deep_break   # 不是高 IV → 通过; 是高 IV → 需深破


def filter_iv_high_bp_low(raw, value, iv_min=25.0):
    """高 IV 时 bp_low 必须 <= value."""
    high_iv = raw["gvz"] >= iv_min
    return (~high_iv) | (raw["bp_low"] <= value)


def filter_ma_trend(raw, value):
    return raw["ma_trend"] >= value


def filter_tier(raw, value):
    """value in {'S','A','B','S+A','ALL'}: 保留对应 tier."""
    if value == "ALL":
        return pd.Series(True, index=raw.index)
    if value == "S+A":
        return raw["signal_tier"].isin(["S", "A"])
    return raw["signal_tier"] == value


def filter_bull_only(raw, _value=None):
    return raw["regime"] == "Bull"


FILTERS = {
    "buy_bp": filter_buy_bp,
    "rv_pctile_max": filter_rv_pctile_max,
    "ret_20d_min": filter_ret_20d_min,
    "ret_20d_max": filter_ret_20d_max,
    "iv_filter_high_min": filter_iv_high_min,
    "iv_high_bp_low_max": filter_iv_high_bp_low,
    "ma_trend_threshold": filter_ma_trend,
    "tier": filter_tier,
}


def apply_filters(raw, filters: dict, base_mask=None) -> pd.DataFrame:
    """filters = {filter_name: value}. 同时应用多个."""
    if base_mask is None:
        base_mask = filter_bull_only(raw)
    for fname, fval in filters.items():
        if fname not in FILTERS:
            raise KeyError(f"unknown filter {fname}")
        base_mask = base_mask & FILTERS[fname](raw, fval)
    return raw[base_mask]


# -------- scoring --------

def score(sub: pd.DataFrame, horizon: int = 10) -> dict:
    col = f"r{horizon}d"
    if col not in sub.columns:
        return {"n": 0, "WR": None, "mean": None, "sum": None, "scoreB": 0,
                 "max_loss": None}
    s = sub[col].dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "WR": None, "mean": None, "sum": None, "scoreB": 0,
                 "max_loss": None}
    wr = (s > 0).mean()
    mean = s.mean()
    return {
        "n": n,
        "WR": round(wr * 100, 1),
        "mean": round(mean, 2),
        "sum": round(s.sum(), 1),
        "max_loss": round(s.min(), 1),
        "scoreB": round((wr ** 2) * math.log(1 + n) * mean, 2),
    }


# -------- walk-forward driver --------

def walk_forward_filter(raw: pd.DataFrame, filter_name: str, grid: list,
                          base_filters: dict = None,
                          train_years: int = 4, horizon: int = 10,
                          min_train_n: int = 8, min_test_n: int = 3) -> pd.DataFrame:
    """单 filter walk-forward.

    base_filters: 固定其他 filter (例 {"buy_bp": 0.30, "ma_trend_threshold": 0.0})
    """
    if base_filters is None:
        base_filters = {}
    all_years = sorted(set(raw.index.year))
    if len(all_years) < train_years + 1:
        return pd.DataFrame()
    folds = []
    for test_year in range(all_years[train_years], all_years[-1] + 1):
        train_lo = pd.Timestamp(f"{test_year - train_years}-01-01")
        train_hi = pd.Timestamp(f"{test_year - 1}-12-31")
        test_lo = pd.Timestamp(f"{test_year}-01-01")
        test_hi = pd.Timestamp(f"{test_year}-12-31")

        train_mask = (raw.index >= train_lo) & (raw.index <= train_hi)
        test_mask = (raw.index >= test_lo) & (raw.index <= test_hi)

        train_universe = raw[train_mask]
        test_universe = raw[test_mask]
        if len(train_universe) < 50:
            continue

        # grid on train
        train_results = []
        for v in grid:
            filters = dict(base_filters); filters[filter_name] = v
            sub = apply_filters(train_universe, filters)
            s = score(sub, horizon)
            train_results.append({"value": v, **s})
        valid = [r for r in train_results if r["n"] >= min_train_n]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["scoreB"])
        best_v = best["value"]

        # apply best to test
        filters_best = dict(base_filters); filters_best[filter_name] = best_v
        sub_test = apply_filters(test_universe, filters_best)
        test_perf = score(sub_test, horizon)

        # prod baseline (use config value)
        prod_val = getattr(get_config(raw["regime"].name) if hasattr(raw["regime"], "name") else None,
                              filter_name, None)
        # prod 值的回测
        if prod_val is not None:
            filters_prod = dict(base_filters); filters_prod[filter_name] = prod_val
            sub_prod = apply_filters(test_universe, filters_prod)
            prod_perf = score(sub_prod, horizon)
        else:
            prod_perf = {"n": 0, "scoreB": 0, "WR": None}

        folds.append({
            "test_year": test_year,
            "train": f"{test_year - train_years}-{test_year - 1}",
            "best_value": best_v,
            "train_n": best["n"], "train_WR": best["WR"],
            "train_scoreB": best["scoreB"],
            "test_n": test_perf["n"], "test_WR": test_perf["WR"],
            "test_sum": test_perf["sum"], "test_mean": test_perf["mean"],
            "test_max_loss": test_perf["max_loss"],
            "test_scoreB": test_perf["scoreB"],
            "prod_value": prod_val,
            "prod_n": prod_perf["n"], "prod_WR": prod_perf["WR"],
            "prod_scoreB": prod_perf["scoreB"],
        })
    return pd.DataFrame(folds)


# -------- trailing windows (含最新数据) --------

LAYER1_WINDOWS = [
    ("10y", 10 * 365),
    ("5y",   5 * 365),
    ("3y",   3 * 365),
    ("1y",   365),
]

LAYER2_WINDOWS = [
    ("1y",  365),
    ("6m",  180),
    ("3m",   90),
]


def trailing_slice(df: pd.DataFrame, days_back: int) -> pd.DataFrame:
    """返回 df 中 [latest - days_back, latest] 的子集 (含最新数据)."""
    if not len(df):
        return df
    latest = df.index.max()
    start = latest - pd.Timedelta(days=days_back)
    return df[df.index >= start]


def trailing_grid(raw: pd.DataFrame, filter_name: str, grid: list,
                     base_filters: dict, window_days: int,
                     horizon: int = 10, min_n: int = 5) -> dict:
    """在 trailing 窗口内 grid 一个 filter, 返回 best + 所有结果."""
    sub_raw = trailing_slice(raw, window_days)
    if not len(sub_raw):
        return {"best": None, "all": pd.DataFrame()}
    rows = []
    for v in grid:
        filters = dict(base_filters); filters[filter_name] = v
        sub = apply_filters(sub_raw, filters)
        s = score(sub, horizon)
        rows.append({"value": v, **s})
    df = pd.DataFrame(rows)
    valid = df[df["n"] >= min_n]
    if not len(valid):
        return {"best": None, "all": df}
    best = valid.loc[valid["scoreB"].idxmax()].to_dict()
    return {"best": best, "all": df}


def multi_window_filter(raw: pd.DataFrame, filter_name: str, grid: list,
                           base_filters: dict, windows: list = None,
                           horizon: int = 10, min_n: int = 5) -> pd.DataFrame:
    """跨多 trailing 窗口对一个 filter 做 grid, 返回每窗最佳."""
    if windows is None: windows = LAYER1_WINDOWS
    rows = []
    for label, days in windows:
        result = trailing_grid(raw, filter_name, grid, base_filters,
                                      days, horizon, min_n)
        b = result["best"]
        row = {"window": label, "days": days,
               "best_value": None, "n": 0, "WR": None, "mean": None,
               "sum": None, "max_loss": None, "scoreB": 0}
        if b is not None:
            row.update(b)
            row["best_value"] = b.get("value")
        rows.append(row)
    return pd.DataFrame(rows)


def cross_asset_pivot(raw_src: pd.DataFrame, ohlc_dst: pd.DataFrame,
                          tier: str, horizon: int) -> dict:
    """src 出现 tier 信号当日, dst spot 前向回报."""
    bs = raw_src["buy_signal"].fillna(False).astype(bool)
    st_col = raw_src["signal_tier"].fillna("")
    if tier == "ALL":
        src_dates = raw_src.index[bs]
    elif tier == "S+A":
        src_dates = raw_src.index[bs & st_col.isin(["S", "A"])]
    elif tier == "no_filter":
        src_dates = raw_src.index[raw_src["regime"] == "Bull"]
    else:
        src_dates = raw_src.index[bs & (st_col == tier)]
    # v3.7.227: dst spot return — entry 改下一日 Open (严格无 look-ahead)
    dst_returns = []
    for d in src_dates:
        if d not in ohlc_dst.index: continue
        i = ohlc_dst.index.get_loc(d)
        # 入场 = 信号日+1 Open (严格事后, 信号 EOD 才完整)
        if i + 1 >= len(ohlc_dst): continue
        ent = float(ohlc_dst.iloc[i + 1]["Open"])
        # 出场 = 信号日+1+horizon Close
        if i + 1 + horizon < len(ohlc_dst):
            ext = float(ohlc_dst.iloc[i + 1 + horizon]["Close"])
            dst_returns.append((ext / ent - 1) * 100)
    if not dst_returns:
        return {"n": 0, "WR": None, "mean": None, "sum": None}
    s = pd.Series(dst_returns)
    return {
        "n": len(s),
        "WR": round((s > 0).mean() * 100, 1),
        "mean": round(s.mean(), 2),
        "sum": round(s.sum(), 1),
        "max_loss": round(s.min(), 1),
        "max_gain": round(s.max(), 1),
    }
