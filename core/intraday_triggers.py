"""盘中触发检测 (参数化, 用于回测对比规则/时间尺度的成功率与收益).

核心模型:
  - 日线层 (今日预测) 给出当日 bp030 / bp090 阈值 → 仅是 "开窗阈值"
  - 盘中层 (本模块) 实时监控盘中 K线, 当价格在阈值外侧 + 指标确认时触发
  - 一天可触发多次, 每次记录真实成交价 + 时间戳 + 规则
  - 历史代表价: 买入取最差 (max)、退出取最差 (min)

所有规则、时间尺度、交易时段都通过参数控制, 不硬编码.

公开函数:
    compute_indicators(kline, ...): 计算 Stoch RSI / MACD / KDJ
    detect_triggers(...): 主入口, 返回触发事件 DataFrame
    worst_of_day(...): 按日聚合到代表价
    backfill(...): 历史回填到 parquet log

规则集合 (RULES 常量) 包含所有可选规则; 用 rule_set 参数控制启用哪些;
用 confirm_mode (any/all/k_of_n) 控制确认强度.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import time, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ── 默认参数 (全部可被调用者覆盖) ────────────────────────────

# Stoch RSI(14, 14, 3, 3) — 与日线/MTF 面板一致
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_SMOOTH_K = 3
DEFAULT_STOCH_RSI_SMOOTH_D = 3
DEFAULT_STOCH_OVERSOLD = 20
DEFAULT_STOCH_OVERBOUGHT = 80

# MACD(12, 26, 9)
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9

# KDJ(9, 3, 3)
DEFAULT_KDJ_PERIOD = 9
DEFAULT_KDJ_K_SMOOTH = 3
DEFAULT_KDJ_D_SMOOTH = 3
DEFAULT_KDJ_J_OVERSOLD = 0
DEFAULT_KDJ_J_OVERBOUGHT = 100

# 交易时段 (UTC). 所有时间戳必须先归一化到 UTC.
# 美股期权: 09:30 - 16:00 ET ≈ 14:30 - 21:00 UTC (DST aware 调用方处理)
US_OPTIONS_SESSION_UTC = (time(14, 30), time(21, 0))
# 黄金期货 CME Globex: 23h/天, 周日 23:00 UTC 开盘 ~ 周五 22:00 UTC. 这里给 24h 占位.
FUTURES_SESSION_24H = (time(0, 0), time(23, 59))


# ── 触发规则定义 (全部内置, 启用与否由参数控制) ──────────────

#: 所有可用规则名. 调用方通过 `rule_set=[...]` 指定启用.
RULES_BUY = (
    "stoch_rsi_cross_up_oversold",   # Stoch RSI K 上穿 oversold 阈
    "stoch_rsi_in_oversold",         # Stoch RSI K 当前 < oversold (软条件)
    "macd_bullish_cross",            # MACD line 上穿 signal line
    "macd_hist_turn_up",             # MACD 柱由减弱 (绿减) 转加强 (绿增/红减)
    "kdj_j_cross_up_oversold",       # KDJ J 上穿 oversold
    "kdj_k_cross_d_up",              # KDJ K 上穿 D
)
RULES_EXIT = (
    "stoch_rsi_cross_down_overbought",
    "stoch_rsi_in_overbought",
    "macd_bearish_cross",
    "macd_hist_turn_down",
    "kdj_j_cross_down_overbought",
    "kdj_k_cross_d_down",
)

DEFAULT_BUY_RULES = ("stoch_rsi_cross_up_oversold",
                       "stoch_rsi_in_oversold",       # v3.7.56: 沉默触发 (价格在底 + Stoch oversold 即可)
                       "macd_bullish_cross",
                       "macd_hist_turn_up")
DEFAULT_EXIT_RULES = ("stoch_rsi_cross_down_overbought",
                        "stoch_rsi_in_overbought",     # v3.7.56: 沉默触发对称
                        "macd_bearish_cross")


# ── 指标计算 ──────────────────────────────────────────────

def _stoch_rsi(close: pd.Series,
               period: int = DEFAULT_STOCH_RSI_PERIOD,
               smooth_k: int = DEFAULT_STOCH_RSI_SMOOTH_K,
               smooth_d: int = DEFAULT_STOCH_RSI_SMOOTH_D):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=3).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rlow = rsi.rolling(period, min_periods=3).min()
    rhigh = rsi.rolling(period, min_periods=3).max()
    k_raw = ((rsi - rlow) / (rhigh - rlow).replace(0, np.nan)) * 100
    k = k_raw.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return k, d


def _macd(close: pd.Series,
          fast: int = DEFAULT_MACD_FAST,
          slow: int = DEFAULT_MACD_SLOW,
          signal: int = DEFAULT_MACD_SIGNAL):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = DEFAULT_KDJ_PERIOD,
         k_smooth: int = DEFAULT_KDJ_K_SMOOTH,
         d_smooth: int = DEFAULT_KDJ_D_SMOOTH):
    ll = low.rolling(period, min_periods=1).min()
    hh = high.rolling(period, min_periods=1).max()
    rsv = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def compute_indicators(kline: pd.DataFrame, **kw) -> pd.DataFrame:
    """计算 Stoch RSI / MACD / KDJ 三套指标. 返回与 kline 同 index 的 DataFrame."""
    c = kline["Close"]
    h = kline["High"]
    l = kline["Low"]

    sk, sd = _stoch_rsi(
        c,
        kw.get("stoch_period", DEFAULT_STOCH_RSI_PERIOD),
        kw.get("stoch_smooth_k", DEFAULT_STOCH_RSI_SMOOTH_K),
        kw.get("stoch_smooth_d", DEFAULT_STOCH_RSI_SMOOTH_D))
    macd_line, macd_sig, macd_hist = _macd(
        c,
        kw.get("macd_fast", DEFAULT_MACD_FAST),
        kw.get("macd_slow", DEFAULT_MACD_SLOW),
        kw.get("macd_signal", DEFAULT_MACD_SIGNAL))
    kk, kd, kj = _kdj(
        h, l, c,
        kw.get("kdj_period", DEFAULT_KDJ_PERIOD),
        kw.get("kdj_k_smooth", DEFAULT_KDJ_K_SMOOTH),
        kw.get("kdj_d_smooth", DEFAULT_KDJ_D_SMOOTH))

    return pd.DataFrame({
        "stoch_k": sk, "stoch_d": sd,
        "macd_line": macd_line, "macd_signal": macd_sig, "macd_hist": macd_hist,
        "kdj_k": kk, "kdj_d": kd, "kdj_j": kj,
    }, index=kline.index)


# ── 单条规则的布尔判定 ───────────────────────────────────

def _eval_rule(name: str, ind: pd.DataFrame, i: int,
               oversold: float, overbought: float,
               kdj_oversold: float, kdj_overbought: float) -> bool:
    """评估第 i 条 K 线上指定规则是否成立."""
    if i < 1:
        return False
    sk = ind["stoch_k"].iloc
    sd = ind["stoch_d"].iloc
    ml = ind["macd_line"].iloc
    msg = ind["macd_signal"].iloc
    mh = ind["macd_hist"].iloc
    kk = ind["kdj_k"].iloc
    kd = ind["kdj_d"].iloc
    kj = ind["kdj_j"].iloc

    def _safe(s, idx):
        v = s[idx]
        return None if pd.isna(v) else v

    # ── BUY ──
    if name == "stoch_rsi_cross_up_oversold":
        a, b = _safe(sk, i - 1), _safe(sk, i)
        return a is not None and b is not None and a <= oversold < b
    if name == "stoch_rsi_in_oversold":
        b = _safe(sk, i)
        return b is not None and b < oversold
    if name == "macd_bullish_cross":
        a1, a2 = _safe(ml, i - 1), _safe(msg, i - 1)
        b1, b2 = _safe(ml, i), _safe(msg, i)
        return None not in (a1, a2, b1, b2) and a1 <= a2 and b1 > b2
    if name == "macd_hist_turn_up":
        a, b = _safe(mh, i - 1), _safe(mh, i)
        if a is None or b is None or i < 2:
            return False
        prev = _safe(mh, i - 2)
        return prev is not None and a < prev and b > a
    if name == "kdj_j_cross_up_oversold":
        a, b = _safe(kj, i - 1), _safe(kj, i)
        return a is not None and b is not None and a <= kdj_oversold < b
    if name == "kdj_k_cross_d_up":
        a1, a2 = _safe(kk, i - 1), _safe(kd, i - 1)
        b1, b2 = _safe(kk, i), _safe(kd, i)
        return None not in (a1, a2, b1, b2) and a1 <= a2 and b1 > b2

    # ── EXIT ──
    if name == "stoch_rsi_cross_down_overbought":
        a, b = _safe(sk, i - 1), _safe(sk, i)
        return a is not None and b is not None and a >= overbought > b
    if name == "stoch_rsi_in_overbought":
        b = _safe(sk, i)
        return b is not None and b > overbought
    if name == "macd_bearish_cross":
        a1, a2 = _safe(ml, i - 1), _safe(msg, i - 1)
        b1, b2 = _safe(ml, i), _safe(msg, i)
        return None not in (a1, a2, b1, b2) and a1 >= a2 and b1 < b2
    if name == "macd_hist_turn_down":
        a, b = _safe(mh, i - 1), _safe(mh, i)
        if a is None or b is None or i < 2:
            return False
        prev = _safe(mh, i - 2)
        return prev is not None and a > prev and b < a
    if name == "kdj_j_cross_down_overbought":
        a, b = _safe(kj, i - 1), _safe(kj, i)
        return a is not None and b is not None and a >= kdj_overbought > b
    if name == "kdj_k_cross_d_down":
        a1, a2 = _safe(kk, i - 1), _safe(kd, i - 1)
        b1, b2 = _safe(kk, i), _safe(kd, i)
        return None not in (a1, a2, b1, b2) and a1 >= a2 and b1 < b2

    return False


def _in_session(ts: pd.Timestamp,
                session_utc: tuple[time, time] | None) -> bool:
    if session_utc is None:
        return True
    t = ts.time() if ts.tzinfo is None else ts.tz_convert("UTC").time()
    start, end = session_utc
    if start <= end:
        return start <= t <= end
    # 跨午夜
    return t >= start or t <= end


# ── 主入口: 逐条 K 线扫描触发 ────────────────────────────

@dataclass
class TriggerConfig:
    """触发检测配置 (全部参数化)."""
    timeframe_minutes: int = 60        # 1h
    side: str = "BUY"                   # "BUY" or "EXIT"
    rule_set: Sequence[str] = DEFAULT_BUY_RULES
    confirm_mode: str | int = 2         # v3.7.82: any → 2-of-4 (单笔效率 +60%, win 79→81%)
    oversold: float = DEFAULT_STOCH_OVERSOLD
    overbought: float = DEFAULT_STOCH_OVERBOUGHT
    kdj_oversold: float = DEFAULT_KDJ_J_OVERSOLD
    kdj_overbought: float = DEFAULT_KDJ_J_OVERBOUGHT
    session_utc: tuple[time, time] | None = None  # None = 全天


def detect_triggers(kline: pd.DataFrame,
                    daily_thresholds: pd.DataFrame,
                    config: TriggerConfig,
                    asset: str = "GLD",
                    daily_low: pd.Series = None,
                    daily_high: pd.Series = None) -> pd.DataFrame:
    """扫描 kline, 在 (价格突破阈值) + (规则确认) + (时段过滤) 时记录触发.

    Args:
        kline: 盘中 OHLC, index=DatetimeIndex
        daily_thresholds: index=date, 必有列 'bp030_price', 'bp090_price'
        config: TriggerConfig
        asset: "GLD" / "SLV" / "GC" 等, 仅用作 log 标识

    Returns:
        DataFrame: 列 [date, trigger_time, price, side, asset, timeframe,
                       rules, bp_threshold, n_confirms]
    """
    if kline is None or len(kline) == 0:
        return pd.DataFrame()

    ind = compute_indicators(kline)
    rule_set = list(config.rule_set)

    out = []
    dates = pd.DatetimeIndex(kline.index).normalize()
    for i, ts in enumerate(kline.index):
        d = dates[i]
        # 时段过滤
        if not _in_session(ts, config.session_utc):
            continue
        if d not in daily_thresholds.index:
            continue
        row = daily_thresholds.loc[d]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        bp030 = row.get("bp030_price", np.nan)
        bp090 = row.get("bp090_price", np.nan)

        c = kline["Close"].iloc[i]
        lo = kline["Low"].iloc[i] if "Low" in kline.columns else c
        hi = kline["High"].iloc[i] if "High" in kline.columns else c
        if pd.isna(c):
            continue
        # v3.7.61 yfinance prepost 脏数据过滤 (双层):
        # (a) 1h Low < daily Low → 视为脏 (1h 不应突破 daily); 用 daily Low 兜底
        # (b) 1h Low 比 min(Open, Close) 低 > 3% → 异常 wick
        op = kline["Open"].iloc[i] if "Open" in kline.columns else c
        # (a) daily sanity bound
        if daily_low is not None and not pd.isna(lo):
            d_lo = daily_low.get(d, None)
            if d_lo is not None and not pd.isna(d_lo) and lo < d_lo * 0.99:
                lo = float(d_lo)
        if daily_high is not None and not pd.isna(hi):
            d_hi = daily_high.get(d, None)
            if d_hi is not None and not pd.isna(d_hi) and hi > d_hi * 1.01:
                hi = float(d_hi)
        # (b) intra-bar wick
        if not pd.isna(lo) and not pd.isna(c) and not pd.isna(op):
            ref_min = min(op, c)
            if lo < ref_min * 0.97:
                lo = ref_min
        if not pd.isna(hi) and not pd.isna(c) and not pd.isna(op):
            ref_max = max(op, c)
            if hi > ref_max * 1.03:
                hi = ref_max

        # 价格突破阈值 — v3.7.61 改用 Low (盘中最低), 抓深谷
        if config.side == "BUY":
            if pd.isna(bp030) or pd.isna(bp090) or lo >= bp030:
                continue
            threshold = bp030
            # v3.7.61 深谷直接触发 — 价格深破 bp020 时跳过技术确认
            # bp020 = lower + 0.20 × band_range (从 bp030/bp090 反推)
            band_range = (bp090 - bp030) / 0.60
            bp020 = bp030 - 0.10 * band_range  # 比 bp030 低 10% 范围
            deep_trigger = lo < bp020
        else:
            if pd.isna(bp090) or pd.isna(bp030) or hi <= bp090:
                continue
            threshold = bp090
            band_range = (bp090 - bp030) / 0.60
            bp095 = bp090 + 0.05 * band_range
            deep_trigger = hi > bp095

        # 评估规则
        passed = [r for r in rule_set
                  if _eval_rule(r, ind, i,
                                config.oversold, config.overbought,
                                config.kdj_oversold, config.kdj_overbought)]
        n_pass = len(passed)
        n_total = len(rule_set)

        if config.confirm_mode == "any":
            ok = n_pass >= 1
        elif config.confirm_mode == "all":
            ok = n_pass == n_total
        elif isinstance(config.confirm_mode, int):
            ok = n_pass >= config.confirm_mode
        else:
            ok = n_pass >= 1

        # v3.7.61 深谷自动触发 (覆盖技术确认要求)
        if deep_trigger:
            ok = True
            if not passed:
                passed = ["deep_zone_auto"]

        if not ok:
            continue
        # 触发价 = 低点 (BUY) / 高点 (SELL), 模拟限价单成交价
        trigger_price = lo if config.side == "BUY" else hi

        out.append({
            "date": d,
            "trigger_time": ts,
            "price": float(trigger_price),  # v3.7.61: Low/High 而非 Close (限价单价)
            "side": config.side,
            "asset": asset,
            "timeframe": f"{config.timeframe_minutes}m",
            "rules": ",".join(passed),
            "bp_threshold": float(threshold),
            "n_confirms": n_pass,
        })

    return pd.DataFrame(out)


# ── 按日聚合到代表价 ──────────────────────────────────────

def _agg_of_day(triggers: pd.DataFrame, side: str, mode: str) -> pd.DataFrame:
    """通用按日聚合.

    mode: "worst" (买取 max/卖取 min) / "best" (买取 min/卖取 max) /
          "first" (按时间最早) / "mean" (均值)
    """
    if triggers is None or len(triggers) == 0:
        return pd.DataFrame()
    sub = triggers[triggers["side"] == side].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    if mode == "worst":
        idx = (sub.groupby("date")["price"].idxmax() if side == "BUY"
               else sub.groupby("date")["price"].idxmin())
        rep = sub.loc[idx].set_index("date")
    elif mode == "best":
        idx = (sub.groupby("date")["price"].idxmin() if side == "BUY"
               else sub.groupby("date")["price"].idxmax())
        rep = sub.loc[idx].set_index("date")
    elif mode == "first":
        idx = sub.groupby("date")["trigger_time"].idxmin()
        rep = sub.loc[idx].set_index("date")
    elif mode == "mean":
        rep = sub.groupby("date").agg(
            price=("price", "mean"),
            trigger_time=("trigger_time", "min"),
            bp_threshold=("bp_threshold", "first"),
            rules=("rules", "first"),
            asset=("asset", "first"),
            timeframe=("timeframe", "first"),
            side=("side", "first"),
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    counts = sub.groupby("date").size().rename("n_triggers")
    first_t = sub.groupby("date")["trigger_time"].min().rename("first_time")
    rules_concat = sub.groupby("date")["rules"].apply(
        lambda s: ",".join(sorted(set(",".join(s).split(","))))
    ).rename("rules_all")
    return rep.join([counts, first_t, rules_concat])


def worst_of_day(triggers: pd.DataFrame, side: str = "BUY") -> pd.DataFrame:
    """每日多触发取最差价 (买:最高 / 卖:最低)."""
    return _agg_of_day(triggers, side, "worst")


def dedupe_intraday(triggers: pd.DataFrame, side: str = "BUY",
                       min_drop_pct: float = 1.5) -> pd.DataFrame:
    """日内连续触发去重 (v3.7.67) — 类似日线 dedupe 加仓机制.

    v3.7.81: 默认 0.5 → 1.5 (5m × 60d 网格回测 EU/单笔效率最优).

    规则:
      - 第一笔触发保留
      - 后续触发: BUY 须比上次保留的 entry 价低 ≥ min_drop_pct%
                  SELL 须比上次保留的 entry 价高 ≥ min_drop_pct%
      - 中间反弹超 1% 后再回撤的, 视为新 wave (重置 prev)

    用于:
      - 实盘加仓 (避免每个 1h bar 都加仓)
      - chart 显示 (只标显著入场点)

    Returns: 保留行的 DataFrame (子集)
    """
    if triggers is None or len(triggers) == 0:
        return triggers
    sub = triggers[triggers["side"] == side].copy()
    if not len(sub):
        return sub
    sub = sub.sort_values("trigger_time").reset_index(drop=True)
    keep = []
    prev_kept_price = None
    last_seen_price = None  # 跟踪是否中间反弹了
    for _, r in sub.iterrows():
        p = float(r["price"])
        if prev_kept_price is None:
            keep.append(r); prev_kept_price = p; last_seen_price = p
            continue
        # 中间反弹检测: 比上次 kept 价反向超 1% 后再回到 buy/sell zone
        # → 视为 wave 重置 (类似日线 gap-reset)
        if side == "BUY":
            if last_seen_price is not None and last_seen_price > prev_kept_price * 1.01:
                # 反弹后再触发 → 新 wave
                keep.append(r); prev_kept_price = p; last_seen_price = p
                continue
            # 否则: 必须比上次 kept 跌 > min_drop_pct%
            drop = (prev_kept_price - p) / prev_kept_price * 100
            if drop >= min_drop_pct:
                keep.append(r); prev_kept_price = p
            last_seen_price = p
        else:  # SELL/EXIT
            if last_seen_price is not None and last_seen_price < prev_kept_price * 0.99:
                keep.append(r); prev_kept_price = p; last_seen_price = p
                continue
            rise = (p - prev_kept_price) / prev_kept_price * 100
            if rise >= min_drop_pct:
                keep.append(r); prev_kept_price = p
            last_seen_price = p
    if not keep:
        return sub.iloc[0:0]
    return pd.DataFrame(keep).reset_index(drop=True)


def best_of_day(triggers: pd.DataFrame, side: str = "BUY") -> pd.DataFrame:
    """每日多触发取最优价 (买:最低 / 卖:最高)."""
    return _agg_of_day(triggers, side, "best")


def average_of_day(triggers: pd.DataFrame, side: str = "BUY",
                     dedup_first: bool = True,
                     min_drop_pct: float = 0.5) -> pd.DataFrame:
    """每日多触发取平均价 (= 实际分批加仓的持仓均价).

    v3.7.68: 适合 dedupe_intraday 后的"显著加仓点"取平均 — 比 worst 更贴近实战.

    Args:
        dedup_first: 先 dedupe_intraday 再 average (避免噪音触发拖偏均值)
        min_drop_pct: dedupe 阈值, 仅在 dedup_first=True 时生效

    Returns: 每天一行 DataFrame, price = 当日 dedupe 后触发均价
    """
    if triggers is None or len(triggers) == 0:
        return triggers
    sub = triggers[triggers["side"] == side].copy()
    if not len(sub):
        return sub
    if dedup_first:
        # 按 date 分组分别 dedupe (dedupe_intraday 是日内逻辑)
        out_rows = []
        for d, grp in sub.groupby("date"):
            dd = dedupe_intraday(grp, side=side, min_drop_pct=min_drop_pct)
            if len(dd) > 0:
                avg_p = dd["price"].mean()
                first = dd.iloc[0].copy()
                first["price"] = float(avg_p)
                first["n_intraday"] = len(dd)
                out_rows.append(first)
        if not out_rows:
            return sub.iloc[0:0]
        return pd.DataFrame(out_rows).reset_index(drop=True)
    # 不 dedupe — 直接所有 raw 取平均
    grouped = sub.groupby("date").agg({
        "price": "mean",
        "trigger_time": "first",
        "side": "first",
        "asset": "first",
        "timeframe": "first",
        "rules": "first",
        "bp_threshold": "first",
        "n_confirms": "max",
    }).reset_index()
    return grouped


def first_of_day(triggers: pd.DataFrame, side: str = "BUY") -> pd.DataFrame:
    """每日多触发取最早一次 (按时间)."""
    return _agg_of_day(triggers, side, "first")


# ── 持久化 ────────────────────────────────────────────────

def load_log(log_path: str) -> pd.DataFrame:
    """加载累计 log; 不存在返回空 DataFrame."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=[
            "date", "trigger_time", "price", "side", "asset",
            "timeframe", "rules", "bp_threshold", "n_confirms"])
    df = pd.read_parquet(log_path)
    if "trigger_time" in df.columns:
        df["trigger_time"] = pd.to_datetime(df["trigger_time"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def save_log(df: pd.DataFrame, log_path: str):
    """保存累计 log (覆盖写)."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    df.to_parquet(log_path, index=False)


def upsert_log(new_triggers: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """合并新触发到 log: 按 (date, trigger_time, side, asset, timeframe, rules)
    去重, 后写入覆盖前."""
    existing = load_log(log_path)
    if len(new_triggers) == 0:
        return existing
    combined = pd.concat([existing, new_triggers], ignore_index=True)
    key_cols = ["date", "trigger_time", "side", "asset", "timeframe", "rules"]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(["date", "trigger_time"])
    save_log(combined, log_path)
    return combined


# ── 历史回填 (一次性扫所有 1h 数据) ──────────────────────

def backfill(kline: pd.DataFrame,
             daily_thresholds: pd.DataFrame,
             asset: str,
             timeframe_minutes: int = 60,
             buy_rules: Sequence[str] = DEFAULT_BUY_RULES,
             exit_rules: Sequence[str] = DEFAULT_EXIT_RULES,
             confirm_mode: str | int = 2,  # v3.7.82
             session_utc: tuple[time, time] | None = None,
             log_path: str | None = None) -> pd.DataFrame:
    """全量回填: 对 kline 同时跑 BUY 和 EXIT 配置, 写入 log."""
    cfg_buy = TriggerConfig(
        timeframe_minutes=timeframe_minutes, side="BUY",
        rule_set=buy_rules, confirm_mode=confirm_mode,
        session_utc=session_utc)
    cfg_exit = TriggerConfig(
        timeframe_minutes=timeframe_minutes, side="EXIT",
        rule_set=exit_rules, confirm_mode=confirm_mode,
        session_utc=session_utc)

    buys = detect_triggers(kline, daily_thresholds, cfg_buy, asset=asset)
    exits = detect_triggers(kline, daily_thresholds, cfg_exit, asset=asset)
    out = pd.concat([buys, exits], ignore_index=True)

    if log_path is not None and len(out) > 0:
        upsert_log(out, log_path)

    return out
