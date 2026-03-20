"""事件日历 — FOMC/OPEX/期货交割日/非农等关键日期.

用于:
  1. 图表标注
  2. 波动率策略信号 (Straddle)
  3. 临近事件日收紧买入阈值
"""

import pandas as pd
from datetime import date, timedelta

# ══════════════════════════════════════════════════════════
# 2026 FOMC 日期 (每次会议2天, 第2天发声明)
# 来源: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# ══════════════════════════════════════════════════════════
FOMC_2026 = [
    ("2026-01-28", "2026-01-29"),
    ("2026-03-17", "2026-03-18"),
    ("2026-05-05", "2026-05-06"),
    ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"),
    ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"),
    ("2026-12-15", "2026-12-16"),
]

# 2025 补充 (用于回测)
FOMC_2025 = [
    ("2025-01-28", "2025-01-29"),
    ("2025-03-18", "2025-03-19"),
    ("2025-05-06", "2025-05-07"),
    ("2025-06-17", "2025-06-18"),
    ("2025-07-29", "2025-07-30"),
    ("2025-09-16", "2025-09-17"),
    ("2025-10-28", "2025-10-29"),
    ("2025-12-16", "2025-12-17"),
]

# GLD 月度 OPEX (每月第三个周五)
def get_opex_dates(year):
    """计算指定年份的月度 OPEX 日期 (每月第三个周五)."""
    dates = []
    for month in range(1, 13):
        d = date(year, month, 1)
        # 找第一个周五
        while d.weekday() != 4:  # Friday
            d += timedelta(days=1)
        # 第三个周五
        opex = d + timedelta(weeks=2)
        dates.append(opex)
    return dates

OPEX_2025 = get_opex_dates(2025)
OPEX_2026 = get_opex_dates(2026)

# 非农就业 (每月第一个周五)
def get_nfp_dates(year):
    dates = []
    for month in range(1, 13):
        d = date(year, month, 1)
        while d.weekday() != 4:
            d += timedelta(days=1)
        dates.append(d)
    return dates

NFP_2025 = get_nfp_dates(2025)
NFP_2026 = get_nfp_dates(2026)


def get_all_events(start_date=None, end_date=None):
    """获取指定范围内的所有事件.

    Returns: list of (date, event_type, label)
    """
    events = []

    for y1, y2 in FOMC_2025 + FOMC_2026:
        events.append((pd.Timestamp(y2), "FOMC", "FOMC"))

    for d in OPEX_2025 + OPEX_2026:
        events.append((pd.Timestamp(d), "OPEX", "OPEX"))

    for d in NFP_2025 + NFP_2026:
        events.append((pd.Timestamp(d), "NFP", "NFP"))

    if start_date:
        events = [(d, t, l) for d, t, l in events if d >= pd.Timestamp(start_date)]
    if end_date:
        events = [(d, t, l) for d, t, l in events if d <= pd.Timestamp(end_date)]

    return sorted(events, key=lambda x: x[0])


def days_to_next_event(current_date, event_type=None):
    """距下一个事件的天数.

    Args:
        event_type: None=任意, "FOMC", "OPEX", "NFP"

    Returns: (days, event_type, event_date) or (999, None, None)
    """
    events = get_all_events(start_date=current_date)
    if event_type:
        events = [(d, t, l) for d, t, l in events if t == event_type]

    cd = pd.Timestamp(current_date)
    for d, t, l in events:
        diff = (d - cd).days
        if diff >= 0:
            return diff, t, d

    return 999, None, None


def compute_event_features(dates_index):
    """为每个交易日计算事件特征.

    Returns: DataFrame with columns:
        days_to_fomc, days_to_opex, days_to_nfp, days_to_any_event,
        is_fomc_week, is_opex_week
    """
    records = []
    for d in dates_index:
        d_fomc, _, _ = days_to_next_event(d, "FOMC")
        d_opex, _, _ = days_to_next_event(d, "OPEX")
        d_nfp, _, _ = days_to_next_event(d, "NFP")
        d_any = min(d_fomc, d_opex, d_nfp)

        records.append({
            "days_to_fomc": d_fomc,
            "days_to_opex": d_opex,
            "days_to_nfp": d_nfp,
            "days_to_any_event": d_any,
            "is_fomc_week": 1 if d_fomc <= 5 else 0,
            "is_opex_week": 1 if d_opex <= 5 else 0,
        })

    return pd.DataFrame(records, index=dates_index)


# ══════════════════════════════════════════════════════════
# Straddle 信号
# ══════════════════════════════════════════════════════════

# Straddle 参数
STRADDLE_RV_THRESHOLD = 20.0   # RV 低于此值视为波动率压缩
STRADDLE_RV_ABS_MAX = 25.0     # RV 绝对值上限 (高于此值成本太高)
STRADDLE_EVENT_DAYS = 3        # 距事件日 <= N 天
STRADDLE_RV_DROP_PCT = 30.0    # RV 相对近期下降 > N%
# 事件权重: FOMC 最重要, NFP 次之, OPEX 最低
EVENT_WEIGHT = {"FOMC": 3, "NFP": 2, "OPEX": 1}


STRADDLE_HOLD_DAYS = 5     # 持仓天数
STRADDLE_WIN_MOVE = 0      # 波动 > 成本即为盈利 (0 = 自动用成本)


def detect_straddle_signal(rv_series, dates_index,
                            rv_threshold=STRADDLE_RV_THRESHOLD,
                            rv_abs_max=STRADDLE_RV_ABS_MAX,
                            event_days=STRADDLE_EVENT_DAYS,
                            rv_drop_pct=STRADDLE_RV_DROP_PCT):
    """检测 Straddle (做多波动率) 信号.

    条件 (评分制, score >= 3 触发):
      - RV < rv_threshold (波动率压缩):     +2
      - RV 相对20天均值下降 > rv_drop_pct%: +1
      - 距 FOMC <= event_days 天:           +3
      - 距 NFP <= event_days 天:            +2
      - 距 OPEX <= event_days 天:           +1

    额外硬门槛:
      - RV 绝对值 > rv_abs_max → 不触发 (成本太高)

    Returns: DataFrame with straddle_signal, straddle_reason, score
    """
    rv_ma20 = rv_series.rolling(20, min_periods=5).mean()

    records = []
    for d in dates_index:
        rv = rv_series.get(d, 50)
        rv_avg = rv_ma20.get(d, rv)
        rv_drop = (rv_avg - rv) / rv_avg * 100 if rv_avg > 0 else 0

        # 找最近的各类事件
        d_fomc, _, fomc_d = days_to_next_event(d, "FOMC")
        d_nfp, _, nfp_d = days_to_next_event(d, "NFP")
        d_opex, _, opex_d = days_to_next_event(d, "OPEX")
        d_any = min(d_fomc, d_nfp, d_opex)

        # 最近事件信息
        if d_fomc <= d_nfp and d_fomc <= d_opex:
            ev_type, ev_date = "FOMC", fomc_d
        elif d_nfp <= d_opex:
            ev_type, ev_date = "NFP", nfp_d
        else:
            ev_type, ev_date = "OPEX", opex_d

        score = 0
        reasons = []

        # RV 压缩
        if rv < rv_threshold:
            score += 2
            reasons.append(f"RV={rv:.1f}%<{rv_threshold}%")

        # RV 下降
        if rv_drop > rv_drop_pct:
            score += 1
            reasons.append(f"RV降{rv_drop:.0f}%")

        # 事件接近 (按权重)
        if d_fomc <= event_days:
            score += EVENT_WEIGHT["FOMC"]
            reasons.append(f"距FOMC {d_fomc}天")
        if d_nfp <= event_days:
            score += EVENT_WEIGHT["NFP"]
            reasons.append(f"距NFP {d_nfp}天")
        if d_opex <= event_days:
            score += EVENT_WEIGHT["OPEX"]
            reasons.append(f"距OPEX {d_opex}天")

        # 硬门槛: RV 太高 → 成本太高, 不做
        if rv > rv_abs_max:
            signal = False
            if reasons:
                reasons.append(f"但RV={rv:.0f}%>阈值{rv_abs_max}%,成本过高")
        else:
            signal = score >= 3

        records.append({
            "straddle_signal": signal,
            "straddle_reason": " + ".join(reasons) if signal else "",
            "straddle_score": score,
            "rv": rv,
            "days_to_event": d_any,
            "next_event": f"{ev_type} {ev_date.strftime('%m/%d')}" if ev_date else "",
        })

    return pd.DataFrame(records, index=dates_index)


def backtest_straddle(close, high, low, rv_series, dates_index,
                       hold_days=STRADDLE_HOLD_DAYS, **kwargs):
    """Straddle 信号回测.

    成本估算: price × RV/100 × sqrt(hold_days/252)
    盈利: max(上涨幅度, 下跌幅度) - 成本
    胜率: 波动 > 成本 即为胜

    Returns: list of trade dicts
    """
    import numpy as np

    straddle = detect_straddle_signal(rv_series, dates_index, **kwargs)
    signals = straddle[straddle["straddle_signal"]]

    # 去重: 连续信号只取第一天
    entries = []
    prev = None
    for d in signals.index:
        if prev is None or (d - prev).days > 3:
            entries.append(d)
        prev = d

    sqrt_h252 = np.sqrt(hold_days / 252)
    trades = []

    for d in entries:
        c = close.get(d, 0)
        if c == 0:
            continue
        rv = rv_series.get(d, 20)
        reason = straddle.loc[d, "straddle_reason"]
        next_event = straddle.loc[d, "next_event"]

        cost_pct = rv / 100 * sqrt_h252 * 100

        loc = close.index.get_loc(d)
        if loc + hold_days >= len(close):
            end_loc = len(close) - 1
            partial = True
        else:
            end_loc = loc + hold_days
            partial = False

        window_high = high.iloc[loc + 1:end_loc + 1].max()
        window_low = low.iloc[loc + 1:end_loc + 1].min()
        exit_date = close.index[end_loc]

        move_up = (window_high / c - 1) * 100
        move_down = (1 - window_low / c) * 100
        max_move = max(move_up, move_down)

        pnl_pct = max_move - cost_pct
        direction = "上涨" if move_up > move_down else "下跌"

        trades.append({
            "entry_date": d, "exit_date": exit_date,
            "entry_price": c, "rv": rv,
            "cost_pct": cost_pct,
            "max_move": max_move, "direction": direction,
            "pnl_pct": pnl_pct,
            "reason": reason, "next_event": next_event,
            "partial": partial,
        })

    return trades
