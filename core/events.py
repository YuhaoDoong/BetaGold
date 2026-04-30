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
    ("2026-01-27", "2026-01-28"),  # v3.7.17 修正 (原 1月28-29 错)
    ("2026-03-17", "2026-03-18"),
    ("2026-04-28", "2026-04-29"),  # v3.7.17 修正 (原 5月5-6 错)
    ("2026-06-16", "2026-06-17"),
    ("2026-07-28", "2026-07-29"),
    ("2026-09-15", "2026-09-16"),
    ("2026-10-27", "2026-10-28"),
    ("2026-12-08", "2026-12-09"),  # v3.7.17 修正 (原 12月15-16 错)
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


# COMEX 期货交割日 (交割月倒数第三个交易日, 简化为25号附近)
def get_futures_expiry(year, metal="gold"):
    """COMEX 期货交割月的最后交易日 (近似: 交割月25日前的周五)."""
    if metal == "gold":
        months = [2, 4, 6, 8, 10, 12]
    else:  # silver
        months = [1, 3, 5, 7, 9, 12]
    dates = []
    for m in months:
        # 近似: 交割月25号
        d = date(year, m, 25)
        # 退到最近的工作日
        while d.weekday() > 4:
            d -= timedelta(days=1)
        dates.append(d)
    return dates

FUTURES_GOLD_2025 = get_futures_expiry(2025, "gold")
FUTURES_GOLD_2026 = get_futures_expiry(2026, "gold")
FUTURES_SILVER_2025 = get_futures_expiry(2025, "silver")
FUTURES_SILVER_2026 = get_futures_expiry(2026, "silver")


def get_all_events(start_date=None, end_date=None, asset="gold"):
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

    # 期货交割日
    if asset == "gold":
        for d in FUTURES_GOLD_2025 + FUTURES_GOLD_2026:
            events.append((pd.Timestamp(d), "FUT_EXP", "GC交割"))
    else:
        for d in FUTURES_SILVER_2025 + FUTURES_SILVER_2026:
            events.append((pd.Timestamp(d), "FUT_EXP", "SI交割"))

    if start_date:
        events = [(d, t, l) for d, t, l in events if d >= pd.Timestamp(start_date)]
    if end_date:
        events = [(d, t, l) for d, t, l in events if d <= pd.Timestamp(end_date)]

    return sorted(events, key=lambda x: x[0])


def days_to_next_event(current_date, event_type=None, asset="gold"):
    """距下一个事件的天数.

    Args:
        event_type: None=任意, "FOMC", "OPEX", "NFP", "FUT_EXP"
        asset: "gold" or "silver"

    Returns: (days, event_type, event_date) or (999, None, None)
    """
    events = get_all_events(start_date=current_date, asset=asset)
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


STRADDLE_RV_PCTILE_MAX = 0.50  # v3.7.32: RV %tile > 此值不触发 (实证)

def detect_straddle_signal(rv_series, dates_index,
                            rv_threshold=STRADDLE_RV_THRESHOLD,
                            rv_abs_max=STRADDLE_RV_ABS_MAX,
                            event_days=STRADDLE_EVENT_DAYS,
                            rv_drop_pct=STRADDLE_RV_DROP_PCT,
                            rv_pctile=None,
                            rv_pctile_max=STRADDLE_RV_PCTILE_MAX,
                            asset=None):
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
    if asset is not None:
        try:
            from core.strategy_config import get_config
            _ac = get_config(asset)
            if hasattr(_ac, "straddle_rv_pctile_max"):
                rv_pctile_max = _ac.straddle_rv_pctile_max
        except Exception:
            pass

    rv_ma20 = rv_series.rolling(20, min_periods=5).mean()

    records = []
    for d in dates_index:
        rv = rv_series.get(d, 50)
        rv_avg = rv_ma20.get(d, rv)
        rv_drop = (rv_avg - rv) / rv_avg * 100 if rv_avg > 0 else 0
        # v3.7.32: RV %tile 上限过滤 (高 IV 时 STRADDLE 没 alpha)
        rv_pct_d = (rv_pctile.get(d, 0.5) if rv_pctile is not None else None)

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
        # v3.7.32: RV %tile 太高 → IV 贵, 没 alpha (实证 5y Sharpe 0.55)
        elif rv_pct_d is not None and rv_pct_d > rv_pctile_max:
            signal = False
            if reasons:
                reasons.append(f"但RV%tile={rv_pct_d:.2f}>{rv_pctile_max},IV过贵")
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


# IV crush 实证值 (基于 GVZ 真实数据, 2025-2026 FOMC)
# 详见 core/iv_crush.py 模块.
from core.iv_crush import IV_CRUSH_CONSERVATIVE
IV_CRUSH_FOMC = IV_CRUSH_CONSERVATIVE["FOMC"]   # 0.05
IV_CRUSH_NFP = IV_CRUSH_CONSERVATIVE["NFP"]     # 0.05
IV_CRUSH_OPEX = IV_CRUSH_CONSERVATIVE["OPEX"]   # 0.05


def backtest_straddle(close, high, low, rv_series, dates_index,
                       hold_days=STRADDLE_HOLD_DAYS,
                       iv_crush_adj=False, **kwargs):
    # iv_crush_adj 默认 False (v3.7.14):
    # GLD GVZ 实证 FOMC mean -0.1%, 与 SPX/VIX 30-60% 完全不同.
    # IV crush 修正对 GLD 几乎是白噪声, 故默认关闭.
    # Dashboard 仍显示 IV/RV ratio 作为风险参考, 但不调 P&L.
    # 若用 SPY/QQQ 等强 crush 资产, 可手动 iv_crush_adj=True 启用.
    """Straddle 信号回测 (含 IV crush 修正).

    P&L 模型:
      gross_cost = RV × √(hold/252) × 100  (1σ ATM premium 估算)
      iv_crush_loss = gross_cost × Σ(crush_factor for each event in hold window)
        - FOMC 跨过: 30% 扣除
        - NFP  跨过: 15% 扣除
        - OPEX 跨过: 10% 扣除
      pnl = max_move - gross_cost - iv_crush_loss
      win = max_move > gross_cost + iv_crush_loss

    iv_crush_adj=False 时退回旧模型 (无 IV crush 扣除, 仅供对比).

    Returns: list of trade dicts (含 iv_crush_loss / spans_events 字段)
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

        # 检测持仓窗口 (hold_days 内) 是否跨事件 → IV crush 损失
        d_fomc, _, _ = days_to_next_event(d, "FOMC")
        d_nfp, _, _ = days_to_next_event(d, "NFP")
        d_opex, _, _ = days_to_next_event(d, "OPEX")
        spans_fomc = d_fomc <= hold_days
        spans_nfp = d_nfp <= hold_days
        spans_opex = d_opex <= hold_days

        iv_crush_loss = 0.0
        events_hit = []
        if iv_crush_adj:
            if spans_fomc:
                iv_crush_loss += cost_pct * IV_CRUSH_FOMC
                events_hit.append(f"FOMC(-{IV_CRUSH_FOMC:.0%})")
            if spans_nfp:
                iv_crush_loss += cost_pct * IV_CRUSH_NFP
                events_hit.append(f"NFP(-{IV_CRUSH_NFP:.0%})")
            if spans_opex:
                iv_crush_loss += cost_pct * IV_CRUSH_OPEX
                events_hit.append(f"OPEX(-{IV_CRUSH_OPEX:.0%})")

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

        # 实际 P&L = 价格移动 - 入场成本 - IV crush 损失
        pnl_pct = max_move - cost_pct - iv_crush_loss
        win = max_move > cost_pct + iv_crush_loss
        direction = "上涨" if move_up > move_down else "下跌"

        trades.append({
            "entry_date": d, "exit_date": exit_date,
            "entry_price": c, "rv": rv,
            "cost_pct": cost_pct,
            "iv_crush_loss": iv_crush_loss,
            "events_hit": ",".join(events_hit) if events_hit else "—",
            "max_move": max_move, "direction": direction,
            "pnl_pct": pnl_pct,
            "win": win,
            "reason": reason, "next_event": next_event,
            "partial": partial,
        })

    return trades


# ── Short Vol (做空波动率: Iron Condor 16Δ/5Δ) ──
# 严格化思路 (目标胜率 > 80%):
#   1) 改用 Iron Condor (1.6σ 短 / 3σ 长翼), 不再用 Short Strangle
#   2) RV %tile 收缩到中位窄带 [35%, 65%]
#   3) 必须连续 RV 回落 (3日均值 < 10日均值, 趋势性回落)
#   4) Bull / Range regime 才允许 (Mixed 也不接, Bear 屏蔽)
#   5) 全事件硬门槛: 持仓窗口 (5天) 内不能有 FOMC/NFP/OPEX/FUT_EXP
#   6) Score 门槛提至 7
#   7) 价格平静过滤: 近 5 日日均振幅 < 1.5%
SHORT_VOL_RV_PCTILE_LO = 0.45    # v3.7.29 网格搜索最优 (Sharpe +9%, 总 +14%)
SHORT_VOL_RV_PCTILE_HI = 0.80    # v3.7.29 (从 0.65 拓宽到 0.80)
SHORT_VOL_RV_ABS_MIN = 13.0
SHORT_VOL_RV_ABS_MAX = 28.0      # RV 绝对上限 (从 32 收到 28)
SHORT_VOL_FOMC_BUFFER = 10       # 距 FOMC > 10 天
SHORT_VOL_NFP_BUFFER = 7
SHORT_VOL_OPEX_BUFFER = 5
SHORT_VOL_RV_TREND_DAYS = 3      # 3日均值 < 10日均值 → vol 趋势回落
SHORT_VOL_DAILY_RANGE_MAX = 1.5  # 近5日日均振幅上限 %
SHORT_VOL_SCORE_TRIGGER = 7      # 触发分数门槛 (从 6 提至 7)
SHORT_VOL_STRIKE_SIGMA = 1.6     # IC 短腿 ≈ 16Δ ≈ 1.6σ
SHORT_VOL_WING_SIGMA = 3.0       # IC 长腿 (限制最大亏损)
SHORT_VOL_PREMIUM_RATIO = 0.40   # 16Δ 短 - 5Δ 长 净 credit ≈ 1σ premium 的 40%


def detect_short_vol_signal(rv_series, rv_pctile, dates_index,
                             rv_pctile_lo=SHORT_VOL_RV_PCTILE_LO,
                             rv_pctile_hi=SHORT_VOL_RV_PCTILE_HI,
                             rv_abs_min=SHORT_VOL_RV_ABS_MIN,
                             rv_abs_max=SHORT_VOL_RV_ABS_MAX,
                             fomc_buffer=SHORT_VOL_FOMC_BUFFER,
                             nfp_buffer=SHORT_VOL_NFP_BUFFER,
                             opex_buffer=SHORT_VOL_OPEX_BUFFER,
                             daily_range_max=SHORT_VOL_DAILY_RANGE_MAX,
                             score_trigger=SHORT_VOL_SCORE_TRIGGER,
                             regime=None,
                             daily_range=None,
                             asset=None):
    if asset is not None:
        try:
            from core.strategy_config import get_config
            _ac = get_config(asset)
            rv_pctile_lo = _ac.short_vol_rv_pctile_lo
            rv_pctile_hi = _ac.short_vol_rv_pctile_hi
            rv_abs_min = _ac.short_vol_rv_abs_min
            rv_abs_max = _ac.short_vol_rv_abs_max
            fomc_buffer = _ac.short_vol_fomc_buffer
            nfp_buffer = _ac.short_vol_nfp_buffer
            opex_buffer = _ac.short_vol_opex_buffer
            score_trigger = _ac.short_vol_score_trigger
        except Exception:
            pass
    """检测做空波动率信号 (Iron Condor 16Δ/5Δ, 严格时机).

    评分 (score >= score_trigger 触发):
      - RV %tile ∈ [lo, hi] 中位窄带:           +2
      - RV ∈ [abs_min, abs_max]:                +1
      - RV 3日均值 < 10日均值 (趋势回落):       +2
      - 距 FOMC > fomc_buffer+5 天:             +2
      - 距 NFP > nfp_buffer 天:                 +1
      - 距 OPEX > opex_buffer 天:               +1
      - regime ∈ {Bull, Range} (非 Mixed/Bear): +1
      - 近5日日均振幅 < daily_range_max%:       +1

    硬门槛 (任意命中 → 不触发):
      - RV %tile > 0.75 (高位, 反弹风险)
      - RV %tile < 0.25 (premium 太薄)
      - RV 越界
      - 距 FOMC ≤ fomc_buffer
      - 距 NFP ≤ nfp_buffer
      - regime == Bear (尾部风险)
      - 持仓窗口 (5 交易日) 内有任何主要事件
    """
    rv_ma3 = rv_series.rolling(3, min_periods=2).mean()
    rv_ma10 = rv_series.rolling(10, min_periods=3).mean()

    records = []
    for d in dates_index:
        rv = rv_series.get(d, 20)
        rv_pct = rv_pctile.get(d, 0.5) if rv_pctile is not None else 0.5
        rv_3d = rv_ma3.get(d, rv)
        rv_10d_ma = rv_ma10.get(d, rv)
        rv_falling = (rv_3d < rv_10d_ma)
        regime_d = (regime.get(d, "Range") if regime is not None
                    else "Range")

        d_fomc, _, _ = days_to_next_event(d, "FOMC")
        d_nfp, _, _ = days_to_next_event(d, "NFP")
        d_opex, _, _ = days_to_next_event(d, "OPEX")

        # 近5日日均振幅 (high-low)/close
        if daily_range is not None:
            dr_5d = daily_range.rolling(5, min_periods=3).mean().get(d, 99)
        else:
            dr_5d = 0  # 无数据则 +1 跳过

        score = 0
        reasons = []

        if rv_pctile_lo <= rv_pct <= rv_pctile_hi:
            score += 2
            reasons.append(f"RV%tile={rv_pct:.0%}∈[{rv_pctile_lo:.0%},{rv_pctile_hi:.0%}]")

        if rv_abs_min <= rv <= rv_abs_max:
            score += 1
            reasons.append(f"RV={rv:.0f}%适中")

        if rv_falling:
            score += 2
            reasons.append("RV趋势回落(3d<10d)")

        if d_fomc > fomc_buffer + 5:
            score += 2
            reasons.append(f"距FOMC {d_fomc}天")
        if d_nfp > nfp_buffer:
            score += 1
        if d_opex > opex_buffer:
            score += 1

        if regime_d in ("Bull", "Range"):
            score += 1
            reasons.append(f"{regime_d}稳定")

        if daily_range is not None and dr_5d < daily_range_max:
            score += 1
            reasons.append(f"近5日振幅{dr_5d:.2f}%<{daily_range_max}%")

        # 硬门槛
        block = None
        if rv_pct > 0.75:
            block = f"RV%tile={rv_pct:.0%}>75%(高位)"
        elif rv_pct < 0.25:
            block = f"RV%tile={rv_pct:.0%}<25%(premium太薄)"
        elif rv < rv_abs_min:
            block = f"RV={rv:.0f}%<{rv_abs_min}%"
        elif rv > rv_abs_max:
            block = f"RV={rv:.0f}%>{rv_abs_max}%"
        elif d_fomc <= fomc_buffer:
            block = f"距FOMC仅{d_fomc}天≤{fomc_buffer}"
        elif d_nfp <= nfp_buffer:
            block = f"距NFP仅{d_nfp}天≤{nfp_buffer}"
        elif regime_d == "Bear":
            block = "Bear regime"
        elif min(d_fomc, d_nfp, d_opex) <= 5:
            block = f"窗口内有事件(min={min(d_fomc,d_nfp,d_opex)}天)"

        if block:
            signal = False
            reasons = [block]
        else:
            signal = score >= score_trigger

        records.append({
            "short_vol_signal": signal,
            "short_vol_reason": " + ".join(reasons) if signal else (
                block if block else ""),
            "short_vol_score": score,
            "rv": rv,
            "rv_pctile": rv_pct,
        })

    return pd.DataFrame(records, index=dates_index)


def backtest_short_vol(close, high, low, rv_series, rv_pctile, dates_index,
                        hold_days=STRADDLE_HOLD_DAYS,
                        strike_sigma=SHORT_VOL_STRIKE_SIGMA,
                        wing_sigma=SHORT_VOL_WING_SIGMA,
                        premium_ratio=SHORT_VOL_PREMIUM_RATIO,
                        **kwargs):
    """做空波动率回测 (Iron Condor 1.6σ 短 / 3σ 长翼).

    结构:
      - 卖 OTM Put (1.6σ 下方) + 买 OTM Put (3σ 下方)
      - 卖 OTM Call (1.6σ 上方) + 买 OTM Call (3σ 上方)
    净收 credit ≈ premium_ratio × σ_pct
    P&L:
      - 波动 < 1.6σ → 留全部 credit (赢)
      - 波动 ∈ [1.6σ, 3σ] → credit - (max_move - 1.6σ)
      - 波动 > 3σ → max loss = (3σ - 1.6σ) - credit (翼宽锁定)
    """
    import numpy as np

    # 取 daily_range 给信号检测用
    if 'daily_range' not in kwargs:
        kwargs['daily_range'] = ((high - low) / close * 100)

    sv = detect_short_vol_signal(rv_series, rv_pctile, dates_index, **kwargs)
    signals = sv[sv["short_vol_signal"]]

    # 去重: 同向 ≤ 3 天连续视为同一笔
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
        reason = sv.loc[d, "short_vol_reason"]

        sigma_pct = rv * sqrt_h252                  # 1σ %
        short_strike = sigma_pct * strike_sigma     # 1.6σ
        wing_strike = sigma_pct * wing_sigma        # 3σ
        wing_width = wing_strike - short_strike     # 1.4σ
        credit = sigma_pct * premium_ratio          # 净 credit

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

        # IC P&L
        if max_move <= short_strike:
            pnl_pct = credit
            win = True
        elif max_move >= wing_strike:
            pnl_pct = credit - wing_width
            win = False
        else:
            pnl_pct = credit - (max_move - short_strike)
            win = pnl_pct > 0

        trades.append({
            "entry_date": d, "exit_date": exit_date,
            "entry_price": c, "rv": rv,
            "sigma_pct": sigma_pct,
            "short_strike_pct": short_strike,
            "wing_strike_pct": wing_strike,
            "credit_pct": credit,
            "max_move": max_move,
            "pnl_pct": pnl_pct,
            "win": win,
            "reason": reason,
            "partial": partial,
        })

    return trades
