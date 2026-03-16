"""OI 因子计算 + 区间修正.

从 EOD 期权快照提取:
  - Max Pain (pin 效应)
  - Call Wall / Put Wall (方向性阻力)
  - Net Gamma Exposure (波动压缩/放大)

用于修正模型预测区间和 Hybrid Band.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# SGT/北京时区 (UTC+8)
_TZ_SGT = timezone(timedelta(hours=8))


def _recalc_dte(eod_df, ref_date=None):
    """根据 ref_date 重算 DTE (从 strike_time 列).

    ref_date: datetime.date 或 None (默认今日 SGT).
    返回带有更新 dte 列的 DataFrame 副本.
    """
    if ref_date is None:
        ref_date = datetime.now(_TZ_SGT).date()

    df = eod_df.copy()
    if "strike_time" in df.columns:
        expiry = pd.to_datetime(df["strike_time"].astype(str).str[:10])
        ref_ts = pd.Timestamp(ref_date)
        df["dte"] = (expiry - ref_ts).dt.days
    return df


def compute_oi_factors(eod_df, spot, dte_max=60, ref_date=None):
    """从 EOD 快照计算 OI 因子.

    以 OI 最集中的到期日 (dominant_dte) 为核心, 而非最近到期日.
    月度 OPEX 通常占 60-80% 的 OI, 是真正驱动 pin 效应的力量.

    Args:
        ref_date: datetime.date, 用于重算 DTE. 默认今日 (SGT).

    Returns dict or None.
    """
    if eod_df is None or len(eod_df) == 0:
        return None

    # 用 ref_date 重算 DTE
    eod_df = _recalc_dte(eod_df, ref_date)

    df = eod_df[(eod_df["dte"] >= 1) &
                (eod_df["dte"] <= dte_max) &
                (eod_df["option_open_interest"] > 0)].copy()

    if len(df) == 0:
        return None

    # ── 按到期日统计 OI ──
    dte_oi = df.groupby("dte")["option_open_interest"].sum().sort_index()
    total_oi = dte_oi.sum()
    dominant_dte = int(dte_oi.idxmax())  # OI 最大的到期日
    dominant_oi = int(dte_oi.max())
    nearest_dte = int(dte_oi.index.min())

    # 到期日明细 (前5个)
    expiry_breakdown = []
    # 获取 DTE → 日期映射
    dte_to_date = {}
    if "strike_time" in df.columns:
        for dte_val in dte_oi.index:
            sub = df[df["dte"] == dte_val]
            if len(sub) > 0:
                dte_to_date[int(dte_val)] = str(
                    sub["strike_time"].iloc[0])[:10]
    for dte_val in dte_oi.nlargest(5).index:
        dte_int = int(dte_val)
        oi_val = int(dte_oi[dte_val])
        pct = oi_val / total_oi * 100
        label = "月度OPEX" if pct > 30 else "周度"
        expiry_breakdown.append({
            "dte": dte_int,
            "date": dte_to_date.get(dte_int, ""),
            "oi": oi_val,
            "pct": pct,
            "label": label,
        })
    expiry_breakdown.sort(key=lambda x: x["dte"])

    # ── 用 dominant expiry 的 OI 计算核心因子 ──
    # 包含 dominant_dte 附近 ±2 天 (同一周), 以及更远的到期日
    core_df = df[df["dte"] >= min(dominant_dte - 2, nearest_dte)]
    calls = core_df[core_df["option_type"] == "CALL"]
    puts = core_df[core_df["option_type"] == "PUT"]

    call_oi = calls.groupby("option_strike_price")[
        "option_open_interest"].sum()
    put_oi = puts.groupby("option_strike_price")[
        "option_open_interest"].sum()

    if len(call_oi) == 0 or len(put_oi) == 0:
        return None

    # Max Pain
    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    pain = []
    for k in all_strikes:
        c_pain = sum(max(k - s, 0) * oi for s, oi in call_oi.items())
        p_pain = sum(max(s - k, 0) * oi for s, oi in put_oi.items())
        pain.append((k, c_pain + p_pain))
    max_pain = min(pain, key=lambda x: x[1])[0]

    # Call Wall / Put Wall
    call_wall = float(call_oi.idxmax())
    put_wall = float(put_oi.idxmax())

    # Net Gamma Exposure (近 ATM ±5%)
    atm_range = (spot * 0.95, spot * 1.05)
    near_atm = core_df[
        (core_df["option_strike_price"] >= atm_range[0]) &
        (core_df["option_strike_price"] <= atm_range[1])]
    call_gex = near_atm[near_atm["option_type"] == "CALL"].apply(
        lambda r: r["option_gamma"] * r["option_open_interest"] * 100,
        axis=1).sum() if len(near_atm) > 0 else 0
    put_gex = near_atm[near_atm["option_type"] == "PUT"].apply(
        lambda r: r["option_gamma"] * r["option_open_interest"] * 100,
        axis=1).sum() if len(near_atm) > 0 else 0

    # PCR
    pcr = put_oi.sum() / call_oi.sum() if call_oi.sum() > 0 else 1.0

    # Top3 strikes
    top3_calls = call_oi.nlargest(3)
    top3_puts = put_oi.nlargest(3)

    return {
        "max_pain": max_pain,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_gex": call_gex,
        "put_gex": put_gex,
        "net_gex": call_gex - put_gex,
        "nearest_dte": nearest_dte,
        "dominant_dte": dominant_dte,
        "dominant_oi": dominant_oi,
        "dominant_oi_pct": dominant_oi / total_oi * 100,
        "pcr": pcr,
        "total_call_oi": int(call_oi.sum()),
        "total_put_oi": int(put_oi.sum()),
        "top3_call_strikes": list(top3_calls.index),
        "top3_call_oi": list(top3_calls.values),
        "top3_put_strikes": list(top3_puts.index),
        "top3_put_oi": list(top3_puts.values),
        "expiry_breakdown": expiry_breakdown,
    }


def _apply_oi_adj(upper, lower, spot, oi, expiry_factor):
    """OI 修正核心逻辑 (内部用).

    Returns: (adj_upper, adj_lower, cw_adj, pw_adj, adjustments)
    """
    max_pain = oi["max_pain"]
    call_wall = oi["call_wall"]
    put_wall = oi["put_wall"]
    net_gex = oi["net_gex"]

    adj_upper = upper
    adj_lower = lower
    adjustments = []

    # 1. Max Pain 引力
    mp_dist_pct = (spot - max_pain) / spot
    gravity_strength = 0.15 * expiry_factor

    if mp_dist_pct > 0:
        gravity = mp_dist_pct * gravity_strength
        adj_upper = upper * (1 - gravity)
        adjustments.append(
            f"Max Pain ${max_pain:.0f}在下方 → 压缩上界{gravity*100:.1f}%")
    else:
        gravity = mp_dist_pct * gravity_strength
        adj_lower = lower * (1 - gravity)
        adjustments.append(
            f"Max Pain ${max_pain:.0f}在上方 → 抬升下界{abs(gravity)*100:.1f}%")

    # 2. Call Wall 压制
    cw_adj = False
    if call_wall < adj_upper:
        blend = 0.3 * expiry_factor
        adj_upper = adj_upper * (1 - blend) + call_wall * blend
        cw_adj = True
        adjustments.append(
            f"Call Wall ${call_wall:.0f} 压制上界 (权重{blend:.0%})")

    # 3. Put Wall 支撑
    pw_adj = False
    if put_wall > adj_lower:
        blend = 0.2 * expiry_factor
        adj_lower = adj_lower * (1 - blend) + put_wall * blend
        pw_adj = True
        adjustments.append(
            f"Put Wall ${put_wall:.0f} 支撑下界 (权重{blend:.0%})")

    # 4. Gamma 效应
    if net_gex > 0:
        compress = min(net_gex / 1e6 * 0.002, 0.03) * expiry_factor
        mid = (adj_upper + adj_lower) / 2
        adj_upper = mid + (adj_upper - mid) * (1 - compress)
        adj_lower = mid + (adj_lower - mid) * (1 - compress)
        adjustments.append(
            f"Long gamma → 压缩区间{compress*100:.1f}%")
    elif net_gex < 0:
        expand = min(abs(net_gex) / 1e6 * 0.002, 0.03) * expiry_factor
        mid = (adj_upper + adj_lower) / 2
        adj_upper = mid + (adj_upper - mid) * (1 + expand)
        adj_lower = mid + (adj_lower - mid) * (1 + expand)
        adjustments.append(
            f"Short gamma → 扩大区间{expand*100:.1f}%")

    return adj_upper, adj_lower, cw_adj, pw_adj, adjustments


def adjust_range(upper_price, lower_price, spot, oi):
    """用 OI 因子修正预测区间 (单一值).

    Returns: (adj_upper, adj_lower, details_dict)
    """
    if oi is None:
        return upper_price, lower_price, {"adjusted": False}

    # 用 dominant_dte (OI最大到期日) 驱动 expiry_factor
    ref_dte = oi.get("dominant_dte", oi["nearest_dte"])
    expiry_factor = np.clip(1.0 - (ref_dte - 7) / 30, 0.1, 1.0)

    adj_upper, adj_lower, cw_adj, pw_adj, adjustments = \
        _apply_oi_adj(upper_price, lower_price, spot, oi, expiry_factor)

    return adj_upper, adj_lower, {
        "adjusted": True,
        "upper_change_pct": (adj_upper / upper_price - 1) * 100,
        "lower_change_pct": (adj_lower / lower_price - 1) * 100,
        "expiry_factor": expiry_factor,
        "cw_adj": cw_adj,
        "pw_adj": pw_adj,
        "adjustments": adjustments,
    }


def adjust_range_daily(upper_price, lower_price, spot, oi, n_days=5):
    """逐日修正预测区间: sqrt缩放 + OI因子随DTE倒计时变化.

    到期前: 压制增强 (expiry_factor 升高)
    到期日: 最强 pin 效应
    到期后: 压制释放, 区间回归模型预测

    Args:
        upper_price: 模型5日上界 (绝对价格)
        lower_price: 模型5日下界
        spot: 当前价格
        oi: compute_oi_factors() 返回值 (可为 None)
        n_days: 预测天数 (默认5)

    Returns: (daily_ranges, events)
        daily_ranges: [(day1_upper, day1_lower), ...] 共 n_days 项
        events: [(day_idx, description), ...] 到期/释放事件
    """
    mid = (upper_price + lower_price) / 2
    half = (upper_price - lower_price) / 2

    daily = []
    events = []

    for d in range(1, n_days + 1):
        # 基础: sqrt 缩放 — 不确定性随时间增长
        scale = np.sqrt(d / n_days)
        day_u = mid + half * scale
        day_l = mid - half * scale

        if oi is not None:
            ref_dte = oi.get("dominant_dte", oi["nearest_dte"])
            eff_dte = ref_dte - d

            if eff_dte > 0:
                # 到期前: OI 修正, 效应随 DTE 减小而增强
                ef = np.clip(1.0 - (eff_dte - 7) / 30, 0.1, 1.0)
                day_u, day_l, _, _, _ = _apply_oi_adj(
                    day_u, day_l, spot, oi, ef)
            elif eff_dte == 0:
                # 到期日: 最强 pin 效应
                day_u, day_l, _, _, _ = _apply_oi_adj(
                    day_u, day_l, spot, oi, 1.0)
                events.append((d, "OPEX 期权到期 — 最强pin效应"))
            else:
                # 到期后: OI 压制释放, 不再修正 (保留 sqrt 基础区间)
                if eff_dte == -1:
                    events.append((d, "到期后压力释放 — 区间扩大"))

        daily.append((day_u, day_l))

    return daily, events


def adjust_band_history(upper_band, lower_band, close, snapshots):
    """用历史 EOD 快照修正 band 上下界.

    每个快照日计算 OI 因子, 向后持续生效直到下一个快照日.

    Args:
        upper_band: pd.Series (date → upper)
        lower_band: pd.Series (date → lower)
        close: pd.Series (date → close price)
        snapshots: {pd.Timestamp: eod_df} 全部历史快照

    Returns: (adj_upper: pd.Series, adj_lower: pd.Series)
             仅包含有 OI 修正的日期; 无快照覆盖的日期不包含.
    """
    if not snapshots:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # 按日期排序快照, 计算每个快照的 OI 因子
    snap_dates = sorted(snapshots.keys())
    oi_by_snap = {}
    for sd in snap_dates:
        spot = close.get(sd, None)
        if spot is None or spot == 0:
            continue
        oi = compute_oi_factors(snapshots[sd], spot,
                                ref_date=sd.date() if hasattr(sd, 'date') else sd)
        if oi is not None:
            oi_by_snap[sd] = oi

    if not oi_by_snap:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # 对 band 中的每个日期, 找到最近 (<=) 的快照并修正
    adj_u = {}
    adj_l = {}
    sorted_snap = sorted(oi_by_snap.keys())

    for d in upper_band.index:
        if d not in lower_band.index:
            continue
        # 找最近的快照 (bisect)
        candidates = [s for s in sorted_snap if s <= d]
        if not candidates:
            continue
        snap_d = candidates[-1]
        oi = oi_by_snap[snap_d]
        u = upper_band[d]
        l = lower_band[d]
        spot = close.get(d, 0)
        if spot == 0 or u <= l:
            continue
        au, al, _, _, _ = _apply_oi_adj(u, l, spot, oi,
                                         np.clip(1.0 - (oi.get("dominant_dte", oi["nearest_dte"]) - 7) / 30, 0.1, 1.0))
        adj_u[d] = au
        adj_l[d] = al

    return pd.Series(adj_u), pd.Series(adj_l)
