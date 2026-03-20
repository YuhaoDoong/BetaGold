"""统一策略选择器 — 方向性 vs Straddle 择优.

当同一天出现多种信号时, 选择胜率最高的策略:
  - 方向性 (BUY CALL / SELL PUT): 看5天后是否上涨
  - Straddle: 看5天内波动幅度是否超过成本
  - EXIT: 退出优先, 不参与选择

选择逻辑:
  1. Straddle score >= 5 (FOMC+RV压缩) → 优先 Straddle
  2. 两者都有但 Straddle score < 5 → 选方向性 (更常见, 成本低)
  3. 只有一种 → 用该信号
"""

import pandas as pd
import numpy as np


STRADDLE_PRIORITY_SCORE = 5  # score >= 此值时 Straddle 优先于方向性


def build_unified_signals(dir_signals, straddle_df, close, high, low,
                           hold_days=5, straddle_cost_pct=3.0):
    """构建统一信号表 (方向性 + Straddle 合并).

    Args:
        dir_signals: from generate_daily_signals()
        straddle_df: from detect_straddle_signal()
        close/high/low: GLD price series
        straddle_cost_pct: Straddle 成本估算 (% of price)

    Returns: DataFrame with unified signals + P&L
    """
    dates = dir_signals.index.intersection(straddle_df.index)
    records = []

    for d in dates:
        dr = dir_signals.loc[d] if d in dir_signals.index else None
        sr = straddle_df.loc[d] if d in straddle_df.index else None

        has_dir = dr is not None and dr.get("buy_signal", False)
        has_exit = dr is not None and dr.get("exit_signal", False)
        has_straddle = sr is not None and sr.get("straddle_signal", False)
        straddle_score = sr["straddle_score"] if sr is not None else 0

        if not has_dir and not has_exit and not has_straddle:
            continue

        # 5天后收益
        loc = close.index.get_loc(d) if d in close.index else -1
        if loc < 0 or loc + hold_days >= len(close):
            ret_5d = None
            max_move = None
        else:
            ret_5d = (close.iloc[loc + hold_days] / close[d] - 1) * 100
            move_up = (high.iloc[loc + 1:loc + hold_days + 1].max() / close[d] - 1) * 100
            move_down = (1 - low.iloc[loc + 1:loc + hold_days + 1].min() / close[d]) * 100
            max_move = max(move_up, move_down)

        # 策略选择
        if has_exit:
            chosen = "EXIT"
            chosen_reason = "退出信号"
        elif has_dir and has_straddle:
            # 两者都有 → 按 score 选
            if straddle_score >= STRADDLE_PRIORITY_SCORE:
                chosen = "STRADDLE"
                chosen_reason = f"Straddle优先(score={straddle_score}≥{STRADDLE_PRIORITY_SCORE})"
            else:
                chosen = dr["buy_type"] if dr["buy_type"] else "BUY CALL"
                chosen_reason = f"方向性优先(Straddle score={straddle_score})"
        elif has_dir:
            chosen = dr["buy_type"] if dr["buy_type"] else "BUY CALL"
            chosen_reason = "仅方向性"
        elif has_straddle:
            chosen = "STRADDLE"
            chosen_reason = f"仅Straddle(score={straddle_score})"
        else:
            continue

        # 盈亏判定
        if ret_5d is not None:
            if chosen == "EXIT":
                # EXIT 后5天没涨 > 3% 就是对的
                win = ret_5d < 3
            elif chosen == "STRADDLE":
                win = max_move > straddle_cost_pct if max_move else None
            else:
                # 方向性: 5天后没跌 > 3%
                win = ret_5d > -3
        else:
            win = None

        records.append({
            "date": d,
            "close": close.get(d, 0),
            "dir_signal": dr["buy_type"] if has_dir else ("EXIT" if has_exit else ""),
            "straddle_signal": has_straddle,
            "straddle_score": straddle_score,
            "chosen": chosen,
            "chosen_reason": chosen_reason,
            "ret_5d": ret_5d,
            "max_move_5d": max_move,
            "win": win,
        })

    return pd.DataFrame(records).set_index("date")


def compute_unified_stats(unified_df):
    """计算统一胜率统计."""
    if len(unified_df) == 0:
        return {}

    # 去重: 连续信号只取第一天
    entries = []
    prev = None
    for d, r in unified_df.iterrows():
        if r["chosen"] == "EXIT":
            entries.append(d)
            prev = None
        elif prev is None or (d - prev).days > 3:
            entries.append(d)
            prev = d

    deduped = unified_df.loc[entries].copy()
    deduped["win_bool"] = deduped["win"].apply(
        lambda x: bool(x) if x is not None else None)
    valid = deduped[deduped["win_bool"].notna()]

    if len(valid) == 0:
        return {"total": len(deduped), "evaluated": 0}

    total = len(valid)
    wins = int(valid["win_bool"].apply(bool).sum())

    # 按策略分组
    by_type = {}
    for chosen in valid["chosen"].unique():
        sub = valid[valid["chosen"] == chosen]
        w = int(sub["win_bool"].apply(bool).sum())
        by_type[chosen] = {
            "n": len(sub),
            "win": w,
            "wr": w / len(sub) if len(sub) > 0 else 0,
        }

    return {
        "total": total,
        "wins": int(wins),
        "win_rate": wins / total,
        "by_type": by_type,
        "deduped_count": len(deduped),
    }
