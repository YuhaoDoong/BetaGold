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


STRADDLE_PRIORITY_SCORE = 5      # 做多波动率 score >= 此值时优先
SHORT_VOL_PRIORITY_SCORE = 5     # 做空波动率 score >= 此值时优先
VOL_DIR_BOTH_STRONG = 4          # 波动率与方向性都 >= 此值时同时推荐 (MIXED)


def dedupe_unified(unified_df, close_d, log_price_fn=None,
                   add_drop_pct=2.0, dir_gap_days=5,
                   straddle_gap_days=3):
    """对 build_unified_signals 输出去重, 返回保留行的子集 + entry_p 列.

    规则:
      - EXIT: 重置 _prev_entry, 全部保留
      - STRADDLE: 同向连续 ≤ straddle_gap_days 天, 仅 score 升级时保留;
        否则只保留首个
      - 方向性 (BUY/SELL): 同向 ≤ dir_gap_days 天内, 价格跌 > add_drop_pct%
        视为加仓 (保留并标 is_add=True), 否则视为横盘 (suppress)

    Args:
        log_price_fn: callable (d, side) -> float or None.
            用于取标注价格 (log 真实触发或 close 兜底).
            None 时退到 close_d.

    Returns: DataFrame 保留行, 原 columns + entry_p + is_add (bool).
    """
    if unified_df is None or len(unified_df) == 0:
        return unified_df

    def _price(d, chosen):
        if log_price_fn is not None \
                and "STRADDLE" not in chosen \
                and "SHORT_VOL" not in chosen:
            side = "EXIT" if "EXIT" in chosen else "BUY"
            p = log_price_fn(d, side)
            if p is not None:
                return p
        return close_d.get(d, 0)

    prev = {}  # {chosen_type: (date, price, score)}
    keep_rows = []
    for d, r in unified_df.iterrows():
        chosen = r["chosen"]
        # 取对应 vol score (做多/做空)
        if "SHORT_VOL" in chosen:
            score = r.get("short_vol_score", 0)
        else:
            score = r.get("straddle_score", 0)
        entry_p = _price(d, chosen)

        show = True
        is_add = False

        # 纯波动率信号 (没和方向性混合) 用 vol 去重逻辑
        is_pure_vol = (chosen in ("STRADDLE", "SHORT_VOL"))

        if chosen == "EXIT":
            prev = {}
        elif is_pure_vol:
            p = prev.get(chosen)
            if p and (d - p[0]).days <= straddle_gap_days:
                if score > p[2]:
                    show = True   # score 升级
                else:
                    show = False  # 同 score 连续, 不重复
            prev[chosen] = (d, entry_p, score)
        else:
            p = prev.get(chosen)
            if p and p[1] > 0:
                drop = (p[1] - entry_p) / p[1] * 100
                if drop > add_drop_pct:
                    is_add = True   # 加仓
                elif (d - p[0]).days <= dir_gap_days:
                    show = False    # 横盘忽略
            if show:
                prev[chosen] = (d, entry_p, 0)

        if show:
            new = r.copy()
            new["entry_p"] = entry_p
            new["is_add"] = is_add
            keep_rows.append((d, new))

    if not keep_rows:
        return unified_df.iloc[0:0]
    out = pd.DataFrame([r for _, r in keep_rows],
                       index=[d for d, _ in keep_rows])
    return out


def build_unified_signals(dir_signals, straddle_df, close, high, low,
                           hold_days=5, straddle_cost_pct=3.0,
                           short_vol_df=None):
    """构建统一信号表 (方向性 + 做多波动率 + 做空波动率 合并).

    Args:
        dir_signals: from generate_daily_signals()
        straddle_df: from detect_straddle_signal()  (做多波动率)
        short_vol_df: from detect_short_vol_signal()  (做空波动率, 可选)
        close/high/low: price series
        straddle_cost_pct: Straddle 成本估算 (% of price)

    选择逻辑 (vol = 做多 vs 做空 取强者):
      1. EXIT 优先
      2. vol_score 与 dir_score 都 >= VOL_DIR_BOTH_STRONG → MIXED 同时推荐
      3. vol_score >= 优先阈值 → 选 vol
      4. 仅一个 → 选该信号
    Returns: DataFrame with unified signals + P&L
    """
    dates = dir_signals.index.intersection(straddle_df.index)
    if short_vol_df is not None:
        dates = dates.intersection(short_vol_df.index)
    records = []

    for d in dates:
        dr = dir_signals.loc[d] if d in dir_signals.index else None
        sr = straddle_df.loc[d] if d in straddle_df.index else None
        svr = (short_vol_df.loc[d]
               if short_vol_df is not None and d in short_vol_df.index
               else None)

        has_dir = dr is not None and dr.get("buy_signal", False)
        has_exit = dr is not None and dr.get("exit_signal", False)
        has_straddle = sr is not None and sr.get("straddle_signal", False)
        has_short_vol = svr is not None and svr.get("short_vol_signal", False)
        straddle_score = sr["straddle_score"] if sr is not None else 0
        short_vol_score = (svr["short_vol_score"]
                           if svr is not None else 0)

        if not has_dir and not has_exit and not has_straddle \
                and not has_short_vol:
            continue

        # 5天后收益
        loc = close.index.get_loc(d) if d in close.index else -1
        if loc < 0 or loc + hold_days >= len(close):
            ret_5d = None
            max_move = None
            move_up = None
            move_down = None
        else:
            ret_5d = (close.iloc[loc + hold_days] / close[d] - 1) * 100
            move_up = (high.iloc[loc + 1:loc + hold_days + 1].max() / close[d] - 1) * 100
            move_down = (1 - low.iloc[loc + 1:loc + hold_days + 1].min() / close[d]) * 100
            max_move = max(move_up, move_down)

        # vol 取做多/做空两者较强者
        if has_straddle and has_short_vol:
            # 两者互斥 (高 RV vs 低 RV), 但若都判定为 True (异常), 取 score 高者
            if short_vol_score >= straddle_score:
                vol_type, vol_score = "SHORT_VOL", short_vol_score
            else:
                vol_type, vol_score = "STRADDLE", straddle_score
        elif has_straddle:
            vol_type, vol_score = "STRADDLE", straddle_score
        elif has_short_vol:
            vol_type, vol_score = "SHORT_VOL", short_vol_score
        else:
            vol_type, vol_score = None, 0

        dir_type = (dr["buy_type"] if (has_dir and dr["buy_type"]) else
                    ("BUY CALL" if has_dir else None))

        # 策略选择
        if has_exit:
            chosen = "EXIT"
            chosen_reason = "退出信号"
        elif vol_type and dir_type:
            # Vega 兼容矩阵 (按 vega 方向判断 MIXED 是否合理):
            #   BUY CALL  = long vega  (低 RV 入场, 赌 vol 涨)
            #   SELL PUT  = short vega (高 RV 入场, 收 IV premium)
            #   STRADDLE  = long vega
            #   SHORT_VOL = short vega
            # 只有 vega 方向相同才允许 MIXED, 否则取主动信号 (方向性).
            dir_long_vega = (dir_type == "BUY CALL")
            vol_long_vega = (vol_type == "STRADDLE")
            vega_compatible = (dir_long_vega == vol_long_vega)

            if not vega_compatible:
                # 矛盾: BUY CALL+SHORT_VOL 或 SELL PUT+STRADDLE
                vol_priority = (STRADDLE_PRIORITY_SCORE
                                if vol_type == "STRADDLE"
                                else SHORT_VOL_PRIORITY_SCORE)
                if vol_score >= vol_priority + 2:
                    chosen = vol_type
                    chosen_reason = (f"{vol_type}极强(score={vol_score})"
                                     f"覆盖矛盾方向性")
                else:
                    chosen = dir_type
                    chosen_reason = (f"方向性优先(vega矛盾 vs {vol_type}, "
                                     f"score={vol_score})")
            else:
                # vega 同向 → 可 MIXED
                if vol_score >= VOL_DIR_BOTH_STRONG:
                    chosen = f"{dir_type} + {vol_type}"
                    chosen_reason = (f"方向+{vol_type}同强同向"
                                     f"(vol={vol_score})")
                elif (vol_type == "STRADDLE"
                      and vol_score >= STRADDLE_PRIORITY_SCORE) or \
                     (vol_type == "SHORT_VOL"
                      and vol_score >= SHORT_VOL_PRIORITY_SCORE):
                    chosen = vol_type
                    chosen_reason = f"{vol_type}优先(score={vol_score})"
                else:
                    chosen = dir_type
                    chosen_reason = f"方向性优先({vol_type} score={vol_score})"
        elif dir_type:
            chosen = dir_type
            chosen_reason = "仅方向性"
        elif vol_type:
            chosen = vol_type
            chosen_reason = f"仅{vol_type}(score={vol_score})"
        else:
            continue

        # 盈亏判定 (sigma_pct 用 RV * sqrt(hold/252) 估)
        rv_today = (svr["rv"] if svr is not None
                    else (sr["rv"] if sr is not None else 20))
        sigma_pct = rv_today * (hold_days / 252) ** 0.5
        # SHORT_VOL 用 IC 1.6σ 短腿距离 (与 backtest_short_vol 一致)
        from core.events import (SHORT_VOL_STRIKE_SIGMA,
                                  SHORT_VOL_WING_SIGMA)
        ic_short = sigma_pct * SHORT_VOL_STRIKE_SIGMA
        ic_wing = sigma_pct * SHORT_VOL_WING_SIGMA

        # 各策略 win 定义 (按 vega/delta 实际盈亏, 全部用动态 sigma_pct):
        #   BUY CALL  long delta + long vega: max_up > 1σ
        #             (真涨才赢, 横盘亏 theta+IV crush)
        #   SELL PUT  long delta + short vega: max_down < 1σ
        #             (跌不破 1σ 下方短 put strike, 横盘+上涨都赢)
        #   STRADDLE  long gamma + long vega: |move| > 1σ (≥ premium)
        #   SHORT_VOL short gamma + short vega: max_move < 1.6σ (留 credit)
        #   期货多头 (FUT_LONG)  linear delta, vega=0, theta=0:
        #             ret_5d > 0 即赢 (无 IV 成本)
        def _dir_win(dt):
            if move_up is None or move_down is None:
                return None
            if dt == "BUY CALL":
                return move_up > sigma_pct
            return move_down < sigma_pct  # SELL PUT

        def _vol_win(vt):
            if max_move is None:
                return None
            if vt == "STRADDLE":
                return max_move > sigma_pct
            return max_move < ic_short  # SHORT_VOL

        # 期货多头胜率 (与期权并列, 用于对比)
        fut_win = (ret_5d > 0) if ret_5d is not None else None
        # 期货 + 3% 止损: 5天内 max_down 突破 3% 视为先止损, 否则按 ret_5d
        if ret_5d is not None and move_down is not None:
            if move_down > 3:
                fut_stop_pnl = -3.0
            else:
                fut_stop_pnl = ret_5d
            fut_stop_win = fut_stop_pnl > 0
        else:
            fut_stop_pnl = None
            fut_stop_win = None

        if ret_5d is not None:
            if chosen == "EXIT":
                win = ret_5d < 3
            elif "+" in chosen:
                base_dir, base_vol = chosen.split(" + ")
                dw, vw = _dir_win(base_dir), _vol_win(base_vol)
                if dw is None and vw is None:
                    win = None
                else:
                    win = bool(dw) or bool(vw)
            elif chosen in ("BUY CALL", "SELL PUT"):
                win = _dir_win(chosen)
            elif chosen in ("STRADDLE", "SHORT_VOL"):
                win = _vol_win(chosen)
            else:
                win = ret_5d > -3
        else:
            win = None

        records.append({
            "date": d,
            "close": close.get(d, 0),
            "dir_signal": dr["buy_type"] if has_dir else ("EXIT" if has_exit else ""),
            "straddle_signal": has_straddle,
            "straddle_score": straddle_score,
            "short_vol_signal": has_short_vol,
            "short_vol_score": short_vol_score,
            "fut_win": fut_win,
            "fut_stop_win": fut_stop_win,
            "fut_stop_pnl": fut_stop_pnl,
            "chosen": chosen,
            "chosen_reason": chosen_reason,
            "ret_5d": ret_5d,
            "max_move_5d": max_move,
            "sigma_pct": sigma_pct,
            "win": win,
        })

    if not records:
        return pd.DataFrame(columns=[
            "close", "dir_signal", "straddle_signal", "straddle_score",
            "short_vol_signal", "short_vol_score",
            "fut_win", "fut_stop_win", "fut_stop_pnl",
            "chosen", "chosen_reason",
            "ret_5d", "max_move_5d", "sigma_pct", "win",
        ])
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

    # EXIT 不单独统计胜率 — 只统计入场信号
    entry_signals = deduped[deduped["chosen"] != "EXIT"]
    valid = entry_signals[entry_signals["win_bool"].notna()]

    if len(valid) == 0:
        return {"total": len(entry_signals), "evaluated": 0}

    total = len(valid)
    wins = int(valid["win_bool"].apply(bool).sum())

    # 按策略分组 (不含 EXIT)
    by_type = {}
    for chosen in valid["chosen"].unique():
        sub = valid[valid["chosen"] == chosen]
        w = int(sub["win_bool"].apply(bool).sum())
        by_type[chosen] = {
            "n": len(sub),
            "win": w,
            "wr": w / len(sub) if len(sub) > 0 else 0,
        }

    # 期货独立统计 (按方向性类型拆分: BUY CALL 信号 vs SELL PUT 信号)
    fut_stats = {}
    dir_groups = {
        "BUY CALL 类": ["BUY CALL", "BUY CALL + STRADDLE"],
        "SELL PUT 类": ["SELL PUT", "SELL PUT + SHORT_VOL"],
        "全部方向性": ["BUY CALL", "SELL PUT",
                       "BUY CALL + STRADDLE", "SELL PUT + SHORT_VOL"],
    }
    for grp_name, types in dir_groups.items():
        sub = valid[valid["chosen"].isin(types)]
        fut_valid = sub[sub["fut_win"].notna()]
        if len(fut_valid) == 0:
            continue
        opt_wins = int(fut_valid["win_bool"].apply(bool).sum())
        fut_wins = int(fut_valid["fut_win"].apply(bool).sum())
        fut_stop_wins = int(fut_valid["fut_stop_win"].apply(bool).sum())
        fut_stats[grp_name] = {
            "n": len(fut_valid),
            "opt_win": opt_wins,
            "opt_wr": opt_wins / len(fut_valid),
            "fut_win": fut_wins,
            "fut_wr": fut_wins / len(fut_valid),
            "fut_stop_win": fut_stop_wins,
            "fut_stop_wr": fut_stop_wins / len(fut_valid),
            "fut_stop_total_pnl": float(fut_valid["fut_stop_pnl"].sum()),
            "fut_stop_avg_pnl": float(fut_valid["fut_stop_pnl"].mean()),
        }

    return {
        "total": total,
        "wins": int(wins),
        "win_rate": wins / total,
        "by_type": by_type,
        "futures": fut_stats,
        "deduped_count": len(deduped),
    }
