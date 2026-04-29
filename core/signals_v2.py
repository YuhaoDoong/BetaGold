"""信号系统 — 与 Dashboard 显示策略 1:1 一致.

策略 (与持仓管理表 / K线区显示一致):
  入场: 日线 bp_low < BUY_BP (开窗) + 盘中真实触发 (Stoch RSI / MACD / KDJ 确认)
        入场价从盘中 log 取代表价 (默认 worst); 没 log 兜底当日收盘
  退出 (优先级):
    1. StopLoss: 入场后日内 low 跌破 -STOP_LOSS_PCT%
    2. BandExit: 日线 bp_high > EXIT_BP → 优先 log EXIT 代表价, 兜底 bp090
    3. Pullback: 持仓期间峰值涨幅 > PULLBACK_GAIN% 且当前回撤 >= PULLBACK_DD%
       → 这就是持仓管理 "止盈位" 列显示的那条线
  风控: 连续 CONSECUTIVE_STOP 笔止损后暂停买入 (bp>0.5 恢复)
  安全帽: MAX_HOLD_DAYS 强平 (实际 2-5 天就走完了)

不再用 (这些没在 dashboard 显示):
  ✗ 12h K线 resample (粒度全用日线 high/low, 与持仓管理 peak 对齐)
  ✗ MACD 弱化止盈 (没在任何区块显示)

参数:
  BUY_BP           = 0.30   # 买入阈值 (Band Position)
  EXIT_BP          = 0.90   # 退出阈值
  STOP_LOSS_PCT    = 3.0    # 单笔止损 %
  PULLBACK_GAIN    = 2.0    # Pullback 触发: 峰值涨幅 > N%
  PULLBACK_DD      = 1.5    # Pullback 触发: 从峰值回撤 >= N%
  CONSECUTIVE_STOP = 99     # 熔断默认禁用 (实证: 不提升胜率)
                          # 保留参数, 调小可启用 (eg. =2 启动连续 2 止损熔断)
                          # A/B 实证 (近 5y): 关熔断 +52.6% / CS=2 +47.4%
                          # 详见 docs/EXPERIMENTS.md "熔断 A/B"
  MAX_HOLD_DAYS    = 30     # 安全帽 (远大于实际持仓 2-5 天)
"""

import numpy as np
import pandas as pd
from datetime import timedelta

# ── 可配置参数 ──
BUY_BP = 0.30
EXIT_BP = 0.90
STOP_LOSS_PCT = 3.0
PULLBACK_GAIN = 2.0       # 持仓管理 "止盈位" 列就是基于这俩计算
PULLBACK_DD = 1.5
CONSECUTIVE_STOP = 99     # v3.7.8 默认禁用 (A/B 实证不提升胜率, 错杀赢面)
                          # 改 2 启用经典连续 2 止损熔断
                          # 详见 docs/EXPERIMENTS.md §11 "熔断 A/B"
MAX_HOLD_DAYS = 30
DEFAULT_TZ_OFFSET = 8
# RV 极值过滤 (v3.7.29, 步长 0.025 精细网格搜索后选定):
#   BUY CALL 在 RV < 0.50 (vol 偏低/中性, 期权便宜)
#   SELL PUT 在 RV > 0.80 (vol 高位, 收 IV premium; 从 0.85 降到 0.80)
#   0.50-0.80 屏蔽 (温水中位)
# 5y 网格回测对比:
#   v3.6.1 (0.50/0.85): 37 笔 81% 胜率 +46.4% Sharpe 0.616
#   v3.7.29 (0.50/0.80): 38 笔 82% 胜率 +48.9% Sharpe 0.638  ← 全维度优化
# 副效应: SLV 3月27 (RV%tile=0.84) 边缘 SELL PUT 现可触发
RV_FILTER_LOW = 0.50      # RV %tile < 此值 → BUY CALL
RV_FILTER_HIGH = 0.80     # RV %tile > 此值 → SELL PUT (0.85 → 0.80, v3.7.29)
RV_FILTER_ENABLED = True  # 默认开启 RV 极值过滤

# Bear regime 做空 (实验性, 默认关闭):
#   实证: Bear + bp_high>0.85 期货空头 67% 胜率 (近 5y, n=9 样本小)
#         Bear + bp_low<0.30 追跌做空 60% 胜率 (n=10)
#   样本太小, 默认关闭, 用户可手动开启
BEAR_SHORT_ENABLED = False
BEAR_SHORT_BP_HIGH = 0.85   # 触发阈值 (bp_high > 此值)

# 兼容旧引用 (已不再用)
EXIT_TIMEFRAME = "1d"
MACD_MIN_GAIN = 999


def _macd_hist(c, fast=12, slow=26, sig=9):
    ef = c.ewm(span=fast, min_periods=1).mean()
    es = c.ewm(span=slow, min_periods=1).mean()
    ml = ef - es
    return ml - ml.ewm(span=sig, min_periods=1).mean()


def resample_1h(gld_1h, timeframe):
    """将 1h 数据 resample 到指定时间尺度."""
    if timeframe == "1h":
        return gld_1h
    return gld_1h.resample(timeframe).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()


def generate_daily_signals(close_d, high_d, low_d,
                           upper_band, lower_band,
                           regime, rv_pctile,
                           buy_bp=BUY_BP, exit_bp=EXIT_BP,
                           rv_filter=RV_FILTER_ENABLED,
                           rv_low=RV_FILTER_LOW,
                           rv_high=RV_FILTER_HIGH):
    """日线级别信号: v1.0 Band + H/L 触发 + RV 极值过滤.

    rv_filter=True 时只在 RV %tile < rv_low 或 > rv_high 时触发方向性.
    回测显示: 排除 25-75% 中位后大涨>3% 概率 21% → 32%, 大跌>3% 概率不变 6%.

    每天输出: Band 参数 + 买入/退出触发状态 + 阈值价位.
    """
    bp_dates = upper_band.dropna().index.intersection(
        lower_band.dropna().index)
    records = []

    for d in bp_dates:
        ub, lb = upper_band[d], lower_band[d]
        if ub <= lb:
            continue
        c = close_d.get(d, np.nan)
        h = high_d.get(d, np.nan)
        lo = low_d.get(d, np.nan)
        if np.isnan(c):
            continue

        bp_close = (c - lb) / (ub - lb)
        bp_low = (lo - lb) / (ub - lb)
        bp_high = (h - lb) / (ub - lb)
        bp030 = lb + buy_bp * (ub - lb)
        bp090 = lb + exit_bp * (ub - lb)

        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        buy_sig = is_bull and bp_low < buy_bp
        # RV 极值过滤: 中位区间 (温水区) 屏蔽方向性
        rv_extreme = (rv < rv_low) or (rv > rv_high)
        if rv_filter and buy_sig and not rv_extreme:
            buy_sig = False
        buy_type = None
        if buy_sig:
            if rv_filter:
                # 低 RV → BUY CALL (期权便宜), 高 RV → SELL PUT (收 IV)
                buy_type = "BUY CALL" if rv < rv_low else "SELL PUT"
            else:
                # 兼容旧 v1.0 逻辑: 0.85 为分界
                buy_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"

        exit_sig = bp_high > exit_bp
        # Regime 退出
        if d in regime.index:
            loc = regime.index.get_loc(d)
            if loc > 0 and regime.iloc[loc - 1] == "Bull" \
                    and regime[d] != "Bull":
                exit_sig = True

        parts = []
        if buy_sig:
            parts.append(buy_type)
        if exit_sig:
            parts.append("EXIT")

        records.append({
            "date": d, "close": c, "high": h, "low": lo,
            "upper": ub, "lower": lb,
            "bp_close": bp_close, "bp_low": bp_low, "bp_high": bp_high,
            "bp030_price": bp030, "bp090_price": bp090,
            "buy_signal": buy_sig, "buy_type": buy_type,
            "exit_signal": exit_sig,
            "regime": regime.get(d, "?"), "rv_pctile": rv,
            "signal_text": " + ".join(parts),
        })

    return pd.DataFrame(records).set_index("date")


def run_backtest(close_d, high_d, low_d,
                 upper_band, lower_band,
                 regime, rv_pctile,
                 gld_1h=None,  # 保留参数兼容, 现在不再用 (改 daily 粒度)
                 max_hold_days=MAX_HOLD_DAYS,
                 buy_bp=BUY_BP, exit_bp=EXIT_BP,
                 stop_loss_pct=STOP_LOSS_PCT,
                 pullback_gain=PULLBACK_GAIN,
                 pullback_dd=PULLBACK_DD,
                 consecutive_stop=CONSECUTIVE_STOP,
                 start_date=None,
                 entry_log_lookup=None,
                 exit_log_lookup=None,
                 entry_price_mode="log",
                 rv_filter=RV_FILTER_ENABLED,
                 rv_low=RV_FILTER_LOW,
                 rv_high=RV_FILTER_HIGH,
                 # 兼容旧 kwargs
                 exit_timeframe=None,
                 macd_min_gain=None):
    """真实策略回测 — 与 Dashboard 持仓管理 + 主图显示策略 1:1 一致.

    入场:
      条件: bp_low < buy_bp (开窗) + Bull regime
      入场价: 按 entry_price_mode 选 (默认 log 当日代表价)

    退出 (持仓中每天按以下顺序检查):
      1. StopLoss: 日内 low 跌破入场 -stop_loss_pct%
      2. BandExit: bp_high > exit_bp → 优先 log EXIT 代表价, 兜底 bp090
      3. Pullback: 持仓期峰值涨幅 > pullback_gain% 且当前回撤 >= pullback_dd%
         (与持仓管理 "止盈位" 列同公式)
      4. Timeout: 持仓 >= max_hold_days 天 (安全帽)

    风控: 连续 consecutive_stop 笔止损后暂停 (bp>0.5 恢复)

    Returns: list of trade dicts (含 entry_source / exit_source 标识).
    """
    bp_dates = upper_band.dropna().index.intersection(
        lower_band.dropna().index)
    if start_date:
        bp_dates = bp_dates[bp_dates >= start_date]

    trades = []
    in_trade = False
    entry_dt = entry_price = entry_type = None
    entry_source = None
    exit_source = None
    peak = 0
    consecutive_stops = 0

    def _lookup_log(lookup, d):
        if lookup is None or len(lookup) == 0:
            return None, 0
        d_norm = pd.Timestamp(d).normalize()
        if d_norm not in lookup.index:
            return None, 0
        row = lookup.loc[d_norm]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return float(row["price"]), int(row.get("n_triggers", 1))

    for d in bp_dates:
        u, l = upper_band.get(d, np.nan), lower_band.get(d, np.nan)
        if np.isnan(u) or np.isnan(l) or u <= l:
            continue
        c, h, lo = close_d[d], high_d[d], low_d[d]
        if np.isnan(c) or np.isnan(h) or np.isnan(lo):
            continue
        bp_lo = (lo - l) / (u - l)
        bp_hi = (h - l) / (u - l)
        bp090 = l + exit_bp * (u - l)
        is_bull = regime.get(d, "?") == "Bull"
        rv = rv_pctile.get(d, 0.5)

        # ── 退出 (持仓中) ──
        exit_type = exit_price = None
        if in_trade:
            peak = max(peak, h)

            # 1) StopLoss: 日内 low 跌破入场 -stop_loss_pct%
            if entry_price > 0 and \
               lo / entry_price - 1 < -stop_loss_pct / 100:
                exit_type = "StopLoss"
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                exit_source = "止损线"
            # 2) BandExit: bp_high > exit_bp
            elif bp_hi > exit_bp:
                _lp, _ln = _lookup_log(exit_log_lookup, d)
                if _lp is not None:
                    exit_type, exit_price = "BandExit", _lp
                    exit_source = f"盘中×{_ln}"
                else:
                    exit_type, exit_price = "BandExit", bp090
                    exit_source = "阈值"
            # 3) Pullback: 与持仓管理 "止盈位" 列同公式
            #    峰值涨幅 > pullback_gain% 且日内回撤 >= pullback_dd%
            elif entry_price > 0:
                gain_peak = (peak / entry_price - 1) * 100
                # 日内 low 相对峰值的回撤
                dd = (peak - lo) / peak * 100 if peak > 0 else 0
                if gain_peak > pullback_gain and dd >= pullback_dd:
                    exit_type = "Pullback"
                    # 止盈位 = peak * (1 - dd%); 用日内可能触及的最差情况
                    exit_price = peak * (1 - pullback_dd / 100)
                    exit_source = "止盈位"
            # 4) Timeout 安全帽
            if not exit_type and (d - entry_dt).days >= max_hold_days:
                exit_type, exit_price = "Timeout", c
                exit_source = "收盘 (超期)"

        if in_trade and exit_type:
            gain = (exit_price / entry_price - 1) * 100
            trades.append({
                "entry_date": entry_dt, "exit_date": d,
                "entry_price": entry_price, "exit_price": exit_price,
                "type": entry_type, "exit_type": exit_type,
                "gain": gain,
                "hold_days": (d - entry_dt).days,
                "peak": peak,
                "entry_source": entry_source or "—",
                "exit_source": exit_source or "—",
            })
            in_trade = False
            exit_source = None
            consecutive_stops = (consecutive_stops + 1
                                 if exit_type == "StopLoss" else 0)

        # ── 入场 ──
        if consecutive_stops >= consecutive_stop:
            if bp_hi > exit_bp or (c - l) / (u - l) > 0.50:
                consecutive_stops = 0
            else:
                continue

        # RV 极值过滤: 中位区间 (温水区) 跳过方向性
        rv_extreme = (rv < rv_low) or (rv > rv_high)
        if rv_filter and not rv_extreme:
            continue

        if not in_trade and is_bull and bp_lo < buy_bp:
            if rv_filter:
                entry_type = "BUY CALL" if rv < rv_low else "SELL PUT"
            else:
                entry_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"
            if entry_price_mode == "close":
                entry_price = c
                entry_source = "收盘"
            elif entry_price_mode == "high":
                entry_price = h
                entry_source = "最高"
            else:  # "log" 默认
                _lp, _ln = _lookup_log(entry_log_lookup, d)
                if _lp is not None:
                    entry_price = _lp
                    entry_source = f"盘中×{_ln}"
                else:
                    entry_price = c
                    entry_source = "收盘 (无 log)"
            entry_dt = d
            peak = h
            in_trade = True

    # 末尾若仍在持仓中, 标记为活跃持仓 (供 Dashboard 持仓管理使用)
    # 用 exit_date=None 区分: 已平仓的 trades 都有 exit_date.
    if in_trade:
        trades.append({
            "entry_date": entry_dt,
            "exit_date": None,
            "entry_price": entry_price,
            "exit_price": None,
            "type": entry_type,
            "exit_type": "ACTIVE",
            "gain": None,
            "hold_days": None,
            "peak": peak,
            "entry_source": entry_source or "—",
            "exit_source": None,
            "active": True,
        })

    return trades
