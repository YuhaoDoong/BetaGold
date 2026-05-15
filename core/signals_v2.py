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
                           rv_high=RV_FILTER_HIGH,
                           volume=None,
                           asset=None,
                           gvz_series=None):
    """asset 参数会查 strategy_config 覆盖默认阈值 (per-asset 校准).
    v3.7.117: gvz_series (可选) 接入 IV 三阶过滤.
    """
    _iv_cfg = None
    _sp_cfg = None
    if asset is not None:
        try:
            from core.strategy_config import get_config
            _ac = get_config(asset)
            rv_low = _ac.rv_filter_low
            rv_high = _ac.rv_filter_high
            buy_bp = _ac.buy_bp
            exit_bp = _ac.exit_bp
            if getattr(_ac, "iv_filter_enabled", False):
                _iv_cfg = _ac
            if getattr(_ac, "sp_score_enabled", False):
                _sp_cfg = _ac
        except Exception:
            pass

    # v3.7.123: 预算技术指标 (MACD hist / RSI 14 / Stoch %K) — 喂 sp_score
    _ind_macd = _ind_rsi = _ind_stoch_k = None
    if _sp_cfg is not None:
        try:
            from core.dir_indicators import rsi as _rsi_fn, macd as _macd_fn, \
                stoch_kd as _stoch_fn
            _ind_rsi = _rsi_fn(close_d, n=14)
            _, _, _ind_macd = _macd_fn(close_d)
            _ind_stoch_k, _ = _stoch_fn(high_d, low_d, close_d)
        except Exception:
            _sp_cfg = None  # 拿不到指标就退回单切

    # v3.7.127: ma_trend (MA20/MA50) 入场过滤 — 实证最强单因子分化
    # bc_entry_filter_test.py: ma_trend >= 0.99 单一过滤
    #   GLD wr 64% → 76% / 累计 +2997% → +3421%
    #   SLV wr 51% → 91% / 累计 +1611% → +2437%
    # 触发: ma_trend < 0.99 (MA20 < MA50, 下行趋势) 时 BC wr ~0-12%, 必跳过
    _ma_trend = None
    try:
        _ma20 = close_d.rolling(20).mean()
        _ma50 = close_d.rolling(50).mean()
        _ma_trend = _ma20 / _ma50
    except Exception:
        pass

    # v3.7.201: 信号双因子硬过滤 (rv_pctile_max + ret_20d_min)
    # signal_filter_deep.py 3y GLD grid: rv<0.75 + ret>-3% n=82/143 WR +5pp Q1 拦 19/20
    _ret_20d = None
    try:
        _ret_20d = close_d.pct_change(20)
    except Exception:
        pass
    """日线级别信号: v1.0 Band + H/L 触发 + RV 极值过滤.

    rv_filter=True 时只在 RV %tile < rv_low 或 > rv_high 时触发方向性.
    回测显示: 排除 25-75% 中位后大涨>3% 概率 21% → 32%, 大跌>3% 概率不变 6%.

    每天输出: Band 参数 + 买入/退出触发状态 + 阈值价位.
    """
    bp_dates = upper_band.dropna().index.intersection(
        lower_band.dropna().index)

    # v3.7.47: dir_indicators 计算 sizing 依据
    sizing_atr = None
    sizing_confirm = None
    try:
        from core.dir_indicators import atr_ratio_5_20, directional_confirm
        sizing_atr = atr_ratio_5_20(high_d, low_d, close_d)
        if volume is not None:
            sizing_confirm = directional_confirm(
                close_d, high_d, low_d, volume, side='BUY')
    except Exception:
        pass

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
        # v3.7.127/128: ma_trend (MA20/MA50) 过滤 — 下行趋势接飞刀概率高
        # 阈值 per-asset grid 最优: GLD 0.975 / SLV 0.99
        ma_trend_skip = False
        if buy_sig and _ma_trend is not None and d in _ma_trend.index:
            mt = float(_ma_trend.get(d, np.nan))
            mt_thr = 0.99
            if asset is not None:
                try:
                    from core.strategy_config import get_config
                    _ac = get_config(asset)
                    if getattr(_ac, "ma_trend_filter_enabled", True):
                        mt_thr = _ac.ma_trend_threshold
                    else:
                        mt_thr = -1  # 禁用过滤
                except Exception:
                    pass
            if not np.isnan(mt) and mt < mt_thr:
                buy_sig = False
                ma_trend_skip = True

        # v3.7.201: 双因子硬过滤 (rv_pctile_max + ret_20d_min)
        # GLD 3y grid: rv<0.75 + ret>-3% n=82/143 WR +5pp Q1 拦 19/20
        signal_hard_skip_reason = ""
        _r20_val = np.nan
        if _ret_20d is not None and d in _ret_20d.index:
            _r20_val = float(_ret_20d.get(d, np.nan))
        if buy_sig and asset is not None:
            try:
                from core.strategy_config import get_config
                _ac2 = get_config(asset)
                _rv_max = getattr(_ac2, "rv_pctile_max_hard", 1.0)
                _ret_min = getattr(_ac2, "ret_20d_min_hard", -1.0)
                _ret_max = getattr(_ac2, "ret_20d_max_hard", 100.0)
                if rv >= _rv_max:
                    buy_sig = False
                    signal_hard_skip_reason = f"rv_pct {rv:.2f}>={_rv_max} 跳过"
                elif not np.isnan(_r20_val) and _r20_val <= _ret_min:
                    buy_sig = False
                    signal_hard_skip_reason = (
                        f"ret_20d {_r20_val*100:+.1f}%<={_ret_min*100:.0f}% 跳过")
                elif not np.isnan(_r20_val) and _r20_val >= _ret_max:
                    # v3.7.214: 顶部追高过滤
                    buy_sig = False
                    signal_hard_skip_reason = (
                        f"ret_20d {_r20_val*100:+.1f}%>={_ret_max*100:.0f}% 顶部追高 跳过")
            except Exception:
                pass

        # v3.7.207: tier 计算延后到 IV filter 之后 (跟 buy_sig 最终状态一致)
        # 旧 v3.7.202: tier 在 IV filter 前算 → 3/17 tier=S 但 buy_signal=False 矛盾
        signal_tier = ""
        buy_type = None
        iv_filter_reason = ""  # v3.7.117 透明记录 IV 过滤原因
        sp_score = None
        sp_score_breakdown = ""
        if buy_sig:
            if rv_filter:
                # 低 RV → BUY CALL (期权便宜), 高 RV → SELL PUT (收 IV)
                buy_type = "BUY CALL" if rv < rv_low else "SELL PUT"
            else:
                buy_type = "BUY CALL" if rv <= 0.85 else "SELL PUT"
            # v3.7.123: SP score 多因子打分覆盖单切 (paired-grid 验证强信号)
            if _sp_cfg is not None:
                gvz_v = float(gvz_series.get(d, np.nan)) \
                    if (gvz_series is not None and d in gvz_series.index) else np.nan
                # 用 GVZ - RV(年化) 近似 IV-RV gap
                rv_abs = rv * 30.0  # rv_pctile 0-1 不等于绝对 IV; 仅启发用
                iv_rv_gap = (gvz_v - rv_abs) if not np.isnan(gvz_v) else 0.0
                rsi_v = float(_ind_rsi.get(d, np.nan)) if _ind_rsi is not None else np.nan
                macd_v = float(_ind_macd.get(d, np.nan)) if _ind_macd is not None else np.nan
                stoch_v = float(_ind_stoch_k.get(d, np.nan)) \
                    if _ind_stoch_k is not None else np.nan

                hits = []
                score = 0.0
                if iv_rv_gap > 0:
                    score += _sp_cfg.sp_score_w_iv_rv_gap
                    hits.append(f"IV-RV+{iv_rv_gap:.1f}*{_sp_cfg.sp_score_w_iv_rv_gap}")
                if bp_low < 0.05:
                    score += _sp_cfg.sp_score_w_bp_low_deep
                    hits.append(f"bp_low{bp_low:.2f}*{_sp_cfg.sp_score_w_bp_low_deep}")
                if bp_close < 0.30:
                    score += _sp_cfg.sp_score_w_bp_close_low
                    hits.append(f"bp_cl{bp_close:.2f}*{_sp_cfg.sp_score_w_bp_close_low}")
                if not np.isnan(gvz_v) and gvz_v >= 28:
                    score += _sp_cfg.sp_score_w_gvz_high
                    hits.append(f"GVZ{gvz_v:.0f}*{_sp_cfg.sp_score_w_gvz_high}")
                if not np.isnan(rsi_v) and rsi_v < 30:
                    score += _sp_cfg.sp_score_w_rsi_oversold
                    hits.append(f"RSI{rsi_v:.0f}*{_sp_cfg.sp_score_w_rsi_oversold}")
                if not np.isnan(stoch_v) and stoch_v < 40:
                    score += _sp_cfg.sp_score_w_stoch_low
                    hits.append(f"K{stoch_v:.0f}*{_sp_cfg.sp_score_w_stoch_low}")
                if not np.isnan(macd_v) and macd_v < -0.5:
                    score += _sp_cfg.sp_score_w_macd_bear
                    hits.append(f"MACD{macd_v:+.1f}*{_sp_cfg.sp_score_w_macd_bear}")

                sp_score = round(score, 2)
                sp_score_breakdown = " + ".join(hits) if hits else "无命中"
                buy_type = "SELL PUT" if score >= _sp_cfg.sp_score_threshold else "BUY CALL"
            # v3.7.117: GVZ IV 三阶过滤 (实证 BC 高 IV 全错向)
            if _iv_cfg is not None and gvz_series is not None and d in gvz_series.index:
                gvz = float(gvz_series.get(d, 0))
                if gvz >= _iv_cfg.iv_filter_high_min:
                    # 高 IV (>=28): 必需深破 + 强制 SP
                    if bp_low > _iv_cfg.iv_high_bp_low_max:
                        buy_sig = False
                        buy_type = None
                        iv_filter_reason = f"高IV {gvz:.0f} bp_low {bp_low:.2f}>{_iv_cfg.iv_high_bp_low_max} 跳过"
                    else:
                        if _iv_cfg.iv_high_force_sp:
                            buy_type = "SELL PUT"
                            iv_filter_reason = f"高IV {gvz:.0f} 深破 {bp_low:.2f} 强制 SP"
                elif gvz >= _iv_cfg.iv_filter_low_max:
                    # 中 IV (22-28): 二次确认 (用 sizing_confirm 已有 RSI/MACD/Stoch)
                    if _iv_cfg.iv_mid_dual_confirm and sizing_confirm is not None and d in sizing_confirm.index:
                        cnt = sizing_confirm['confirm_count'].get(d, 0)
                        if cnt < 2:  # 不足 2 个技术指标 align → skip
                            buy_sig = False
                            buy_type = None
                            iv_filter_reason = f"中IV {gvz:.0f} confirm {int(cnt)}/4<2 跳过"
                        else:
                            iv_filter_reason = f"中IV {gvz:.0f} confirm {int(cnt)}/4 通过"
                else:
                    iv_filter_reason = f"低IV {gvz:.0f} 正常"

        # v3.7.207: tier 计算延后到这里 — IV filter 之后, 跟 buy_sig 最终状态对齐
        # 旧 v3.7.202 tier 在 IV filter 前算 → 3/17 等"IV 拦下"信号仍标 S, 误导
        if buy_sig and asset is not None:
            try:
                from core.strategy_config import get_config
                _ac3 = get_config(asset)
                s_rv = getattr(_ac3, "tier_s_rv_max", 0.65)
                s_ret = getattr(_ac3, "tier_s_ret_20d_min", 0.0)
                s_bp = getattr(_ac3, "tier_s_bp_low_max", 0.20)
                a_rv = getattr(_ac3, "tier_a_rv_max", 0.75)
                a_ret = getattr(_ac3, "tier_a_ret_20d_min", -0.01)
                a_bp = getattr(_ac3, "tier_a_bp_low_max", 0.20)
                _r20_safe = _r20_val if not np.isnan(_r20_val) else 0.0
                if (rv < s_rv and _r20_safe > s_ret and bp_low <= s_bp):
                    signal_tier = "S"
                elif (rv < a_rv and _r20_safe > a_ret and bp_low <= a_bp):
                    signal_tier = "A"
                else:
                    signal_tier = "B"
            except Exception:
                signal_tier = "B"

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

        # v3.7.47: sizing 倍数 (基础 1×, 加分: ATR 收缩 +1×, 反转齐心 +2×)
        sizing = 1.0
        sizing_reasons = []
        if sizing_atr is not None and d in sizing_atr.index:
            atr_v = sizing_atr.get(d, np.nan)
            if not np.isnan(atr_v) and atr_v < 1.0:
                sizing += 1.0
                sizing_reasons.append(f"ATR收缩({atr_v:.2f})")
        if sizing_confirm is not None and d in sizing_confirm.index:
            cnt = sizing_confirm['confirm_count'].get(d, 0)
            if cnt >= 3:
                sizing += 2.0
                sizing_reasons.append(f"反转齐心({int(cnt)}/4)")
            elif cnt >= 2:
                sizing += 1.0
                sizing_reasons.append(f"反转部分({int(cnt)}/4)")

        records.append({
            "date": d, "close": c, "high": h, "low": lo,
            "upper": ub, "lower": lb,
            "bp_close": bp_close, "bp_low": bp_low, "bp_high": bp_high,
            "bp030_price": bp030, "bp090_price": bp090,
            "buy_signal": buy_sig, "buy_type": buy_type,
            "exit_signal": exit_sig,
            "regime": regime.get(d, "?"), "rv_pctile": rv,
            "signal_text": " + ".join(parts),
            "sizing": round(sizing, 1),
            "sizing_reasons": ", ".join(sizing_reasons) if sizing_reasons else "",
            "iv_filter_reason": iv_filter_reason,
            "sp_score": sp_score,
            "sp_score_breakdown": sp_score_breakdown,
            "ma_trend": float(_ma_trend.get(d, np.nan))
                         if (_ma_trend is not None and d in _ma_trend.index)
                         else np.nan,
            "ma_trend_skip": ma_trend_skip,
            "signal_hard_skip_reason": signal_hard_skip_reason,
            "ret_20d": float(_ret_20d.get(d, np.nan))
                         if (_ret_20d is not None and d in _ret_20d.index)
                         else np.nan,
            "signal_tier": signal_tier,  # v3.7.202: S/A/B
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
                 asset=None,
                 # 兼容旧 kwargs
                 exit_timeframe=None,
                 macd_min_gain=None):
    if asset is not None:
        try:
            from core.strategy_config import get_config
            _ac = get_config(asset)
            rv_low = _ac.rv_filter_low
            rv_high = _ac.rv_filter_high
            buy_bp = _ac.buy_bp
            exit_bp = _ac.exit_bp
            stop_loss_pct = _ac.stop_loss_pct
            pullback_gain = _ac.pullback_gain
            pullback_dd = _ac.pullback_dd
            consecutive_stop = _ac.consecutive_stop
            max_hold_days = _ac.max_hold_days
        except Exception:
            pass
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
