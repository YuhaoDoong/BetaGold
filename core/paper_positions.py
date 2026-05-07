"""Paper-trading 持仓管理 — 每个盘中触发模拟一次交易, 累计到持仓.

数据流:
  1. detect_triggers 输出 BUY/EXIT triggers (来自 intraday log)
  2. open_position(asset, side, trigger_time, ul_price, strategy, option_symbol=None)
     - 写入 paper_positions.parquet, status=OPEN
     - 如有 option_symbol 同步获取 entry option price
  3. close_position(asset, exit_time, ul_price)
     - 找匹配的 OPEN 持仓 (FIFO), 写 exit_*, status=CLOSED
  4. mark_to_market(open_positions) — dashboard 实时计算未实现 P&L
     - 期货策略: 直接用当前 spot vs entry
     - 期权策略: 当前期权 quote vs entry option price (回退 spot 模型估算)

存储 schema (paper_positions.parquet):
  trade_id (uuid), asset, strategy, side, qty,
  open_time (UTC), open_ul_price, open_option_symbol, open_option_price,
  close_time, close_ul_price, close_option_price,
  status (OPEN/CLOSED), realized_pnl_pct, source (intraday_log)

策略推断 — 从当日 chosen 决定 option contract:
  BUY CALL  → ATM call, ~45 DTE
  SELL PUT  → 30Δ put, ~45 DTE
  STRADDLE  → ATM call+put, ~14 DTE
  SHORT_VOL → IC 1.6σ short / 3σ wing
  无 chosen → 仅记 underlying spot
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

POSITIONS_FILENAME = "paper_positions.parquet"


@dataclass
class PaperPosition:
    trade_id: str
    asset: str
    strategy: str           # BUY CALL / SELL PUT / STRADDLE / SHORT_VOL / SPOT
    side: str               # BUY / EXIT
    qty: int
    open_time: pd.Timestamp
    open_ul_price: float
    open_option_symbol: Optional[str]
    open_option_price: Optional[float]
    close_time: Optional[pd.Timestamp] = None
    close_ul_price: Optional[float] = None
    close_option_price: Optional[float] = None
    status: str = "OPEN"
    realized_pnl_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "strategy": self.strategy,
            "side": self.side,
            "qty": self.qty,
            "open_time": self.open_time,
            "open_ul_price": self.open_ul_price,
            "open_option_symbol": self.open_option_symbol,
            "open_option_price": self.open_option_price,
            "close_time": self.close_time,
            "close_ul_price": self.close_ul_price,
            "close_option_price": self.close_option_price,
            "status": self.status,
            "realized_pnl_pct": self.realized_pnl_pct,
        }


def _path(data_root: str) -> str:
    return os.path.join(data_root, POSITIONS_FILENAME)


def load_positions(data_root: str) -> pd.DataFrame:
    p = _path(data_root)
    if not os.path.exists(p):
        return pd.DataFrame(columns=[
            "trade_id", "asset", "strategy", "side", "qty",
            "open_time", "open_ul_price",
            "open_option_symbol", "open_option_price",
            "close_time", "close_ul_price", "close_option_price",
            "status", "realized_pnl_pct",
        ])
    return pd.read_parquet(p)


def save_positions(df: pd.DataFrame, data_root: str) -> None:
    df = df.copy()
    # 强制类型 — pyarrow 不接受混合 dtype
    for c in ["open_time", "close_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["open_ul_price", "open_option_price",
              "close_ul_price", "close_option_price",
              "realized_pnl_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(_path(data_root), index=False)


def open_position(data_root: str, asset: str, strategy: str,
                  trigger_time: pd.Timestamp, ul_price: float,
                  side: str = "BUY", qty: int = 1,
                  option_symbol: Optional[str] = None,
                  option_price: Optional[float] = None) -> PaperPosition:
    """开仓 (幂等: 相同 asset+side+trigger_time+strategy 只记一次)."""
    df = load_positions(data_root)
    # 幂等检查
    if len(df) > 0:
        dup = df[
            (df["asset"] == asset)
            & (df["side"] == side)
            & (df["strategy"] == strategy)
            & (pd.to_datetime(df["open_time"]) == pd.Timestamp(trigger_time))
        ]
        if len(dup) > 0:
            row = dup.iloc[0]
            return PaperPosition(**{k: row[k] for k in
                                     ["trade_id", "asset", "strategy", "side",
                                      "qty", "open_time", "open_ul_price",
                                      "open_option_symbol", "open_option_price",
                                      "close_time", "close_ul_price",
                                      "close_option_price", "status",
                                      "realized_pnl_pct"] if k in row})
    pos = PaperPosition(
        trade_id=str(uuid.uuid4())[:8],
        asset=asset, strategy=strategy, side=side, qty=qty,
        open_time=pd.Timestamp(trigger_time),
        open_ul_price=float(ul_price),
        open_option_symbol=option_symbol,
        open_option_price=(float(option_price)
                            if option_price is not None else None),
    )
    df = pd.concat([df, pd.DataFrame([pos.to_dict()])], ignore_index=True)
    save_positions(df, data_root)
    return pos


def close_position(data_root: str, asset: str, exit_time: pd.Timestamp,
                   ul_price: float,
                   option_price: Optional[float] = None,
                   skip_strategies: tuple = ("STRADDLE", "SHORT_VOL")) -> int:
    """平仓 BUY 持仓 (不平 STRADDLE/SHORT_VOL — 它们走 DTE/profit 自 exit).
    返回平仓笔数.
    """
    df = load_positions(data_root)
    if len(df) == 0:
        return 0
    mask = (df["asset"] == asset) & (df["status"] == "OPEN") & (df["side"] == "BUY")
    # v3.7.90: STRADDLE/SHORT_VOL 跟方向性 EXIT 触发分离
    if skip_strategies:
        _skip_mask = df["strategy"].astype(str).apply(
            lambda s: any(skip in s for skip in skip_strategies))
        mask = mask & ~_skip_mask
    if not mask.any():
        return 0
    n_closed = 0
    for idx in df.index[mask]:
        df.loc[idx, "close_time"] = pd.Timestamp(exit_time)
        df.loc[idx, "close_ul_price"] = float(ul_price)
        if option_price is not None:
            df.loc[idx, "close_option_price"] = float(option_price)
        df.loc[idx, "status"] = "CLOSED"
        # 计算 realized P&L
        entry = df.loc[idx, "open_option_price"]
        exit_p = option_price
        if entry and exit_p and entry > 0:
            df.loc[idx, "realized_pnl_pct"] = float(
                (exit_p / entry - 1) * 100)
        else:
            entry_ul = df.loc[idx, "open_ul_price"]
            strategy = df.loc[idx, "strategy"]
            sign = 1 if strategy in ("BUY CALL", "SPOT") else -1
            if entry_ul > 0:
                df.loc[idx, "realized_pnl_pct"] = float(
                    sign * (ul_price / entry_ul - 1) * 100)
        n_closed += 1
    save_positions(df, data_root)
    return n_closed


def _strategy_pnl_formula(strategy: str, spot_ratio: float) -> float:
    """根据 strategy 推 P&L (% of underlying move).
    spot_ratio = current_spot / entry_spot - 1 (e.g. 0.02 = +2%)
    返回 P&L %.
    delta 近似 (粗模型, 仅作 MTM 显示用):
      BUY CALL ≈ +50% delta   (ATM call ATM spot move 1% ≈ +0.5% NAV)
      SELL PUT ≈ +30% delta   (反 short put 价值随 spot ↑)
      STRADDLE ≈ ±50% (long vol, |move| × 2 - theta)
      SHORT_VOL ≈ ∓50% (short vol)
      FUTURES_LONG / SPOT ≈ 100% delta
    """
    move_pct = spot_ratio * 100
    s = strategy.upper()
    if s in ("BUY CALL", "BUY_CALL"):
        return move_pct * 0.5 / 1.0  # 简化, 实际 OTM 杠杆更高
    if s in ("SELL PUT", "SELL_PUT"):
        return move_pct * 0.3
    if s == "STRADDLE":
        # long vol: 任一方向都赚 (近似)
        return abs(move_pct) * 1.0 - 0.3  # 减 theta 衰减估算
    if s in ("SHORT_VOL", "SHORT VOL"):
        return -abs(move_pct) * 0.8 + 0.5  # short vol + theta 收
    return move_pct  # FUTURES_LONG / SPOT 1:1


def mark_to_market(data_root: str, current_quotes: dict) -> pd.DataFrame:
    """对 OPEN 持仓计算未实现 P&L.
    current_quotes: {asset: spot_price, option_symbol: option_price, ...}
    返回 OPEN positions 的 DataFrame, 加 列: current_ul / current_option / unrealized_pct
    """
    df = load_positions(data_root)
    if len(df) == 0:
        return df
    open_df = df[df["status"] == "OPEN"].copy()
    if not len(open_df):
        return open_df
    open_df["current_ul"] = open_df["asset"].map(current_quotes)
    open_df["current_option"] = open_df["open_option_symbol"].map(current_quotes)
    rows = []
    for _, row in open_df.iterrows():
        entry_opt = row["open_option_price"]
        cur_opt = row.get("current_option")
        entry_ul = row["open_ul_price"]
        cur_ul = row.get("current_ul")
        if pd.notna(cur_opt) and pd.notna(entry_opt) and entry_opt > 0:
            # 直接用 option quote (live)
            unr = (cur_opt / entry_opt - 1) * 100
        elif pd.notna(cur_ul) and entry_ul and entry_ul > 0:
            # 退化用 underlying-move + strategy delta 近似
            unr = _strategy_pnl_formula(row["strategy"],
                                          cur_ul / entry_ul - 1)
        else:
            unr = float("nan")
        rec = row.to_dict(); rec["unrealized_pct"] = unr
        rows.append(rec)
    return pd.DataFrame(rows)


def bs_price(spot: float, strike: float, T_years: float,
              r: float, sigma: float, right: str) -> float:
    """Black-Scholes 期权定价 (美式 ETF 视为欧式估算).
    right: 'C' / 'P'
    """
    import math
    from scipy.stats import norm
    if T_years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        intrinsic = (max(spot - strike, 0)
                       if right.upper() == "C" else max(strike - spot, 0))
        return float(intrinsic)
    d1 = ((math.log(spot / strike) + (r + 0.5 * sigma**2) * T_years)
            / (sigma * math.sqrt(T_years)))
    d2 = d1 - sigma * math.sqrt(T_years)
    if right.upper() == "C":
        return float(spot * norm.cdf(d1)
                      - strike * math.exp(-r * T_years) * norm.cdf(d2))
    return float(strike * math.exp(-r * T_years) * norm.cdf(-d2)
                  - spot * norm.cdf(-d1))


def estimate_straddle_premium(spot: float, strike: float,
                                 dte_days: int, iv: float = 0.20,
                                 r: float = 0.04) -> tuple[float, float]:
    """估 ATM straddle (call + put) 入场总权利金.
    返回 (call_price, put_price).
    """
    T = max(dte_days, 1) / 365.0
    return (bs_price(spot, strike, T, r, iv, "C"),
            bs_price(spot, strike, T, r, iv, "P"))


def find_active_expiry_near(asset: str, target_date: pd.Timestamp,
                              tolerance_days: int = 7) -> Optional[str]:
    """从 yfinance 当前 expiry 列表找最接近 target_date 的 (YYYY-MM-DD).
    """
    try:
        import yfinance as yf
        opts = yf.Ticker(asset).options
        if not opts: return None
        target = pd.Timestamp(target_date).normalize()
        best = None; best_d = 999
        for o in opts:
            d = abs((pd.Timestamp(o) - target).days)
            if d < best_d:
                best_d = d; best = o
        if best_d > tolerance_days: return None
        return best
    except Exception:
        return None


def fetch_chain_atm_premium(asset: str, expiry_str: str,
                              spot: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """拉 option_chain ATM call + put lastPrice. 返回 (call, put, strike).
    """
    try:
        import yfinance as yf
        chain = yf.Ticker(asset).option_chain(expiry_str)
        # ATM strike = closest to spot
        for df, _name in [(chain.calls, "calls"), (chain.puts, "puts")]:
            df["_dist"] = (df["strike"] - spot).abs()
        atm_strike = chain.calls.loc[chain.calls["_dist"].idxmin(), "strike"]
        call_row = chain.calls[chain.calls["strike"] == atm_strike]
        put_row = chain.puts[chain.puts["strike"] == atm_strike]
        c_p = (float(call_row.iloc[0]["lastPrice"])
                if len(call_row) and call_row.iloc[0]["lastPrice"] > 0 else None)
        p_p = (float(put_row.iloc[0]["lastPrice"])
                if len(put_row) and put_row.iloc[0]["lastPrice"] > 0 else None)
        return (c_p, p_p, float(atm_strike))
    except Exception:
        return (None, None, None)


_KLINE_DB_PATH = "/Users/yhdong/Gold/data/raw/options_history/kline_db/all_klines.parquet"
_KLINE_DB_CACHE: Optional[pd.DataFrame] = None
_KLINE_DB_MTIME: Optional[float] = None  # v3.7.156: 文件 mtime 守卫
_KLINE_DB_SIZE: Optional[int] = None     # v3.7.157: 文件 size 双重守卫


def _load_kline_db() -> Optional[pd.DataFrame]:
    """加载 EOD 期权 OHLC kline_db (cached)."""
    # v3.7.156: 加 mtime 守卫 — 文件变 → cache 自动重载
    global _KLINE_DB_CACHE, _KLINE_DB_MTIME
    import os
    if not os.path.exists(_KLINE_DB_PATH):
        return None
    cur_mtime = os.path.getmtime(_KLINE_DB_PATH)
    if _KLINE_DB_CACHE is not None and _KLINE_DB_MTIME == cur_mtime:
        return _KLINE_DB_CACHE
    df = pd.read_parquet(_KLINE_DB_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    _KLINE_DB_CACHE = df
    _KLINE_DB_MTIME = cur_mtime
    return df


def pick_liquid_monthly_option(asset: str, signal_date: pd.Timestamp,
                                  ul_price: float, right: str,
                                  dte_target: int = 45,
                                  strike_tolerance: float = 8.0) -> Optional[dict]:
    """从 kline_db 挑期权: strike 接近 ul_price 优先 (覆盖 ATM), 月度 opex 偏好,
    DTE 接近 target, 成交量降序.

    v3.7.96: 重排优先级避开 strike-out-of-coverage bug. 之前 expiry 优先时
    可能选到 strikes 不覆盖 ATM 的 expiry (e.g. 3-5 GLD spot \$466 选 4-17
    expiry 但 strikes 只到 \$320 → 实际应选 5-15 expiry 的 \$465 strike).
    """
    db = _load_kline_db()
    if db is None: return None
    sig_d = pd.Timestamp(signal_date).normalize()
    day_data = db[db["date"] == sig_d]
    # v3.7.145: 没今日 EOD options 数据时, fallback 到最近可用日
    # 用 user 模型: 信号一触发就该开仓, 即使 kline_db 一两日滞后
    if not len(day_data):
        avail_dates = db.loc[db["date"] <= sig_d, "date"]
        if not len(avail_dates): return None
        nearest = avail_dates.max()
        day_data = db[db["date"] == nearest]
        if not len(day_data): return None
    asset_data = day_data[day_data["code"].str.contains(asset, na=False)]
    if not len(asset_data): return None
    type_str = "CALL" if right.upper() == "C" else "PUT"
    pool = asset_data[asset_data["option_type"] == type_str].copy()
    if not len(pool): return None
    # 1. Strike 必须接近 ul_price (核心硬过滤)
    pool["strike_diff"] = (pool["strike"] - ul_price).abs()
    near = pool[pool["strike_diff"] <= strike_tolerance]
    if not len(near):
        # 扩到最接近 5 个 strike
        near = pool.nsmallest(5, "strike_diff")
    if not len(near): return None
    near = near.copy()
    # 2. 月度 opex 优先 (第三周五)
    near["is_monthly"] = near["expiry"].apply(
        lambda d: 1 if (d.weekday() == 4 and 15 <= d.day <= 21) else 0)
    monthly = near[near["is_monthly"] == 1]
    if len(monthly):
        near = monthly
    # 3. DTE 接近 target
    near["dte_diff"] = (near["dte_at_date"] - dte_target).abs()
    near = near[near["dte_diff"] <= 20]
    if not len(near):
        near = monthly if len(monthly) else pool[pool["strike_diff"] <= strike_tolerance]
        if not len(near): return None
        near = near.copy()
        near["dte_diff"] = (near["dte_at_date"] - dte_target).abs()
    # 4. 综合排序: strike_diff 升序, dte_diff 升序, volume 降序
    near = near.sort_values(["strike_diff", "dte_diff", "volume"],
                              ascending=[True, True, False])
    row = near.iloc[0]
    return {
        "code": row["code"],
        "strike": float(row["strike"]),
        "expiry": row["expiry"].strftime("%Y-%m-%d"),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "dte_at_date": int(row["dte_at_date"]),
        "volume": float(row["volume"]),
    }


def interpolate_option_intraday(option_info: dict,
                                  spot_open: float, spot_close: float,
                                  spot_intraday: float,
                                  spot_high: Optional[float] = None,
                                  spot_low: Optional[float] = None) -> float:
    """用 daily option O/C + spot 在 daily range 中位置插值 intraday 期权价.

    精度: 假设期权随 spot 线性 (delta-locked). intraday 内 IV/gamma 微动忽略.
    误差: 单日 spot 移动 < 2% 时 < 5%; spot 大幅 (3%+) 时可能偏 5-10%.

    优于 BS+假IV 估算 (后者偏 20%+).
    """
    if abs(spot_close - spot_open) < 1e-6:
        # 平日 - 用 H/L 范围
        if spot_high and spot_low and spot_high > spot_low:
            ratio = (spot_intraday - spot_low) / (spot_high - spot_low)
            ratio = max(0.0, min(1.0, ratio))
            return float(option_info["low"]
                          + ratio * (option_info["high"] - option_info["low"]))
        return float(option_info["close"])  # fallback: 收盘价
    ratio = (spot_intraday - spot_open) / (spot_close - spot_open)
    ratio = max(0.0, min(1.0, ratio))  # clip 0~1
    return float(option_info["open"]
                  + ratio * (option_info["close"] - option_info["open"]))


def price_strategy_at(asset: str, strategy: str,
                         signal_date: pd.Timestamp,
                         trigger_time: pd.Timestamp,
                         spot_at_trigger: float,
                         daily_O: float, daily_C: float,
                         daily_H: Optional[float] = None,
                         daily_L: Optional[float] = None,
                         dte_target: int = 45) -> dict:
    """统一定价: 拉 kline_db 找最佳合约, 插值算 trigger 时点期权价.

    BUY CALL  → ATM long call
    SELL PUT  → put credit spread (-ATM/+ -5%)
    STRADDLE  → ATM long call + long put (combined)
    其他 → SPOT
    返回 {legs, entry_price, daily_open_price, daily_close_price, kline_codes, source}
    """
    out = {"legs": [], "entry_price": 0.0, "daily_open_price": 0.0,
           "daily_close_price": 0.0, "kline_codes": [], "source": "—"}
    if "BUY CALL" in strategy:
        # v3.7.97: 根据"买点深度+IV"决定单腿还是价差
        # 单腿 BUY CALL: 买点深 (spot 低) + IV 小 → 期权便宜, 杠杆好
        # 否则 bull call spread (+ATM call / -OTM +5% call): 期权贵时降本
        # 简化判定: 若 use_spread=True (调用方传入) → spread; 否则 single
        _use_spread = bool(out.get("force_spread", False))
        c = pick_liquid_monthly_option(asset, signal_date, spot_at_trigger,
                                          "C", dte_target)
        if c and not _use_spread:
            entry = interpolate_option_intraday(c, daily_O, daily_C,
                                                   spot_at_trigger,
                                                   daily_H, daily_L)
            _e = c["expiry"][5:].replace("-", "/")
            out.update(legs=[("long_call", c["code"], c["strike"], 1)],
                        entry_price=entry,
                        leg_prices=[("long_call", entry)],
                        daily_open_price=c["open"],
                        daily_close_price=c["close"],
                        kline_codes=[c["code"]],
                        source=f"C${c['strike']:.0f} ({_e})")
        elif c:  # bull call spread
            short_K = round(c["strike"] * 1.05)
            sc = pick_liquid_monthly_option(asset, signal_date, short_K,
                                               "C", dte_target)
            if sc and sc["strike"] > c["strike"]:
                lc_intra = interpolate_option_intraday(c, daily_O, daily_C,
                                                            spot_at_trigger,
                                                            daily_H, daily_L)
                sc_intra = interpolate_option_intraday(sc, daily_O, daily_C,
                                                            spot_at_trigger,
                                                            daily_H, daily_L)
                debit = lc_intra - sc_intra  # 净付出
                _e = c["expiry"][5:].replace("-", "/")
                out.update(legs=[("long_call", c["code"], c["strike"], 1),
                                  ("short_call", sc["code"], sc["strike"], -1)],
                            entry_price=debit,
                            leg_prices=[("long_call", lc_intra),
                                         ("short_call", sc_intra)],
                            daily_open_price=c["open"] - sc["open"],
                            daily_close_price=c["close"] - sc["close"],
                            kline_codes=[c["code"], sc["code"]],
                            source=f"+C${c['strike']:.0f}/-C${sc['strike']:.0f} ({_e})")
    elif "SHORT_VOL" in strategy:
        # v3.7.141: SHORT_VOL 真 Iron Condor 4-leg (-ATM put / +OTM put / -ATM call / +OTM call)
        # 前版误把 SHORT_VOL 当 SP credit spread (只 put 边). 现在分开:
        #   put side:  short ~ATM-3%, long ~ATM-7% (wing)
        #   call side: short ~ATM+3%, long ~ATM+7% (wing)
        # 收双边 credit, 上下各有保护 wing, 横盘获最大 profit.
        sp_strike = round(spot_at_trigger * 0.97)  # short put -3%
        lp_strike = round(spot_at_trigger * 0.93)  # long put -7%
        sc_strike = round(spot_at_trigger * 1.03)  # short call +3%
        lc_strike = round(spot_at_trigger * 1.07)  # long call +7%
        sp = pick_liquid_monthly_option(asset, signal_date, sp_strike, "P", dte_target)
        lp = pick_liquid_monthly_option(asset, signal_date, lp_strike, "P", dte_target)
        sc = pick_liquid_monthly_option(asset, signal_date, sc_strike, "C", dte_target)
        lc = pick_liquid_monthly_option(asset, signal_date, lc_strike, "C", dte_target)
        if sp and lp and sc and lc:
            sp_intra = interpolate_option_intraday(sp, daily_O, daily_C,
                                                        spot_at_trigger, daily_H, daily_L)
            lp_intra = interpolate_option_intraday(lp, daily_O, daily_C,
                                                        spot_at_trigger, daily_H, daily_L)
            sc_intra = interpolate_option_intraday(sc, daily_O, daily_C,
                                                        spot_at_trigger, daily_H, daily_L)
            lc_intra = interpolate_option_intraday(lc, daily_O, daily_C,
                                                        spot_at_trigger, daily_H, daily_L)
            put_credit = sp_intra - lp_intra
            call_credit = sc_intra - lc_intra
            credit = put_credit + call_credit
            _e = sp["expiry"][5:].replace("-", "/")
            out.update(legs=[("short_put", sp["code"], sp["strike"], -1),
                              ("long_put", lp["code"], lp["strike"], 1),
                              ("short_call", sc["code"], sc["strike"], -1),
                              ("long_call", lc["code"], lc["strike"], 1)],
                        entry_price=credit,
                        leg_prices=[("short_put", sp_intra), ("long_put", lp_intra),
                                     ("short_call", sc_intra), ("long_call", lc_intra)],
                        daily_open_price=(sp["open"] - lp["open"]
                                            + sc["open"] - lc["open"]),
                        daily_close_price=(sp["close"] - lp["close"]
                                             + sc["close"] - lc["close"]),
                        kline_codes=[sp["code"], lp["code"], sc["code"], lc["code"]],
                        source=(f"IC -P${sp['strike']:.0f}/+P${lp['strike']:.0f}"
                                 f" -C${sc['strike']:.0f}/+C${lc['strike']:.0f} ({_e})"))
        return out
    elif "SELL PUT" in strategy:
        sp = pick_liquid_monthly_option(asset, signal_date, spot_at_trigger,
                                           "P", dte_target)
        # OTM put -5% (流动性容差)
        long_strike = round(spot_at_trigger * 0.95)
        lp = pick_liquid_monthly_option(asset, signal_date, long_strike,
                                           "P", dte_target)
        # 防重: 长腿 strike 必须低于短腿
        if sp and lp and lp["strike"] >= sp["strike"]:
            # 强制选低 strike — 重新过滤
            db = _load_kline_db()
            if db is not None:
                day_data = db[(db["date"] == pd.Timestamp(signal_date).normalize())
                                & (db["code"].str.contains(asset, na=False))
                                & (db["option_type"] == "PUT")
                                & (db["strike"] < sp["strike"])]
                if len(day_data):
                    day_data = day_data.copy()
                    day_data["dte_diff"] = (day_data["dte_at_date"] - dte_target).abs()
                    day_data["target_dist"] = (day_data["strike"] - long_strike).abs()
                    day_data = day_data.sort_values(["dte_diff", "target_dist"]).head(1)
                    r = day_data.iloc[0]
                    lp = {"code": r["code"], "strike": float(r["strike"]),
                          "expiry": r["expiry"].strftime("%Y-%m-%d"),
                          "open": float(r["open"]), "high": float(r["high"]),
                          "low": float(r["low"]), "close": float(r["close"]),
                          "dte_at_date": int(r["dte_at_date"]),
                          "volume": float(r["volume"])}
        if sp and lp:
            sp_intra = interpolate_option_intraday(sp, daily_O, daily_C,
                                                       spot_at_trigger,
                                                       daily_H, daily_L)
            lp_intra = interpolate_option_intraday(lp, daily_O, daily_C,
                                                       spot_at_trigger,
                                                       daily_H, daily_L)
            credit = sp_intra - lp_intra
            _e = sp["expiry"][5:].replace("-", "/")
            out.update(legs=[("short_put", sp["code"], sp["strike"], -1),
                              ("long_put", lp["code"], lp["strike"], 1)],
                        entry_price=credit,
                        leg_prices=[("short_put", sp_intra),
                                     ("long_put", lp_intra)],
                        daily_open_price=sp["open"] - lp["open"],
                        daily_close_price=sp["close"] - lp["close"],
                        kline_codes=[sp["code"], lp["code"]],
                        source=f"-P${sp['strike']:.0f}/+P${lp['strike']:.0f} ({_e})")
    elif "STRADDLE" in strategy:
        c = pick_liquid_monthly_option(asset, signal_date, spot_at_trigger,
                                          "C", 14)  # STRADDLE 短 DTE
        p = pick_liquid_monthly_option(asset, signal_date, spot_at_trigger,
                                          "P", 14)
        if c and p:
            c_intra = interpolate_option_intraday(c, daily_O, daily_C,
                                                       spot_at_trigger,
                                                       daily_H, daily_L)
            p_intra = interpolate_option_intraday(p, daily_O, daily_C,
                                                       spot_at_trigger,
                                                       daily_H, daily_L)
            total = c_intra + p_intra
            _e = c["expiry"][5:].replace("-", "/")
            out.update(legs=[("long_call", c["code"], c["strike"], 1),
                              ("long_put", p["code"], p["strike"], 1)],
                        entry_price=total,
                        leg_prices=[("long_call", c_intra),
                                     ("long_put", p_intra)],
                        daily_open_price=c["open"] + p["open"],
                        daily_close_price=c["close"] + p["close"],
                        kline_codes=[c["code"], p["code"]],
                        source=f"C${c['strike']:.0f}+P${p['strike']:.0f} ({_e})")
    elif "FUTURES" in strategy:
        # v3.7.107: 期货多头 — 直接 spot, 无 leg/kline_db
        out.update(legs=[("futures_long", f"{asset}_FUT", spot_at_trigger, 1)],
                    entry_price=spot_at_trigger,
                    leg_prices=[("futures_long", spot_at_trigger)],
                    daily_open_price=daily_O,
                    daily_close_price=daily_C,
                    kline_codes=[],
                    source=f"{asset} 期货多头 @ ${spot_at_trigger:.2f}")
    return out


def simulate_option_exit(entry_pricing: dict, signal_date: pd.Timestamp,
                            strategy: str,
                            today_dt: pd.Timestamp,
                            live_spot: float = None,
                            live_high: float = None,
                            live_low: float = None) -> dict:
    """逐日扫 kline_db 真实期权 OHLC, 应用真实退出规则.

    SELL PUT credit spread:
      +50% (cur_credit <= entry × 0.5) — 早平
      -100% (cur_credit >= entry × 2) — 止损 (full credit lost)
      expiry — 强平
    BUY CALL:
      +100% / -50% / expiry
    STRADDLE:
      +100% / 持仓 14d / expiry
    SHORT_VOL:
      +50% / -50% / 30d / expiry

    返回 {is_closed, exit_date, exit_value, exit_reason, pnl_pct}
    """
    # v3.7.132: FUTURES_LONG 委托 core/strategies/futures_long 模块
    # 替代内联逻辑, 走独立模块 (爆仓-100% / leverage 参数化 / SL 自动收紧)
    if "FUTURES" in strategy:
        try:
            from core.strategies.futures_long import (
                simulate_long_position, FuturesConfig)
            asset_key = entry_pricing["legs"][0][1].split("_")[0]
            csv_p = f"/Users/yhdong/Gold/data/raw/market/{asset_key.lower()}.csv"
            spot_df = pd.read_csv(csv_p, index_col=0, parse_dates=True)
            sig_d = pd.Timestamp(signal_date).normalize()
            entry_value = entry_pricing["entry_price"]
            cfg = FuturesConfig(leverage=20)  # Binance XAUUSDT default
            # v3.7.134: 期货 24h 可交易, 用 live spot/high/low 检查 intraday 退出
            res = simulate_long_position(sig_d, entry_value, spot_df, today_dt, cfg,
                                              live_spot=live_spot,
                                              live_high=live_high,
                                              live_low=live_low)
            if res.get("closed"):
                return {"is_closed": True, "exit_date": res["exit_date"],
                         "exit_value": res["exit_price"],
                         "exit_reason": res["reason"],
                         "pnl_pct": res["ret_levered_pct"],  # ROI on margin (lev)
                         "is_liquidation": res.get("is_liquidation", False),
                         "leg_prices": [("futures_long", res["exit_price"])]}
            # OPEN — 用 live MTM 而非上日 close
            cur = res.get("exit_price") or float(spot_df["Close"].iloc[-1])
            return {"is_closed": False, "current_value": cur,
                     "hold_days": res.get("hold_days", 0),
                     "pnl_pct": res.get("ret_levered_pct",
                                          (cur / entry_value - 1) * 100 * cfg.leverage),
                     "leg_prices": [("futures_long", cur)]}
        except Exception as e:
            return {"is_closed": False, "reason": f"futures sim err: {e}"}
    # v3.7.132: 委托给独立模块 (BC / SP / STRADDLE / SHORT_VOL)
    db = _load_kline_db()
    if db is None or not entry_pricing.get("legs"):
        return {"is_closed": False}
    try:
        if "BUY CALL" in strategy:
            from core.strategies.buy_call import simulate_bc_position
            return simulate_bc_position(entry_pricing, signal_date, today_dt, db)
        if "SELL PUT" in strategy:
            from core.strategies.sell_put import simulate_sp_position
            return simulate_sp_position(entry_pricing, signal_date, today_dt, db)
        if "STRADDLE" in strategy:
            from core.strategies.straddle import simulate_straddle_position
            return simulate_straddle_position(entry_pricing, signal_date, today_dt, db)
        if "SHORT_VOL" in strategy:
            from core.strategies.short_vol import simulate_short_vol_position
            return simulate_short_vol_position(entry_pricing, signal_date, today_dt, db)
    except Exception as e:
        # fallback 到旧 inline 逻辑 (debug 用)
        pass
    # ── 兜底: 旧 inline 逻辑 (保留作为回退) ──
    entry_value = entry_pricing["entry_price"]
    legs = entry_pricing["legs"]  # [(label, code, K, qty), ...]
    is_credit = "SELL PUT" in strategy or "SHORT_VOL" in strategy
    is_long_vol = "STRADDLE" in strategy
    is_long_dir = "BUY CALL" in strategy
    # v3.7.120: credit spread max_risk = spread_width - credit (Reg-T margin)
    # PnL% 用 max_risk 当分母 (跟 BUY CALL 用 premium 当分母对称)
    spread_width = 0.0
    if is_credit and len(legs) >= 2:
        # short put @ K_short / long put @ K_long < K_short
        ks = [l[2] for l in legs if "short" in l[0]]
        kl = [l[2] for l in legs if "put" in l[0] and "short" not in l[0]]
        if ks and kl:
            spread_width = abs(ks[0] - kl[0])  # e.g. $20
    max_risk = (spread_width - entry_value) if (is_credit and spread_width > 0) \
                else entry_value
    if max_risk <= 0: max_risk = max(0.01, entry_value)
    # 取首 leg 的 expiry
    first_code = legs[0][1]
    first_kdb = db[db["code"] == first_code]
    if not len(first_kdb): return {"is_closed": False}
    expiry_dt = pd.Timestamp(first_kdb.iloc[0]["expiry"])
    # 退出规则 (基于 max_risk 比例, 跟券商 margin 一致)
    if is_credit:
        # +50% on margin → cur_credit ≤ 0.5 × entry (收一半利润)
        profit_target = entry_value * 0.5
        # -50% on margin → loss = 0.5 × max_risk → cur_credit = entry + 0.5 × max_risk
        stop_loss = entry_value + 0.5 * max_risk
    elif is_long_vol:
        profit_target = entry_value * 2.0
        stop_loss = None
    elif is_long_dir:
        profit_target = entry_value * 2.0
        stop_loss = entry_value * 0.5
    else:
        profit_target = stop_loss = None

    def _pnl_pct(cv):
        """统一 PnL% 公式 (max_risk 分母, 跟券商 margin 一致)."""
        if is_credit:
            return (entry_value - cv) / max_risk * 100
        return (cv / entry_value - 1) * 100
    # 逐日 MTM
    sig_d = pd.Timestamp(signal_date).normalize()
    days = sorted(set(db[db["code"].isin([l[1] for l in legs])]["date"].unique()))
    days_after = [d for d in days if pd.Timestamp(d) > sig_d]
    hold_days = 0
    leg_prices_at_exit = []  # 出场时各 leg 真实 close
    for d in days_after:
        d_ts = pd.Timestamp(d)
        if d_ts > today_dt: break
        # 每 leg close 算 cur_value
        cur_total = 0.0; ok = True
        leg_prices_today = []
        for _lab, _code, _K, _qty in legs:
            r = db[(db["code"] == _code) & (db["date"] == d_ts)]
            if not len(r): ok = False; break
            _p = float(r.iloc[0]["close"])
            leg_prices_today.append((_lab, _p))
            cur_total += _qty * _p
        if not ok: continue
        leg_prices_at_exit = leg_prices_today
        cur_value = -cur_total if is_credit else cur_total
        hold_days += 1
        # 检查退出
        if profit_target is not None and (
            (is_credit and cur_value <= profit_target) or
            (is_long_vol and cur_value >= profit_target) or
            (is_long_dir and cur_value >= profit_target)):
            return {"is_closed": True, "exit_date": d_ts,
                     "exit_value": cur_value, "exit_reason": "+50% profit"
                     if is_credit else "+100% profit",
                     "pnl_pct": _pnl_pct(cur_value),
                     "leg_prices": leg_prices_at_exit}
        if stop_loss is not None and (
            (is_credit and cur_value >= stop_loss) or
            (is_long_dir and cur_value <= stop_loss)):
            return {"is_closed": True, "exit_date": d_ts,
                     "exit_value": cur_value,
                     "exit_reason": "stop loss",
                     "pnl_pct": _pnl_pct(cur_value),
                     "leg_prices": leg_prices_at_exit}
        # STRADDLE 持仓 14d
        if is_long_vol and hold_days >= 14:
            return {"is_closed": True, "exit_date": d_ts,
                     "exit_value": cur_value,
                     "exit_reason": "14d 定时",
                     "pnl_pct": _pnl_pct(cur_value)}
        # SHORT_VOL 30d
        if is_credit and "SHORT_VOL" in strategy and hold_days >= 30:
            return {"is_closed": True, "exit_date": d_ts,
                     "exit_value": cur_value,
                     "exit_reason": "30d 定时",
                     "pnl_pct": _pnl_pct(cur_value)}
        # expiry
        if d_ts >= expiry_dt:
            return {"is_closed": True, "exit_date": d_ts,
                     "exit_value": cur_value,
                     "exit_reason": "expiry",
                     "pnl_pct": _pnl_pct(cur_value),
                     "leg_prices": leg_prices_at_exit}
    # 未触发退出 → OPEN (用最新 close 算 mark-to-market)
    if hold_days > 0 and ok:
        return {"is_closed": False, "current_value": cur_value,
                 "hold_days": hold_days,
                 "pnl_pct": ((entry_value - cur_value) / entry_value * 100
                              if is_credit
                              else (cur_value / entry_value - 1) * 100),
                 "leg_prices": leg_prices_at_exit}
    return {"is_closed": False}


def lookup_gld_intraday_price(trigger_time: pd.Timestamp,
                                 gld_intra_df: pd.DataFrame) -> Optional[float]:
    """从 GLD 5m intraday df 查 trigger 时点 spot. 失败 (off-hours) 返回 None."""
    if gld_intra_df is None or not len(gld_intra_df): return None
    matches = gld_intra_df[gld_intra_df.index <= pd.Timestamp(trigger_time)]
    return float(matches.iloc[-1]["Close"]) if len(matches) else None


def fetch_realtime_option_chain_quote(asset: str, expiry_str: str,
                                         strike: float, right: str) -> Optional[float]:
    """从 yfinance option_chain 拉当前 lastPrice (历史时点拿不到, 仅 live).
    asset: GLD / SLV
    expiry_str: 'YYYY-MM-DD'
    right: 'C' / 'P'
    """
    try:
        import yfinance as yf
        t = yf.Ticker(asset)
        if expiry_str not in t.options:
            return None
        chain = t.option_chain(expiry_str)
        df = chain.calls if right.upper() == "C" else chain.puts
        match = df[df["strike"] == strike]
        if not len(match):
            return None
        return float(match.iloc[0]["lastPrice"])
    except Exception:
        return None


def fetch_realtime_option_price(option_symbol: str) -> Optional[float]:
    """yfinance 拉期权实时报价 (last 1m close).
    option_symbol 格式: 标准 OCC, e.g. GLD260620C00450000.
    """
    try:
        import yfinance as yf
        df = yf.Ticker(option_symbol).history(period="1d", interval="1m")
        if df is None or not len(df):
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def infer_option_symbol(asset: str, strategy: str, ul_price: float,
                          trigger_time: pd.Timestamp,
                          dte_target: int = 45) -> Optional[str]:
    """根据策略 + 触发时 underlying 价 推断 option ticker (OCC 格式).
    策略 → strike/right 简化:
      BUY CALL: ATM call (strike = round(ul, 1) for $100+ stock,
                          else round to $0.5)
      SELL PUT: 30Δ put ≈ ATM × 0.95
    DTE: 45 default. 找最近月度第三周 expiry.
    """
    if strategy not in ("BUY CALL", "SELL PUT", "STRADDLE"):
        return None
    # 第三周五 (monthly opex)
    target_d = pd.Timestamp(trigger_time) + pd.Timedelta(days=dte_target)
    # 跳到下一个 monthly opex
    month = target_d.replace(day=1)
    first_fri = month + pd.Timedelta(days=(4 - month.weekday()) % 7)
    third_fri = first_fri + pd.Timedelta(days=14)
    if third_fri < target_d - pd.Timedelta(days=10):
        # too far back, 推到下月
        nm = month + pd.offsets.MonthBegin(1)
        first_fri = nm + pd.Timedelta(days=(4 - nm.weekday()) % 7)
        third_fri = first_fri + pd.Timedelta(days=14)
    expiry_str = third_fri.strftime("%y%m%d")
    if strategy == "BUY CALL":
        right = "C"
        strike = round(ul_price, 0) if ul_price > 50 else round(ul_price * 2) / 2
    elif strategy == "SELL PUT":
        right = "P"
        strike = round(ul_price * 0.95, 0)
    else:  # STRADDLE — pick ATM call (put symbol could be derived同位)
        right = "C"
        strike = round(ul_price, 0)
    strike_str = f"{int(strike * 1000):08d}"
    return f"{asset}{expiry_str}{right}{strike_str}"
