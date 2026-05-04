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
