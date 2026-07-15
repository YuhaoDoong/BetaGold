"""
Phase 4B: GLD 期权策略回测 — 真实期权价格

基于 Phase 4A 信号 (Buy Call / Sell Put / Exit) + Moomoo 历史期权K线.
不使用 BS 模拟 — 所有期权价格来自真实市场数据.

数据来源:
- 信号: Phase 4A Hybrid Band + Regime + RV
- 期权价格: Moomoo API 历史K线 (data/raw/options_history/kline_db/)
- 日终快照: Moomoo EOD snapshots (data/raw/options_history/YYYY-MM-DD/)

用法:
    conda activate gold
    python src/backtest/options_backtest.py
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.analysis_method_compare import (
    build_band, generate_v2_signals, compute_rv_pctile,
    compute_tp_indicators, find_tp_exits,
)

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase4B")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "backtest")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================================
# 期权价格数据库
# ==========================================================
class OptionPriceDB:
    """基于真实K线数据的期权价格查询"""

    def __init__(self, config: dict = None):
        config = config or load_config()
        raw_path = config["paths"]["raw_data"]
        self.kline_path = os.path.join(
            raw_path, "options_history", "kline_db", "all_klines.parquet")
        self.snapshot_dir = os.path.join(raw_path, "options_history")
        self._klines = None
        self._date_index = None

    def load(self):
        """加载K线数据库"""
        if not os.path.exists(self.kline_path):
            raise FileNotFoundError(
                f"期权K线数据库不存在: {self.kline_path}\n"
                "请先运行: python -m src.data.options.options_history_builder")

        self._klines = pd.read_parquet(self.kline_path)
        self._klines["date"] = pd.to_datetime(self._klines["date"])
        self._klines["expiry_date"] = pd.to_datetime(self._klines["expiry"])

        # 构建按日期的索引 (加速查询)
        self._date_index = {}
        for date, group in self._klines.groupby("date"):
            self._date_index[date.date()] = group

        dates = sorted(self._date_index.keys())
        print(f"  期权价格DB: {len(self._klines):,} 行, "
              f"{self._klines['code'].nunique()} 合约, "
              f"{dates[0]} ~ {dates[-1]}")
        return self

    def get_available_dates(self) -> list:
        """返回有数据的日期列表"""
        return sorted(self._date_index.keys())

    def find_contract(self, date, gld_price: float,
                      option_type: str = "CALL",
                      target_dte: int = 21,
                      dte_range: tuple = (14, 45),
                      moneyness: str = "atm") -> dict | None:
        """
        在指定日期找到最合适的期权合约.

        Args:
            date: 查询日期
            gld_price: 当日GLD价格
            option_type: 'CALL' or 'PUT'
            target_dte: 目标DTE
            dte_range: 可接受的DTE范围
            moneyness: 'atm' = 最接近ATM, 'otm_near' = 轻度OTM

        Returns:
            dict with code, strike, expiry, dte, close, volume
            or None if no contract found
        """
        if isinstance(date, pd.Timestamp):
            date = date.date()

        candidates = self._date_index.get(date)
        if candidates is None:
            return None

        # 筛选 option_type
        mask = candidates["option_type"] == option_type

        # 筛选 DTE 范围
        mask &= candidates["dte_at_date"].between(dte_range[0], dte_range[1])

        filtered = candidates[mask]
        if filtered.empty:
            return None

        # 选择 strike: ATM 优先
        if moneyness == "atm":
            # 最接近 ATM 的 strike
            filtered = filtered.copy()
            filtered["strike_diff"] = (filtered["strike"] - gld_price).abs()
            # 按 |strike - ATM| 排序, 次选 DTE 最接近目标
            filtered["dte_diff"] = (filtered["dte_at_date"] - target_dte).abs()
            filtered = filtered.sort_values(
                ["strike_diff", "dte_diff"])
        elif moneyness == "otm_near":
            # 轻度 OTM: call → strike > S, put → strike < S
            filtered = filtered.copy()
            if option_type == "CALL":
                otm = filtered[filtered["strike"] >= gld_price]
                if otm.empty:
                    otm = filtered
                filtered = otm
            else:
                otm = filtered[filtered["strike"] <= gld_price]
                if otm.empty:
                    otm = filtered
                filtered = otm

            filtered["strike_diff"] = (filtered["strike"] - gld_price).abs()
            filtered["dte_diff"] = (filtered["dte_at_date"] - target_dte).abs()
            filtered = filtered.sort_values(["dte_diff", "strike_diff"])

        # 过滤成交量=0的合约 (可能无流动性)
        has_volume = filtered[filtered["volume"] > 0]
        if not has_volume.empty:
            best = has_volume.iloc[0]
        else:
            best = filtered.iloc[0]

        return {
            "code": best["code"],
            "strike": best["strike"],
            "expiry": best["expiry"],
            "expiry_date": best["expiry_date"],
            "dte": int(best["dte_at_date"]),
            "close": best["close"],
            "high": best["high"],
            "low": best["low"],
            "volume": int(best["volume"]),
        }

    def get_price(self, code: str, date) -> dict | None:
        """获取指定合约在指定日期的价格"""
        if isinstance(date, pd.Timestamp):
            date = date.date()

        day_data = self._date_index.get(date)
        if day_data is None:
            return None

        match = day_data[day_data["code"] == code]
        if match.empty:
            return None

        row = match.iloc[0]
        return {
            "close": row["close"],
            "high": row["high"],
            "low": row["low"],
            "volume": int(row["volume"]),
            "dte": int(row["dte_at_date"]),
        }

    def get_price_range(self, code: str,
                        start_date, end_date) -> pd.DataFrame:
        """获取指定合约在日期范围内的所有价格"""
        mask = self._klines["code"] == code
        df = self._klines[mask].copy()
        if df.empty:
            return df

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.sort_values("date")


# ==========================================================
# 数据加载
# ==========================================================
def load_data():
    config = load_config()
    features, _ = load_dataset(config)
    raw_dir = config["paths"]["raw_data"]

    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)

    range_df = pd.read_parquet(
        os.path.join(PROJECT_ROOT, "data", "models", "dl_range_v2_oos.parquet"))

    # Regime & RV
    feat_cols = [c for c in features.columns if not c.startswith("fwd_")]
    regime = RegimeClassifier().classify(features[feat_cols])["regime"]
    rv_10d = features["rv_10d"] if "rv_10d" in features.columns else None
    rv_pctile = compute_rv_pctile(rv_10d)

    return {
        "gld": gld,
        "range_df": range_df, "regime": regime,
        "rv_10d": rv_10d, "rv_pctile": rv_pctile,
        "features": features, "config": config,
    }


# ==========================================================
# 信号生成
# ==========================================================
def generate_signals(data):
    """生成 Phase 4A 信号 + TP 指标."""
    close = data["gld"]["Close"]
    high = data["gld"]["High"]

    # Hybrid D/L3 band
    upper_band, lower_band, bp = build_band(
        data["range_df"], close,
        upper_lags=(1,), lower_lags=(1, 2, 3))

    dates = bp.dropna().index
    bp_s = bp.reindex(dates)
    rv_p = data["rv_pctile"].reindex(dates)
    is_bull = (data["regime"].reindex(dates) == "Bull")

    buy_call, sell_put, exit_sig = generate_v2_signals(bp_s, rv_p, is_bull)

    # TP 指标
    tp_ind = compute_tp_indicators(close, high, data["gld"]["Low"])

    return {
        "dates": dates,
        "buy_call": buy_call, "sell_put": sell_put, "exit": exit_sig,
        "bp": bp_s, "rv_pctile": rv_p, "is_bull": is_bull,
        "tp_ind": tp_ind,
    }


# ==========================================================
# 交易记录
# ==========================================================
@dataclass
class Trade:
    entry_date: pd.Timestamp
    signal_type: str          # 'buy_call' or 'sell_put'
    option_type: str          # 'CALL' or 'PUT'
    direction: int            # +1 long, -1 short
    code: str                 # 期权合约代码
    strike: float
    expiry: str
    dte: int                  # DTE at entry
    entry_underlying: float   # GLD price at entry
    entry_premium: float      # 期权价格 (real)
    contracts: int = 1

    exit_date: pd.Timestamp = None
    exit_premium: float = None
    exit_underlying: float = None
    exit_reason: str = None
    hold_days: int = 0
    pnl_dollar: float = 0.0
    pnl_pct: float = 0.0
    underlying_move_pct: float = 0.0
    daily_prices: list = field(default_factory=list)


# ==========================================================
# 回测引擎 — 真实价格
# ==========================================================
class OptionsBacktest:
    def __init__(self,
                 target_dte=21,
                 dte_range=(14, 45),
                 max_hold=10,
                 max_positions=3,
                 commission=0.65,
                 slippage_pct=0.02,
                 capital=100_000,
                 position_size_pct=0.05):
        self.target_dte = target_dte
        self.dte_range = dte_range
        self.max_hold = max_hold
        self.max_positions = max_positions
        self.commission = commission
        self.slippage_pct = slippage_pct
        self.capital = capital
        self.position_size_pct = position_size_pct

    def _check_tp_exit(self, trade, d_loc, close, tp_ind):
        """检查止盈拐点 (逐日检查)."""
        all_dates = close.index
        entry_loc = all_dates.get_loc(trade.entry_date)

        cur_close = close.iloc[d_loc]
        entry_price = trade.entry_underlying
        cur_gain = (cur_close / entry_price - 1) * 100

        # 计算 peak
        peak = entry_price
        for loc in range(entry_loc + 1, d_loc + 1):
            if loc < len(all_dates):
                peak = max(peak, close.iloc[loc])
        peak_gain = (peak / entry_price - 1) * 100

        macd_hist = tp_ind["macd_hist"]
        rsi = tp_ind["rsi"]
        d = all_dates[d_loc]

        prev_d = all_dates[d_loc - 1] if d_loc > 0 else None
        prev2_d = all_dates[d_loc - 2] if d_loc > 1 else None

        mh_cur = macd_hist.get(d, 0)
        mh_prev = macd_hist.get(prev_d, 0) if prev_d else 0
        mh_prev2 = macd_hist.get(prev2_d, 0) if prev2_d else 0
        rsi_cur = rsi.get(d, 50)
        rsi_prev = rsi.get(prev_d, 50) if prev_d else 50

        # MACD hist 由正转负
        if mh_prev >= 0 and mh_cur < 0 and cur_gain > 0.3:
            return "MACD"
        # MACDweak: 连续2天缩小
        if (cur_gain > 1.0 and mh_cur > 0
                and mh_cur < mh_prev and mh_prev < mh_prev2
                and mh_prev2 > 0):
            return "MACDweak"
        # RSI 超买回落
        if rsi_prev > 70 and rsi_cur < 60 and cur_gain > 0:
            return "RSI"
        # Pullback
        if peak_gain > 2.0:
            drawdown = (peak - cur_close) / peak * 100
            if drawdown >= 1.5:
                return "Pullback"
        return None

    def run(self, data, signals, price_db: OptionPriceDB):
        """主回测循环 — 使用真实期权价格."""
        close = data["gld"]["Close"]
        tp_ind = signals["tp_ind"]
        all_dates = close.index

        buy_call = signals["buy_call"]
        sell_put = signals["sell_put"]
        exit_sig = signals["exit"]

        # 仅回测有期权数据的日期范围
        avail_dates = set(price_db.get_available_dates())
        signal_dates = signals["dates"]
        valid_dates = [d for d in signal_dates
                       if d.date() in avail_dates]

        if not valid_dates:
            print("  WARNING: 没有信号日期与期权数据重叠!")
            print(f"    信号范围: {signal_dates[0].date()} ~ "
                  f"{signal_dates[-1].date()}")
            opt_dates = sorted(avail_dates)
            print(f"    期权数据: {opt_dates[0]} ~ {opt_dates[-1]}")
            return [], pd.Series(dtype=float)

        print(f"\n  回测期间 (有期权数据): "
              f"{valid_dates[0].strftime('%Y-%m-%d')} ~ "
              f"{valid_dates[-1].strftime('%Y-%m-%d')}")

        # 统计有效信号
        n_bc = sum(1 for d in valid_dates if buy_call.get(d, False))
        n_sp = sum(1 for d in valid_dates if sell_put.get(d, False))
        n_ex = sum(1 for d in valid_dates if exit_sig.get(d, False))
        print(f"  有效信号: Buy Call={n_bc}, Sell Put={n_sp}, Exit={n_ex}")

        trades = []
        open_positions = []
        equity_curve = {}
        last_entry_date = None
        skipped_no_contract = 0

        for d in valid_dates:
            S = close.get(d, np.nan)
            if np.isnan(S):
                continue

            d_loc = all_dates.get_loc(d)

            # --- 1. 检查持仓退出 ---
            for pos in open_positions[:]:
                days_held = (d - pos.entry_date).days
                remaining_dte = max(pos.dte - days_held, 0)

                # 获取真实价格
                price_data = price_db.get_price(pos.code, d)
                if price_data is not None:
                    opt_price = price_data["close"]
                else:
                    # 该日无交易数据 → 用上一个已知价格
                    opt_price = (pos.daily_prices[-1]
                                 if pos.daily_prices else pos.entry_premium)

                pos.daily_prices.append(opt_price)

                exit_reason = None

                # a) Exit signal
                if exit_sig.get(d, False):
                    exit_reason = "exit_signal"

                # b) Max hold
                if exit_reason is None and days_held >= self.max_hold:
                    exit_reason = "max_hold"

                # c) TP inflection (only for long positions)
                if (exit_reason is None and pos.direction == 1
                        and days_held >= 2):
                    tp_type = self._check_tp_exit(pos, d_loc, close, tp_ind)
                    if tp_type:
                        exit_reason = f"tp_{tp_type}"

                # d) Near expiry
                if exit_reason is None and remaining_dte <= 1:
                    exit_reason = "expiry"

                if exit_reason:
                    pos.exit_date = d
                    pos.exit_premium = opt_price
                    pos.exit_underlying = S
                    pos.exit_reason = exit_reason
                    pos.hold_days = days_held
                    pos.underlying_move_pct = (
                        (S / pos.entry_underlying - 1) * 100)

                    # P&L (真实价格, 含滑点和佣金)
                    if pos.direction == 1:  # long call
                        cost = pos.entry_premium * (1 + self.slippage_pct)
                        proceeds = opt_price * (1 - self.slippage_pct)
                        pnl = (proceeds - cost) * pos.contracts * 100
                    else:  # short put
                        credit = pos.entry_premium * (1 - self.slippage_pct)
                        buy_back = opt_price * (1 + self.slippage_pct)
                        pnl = (credit - buy_back) * pos.contracts * 100

                    pnl -= 2 * self.commission * pos.contracts
                    pos.pnl_dollar = pnl
                    denom = pos.entry_premium * 100 * pos.contracts
                    pos.pnl_pct = (pnl / denom * 100) if denom > 0 else 0

                    trades.append(pos)
                    open_positions.remove(pos)

            # --- 2. 新建仓 ---
            sig_type = None
            if buy_call.get(d, False):
                sig_type = "buy_call"
            elif sell_put.get(d, False):
                sig_type = "sell_put"

            if sig_type and len(open_positions) < self.max_positions:
                if (last_entry_date is not None
                        and (d - last_entry_date).days < 2):
                    sig_type = None

            if sig_type:
                if sig_type == "buy_call":
                    opt_type = "CALL"
                    direction = 1
                    moneyness = "atm"
                else:
                    opt_type = "PUT"
                    direction = -1
                    moneyness = "otm_near"

                contract = price_db.find_contract(
                    d, S,
                    option_type=opt_type,
                    target_dte=self.target_dte,
                    dte_range=self.dte_range,
                    moneyness=moneyness,
                )

                if contract is None:
                    skipped_no_contract += 1
                    continue

                premium = contract["close"]
                if premium < 0.10:
                    continue  # 价格太低, 流动性差

                # 仓位大小
                budget = self.capital * self.position_size_pct
                n_contracts = max(1, int(budget / (premium * 100)))
                if direction == -1:
                    margin = 0.20 * S * 100 * n_contracts
                    if margin > budget * 3:
                        n_contracts = max(
                            1, int(budget * 3 / (0.20 * S * 100)))

                trade = Trade(
                    entry_date=d,
                    signal_type=sig_type,
                    option_type=opt_type,
                    direction=direction,
                    code=contract["code"],
                    strike=contract["strike"],
                    expiry=contract["expiry"],
                    dte=contract["dte"],
                    entry_underlying=S,
                    entry_premium=premium,
                    contracts=n_contracts,
                )
                open_positions.append(trade)
                last_entry_date = d

            # --- 3. Daily equity ---
            open_value = 0
            for pos in open_positions:
                price_data = price_db.get_price(pos.code, d)
                if price_data is not None:
                    opt_price = price_data["close"]
                else:
                    opt_price = (pos.daily_prices[-1]
                                 if pos.daily_prices else pos.entry_premium)

                if pos.direction == 1:
                    open_value += (
                        (opt_price - pos.entry_premium)
                        * pos.contracts * 100)
                else:
                    open_value += (
                        (pos.entry_premium - opt_price)
                        * pos.contracts * 100)

            realized = sum(t.pnl_dollar for t in trades)
            equity_curve[d] = self.capital + realized + open_value

        # 强制平仓剩余持仓
        if open_positions:
            last_d = valid_dates[-1]
            S = close.get(last_d, 0)
            for pos in open_positions:
                price_data = price_db.get_price(pos.code, last_d)
                opt_price = (price_data["close"] if price_data
                             else pos.daily_prices[-1]
                             if pos.daily_prices else pos.entry_premium)

                pos.exit_date = last_d
                pos.exit_premium = opt_price
                pos.exit_underlying = S
                pos.exit_reason = "end_of_backtest"
                pos.hold_days = (last_d - pos.entry_date).days
                pos.underlying_move_pct = (
                    (S / pos.entry_underlying - 1) * 100)

                if pos.direction == 1:
                    cost = pos.entry_premium * (1 + self.slippage_pct)
                    proceeds = opt_price * (1 - self.slippage_pct)
                    pnl = (proceeds - cost) * pos.contracts * 100
                else:
                    credit = pos.entry_premium * (1 - self.slippage_pct)
                    buy_back = opt_price * (1 + self.slippage_pct)
                    pnl = (credit - buy_back) * pos.contracts * 100

                pnl -= 2 * self.commission * pos.contracts
                pos.pnl_dollar = pnl
                denom = pos.entry_premium * 100 * pos.contracts
                pos.pnl_pct = (pnl / denom * 100) if denom > 0 else 0
                trades.append(pos)

        if skipped_no_contract > 0:
            print(f"  [!] {skipped_no_contract} 个信号因无匹配合约被跳过")

        equity = pd.Series(equity_curve).sort_index()
        return trades, equity


# ==========================================================
# 统计分析
# ==========================================================
def compute_metrics(trades, equity, capital=100_000):
    """计算回测指标."""
    if len(trades) == 0:
        return {}, pd.DataFrame()

    df = pd.DataFrame([{
        "entry": t.entry_date, "exit": t.exit_date,
        "signal": t.signal_type, "direction": t.direction,
        "code": t.code,
        "strike": t.strike, "expiry": t.expiry, "dte": t.dte,
        "entry_prem": t.entry_premium, "exit_prem": t.exit_premium,
        "entry_S": t.entry_underlying, "exit_S": t.exit_underlying,
        "contracts": t.contracts,
        "hold_days": t.hold_days, "exit_reason": t.exit_reason,
        "pnl": t.pnl_dollar, "pnl_pct": t.pnl_pct,
        "und_move": t.underlying_move_pct,
    } for t in trades])

    total_pnl = df["pnl"].sum()
    total_return = total_pnl / capital * 100
    n_trades = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    # Equity-based
    if len(equity) > 1:
        daily_ret = equity.pct_change().dropna()
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (((equity.iloc[-1] / capital) ** (1 / years) - 1) * 100
                if years > 0 else 0)
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)
        drawdown = (equity / equity.cummax() - 1)
        max_dd = drawdown.min() * 100
    else:
        cagr = sharpe = max_dd = 0.0

    # Profit factor
    total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
    total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
    pf = total_wins / total_losses if total_losses > 0 else float("inf")

    metrics = {
        "n_trades": n_trades,
        "n_buy_call": len(df[df["signal"] == "buy_call"]),
        "n_sell_put": len(df[df["signal"] == "sell_put"]),
        "total_pnl": total_pnl,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0,
        "profit_factor": pf,
        "avg_hold_days": df["hold_days"].mean(),
        "avg_pnl_pct": df["pnl_pct"].mean(),
    }
    return metrics, df


def print_metrics(metrics, df):
    """打印回测结果."""
    if not metrics:
        print("\n  没有交易!")
        return

    print(f"\n{'='*60}")
    print(f"  Phase 4B 期权回测结果 (真实价格)")
    print(f"{'='*60}")
    print(f"  总交易: {metrics['n_trades']} "
          f"(Buy Call: {metrics['n_buy_call']}, "
          f"Sell Put: {metrics['n_sell_put']})")
    print(f"  总P&L: ${metrics['total_pnl']:,.0f} "
          f"({metrics['total_return_pct']:+.1f}%)")
    print(f"  CAGR: {metrics['cagr_pct']:+.2f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Win Rate: {metrics['win_rate_pct']:.0f}%")
    print(f"  Avg Win: ${metrics['avg_win']:+,.0f}, "
          f"Avg Loss: ${metrics['avg_loss']:+,.0f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Avg Hold: {metrics['avg_hold_days']:.1f} days")
    print(f"  Avg P&L/trade: {metrics['avg_pnl_pct']:+.1f}%")

    # 分策略
    for sig in ["buy_call", "sell_put"]:
        sub = df[df["signal"] == sig]
        if len(sub) == 0:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        avg_pnl = sub["pnl_pct"].mean()
        total = sub["pnl"].sum()
        print(f"\n  --- {sig.replace('_', ' ').title()} (n={len(sub)}) ---")
        print(f"    Win Rate: {wr:.0f}%, Avg P&L: {avg_pnl:+.1f}%, "
              f"Total: ${total:+,.0f}")
        print(f"    Avg Hold: {sub['hold_days'].mean():.1f}d, "
              f"Avg Und Move: {sub['und_move'].mean():+.2f}%")

    # 退出原因
    if len(df) > 0:
        print(f"\n  退出原因分布:")
        for reason, sub in df.groupby("exit_reason"):
            wr = (sub["pnl"] > 0).mean() * 100
            print(f"    {reason:20s}: n={len(sub):3d}, WR={wr:.0f}%, "
                  f"avg={sub['pnl_pct'].mean():+.1f}%")

    # 合约明细
    if len(df) <= 30:
        print(f"\n  交易明细:")
        for _, row in df.iterrows():
            entry_d = row["entry"].strftime("%m/%d")
            exit_d = row["exit"].strftime("%m/%d") if pd.notna(row["exit"]) else "?"
            print(f"    {entry_d}→{exit_d} {row['signal']:9s} "
                  f"{row['code'][-15:]:15s} "
                  f"K={row['strike']:.0f} "
                  f"entry=${row['entry_prem']:.2f} "
                  f"exit=${row['exit_prem']:.2f} "
                  f"P&L={row['pnl_pct']:+.0f}% "
                  f"[{row['exit_reason']}]")


# ==========================================================
# 可视化
# ==========================================================
def plot_results(trades_df, equity, capital, out_dir):
    """生成回测报告图."""
    if trades_df.empty or equity.empty:
        print("  无数据，跳过可视化")
        return

    n_subplots = min(4, 2 + (len(trades_df) > 0) + (len(equity) > 5))
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(equity.index, equity.values / 1000,
            color="steelblue", linewidth=1.5)
    ax.axhline(capital / 1000, color="gray", linewidth=1,
               linestyle="--", alpha=0.5)
    ax.set_title("Equity Curve (Real Options Prices)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio ($K)")
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    dd = (equity / equity.cummax() - 1) * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="salmon")
    ax.set_title("Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    # 3. P&L scatter
    ax = axes[1, 0]
    for sig, color, marker in [("buy_call", "#2196F3", "^"),
                                ("sell_put", "#FF9800", "v")]:
        sub = trades_df[trades_df["signal"] == sig]
        if len(sub) > 0:
            ax.scatter(sub["hold_days"], sub["pnl_pct"],
                       c=color, marker=marker, s=60, alpha=0.7,
                       edgecolors="gray", linewidths=0.3,
                       label=f"{sig.replace('_', ' ').title()} "
                             f"(n={len(sub)})")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Hold Days")
    ax.set_ylabel("P&L (%)")
    ax.set_title("Trade P&L vs Hold Days", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Trade P&L bar
    ax = axes[1, 1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c"
              for v in trades_df["pnl_pct"]]
    ax.bar(range(len(trades_df)), trades_df["pnl_pct"],
           color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (%)")
    ax.set_title("Per-Trade P&L (%)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "01_backtest_real.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==========================================================
# main
# ==========================================================
def main():
    print("=" * 60)
    print("  Phase 4B: GLD 期权策略回测 (真实价格)")
    print("=" * 60)

    # 1. 加载数据
    data = load_data()
    signals = generate_signals(data)

    # 2. 加载期权价格数据库
    print("\n  加载期权价格数据库...")
    price_db = OptionPriceDB(config=data["config"])
    price_db.load()

    # 3. 运行回测
    engine = OptionsBacktest(
        target_dte=30,
        dte_range=(14, 400),   # 宽范围: 匹配当前可用数据
        max_hold=10,
        max_positions=3,
        commission=0.65,
        slippage_pct=0.02,
        capital=100_000,
        position_size_pct=0.05,
    )

    trades, equity = engine.run(data, signals, price_db)
    metrics, trades_df = compute_metrics(trades, equity, engine.capital)
    print_metrics(metrics, trades_df)

    # 4. 可视化
    if not trades_df.empty:
        print(f"\n  生成可视化...")
        plot_results(trades_df, equity, engine.capital, OUT_DIR)

        # 保存数据
        trades_df.to_parquet(
            os.path.join(DATA_DIR, "phase4b_trades.parquet"))
        equity.to_frame("equity").to_parquet(
            os.path.join(DATA_DIR, "phase4b_equity.parquet"))
        print(f"  数据保存: data/backtest/phase4b_*.parquet")
    else:
        print("\n  没有交易完成. 需要更多期权历史数据.")
        print("  期权K线数据库正在积累中 — 请每天运行采集脚本.")


if __name__ == "__main__":
    main()
