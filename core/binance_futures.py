"""Binance USDT-M Futures 数据接入 (XAUUSDT 黄金 + XAGUSDT 白银, 20× 杠杆模拟).

不需 API key (公开 endpoints): mark price, funding rate, exchange info.
"""
from __future__ import annotations
import requests
from typing import Optional
from functools import lru_cache
import time

_FAPI = "https://fapi.binance.com/fapi/v1"

# v3.7.163: asset → Binance 永续合约 symbol 映射
ASSET_SYMBOL = {
    "GLD": "XAUUSDT",   # 黄金永续
    "SLV": "XAGUSDT",   # 白银永续
}


def fetch_perp_realtime(symbol: str) -> Optional[dict]:
    """通用拉 Binance 永续实时数据.
    symbol 可以 'XAUUSDT' / 'XAGUSDT' / 任意 Binance USDT-M 永续 symbol.
    """
    try:
        r = requests.get(f"{_FAPI}/premiumIndex",
                          params={"symbol": symbol}, timeout=8)
        r.raise_for_status()
        d = r.json()
        return {
            "symbol": d["symbol"],
            "mark_price": float(d["markPrice"]),
            "index_price": float(d["indexPrice"]),
            "funding_rate": float(d["lastFundingRate"]),
            "next_funding_ms": int(d["nextFundingTime"]),
        }
    except Exception:
        return None


def fetch_realtime_for_asset(asset: str) -> Optional[dict]:
    """根据 asset (GLD/SLV) 拉对应 Binance 永续 (XAU/XAGUSDT)."""
    sym = ASSET_SYMBOL.get(asset.upper())
    if not sym: return None
    return fetch_perp_realtime(sym)


# 兼容旧 API
def fetch_xauusdt_realtime() -> Optional[dict]:
    """旧 API — 仅黄金 XAUUSDT (兼容). 新代码用 fetch_realtime_for_asset(asset)."""
    return fetch_perp_realtime("XAUUSDT")


@lru_cache(maxsize=1)
def fetch_xauusdt_specs(_cache_token: int = 0) -> Optional[dict]:
    """XAUUSDT exchange info (cached). taker/maker fee, min qty 等."""
    try:
        r = requests.get(f"{_FAPI}/exchangeInfo", timeout=8)
        r.raise_for_status()
        for s in r.json()["symbols"]:
            if s["symbol"] == "XAUUSDT":
                return {
                    "tick_size": float(next(f["tickSize"] for f in s["filters"] if f["filterType"] == "PRICE_FILTER")),
                    "min_qty": float(next(f["minQty"] for f in s["filters"] if f["filterType"] == "LOT_SIZE")),
                    # taker/maker fee 公开 endpoint 无 — 用文档默认 (regular tier)
                    "taker_fee": 0.0005,    # 0.05% (USDC pair 0.018%)
                    "maker_fee": 0.0002,    # 0.02%
                    "leverage_max": 20,      # XAUUSDT max leverage
                    "maintenance_margin_rate": 0.005,  # 0.5% (Bracket 1, < $50k notional)
                }
    except Exception:
        pass
    return None


def compute_liquidation_price(entry_price: float, leverage: int = 20,
                                 side: str = "long",
                                 mm_rate: float = 0.005) -> float:
    """20× long XAUUSDT 爆仓价 (cross margin 简化公式).

    Long 爆仓: liq ≈ entry × (1 - 1/lev + mm_rate)
    Short 爆仓: liq ≈ entry × (1 + 1/lev - mm_rate)
    """
    if side.lower() == "long":
        return entry_price * (1 - 1.0 / leverage + mm_rate)
    return entry_price * (1 + 1.0 / leverage - mm_rate)


def estimate_futures_pnl(entry: float, current: float,
                            qty: float = 1.0, leverage: int = 20,
                            funding_rate_8h: float = 0.0,
                            hold_hours: float = 0.0,
                            taker_fee: float = 0.0005) -> dict:
    """估算 XAUUSDT 永续 long 持仓 P&L (全 USDT 计价).

    Args:
        entry, current: 入场/现价 (USDT)
        qty: 持仓数量 (XAU)
        leverage: 杠杆
        funding_rate_8h: 当前 funding rate (8h)
        hold_hours: 已持仓小时
        taker_fee: 单边手续费率

    Returns:
        gross_pnl_usdt: 不含费 P&L
        funding_cost_usdt: 累计资金费
        fee_usdt: 双边手续费 (entry + estimated exit)
        net_pnl_usdt: 净
        roi_pct: ROI on margin (P&L / margin)
        liq_price: 爆仓价
    """
    notional = entry * qty
    margin = notional / leverage
    gross_pnl = (current - entry) * qty
    # 资金费按 8h 累加 (long 付费 if rate>0, 收 if <0)
    n_funding = max(0, int(hold_hours / 8))
    funding_cost = notional * funding_rate_8h * n_funding
    # 手续费: entry + exit 双边
    fee = notional * taker_fee + (current * qty) * taker_fee
    net_pnl = gross_pnl - funding_cost - fee
    roi = (net_pnl / margin * 100) if margin > 0 else 0.0
    return {
        "notional_usdt": notional,
        "margin_usdt": margin,
        "gross_pnl_usdt": gross_pnl,
        "funding_cost_usdt": funding_cost,
        "fee_usdt": fee,
        "net_pnl_usdt": net_pnl,
        "roi_pct": roi,
        "liq_price": compute_liquidation_price(entry, leverage),
    }
