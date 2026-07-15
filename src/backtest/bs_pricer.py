"""
Black-Scholes 期权定价模块

用途: Phase 4B 期权回测, 用GVZ作为IV代理模拟期权价格和Greeks.
GLD ETF 不分红 (q=0).
"""

import numpy as np
from scipy.stats import norm


def bs_d1d2(S, K, T, r, sigma):
    """计算 d1, d2."""
    if T <= 0 or sigma <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes 期权价格.

    Args:
        S: 标的价格
        K: 行权价
        T: 到期时间 (年)
        r: 无风险利率 (年化, decimal)
        sigma: 波动率 (年化, decimal)
        option_type: 'call' or 'put'
    Returns:
        期权理论价格
    """
    if T <= 0:
        # 到期 → 内在价值
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return 0.0
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="call"):
    """Delta."""
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = bs_d1d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return 0.0
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """计算全套Greeks.

    Returns:
        dict with delta, gamma, theta (per day), vega (per 1% IV), rho
    """
    if T <= 0:
        return {
            "delta": bs_delta(S, K, T, r, sigma, option_type),
            "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0,
        }
    d1, d2 = bs_d1d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    nd1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    # Gamma (same for call and put)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Vega (per 1% IV change = 0.01 sigma)
    vega = S * nd1 * sqrt_T * 0.01

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * nd1 * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * nd1 * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "rho": rho}


def find_strike_for_delta(S, T, r, sigma, target_delta,
                          option_type="call", strike_step=1.0):
    """找到使 delta 接近目标值的行权价.

    Args:
        target_delta: 目标delta (call: 正值如0.40, put: 负值如-0.30)
        strike_step: 行权价步长 ($1 for GLD)
    Returns:
        最接近目标delta的行权价
    """
    if T <= 0 or sigma <= 0:
        return round(S / strike_step) * strike_step

    # 搜索范围: ATM ± 20%
    k_low = S * 0.80
    k_high = S * 1.20

    # 生成候选行权价
    k_start = np.ceil(k_low / strike_step) * strike_step
    k_end = np.floor(k_high / strike_step) * strike_step
    strikes = np.arange(k_start, k_end + strike_step, strike_step)

    best_k = round(S / strike_step) * strike_step  # ATM default
    best_diff = float("inf")

    for k in strikes:
        d = bs_delta(S, k, T, r, sigma, option_type)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_k = k

    return best_k
