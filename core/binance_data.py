"""币安行情数据 — XAU/USDT + XAG/USDT.

通过 ccxt 获取币安金银合约/现货实时行情.
用于 Dashboard 实时价格展示.
"""

import logging

logger = logging.getLogger(__name__)

BINANCE_API_KEY = "***REDACTED-BINANCE-KEY***"
BINANCE_SECRET = "***REDACTED-BINANCE-SECRET***"

# 金银交易对
SYMBOLS = {
    "gold": {
        "futures": "XAU/USDT:USDT",    # 黄金合约
        "spot": "PAXG/USDT",            # 黄金代币 (≈1盎司)
    },
    "silver": {
        "futures": "XAG/USDT:USDT",     # 白银合约
    },
}


def fetch_binance_prices():
    """获取币安金银实时行情.

    Returns: dict or None
        {
            "xau_futures": float,  # XAU/USDT 合约价
            "xau_spot": float,     # PAXG/USDT 现货价
            "xag_futures": float,  # XAG/USDT 合约价
            "gold_silver_ratio": float,
            "timestamp": str,
        }
    """
    try:
        import ccxt
        from datetime import datetime

        exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET,
            "enableRateLimit": True,
            "timeout": 10000,
        })

        result = {"timestamp": datetime.now().strftime("%H:%M:%S")}

        for name, symbols in SYMBOLS.items():
            for stype, symbol in symbols.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    key = f"{'xau' if name == 'gold' else 'xag'}_{stype}"
                    result[key] = float(ticker["last"])
                except Exception:
                    pass

        # 金银比
        if "xau_futures" in result and "xag_futures" in result and result["xag_futures"] > 0:
            result["gold_silver_ratio"] = result["xau_futures"] / result["xag_futures"]

        return result if len(result) > 1 else None

    except ImportError:
        logger.warning("ccxt not installed")
        return None
    except Exception as e:
        logger.warning("Binance fetch failed: %s", e)
        return None
