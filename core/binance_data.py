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
    """获取币安金银合约实时行情 + 持仓量.

    Returns: dict or None
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

        # 只获取合约
        for key, symbol in [("xau", "XAU/USDT:USDT"), ("xag", "XAG/USDT:USDT")]:
            try:
                ticker = exchange.fetch_ticker(symbol)
                result[f"{key}_price"] = float(ticker["last"])
                result[f"{key}_change"] = float(ticker.get("percentage", 0) or 0)
                result[f"{key}_volume"] = float(ticker.get("quoteVolume", 0) or 0)
                # 持仓量 (open interest)
                if hasattr(exchange, "fetch_open_interest"):
                    try:
                        oi = exchange.fetch_open_interest(symbol)
                        result[f"{key}_oi"] = float(oi.get("openInterest", 0) or 0)
                    except Exception:
                        pass
            except Exception:
                pass

        # 金银比
        if "xau_price" in result and "xag_price" in result and result["xag_price"] > 0:
            result["gold_silver_ratio"] = result["xau_price"] / result["xag_price"]

        return result if len(result) > 1 else None

    except ImportError:
        logger.warning("ccxt not installed")
        return None
    except Exception as e:
        logger.warning("Binance fetch failed: %s", e)
        return None
