"""COT (Commitments of Traders) 数据自动爬虫.

来源: CFTC 每周五发布
  https://www.cftc.gov/dea/newcot/deafut.txt

用法:
    python scripts/fetch_cot.py

输出: data/raw/cot/gold_cot.csv
"""

import os
import logging

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CFTC_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"
GOLD_CODE = "088691"
SILVER_CODE = "084691"

DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Gold", "data", "raw", "cot"))


def fetch_and_parse(commodity_code=GOLD_CODE):
    """从 CFTC 下载并解析 COT 报告."""
    resp = requests.get(CFTC_URL, timeout=30)
    resp.raise_for_status()

    for line in resp.text.strip().split("\n"):
        if commodity_code not in line:
            continue

        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 15:
            continue

        try:
            date = pd.Timestamp(parts[2])
            return {
                "date": date,
                "NonComm_Long": int(parts[8]),
                "NonComm_Short": int(parts[9]),
                "NonComm_Spread": int(parts[10]),
                "Comm_Long": int(parts[11]),
                "Comm_Short": int(parts[12]),
                "Open_Interest": int(parts[7]),
                "NonComm_Net": int(parts[8]) - int(parts[9]),
                "Comm_Net": int(parts[11]) - int(parts[12]),
            }
        except (ValueError, IndexError):
            continue

    return None


def update_cot(commodity="gold"):
    """更新 COT 数据."""
    os.makedirs(DATA_DIR, exist_ok=True)

    code = GOLD_CODE if commodity == "gold" else SILVER_CODE
    fname = f"{commodity}_cot.csv"
    path = os.path.join(DATA_DIR, fname)

    data = fetch_and_parse(code)
    if data is None:
        logger.warning(f"No {commodity} COT data found")
        return False

    new_row = pd.DataFrame({k: [v] for k, v in data.items() if k != "date"},
                            index=pd.DatetimeIndex([data["date"]], name="Date"))

    if os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        if data["date"] <= existing.index[-1]:
            logger.info(f"{commodity} COT already up to date ({existing.index[-1].date()})")
            return True
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_csv(path)
        logger.info(f"{commodity} COT updated: {existing.index[-1].date()} → {data['date'].date()}")
    else:
        new_row.to_csv(path)
        logger.info(f"{commodity} COT created: {data['date'].date()}")

    return True


if __name__ == "__main__":
    update_cot("gold")
    update_cot("silver")
