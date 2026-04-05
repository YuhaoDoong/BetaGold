"""COT Disaggregated 报告爬虫 — Managed Money/Swap/Producer 分类持仓.

来源: CFTC https://www.cftc.gov/dea/newcot/f_disagg.txt
比 deafut.txt 更详细: 区分 Producer/Swap/Managed Money/Other Reportable

用法:
    python scripts/fetch_cot_disagg.py

输出: data/raw/cot/gold_cot_disagg.csv + silver_cot_disagg.csv
"""

import os
import logging

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CFTC_DISAGG_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
GOLD_CODE = "088691"
SILVER_CODE = "084691"

DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "Gold", "data", "raw", "cot"))


def fetch_and_parse(commodity_code):
    """解析 CFTC Disaggregated 报告."""
    resp = requests.get(CFTC_DISAGG_URL, timeout=30)
    resp.raise_for_status()

    for line in resp.text.strip().split("\n"):
        if commodity_code not in line:
            continue

        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 20:
            continue

        try:
            date = pd.Timestamp(parts[2])
            oi = int(parts[7])

            return {
                "date": date,
                "Open_Interest": oi,
                "Prod_Long": int(parts[8]),
                "Prod_Short": int(parts[9]),
                "Prod_Spread": int(parts[10]),
                "Swap_Long": int(parts[11]),
                "Swap_Short": int(parts[12]),
                "Swap_Spread": int(parts[13]),
                "MM_Long": int(parts[14]),     # Managed Money
                "MM_Short": int(parts[15]),
                "MM_Spread": int(parts[16]),
                "Other_Long": int(parts[17]),   # Other Reportable
                "Other_Short": int(parts[18]),
                "Other_Spread": int(parts[19]),
                "MM_Net": int(parts[14]) - int(parts[15]),
                "Swap_Net": int(parts[11]) - int(parts[12]),
                "Prod_Net": int(parts[8]) - int(parts[9]),
            }
        except (ValueError, IndexError):
            continue

    return None


def update_cot_disagg(commodity="gold"):
    """更新 Disaggregated COT 数据."""
    os.makedirs(DATA_DIR, exist_ok=True)

    code = GOLD_CODE if commodity == "gold" else SILVER_CODE
    fname = f"{commodity}_cot_disagg.csv"
    path = os.path.join(DATA_DIR, fname)

    data = fetch_and_parse(code)
    if data is None:
        logger.warning(f"No {commodity} disagg COT data")
        return False

    new_row = pd.DataFrame({k: [v] for k, v in data.items() if k != "date"},
                            index=pd.DatetimeIndex([data["date"]], name="Date"))

    if os.path.exists(path):
        existing = pd.read_csv(path, index_col=0, parse_dates=True)
        if data["date"] <= existing.index[-1]:
            logger.info(f"{commodity} disagg COT up to date ({existing.index[-1].date()})")
            return True
        combined = pd.concat([existing, new_row])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_csv(path)
        logger.info(f"{commodity} disagg COT: → {data['date'].date()}")
    else:
        new_row.to_csv(path)
        logger.info(f"{commodity} disagg COT created: {data['date'].date()}")

    return True


if __name__ == "__main__":
    update_cot_disagg("gold")
    update_cot_disagg("silver")
