"""v3.7.224: 二维 walk-forward grid — 看参数交互效应.

测试组合 (跨 fold 联合优化):
  1. (iv_filter_high_min, iv_high_bp_low_max) — IV filter 交互
  2. (tier_s_rv_max, tier_s_bp_low_max) — Tier S 边界
  3. (rv_pctile_max_hard, ret_20d_max_hard) — Hard filter 交互

每 fold: train 上扫所有组合, 选 scoreB max, 应用到 test, 比 prod 固定组合.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np

from scripts.walk_forward_full import (
    build_inputs, run_with_override, fwd_returns, score)
from core.strategy_config import get_config


def walk_forward_2d(asset: str, inputs: dict,
                       param1: str, grid1: list,
                       param2: str, grid2: list,
                       train_years: int = 4,
                       horizon: int = 10) -> pd.DataFrame:
    ohlc = inputs["ohlc"]
    all_years = sorted(set(ohlc.index.year))
    folds = []
    for test_year in range(all_years[train_years], all_years[-1] + 1):
        train_start = test_year - train_years
        train_lo = pd.Timestamp(f"{train_start}-01-01")
        train_hi = pd.Timestamp(f"{test_year - 1}-12-31")
        test_lo = pd.Timestamp(f"{test_year}-01-01")
        test_hi = pd.Timestamp(f"{test_year}-12-31")

        # train 上扫 grid1 × grid2
        sig_cache = {}  # (v1, v2) -> sig
        train_results = []
        for v1 in grid1:
            for v2 in grid2:
                sig = run_with_override(asset, inputs,
                                              {param1: v1, param2: v2})
                sig_cache[(v1, v2)] = sig
                buy = sig[sig["buy_signal"]
                           & (sig.index >= train_lo)
                           & (sig.index <= train_hi)]
                if not len(buy):
                    train_results.append({"v1": v1, "v2": v2, "n": 0,
                                              "scoreB": -1e9})
                    continue
                r = fwd_returns(buy.index, ohlc, [horizon])
                s = score(r[f"r{horizon}d"])
                train_results.append({"v1": v1, "v2": v2, **s})

        valid = [r for r in train_results if r["n"] >= 8]
        if not valid:
            continue
        best = max(valid, key=lambda r: r["scoreB"])
        best_v1, best_v2 = best["v1"], best["v2"]

        # 应用到 test
        sig_best = sig_cache[(best_v1, best_v2)]
        buy_test = sig_best[sig_best["buy_signal"]
                              & (sig_best.index >= test_lo)
                              & (sig_best.index <= test_hi)]
        if len(buy_test):
            r_test = fwd_returns(buy_test.index, ohlc, [horizon])
            test_perf = score(r_test[f"r{horizon}d"])
        else:
            test_perf = {"n": 0, "WR": None, "scoreB": 0, "sum": None}

        # prod baseline
        prod_v1 = getattr(get_config(asset), param1, None)
        prod_v2 = getattr(get_config(asset), param2, None)
        if (prod_v1, prod_v2) in sig_cache:
            sig_prod = sig_cache[(prod_v1, prod_v2)]
        else:
            sig_prod = run_with_override(asset, inputs,
                                                {param1: prod_v1, param2: prod_v2})
        buy_prod = sig_prod[sig_prod["buy_signal"]
                              & (sig_prod.index >= test_lo)
                              & (sig_prod.index <= test_hi)]
        if len(buy_prod):
            r_prod = fwd_returns(buy_prod.index, ohlc, [horizon])
            prod_perf = score(r_prod[f"r{horizon}d"])
        else:
            prod_perf = {"n": 0, "scoreB": 0}

        folds.append({
            "test_year": test_year, "train": f"{train_start}-{test_year-1}",
            "best_v1": best_v1, "best_v2": best_v2,
            "train_n": best["n"], "train_WR": best["WR"],
            "train_scoreB": best["scoreB"],
            "test_n": test_perf["n"], "test_WR": test_perf.get("WR"),
            "test_sum": test_perf.get("sum"), "test_scoreB": test_perf["scoreB"],
            "prod_v1": prod_v1, "prod_v2": prod_v2,
            "prod_n": prod_perf["n"], "prod_scoreB": prod_perf["scoreB"],
        })
    return pd.DataFrame(folds)


COMBOS = [
    # (param1, grid1, param2, grid2)
    ("iv_filter_high_min", [22, 25, 28, 99],
     "iv_high_bp_low_max", [0.05, 0.10, 0.20, 0.30, 1.00]),
    ("rv_pctile_max_hard", [0.60, 0.75, 0.90, 1.00],
     "ret_20d_max_hard", [-0.03, 0.0, 0.03, 0.10, 1.00]),
    ("tier_s_rv_max", [0.50, 0.65, 0.80, 2.00],
     "tier_s_bp_low_max", [0.05, 0.10, 0.15, 0.20]),
]


def main():
    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset}\n{'='*100}")
        inputs = build_inputs(asset)
        for p1, g1, p2, g2 in COMBOS:
            print(f"\n--- {asset} 2D: {p1} × {p2} ---")
            print(f"   prod ({getattr(get_config(asset), p1, '?')}, "
                  f"{getattr(get_config(asset), p2, '?')})")
            df = walk_forward_2d(asset, inputs, p1, g1, p2, g2,
                                       train_years=4)
            if not len(df):
                print("   ⚠️ 无 fold")
                continue
            disp = ["test_year", "best_v1", "best_v2",
                     "train_n", "train_WR", "train_scoreB",
                     "test_n", "test_WR", "test_scoreB",
                     "prod_n", "prod_scoreB"]
            print(df[disp].to_string(index=False))
            # 跨 fold 一致性
            combo_dist = df.groupby(["best_v1", "best_v2"]).size().to_dict()
            print(f"   跨 fold best 组合分布: {combo_dist}")
            avg_test = df[df["test_n"] > 0]["test_scoreB"].mean()
            avg_prod = df[df["prod_n"] > 0]["prod_scoreB"].mean()
            print(f"   avg test scoreB: {avg_test:.2f}  vs prod: {avg_prod:.2f}")
            out = Path(f"/Users/yhdong/Gold/data/backtest_history/"
                         f"wf2d_{asset}_{p1}_X_{p2}.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
