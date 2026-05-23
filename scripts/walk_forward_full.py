"""v3.7.224: Walk-forward no-look-ahead grid validation.

设计:
  对每个 (asset, param) 跑滚动 train/test:
    train = 前 N 年, test = 1 年, 滚动 N+1 → 最新
    每 fold 在 train 上 grid 找最优, 应用到 test 看 OOS performance.

输入端确认无 look-ahead:
  - rv_pctile = rolling 252 ✓
  - OOS predictions = walk-forward trained ✓
  - build_band = shift(1,2,3) ✓
  - regime = min_hold_days=1 (避免 forward 20d look-ahead) ★ 强制
  - features_all = point-in-time (假设, 见下)

参数列表 (按优先级):
  1. iv_filter_high_min (GLD prod=25, SLV prod=28)
  2. iv_high_bp_low_max (prod=0.10)
  3. rv_pctile_max_hard (GLD prod=0.75)
  4. sp_score_threshold (GLD prod=3.5, SLV prod=2.5)
  5. ret_20d_max_hard (GLD prod=0.03)

判定:
  - Robust: 多 fold 选同 value, train_WR ≈ test_WR
  - Overfit: fold 间选不同 value, OR train_WR >> test_WR
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, yfinance as yf, numpy as np

from core.data import load_oos_predictions, load_config
from core.signals_v2 import generate_daily_signals
from core.signals import build_band, compute_rv_pctile
from core.regime import RegimeClassifier
from core.strategy_config import get_config


def build_inputs(asset: str) -> dict:
    """加载 + 计算所有 no-look-ahead 输入."""
    cfg = load_config()
    if asset == "GLD":
        oos = load_oos_predictions(cfg)
    else:
        oos = pd.read_parquet(Path(cfg["data_root"]) / "models/dl_range_slv_oos.parquet")
    feat_path = ("/Users/yhdong/Gold/data/processed/features_all.parquet"
                  if asset == "GLD" else
                  "/Users/yhdong/Gold/data/processed/features_slv.parquet")
    feat = pd.read_parquet(feat_path)
    ohlc = pd.read_csv(f"/Users/yhdong/Gold/data/raw/market/{asset.lower()}.csv",
                          index_col=0, parse_dates=True)
    common = ohlc.index.intersection(feat.index).intersection(oos.index)
    close = ohlc.loc[common, "Close"]; high = ohlc.loc[common, "High"]; low = ohlc.loc[common, "Low"]
    upper, lower, _ = build_band(oos.loc[common], close)
    rv_p = compute_rv_pctile(feat.loc[common, "rv_10d"])
    feat_cols = [c for c in feat.columns if not c.startswith("fwd_")]
    # ★ min_hold_days=1 消除 forward look-ahead
    regime = RegimeClassifier(min_hold_days=1).classify(
        feat.loc[common, feat_cols])["regime"]
    gvz = yf.Ticker("^GVZ").history(period="10y")
    gvz.index = pd.to_datetime(gvz.index).tz_localize(None).normalize()
    return {
        "close": close, "high": high, "low": low,
        "upper": upper, "lower": lower,
        "rv_p": rv_p, "regime": regime,
        "gvz": gvz["Close"], "ohlc": ohlc,
    }


def run_with_override(asset: str, inputs: dict, overrides: dict) -> pd.DataFrame:
    """临时改 config 参数后跑 generate_daily_signals."""
    cfg = get_config(asset)
    old = {}
    try:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                old[k] = getattr(cfg, k)
                setattr(cfg, k, v)
        sig = generate_daily_signals(
            inputs["close"], inputs["high"], inputs["low"],
            inputs["upper"], inputs["lower"],
            inputs["regime"], inputs["rv_p"],
            asset=asset, gvz_series=inputs["gvz"])
        return sig
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def fwd_returns(buy_idx: pd.DatetimeIndex, ohlc: pd.DataFrame,
                  horizons: list) -> pd.DataFrame:
    """计算 buy_idx 各日的前向 5/10/20 日 close return (open 入场)."""
    out = pd.DataFrame(index=buy_idx)
    for h in horizons:
        for d in buy_idx:
            if d not in ohlc.index: continue
            i = ohlc.index.get_loc(d)
            ent = float(ohlc.iloc[i]["Open"])
            if i + h < len(ohlc):
                ext = float(ohlc.iloc[i + h]["Close"])
                out.loc[d, f"r{h}d"] = (ext / ent - 1) * 100
            else:
                out.loc[d, f"r{h}d"] = np.nan
    return out


def score(returns: pd.Series) -> dict:
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "WR": None, "mean": None, "sum": None, "scoreB": 0,
                 "max_loss": None}
    wr = (s > 0).mean()
    mean = s.mean()
    return {
        "n": n,
        "WR": round(wr * 100, 1),
        "mean": round(mean, 2),
        "sum": round(s.sum(), 1),
        "max_loss": round(s.min(), 1),
        "scoreB": round((wr ** 2) * math.log(1 + n) * mean, 2),
    }


def walk_forward_univariate(asset: str, inputs: dict, param: str,
                              grid: list, train_years: int = 4,
                              horizon: int = 10) -> pd.DataFrame:
    """对单个参数做 walk-forward grid."""
    ohlc = inputs["ohlc"]
    all_years = sorted(set(ohlc.index.year))
    folds = []
    for test_year in range(all_years[train_years], all_years[-1] + 1):
        train_start = test_year - train_years
        train_lo = pd.Timestamp(f"{train_start}-01-01")
        train_hi = pd.Timestamp(f"{test_year - 1}-12-31")
        test_lo = pd.Timestamp(f"{test_year}-01-01")
        test_hi = pd.Timestamp(f"{test_year}-12-31")

        # 对每个 value 跑 sig + filter to train window + score
        train_results = []
        sig_by_value = {}
        for v in grid:
            sig = run_with_override(asset, inputs, {param: v})
            sig_by_value[v] = sig
            buy = sig[sig["buy_signal"]
                       & (sig.index >= train_lo) & (sig.index <= train_hi)]
            if not len(buy):
                train_results.append({"value": v, "scoreB": -1e9, "n": 0})
                continue
            r = fwd_returns(buy.index, ohlc, [horizon])
            s = score(r[f"r{horizon}d"])
            train_results.append({"value": v, **s})

        # 选 train 上 scoreB 最大 (n>=8 保证统计)
        valid = [r for r in train_results if r["n"] >= 8]
        if not valid:
            continue
        best_train = max(valid, key=lambda r: r["scoreB"])
        best_value = best_train["value"]

        # 应用到 test
        sig_best = sig_by_value[best_value]
        buy_test = sig_best[sig_best["buy_signal"]
                              & (sig_best.index >= test_lo)
                              & (sig_best.index <= test_hi)]
        if len(buy_test):
            r_test = fwd_returns(buy_test.index, ohlc, [horizon])
            test_perf = score(r_test[f"r{horizon}d"])
        else:
            test_perf = {"n": 0, "WR": None, "scoreB": 0, "sum": None,
                          "mean": None, "max_loss": None}

        # baseline (用生产值)
        prod_value = getattr(get_config(asset), param, None)
        if prod_value is not None and prod_value in sig_by_value:
            sig_prod = sig_by_value[prod_value]
        else:
            sig_prod = run_with_override(asset, inputs, {param: prod_value})
        buy_test_prod = sig_prod[sig_prod["buy_signal"]
                                    & (sig_prod.index >= test_lo)
                                    & (sig_prod.index <= test_hi)]
        if len(buy_test_prod):
            r_prod = fwd_returns(buy_test_prod.index, ohlc, [horizon])
            prod_perf = score(r_prod[f"r{horizon}d"])
        else:
            prod_perf = {"n": 0, "WR": None, "scoreB": 0}

        folds.append({
            "test_year": test_year,
            "train": f"{train_start}-{test_year-1}",
            "best_value": best_value,
            "train_n": best_train["n"], "train_WR": best_train["WR"],
            "train_scoreB": best_train["scoreB"],
            "test_n": test_perf["n"], "test_WR": test_perf["WR"],
            "test_sum": test_perf["sum"], "test_mean": test_perf["mean"],
            "test_scoreB": test_perf["scoreB"],
            "test_max_loss": test_perf["max_loss"],
            "prod_value": prod_value, "prod_n": prod_perf["n"],
            "prod_WR": prod_perf["WR"], "prod_scoreB": prod_perf["scoreB"],
        })
    return pd.DataFrame(folds)


PARAMS_TO_TEST = [
    # (param_name, grid, horizon)
    ("iv_filter_high_min", [22, 24, 25, 26, 27, 28, 30, 99], 10),
    ("iv_high_bp_low_max", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00], 10),
    ("rv_pctile_max_hard", [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00], 10),
    ("sp_score_threshold", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0], 10),
    ("ret_20d_max_hard", [-0.05, -0.03, 0.0, 0.03, 0.05, 0.10, 1.00], 10),
    # v3.7.225 task C 入场层补充
    ("buy_bp", [0.20, 0.25, 0.30, 0.35, 0.40], 10),
    ("ma_trend_threshold", [0.0, 0.95, 0.97, 0.975, 0.99, 1.0], 10),
]


def main():
    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset}\n{'='*100}")
        print(f"加载输入 (min_hold_days=1, no look-ahead) ...")
        inputs = build_inputs(asset)
        print(f"  date range: {inputs['ohlc'].index.min().date()} → "
              f"{inputs['ohlc'].index.max().date()}")

        for param, grid, horizon in PARAMS_TO_TEST:
            print(f"\n--- {asset} param: {param} (h={horizon}d) ---")
            cfg_attr = getattr(get_config(asset), param, None)
            print(f"   生产值 = {cfg_attr}, grid = {grid}")
            df = walk_forward_univariate(asset, inputs, param, grid,
                                            train_years=4, horizon=horizon)
            if not len(df):
                print(f"   ⚠️ 无 fold 数据")
                continue
            # show
            disp_cols = ["test_year", "train", "best_value",
                          "train_n", "train_WR", "train_scoreB",
                          "test_n", "test_WR", "test_sum", "test_max_loss", "test_scoreB",
                          "prod_n", "prod_WR", "prod_scoreB"]
            print(df[disp_cols].to_string(index=False))
            best_dist = df["best_value"].value_counts().to_dict()
            print(f"   跨 fold best 分布: {best_dist}")
            avg_test = df[df["test_n"] > 0]["test_scoreB"].mean()
            avg_prod = df[df["prod_n"] > 0]["prod_scoreB"].mean()
            print(f"   平均 test scoreB: {avg_test:.2f}  vs prod scoreB: {avg_prod:.2f}")

            # save
            out_path = Path(
                f"/Users/yhdong/Gold/data/backtest_history/"
                f"wf_{asset}_{param}.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
