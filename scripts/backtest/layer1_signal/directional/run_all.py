"""v3.7.229 Layer 1 主驱动: trailing windows 10y/5y/3y/1y.

每个 filter 在 4 个 trailing 窗 (全含最新数据) 独立 grid, 选 best 值.
判定: 跨窗 best 一致 → robust; 不一致 → 看趋势 (近窗倾向哪边).

数据: GLD/SLV ETF spot 现货价 OHLC
方法: 严格无 look-ahead (entry=信号日+1 Open, regime min_hold=1, rv_pctile rolling 252)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
import pandas as pd
from scripts.backtest.framework import (build_raw_universe, multi_window_filter,
                                              LAYER1_WINDOWS, apply_filters, score,
                                              trailing_slice, cross_asset_pivot)


FILTER_GRIDS = [
    ("buy_bp",              [0.20, 0.25, 0.30, 0.35, 0.40]),
    ("rv_pctile_max",       [0.50, 0.60, 0.70, 0.75, 0.80, 1.00]),
    ("ret_20d_min",         [-0.20, -0.10, -0.05, -0.03]),
    ("ret_20d_max",         [-0.03, 0.0, 0.03, 0.05, 0.10, 1.00]),
    ("iv_filter_high_min",  [22, 25, 28, 30, 99]),
    ("ma_trend_threshold",  [0.0, 0.95, 0.975, 0.99, 1.0]),
]


def main():
    out_dir = Path("/Users/yhdong/Gold/data/backtest_history/v3.7.229_layer1")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = []

    for asset in ["GLD", "SLV"]:
        print(f"\n{'='*100}\n资产: {asset}  (trailing 10y/5y/3y/1y, 全含最新数据)\n{'='*100}")
        raw, _ = build_raw_universe(asset)
        print(f"  data: {raw.index.min().date()} → {raw.index.max().date()}")

        # baseline per window (无任何 filter, 仅 Bull)
        print(f"\n  [Baseline (Bull-only, 无 filter)]:")
        for label, days in LAYER1_WINDOWS:
            sub_raw = trailing_slice(raw, days)
            sub = apply_filters(sub_raw, {})
            s = score(sub, 10)
            print(f"    {label} ({sub_raw.index.min().date()} → {sub_raw.index.max().date()}): "
                  f"n={s.get('n')} WR={s.get('WR')}% mean={s.get('mean')}% "
                  f"sum={s.get('sum')} scoreB={s.get('scoreB')}")

        # 各 filter
        for fname, grid in FILTER_GRIDS:
            base = {} if fname == "buy_bp" else {"buy_bp": 0.30}
            df = multi_window_filter(raw, fname, grid, base, LAYER1_WINDOWS,
                                            horizon=10, min_n=5)
            if not len(df) or df["best_value"].notna().sum() == 0:
                continue
            print(f"\n  --- filter: {fname} (baseline={base}) ---")
            disp_cols = ["window", "best_value", "n", "WR", "mean", "sum",
                          "max_loss", "scoreB"]
            have = [c for c in disp_cols if c in df.columns]
            print(df[have].to_string(index=False))
            # 一致性: 跨窗 best 集中度
            best_dist = df["best_value"].dropna().value_counts().to_dict()
            consistent = len(best_dist) == 1
            trend = ""
            if not consistent:
                # 看 trend: 近窗 (1y) 跟远窗 (10y) 是否同方向
                v1 = df[df["window"] == "1y"]["best_value"].iloc[0] \
                     if "1y" in df["window"].values else None
                v10 = df[df["window"] == "10y"]["best_value"].iloc[0] \
                      if "10y" in df["window"].values else None
                if v1 is not None and v10 is not None:
                    trend = f", 1y={v1} vs 10y={v10}"
            print(f"  跨窗 best 分布: {best_dist}  {'★稳健' if consistent else '⚠️不一致'}{trend}")
            all_summary.append({
                "asset": asset, "filter": fname,
                "best_dist": str(best_dist),
                "consistent": consistent,
                "best_10y": df[df["window"] == "10y"]["best_value"].iloc[0] if "10y" in df["window"].values else None,
                "best_5y": df[df["window"] == "5y"]["best_value"].iloc[0] if "5y" in df["window"].values else None,
                "best_3y": df[df["window"] == "3y"]["best_value"].iloc[0] if "3y" in df["window"].values else None,
                "best_1y": df[df["window"] == "1y"]["best_value"].iloc[0] if "1y" in df["window"].values else None,
            })
            df.to_csv(out_dir / f"{asset}_{fname}.csv", index=False)

    # 总览
    print(f"\n\n=== Layer 1 总览 ===")
    df_sum = pd.DataFrame(all_summary)
    print(df_sum.to_string(index=False))
    df_sum.to_csv(out_dir / "_summary.csv", index=False)

    # Cross-asset (SLV→GLD, GLD→SLV) 跨 trailing window
    print(f"\n\n{'='*100}\nCross-asset: 各 trailing 窗 各 tier 同步性\n{'='*100}")
    raw_gld, ohlc_gld = build_raw_universe("GLD")
    raw_slv, ohlc_slv = build_raw_universe("SLV")
    cross_rows = []
    for label, days in LAYER1_WINDOWS:
        for src_asset, raw_src, ohlc_dst, dst_label in [
            ("SLV", raw_slv, ohlc_gld, "GLD"),
            ("GLD", raw_gld, ohlc_slv, "SLV"),
        ]:
            sub_src = trailing_slice(raw_src, days)
            for tier in ["S", "A", "S+A", "B", "ALL"]:
                s = cross_asset_pivot(sub_src, ohlc_dst, tier, 10)
                if s["n"] >= 3:
                    cross_rows.append({
                        "window": label, "src": src_asset, "tier": tier,
                        "dst": dst_label, **s})
    df_cross = pd.DataFrame(cross_rows)
    print(df_cross.to_string(index=False))
    df_cross.to_csv(out_dir / "_cross_asset.csv", index=False)


if __name__ == "__main__":
    main()
