"""月度自动重测 — 跑所有资产 + 全部 grid search + 保存报告.

用法:
    python scripts/monthly_retune.py                  # 全部资产
    python scripts/monthly_retune.py --asset GLD      # 单个
    python scripts/monthly_retune.py --dry-run        # 不写文件

输出:
    - data/tune_history/<asset>_<date>.json: 单次完整结果
    - data/tune_history/latest.json: 最新汇总 (供 dashboard 读取)
    - 控制台: Top 配置 + 当前对比 + 建议改动

定时调度建议:
    Cron (Linux/Mac):
        # 每月 1 号 04:00 SGT (UTC 20:00 月末)
        0 20 28-31 * * cd /Users/yhdong/GoldDash && \\
            python scripts/monthly_retune.py >> /var/log/golddash_tune.log 2>&1

    Launchd (macOS):
        See ~/Library/LaunchAgents/com.golddash.retune.plist (示例附后)

    手动: streamlit dashboard 侧边栏看到"重测过期"提示后点击触发
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from core.strategy_config import ASSET_CONFIGS, get_config


def load_asset_data(asset):
    """加载资产数据 (复用 tune_thresholds.load_asset)."""
    from scripts.tune_thresholds import load_asset
    return load_asset(asset)


def run_grid_search(asset, years=5, step=0.025):
    """跑全部 grid search, 返回 dict."""
    from scripts.tune_thresholds import (
        grid_directional, grid_short_vol, grid_straddle_pctile,
    )
    from core.strategy_config import GRID_PRECISION
    data = load_asset_data(asset)
    cur = get_config(asset)

    print(f"\n[{asset}] 方向性网格 (5y, step={step})...")
    df_dir = grid_directional(data, years, step)
    print(f"[{asset}] SHORT_VOL 网格...")
    df_sv = grid_short_vol(data, years, step)
    print(f"[{asset}] STRADDLE pctile 网格 (step={GRID_PRECISION['rv_pctile']})...")
    df_st = grid_straddle_pctile(data, years, GRID_PRECISION["rv_pctile"])

    def top_n(df, by, n=5):
        if len(df) == 0:
            return []
        return df.nlargest(n, by).to_dict("records")

    cur_dir = df_dir[(df_dir["lo"] == cur.rv_filter_low)
                     & (df_dir["hi"] == cur.rv_filter_high)]
    cur_dir_row = cur_dir.iloc[0].to_dict() if len(cur_dir) else None
    cur_sv = df_sv[(df_sv["lo"] == cur.short_vol_rv_pctile_lo)
                    & (df_sv["hi"] == cur.short_vol_rv_pctile_hi)]
    cur_sv_row = cur_sv.iloc[0].to_dict() if len(cur_sv) else None
    cur_st = (df_st[df_st["th"] == cur.straddle_rv_pctile_max]
              if len(df_st) else df_st)
    cur_st_row = cur_st.iloc[0].to_dict() if len(cur_st) else None

    return {
        "asset": asset,
        "tune_date": datetime.now().isoformat(timespec="seconds"),
        "years": years,
        "step": step,
        "current_config": {
            "rv_filter_low": cur.rv_filter_low,
            "rv_filter_high": cur.rv_filter_high,
            "short_vol_rv_pctile_lo": cur.short_vol_rv_pctile_lo,
            "short_vol_rv_pctile_hi": cur.short_vol_rv_pctile_hi,
            "straddle_rv_pctile_max": cur.straddle_rv_pctile_max,
            "last_tuned": cur.last_tuned,
        },
        "current_perf": {
            "directional": cur_dir_row,
            "short_vol": cur_sv_row,
            "straddle": cur_st_row,
        },
        "directional_top_sharpe": top_n(df_dir, "sharpe"),
        "directional_top_total": top_n(df_dir, "total"),
        "short_vol_top_sharpe": top_n(df_sv, "sharpe"),
        "short_vol_top_total": top_n(df_sv, "total"),
        "straddle_top_sharpe": top_n(df_st, "sharpe") if len(df_st) else [],
    }


def suggest_changes(result):
    """对比当前 vs Top 配置, 建议是否切换 (≥ 5% 改进才建议)."""
    suggestions = []
    cur_p = result["current_perf"]

    if cur_p["directional"] and result["directional_top_sharpe"]:
        cur_sh = cur_p["directional"]["sharpe"]
        best = result["directional_top_sharpe"][0]
        improve = (best["sharpe"] - cur_sh) / cur_sh * 100 if cur_sh > 0 else 0
        if improve >= 5:
            suggestions.append({
                "type": "directional",
                "current": (cur_p["directional"]["lo"], cur_p["directional"]["hi"]),
                "current_sharpe": cur_sh,
                "best": (best["lo"], best["hi"]),
                "best_sharpe": best["sharpe"],
                "improve_pct": improve,
                "note": f"Sharpe 改进 {improve:.1f}% — 建议切换",
            })

    if cur_p["short_vol"] and result["short_vol_top_sharpe"]:
        cur_sh = cur_p["short_vol"]["sharpe"]
        best = result["short_vol_top_sharpe"][0]
        improve = (best["sharpe"] - cur_sh) / cur_sh * 100 if cur_sh > 0 else 0
        if improve >= 5:
            suggestions.append({
                "type": "short_vol",
                "current": (cur_p["short_vol"]["lo"], cur_p["short_vol"]["hi"]),
                "current_sharpe": cur_sh,
                "best": (best["lo"], best["hi"]),
                "best_sharpe": best["sharpe"],
                "improve_pct": improve,
                "note": f"Sharpe 改进 {improve:.1f}% — 建议切换",
            })

    return suggestions


def save_results(result, out_dir):
    """保存到 data/tune_history/."""
    os.makedirs(out_dir, exist_ok=True)
    asset = result["asset"]
    date = result["tune_date"][:10]
    fpath = os.path.join(out_dir, f"{asset}_{date}.json")
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return fpath


def update_latest_summary(out_dir, results_all):
    """汇总所有资产到 latest.json (dashboard 读取用)."""
    latest = {
        "tune_date": datetime.now().isoformat(timespec="seconds"),
        "assets": {r["asset"]: r for r in results_all},
    }
    fpath = os.path.join(out_dir, "latest.json")
    with open(fpath, "w") as f:
        json.dump(latest, f, indent=2, default=str)
    return fpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None,
                         help="指定单个资产, 默认全部")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--dry-run", action="store_true",
                         help="不写文件")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or str(
        Path(__file__).parent.parent.parent / "Gold" / "data" / "tune_history")

    if args.asset:
        assets = [args.asset]
    else:
        assets = list(ASSET_CONFIGS.keys())

    print(f"=== 月度重测开始 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"资产: {assets}, 年数: {args.years}, 步长: {args.step}")
    print(f"输出目录: {out_dir} {'(dry-run)' if args.dry_run else ''}\n")

    results_all = []
    all_suggestions = []
    for asset in assets:
        try:
            result = run_grid_search(asset, args.years, args.step)
            results_all.append(result)
            suggestions = suggest_changes(result)
            all_suggestions.extend(suggestions)

            # 打印简洁报告
            print(f"\n{'='*60}")
            print(f"  [{asset}] 重测结果")
            print(f"{'='*60}")
            cur_p = result["current_perf"]
            if cur_p["directional"]:
                d = cur_p["directional"]
                print(f"当前方向性: {d['lo']}/{d['hi']} | n={d['n']:.0f} "
                      f"胜{d['wr']*100:.0f}% 总{d['total']:+.1f}% "
                      f"Sharpe{d['sharpe']:.3f}")
            if cur_p["short_vol"]:
                d = cur_p["short_vol"]
                print(f"当前 SHORT_VOL: {d['lo']}/{d['hi']} | n={d['n']:.0f} "
                      f"胜{d['wr']*100:.0f}% 总{d['total']:+.1f}% "
                      f"Sharpe{d['sharpe']:.3f}")
            print(f"\nTop 1 方向性 (Sharpe): "
                  f"{result['directional_top_sharpe'][0] if result['directional_top_sharpe'] else 'N/A'}")
            print(f"Top 1 SHORT_VOL (Sharpe): "
                  f"{result['short_vol_top_sharpe'][0] if result['short_vol_top_sharpe'] else 'N/A'}")

            if not args.dry_run:
                fpath = save_results(result, out_dir)
                print(f"\n保存 → {fpath}")
        except Exception as e:
            print(f"[{asset}] 失败: {e}")

    # 汇总
    if results_all and not args.dry_run:
        latest_path = update_latest_summary(out_dir, results_all)
        print(f"\n汇总 → {latest_path}")

    # 建议
    print(f"\n{'='*60}\n建议汇总\n{'='*60}")
    if all_suggestions:
        for s in all_suggestions:
            print(f"  [{s['type']}] 当前 {s['current']} Sharpe {s['current_sharpe']:.3f}")
            print(f"     建议 → {s['best']} Sharpe {s['best_sharpe']:.3f} "
                  f"(改进 {s['improve_pct']:+.1f}%)")
            print(f"     {s['note']}\n")
    else:
        print("  当前所有配置接近最优 (改进 < 5%), 无需切换\n")

    print(f"=== 月度重测完成 ===\n")


if __name__ == "__main__":
    main()
