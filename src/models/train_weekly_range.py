"""
周频区间交易回测 (Phase 4A Step 3)

两种策略对比:
1. Edge: 区间边缘交易 (逢低做多, 逢高清仓)
2. Overlay: Regime 为主 + 区间风控 (Bull 默认满仓, 过热减仓, 回调加回)

用法:
    conda activate gold
    python src/models/train_weekly_range.py
"""

import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.weekly_range_signal import WeeklyFairValue, WeeklyRangeSignal

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gld_and_gvz(config):
    raw_dir = config["paths"]["raw_data"]
    gld = pd.read_csv(os.path.join(raw_dir, "market", "gld.csv"),
                      index_col=0, parse_dates=True)
    gld_close = gld["Close"].rename("gld_close")
    gvz_path = os.path.join(raw_dir, "volatility", "gvz.csv")
    gvz = None
    if os.path.exists(gvz_path):
        gvz_df = pd.read_csv(gvz_path, index_col=0, parse_dates=True)
        col = "GVZ" if "GVZ" in gvz_df.columns else gvz_df.columns[0]
        gvz = gvz_df[col].rename("gvz")
    return gld_close, gvz


def compute_oos_fair_values(features, gld_close, log_price, dates, n,
                            correction_span=60):
    oos_fv = pd.Series(dtype=float, name="fair_value")
    min_train, test_size, step = 1260, 252, 252
    cutoff = min_train
    while cutoff < n:
        train_idx = dates[:cutoff]
        end = min(cutoff + test_size, n)
        test_idx = dates[cutoff:end]
        model = WeeklyFairValue(correction_span=correction_span)
        model.fit(features.loc[train_idx], log_price.loc[train_idx])
        overlap_start = max(0, cutoff - 120)
        overlap_idx = dates[overlap_start:end]
        fv_full = model.predict_corrected(
            features.loc[overlap_idx], gld_close.loc[overlap_idx])
        oos_fv = pd.concat([oos_fv, fv_full.loc[test_idx]])
        cutoff += step
    return oos_fv[~oos_fv.index.duplicated(keep="last")]


def eval_strategy(signal_valid, daily_ret, name="Strategy"):
    pos = signal_valid["position"]
    strat_ret = pos * daily_ret
    cum = (1 + strat_ret).cumprod()
    total = cum.iloc[-1] - 1
    n_years = (signal_valid.index[-1] - signal_valid.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252) if strat_ret.std() > 0 else 0
    dd = (cum / cum.cummax() - 1).min()
    exp = (pos.abs() > 0).mean()

    # 交易次数 (position changes)
    pos_chg = (pos != pos.shift()).sum()
    return {
        "name": name, "total": total, "cagr": cagr, "sharpe": sharpe,
        "maxdd": dd, "exposure": exp, "trades": pos_chg,
    }


def trade_stats(signal_valid):
    pos = signal_valid["position"]
    price = signal_valid["gld_close"]
    trades = []
    in_trade = False
    for i in range(1, len(pos)):
        prev_p, curr_p = pos.iloc[i-1], pos.iloc[i]
        if prev_p == 0 and curr_p != 0:
            open_i = i
            in_trade = True
        elif in_trade and curr_p == 0:
            hold = (signal_valid.index[i] - signal_valid.index[open_i]).days
            ret = np.sign(pos.iloc[open_i]) * (price.iloc[i] / price.iloc[open_i] - 1)
            trades.append({"hold_days": hold, "return": ret,
                           "direction": "long" if pos.iloc[open_i] > 0 else "short"})
            in_trade = False
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def run():
    config = load_config()
    features, _ = load_dataset(config)
    gld_close, gvz = load_gld_and_gvz(config)

    print("=" * 70)
    print("  Phase 4A Step 3: 周频区间交易回测")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 对齐
    common = features.index.intersection(gld_close.index)
    features, gld_close = features.loc[common], gld_close.loc[common]
    log_price = np.log(gld_close)
    if gvz is not None:
        gvz = gvz.reindex(common)
    ret_20d = features["ret_20d"] if "ret_20d" in features.columns else None

    valid = features.notna().all(axis=1) & log_price.notna()
    features, gld_close, log_price = features[valid], gld_close[valid], log_price[valid]
    if gvz is not None:
        gvz = gvz[valid]
    if ret_20d is not None:
        ret_20d = ret_20d[valid]

    print(f"  样本: {len(features)}, {features.index.min().date()} ~ {features.index.max().date()}")

    # Regime
    regime = RegimeClassifier().classify(features)["regime"]
    for r in ["Bull", "Mixed", "Bear"]:
        print(f"    {r}: {(regime == r).sum()} ({(regime == r).mean():.0%})")

    # Walk-forward 公允价格
    dates = features.index.sort_values()
    oos_fv = compute_oos_fair_values(features, gld_close, log_price, dates, len(dates))
    oos_idx = oos_fv.index
    gvz_oos = gvz.loc[oos_idx] if gvz is not None else None
    ret_oos = ret_20d.loc[oos_idx] if ret_20d is not None else None
    daily_ret = gld_close.loc[oos_idx].pct_change().fillna(0)

    print(f"  OOS: {len(oos_idx)} 天, {oos_idx.min().date()} ~ {oos_idx.max().date()}")
    n_years = (oos_idx[-1] - oos_idx[0]).days / 365.25

    # ========================================
    # 策略对比: 多种配置
    # ========================================
    print(f"\n{'='*70}")
    print("  策略对比")
    print(f"{'='*70}")

    configs = [
        # Edge 模式
        ("Edge: vm2.0 buy0.25 sell0.75 mix1.0",
         dict(mode="edge", vol_multiplier=2.0, buy_zone=0.25, sell_zone=0.75, mixed_position=1.0)),
        ("Edge: vm2.0 buy0.30 sell0.70 mix0.5",
         dict(mode="edge", vol_multiplier=2.0, buy_zone=0.30, sell_zone=0.70, mixed_position=0.5)),
        ("Edge: vm1.5 buy0.25 sell0.75 mix0.0",
         dict(mode="edge", vol_multiplier=1.5, buy_zone=0.25, sell_zone=0.75, mixed_position=0.0)),
        # Overlay 模式
        ("Overlay: base1.0 reduce0.5 dip0.3",
         dict(mode="overlay", vol_multiplier=2.0, bull_base=1.0, bull_reduce=0.5,
              reduce_zone=0.85, dip_zone=0.30, mixed_base=0.0, mixed_dip=0.5)),
        ("Overlay: base1.0 reduce0.5 mix0.3",
         dict(mode="overlay", vol_multiplier=2.0, bull_base=1.0, bull_reduce=0.5,
              reduce_zone=0.85, dip_zone=0.30, mixed_base=0.3, mixed_dip=0.5)),
        ("Overlay: base1.0 reduce0.0 dip0.3",
         dict(mode="overlay", vol_multiplier=2.0, bull_base=1.0, bull_reduce=0.0,
              reduce_zone=0.90, dip_zone=0.30, mixed_base=0.0, mixed_dip=0.5)),
        ("Overlay: base1.0 noredu mix0.5dip",
         dict(mode="overlay", vol_multiplier=2.0, bull_base=1.0, bull_reduce=1.0,
              reduce_zone=0.99, dip_zone=0.30, mixed_base=0.0, mixed_dip=0.5)),
    ]

    # Baselines
    print(f"\n  {'策略':42s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'暴露':>5s} {'换手':>4s}")
    print(f"  {'-'*75}")

    # Buy&Hold
    bh_cum = (1 + daily_ret).cumprod()
    bh_total = bh_cum.iloc[-1] - 1
    bh_cagr = (1 + bh_total) ** (1/n_years) - 1
    bh_sh = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    bh_dd = (bh_cum / bh_cum.cummax() - 1).min()
    print(f"  {'Buy&Hold':42s} {bh_cagr:+7.1%} {bh_sh:+7.2f} {bh_dd:+7.1%} {1.0:5.0%} {1:>4d}")

    # Pure Regime
    regime_oos = regime.loc[oos_idx]
    regime_pos = (regime_oos == "Bull").astype(float)
    rr = regime_pos * daily_ret
    r_cum = (1 + rr).cumprod()
    r_total = r_cum.iloc[-1] - 1
    r_cagr = (1 + r_total) ** (1/n_years) - 1
    r_sh = rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else 0
    r_dd = (r_cum / r_cum.cummax() - 1).min()
    r_exp = (regime_pos > 0).mean()
    r_trades = (regime_pos != regime_pos.shift()).sum()
    print(f"  {'Pure Regime (Bull=1)':42s} {r_cagr:+7.1%} {r_sh:+7.2f} {r_dd:+7.1%} {r_exp:5.0%} {r_trades:>4d}")

    # Regime Bull+Mixed
    bm_pos = pd.Series(0.0, index=oos_idx)
    bm_pos[regime_oos == "Bull"] = 1.0
    bm_pos[regime_oos == "Mixed"] = 0.5
    bm_r = bm_pos * daily_ret
    bm_cum = (1 + bm_r).cumprod()
    bm_total = bm_cum.iloc[-1] - 1
    bm_cagr = (1 + bm_total) ** (1/n_years) - 1
    bm_sh = bm_r.mean() / bm_r.std() * np.sqrt(252)
    bm_dd = (bm_cum / bm_cum.cummax() - 1).min()
    print(f"  {'Regime Bull=1 Mixed=0.5':42s} {bm_cagr:+7.1%} {bm_sh:+7.2f} {bm_dd:+7.1%} {(bm_pos > 0).mean():5.0%} {(bm_pos != bm_pos.shift()).sum():>4d}")

    print(f"  {'-'*75}")

    best_result = None
    best_sharpe = -999
    best_name = ""

    for name, params in configs:
        sg = WeeklyRangeSignal(**params)
        sig = sg.generate(regime=regime_oos, gld_close=gld_close.loc[oos_idx],
                          fair_value=oos_fv, gvz=gvz_oos, ret_20d=ret_oos)
        sig = sig.dropna(subset=["band_position"])
        dr = daily_ret.loc[sig.index]
        stats = eval_strategy(sig, dr, name)
        print(f"  {name:42s} {stats['cagr']:+7.1%} {stats['sharpe']:+7.2f} "
              f"{stats['maxdd']:+7.1%} {stats['exposure']:5.0%} {stats['trades']:>4d}")
        if stats["sharpe"] > best_sharpe:
            best_sharpe = stats["sharpe"]
            best_result = sig
            best_name = name

    # ========================================
    # 最优策略年度明细
    # ========================================
    print(f"\n{'='*70}")
    print(f"  最优: {best_name} (Sharpe={best_sharpe:+.2f})")
    print(f"{'='*70}")

    dr = daily_ret.loc[best_result.index]
    strat_ret = best_result["position"] * dr

    print(f"\n  {'年份':>6s} {'策略':>8s} {'Regime':>8s} {'B&H':>8s} "
          f"{'Bull%':>6s} {'暴露':>5s}")
    print(f"  {'-'*45}")

    for year in sorted(best_result.index.year.unique()):
        ym = best_result.index.year == year
        yr_cum = (1 + strat_ret[ym]).cumprod().iloc[-1] - 1
        yr_regime = (1 + (regime_pos.loc[best_result.index] * dr)[ym]).cumprod().iloc[-1] - 1
        yr_bh = (1 + dr[ym]).cumprod().iloc[-1] - 1
        bull_pct = (best_result.loc[ym, "regime"] == "Bull").mean()
        exp = (best_result.loc[ym, "position"].abs() > 0).mean()
        print(f"  {year:>6d} {yr_cum:+8.1%} {yr_regime:+8.1%} {yr_bh:+8.1%} "
              f"{bull_pct:6.0%} {exp:5.0%}")

    # 交易统计
    tdf = trade_stats(best_result)
    if len(tdf) > 0:
        print(f"\n  交易: {len(tdf)}笔, 均持仓={tdf['hold_days'].mean():.0f}天 "
              f"(中位{tdf['hold_days'].median():.0f}), "
              f"胜率={(tdf['return'] > 0).mean():.0%}, "
              f"均收益={tdf['return'].mean():+.2%}")
        print(f"  持仓分布: <7天={(tdf['hold_days']<7).mean():.0%}, "
              f"7-21天={((tdf['hold_days']>=7)&(tdf['hold_days']<=21)).mean():.0%}, "
              f"21-60天={((tdf['hold_days']>21)&(tdf['hold_days']<=60)).mean():.0%}, "
              f">60天={(tdf['hold_days']>60).mean():.0%}")

    # ========================================
    # Overlay 参数详细扫描
    # ========================================
    print(f"\n{'='*70}")
    print("  Overlay 参数扫描")
    print(f"{'='*70}")

    print(f"\n  {'reduce':>7s} {'reduce_z':>8s} {'dip_z':>6s} {'mx_base':>7s} {'mx_dip':>6s} "
          f"{'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'暴露':>5s}")
    print(f"  {'-'*68}")

    for bull_reduce in [0.0, 0.3, 0.5, 1.0]:
        for reduce_z in [0.80, 0.85, 0.90]:
            for dip_z in [0.20, 0.30]:
                for mx_b, mx_d in [(0.0, 0.0), (0.0, 0.5), (0.3, 0.5)]:
                    sg = WeeklyRangeSignal(
                        mode="overlay", vol_multiplier=2.0,
                        bull_base=1.0, bull_reduce=bull_reduce,
                        reduce_zone=reduce_z, dip_zone=dip_z,
                        mixed_base=mx_b, mixed_dip=mx_d)
                    sig = sg.generate(
                        regime=regime_oos, gld_close=gld_close.loc[oos_idx],
                        fair_value=oos_fv, gvz=gvz_oos, ret_20d=ret_oos)
                    sig = sig.dropna(subset=["band_position"])
                    dr2 = daily_ret.loc[sig.index]
                    st = eval_strategy(sig, dr2)
                    print(f"  {bull_reduce:7.1f} {reduce_z:8.2f} {dip_z:6.2f} "
                          f"{mx_b:7.1f} {mx_d:6.1f} "
                          f"{st['cagr']:+7.1%} {st['sharpe']:+7.2f} {st['maxdd']:+7.1%} "
                          f"{st['exposure']:5.0%}")

    # 保存
    out_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(out_dir, exist_ok=True)
    best_result.to_parquet(os.path.join(out_dir, "weekly_range_signal_oos.parquet"))
    print(f"\n  结果已保存到 {out_dir}/")


if __name__ == "__main__":
    run()
