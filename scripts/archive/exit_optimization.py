"""Exit strategy optimization: find the best exit approach.

Tests multiple exit strategies on all historical buy signals (RV 10d model),
compares avg gain, win rate, avg hold days, and selects the best.
"""
import pandas as pd, numpy as np, sys, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.analysis_method_compare import (
    build_band, generate_v2_signals, compute_rv_pctile,
    compute_tp_indicators, find_tp_exits)


def exit_fixed_tp_sl(entry_dates, close, high, low, tp_pct, sl_pct, max_hold=10):
    """Fixed take-profit / stop-loss exit."""
    all_dates = close.index
    trades = []
    for ed in entry_dates:
        if ed not in all_dates:
            continue
        ep = close[ed]
        loc = all_dates.get_loc(ed)
        window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
        exit_date, exit_type = None, 'Timeout'
        for d in window:
            h, l, c = high.get(d, ep), low.get(d, ep), close.get(d, ep)
            if (h / ep - 1) * 100 >= tp_pct:
                exit_date, exit_type = d, f'TP+{tp_pct}%'
                break
            if (l / ep - 1) * 100 <= -sl_pct:
                exit_date, exit_type = d, f'SL-{sl_pct}%'
                break
        if exit_date is None and len(window) > 0:
            exit_date = window[-1]
        if exit_date is None:
            continue
        xp = close.get(exit_date, ep)
        g = (xp / ep - 1) * 100
        hd = all_dates.get_loc(exit_date) - loc
        trades.append(dict(gain=g, hold_days=hd, exit_type=exit_type))
    return trades


def exit_band_only(entry_dates, close, bp_s, exit_sig, max_hold=10):
    """Exit only on bp > 0.90 or regime exit signal."""
    all_dates = close.index
    trades = []
    for ed in entry_dates:
        if ed not in all_dates:
            continue
        ep = close[ed]
        loc = all_dates.get_loc(ed)
        window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
        exit_date, exit_type = None, 'Timeout'
        for d in window:
            if exit_sig.get(d, False):
                exit_date, exit_type = d, 'BandExit'
                break
        if exit_date is None and len(window) > 0:
            exit_date = window[-1]
        if exit_date is None:
            continue
        xp = close.get(exit_date, ep)
        g = (xp / ep - 1) * 100
        hd = all_dates.get_loc(exit_date) - loc
        trades.append(dict(gain=g, hold_days=hd, exit_type=exit_type))
    return trades


def exit_band_plus_pullback(entry_dates, close, high, exit_sig, max_hold=10,
                            peak_thresh=2.0, dd_thresh=1.5):
    """Band exit + pullback protection."""
    all_dates = close.index
    trades = []
    for ed in entry_dates:
        if ed not in all_dates:
            continue
        ep = close[ed]
        loc = all_dates.get_loc(ed)
        window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
        exit_date, exit_type = None, 'Timeout'
        peak = ep
        for d in window:
            c = close.get(d, ep)
            h = high.get(d, c)
            peak = max(peak, h)
            ppct = (peak / ep - 1) * 100
            dd = (peak - c) / peak * 100
            if exit_sig.get(d, False):
                exit_date, exit_type = d, 'BandExit'
                break
            if ppct > peak_thresh and dd >= dd_thresh:
                exit_date, exit_type = d, 'Pullback'
                break
        if exit_date is None and len(window) > 0:
            exit_date = window[-1]
        if exit_date is None:
            continue
        xp = close.get(exit_date, ep)
        g = (xp / ep - 1) * 100
        hd = all_dates.get_loc(exit_date) - loc
        trades.append(dict(gain=g, hold_days=hd, exit_type=exit_type))
    return trades


def summarize(trades, name):
    if not trades:
        return dict(name=name, n=0, avg=0, wr=0, hold=0)
    tdf = pd.DataFrame(trades)
    return dict(
        name=name, n=len(tdf),
        avg=tdf['gain'].mean(),
        wr=(tdf['gain'] > 0).mean() * 100,
        hold=tdf['hold_days'].mean(),
        median=tdf['gain'].median(),
        worst=tdf['gain'].min(),
    )


def main():
    config = load_config()
    features, _ = load_dataset(config)
    gld = pd.read_csv('data/raw/market/gld.csv', index_col=0, parse_dates=True)
    common = features.index.intersection(gld.index)
    features, gld = features.loc[common], gld.loc[common]
    close, high, low = gld['Close'], gld['High'], gld['Low']

    oos = pd.read_parquet('data/models/dl_range_v2_oos.parquet')
    feat_cols_r = [c for c in features.columns if not c.startswith('fwd_')]
    regime = RegimeClassifier().classify(features[feat_cols_r])['regime']
    rv_10d = features['rv_10d']
    rv_pctile = compute_rv_pctile(rv_10d)

    upper_band, lower_band, bp = build_band(
        oos, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp_dates = bp.dropna().index
    bp_s = bp.reindex(bp_dates)
    rv_p = rv_pctile.reindex(bp_dates)
    is_bull = (regime.reindex(bp_dates) == 'Bull')
    buy_call, sell_put, exit_sig = generate_v2_signals(bp_s, rv_p, is_bull)

    # All buy signals
    buy_dates = []
    for d in bp_dates:
        if buy_call.get(d, False) or sell_put.get(d, False):
            buy_dates.append(d)
    buy_dates = pd.DatetimeIndex(buy_dates)
    print(f"Total buy signals: {len(buy_dates)}")

    tp_ind = compute_tp_indicators(close, high, low)

    # ── Test strategies ──
    results = []

    # 1. Smart TP (all indicators)
    smart_trades = find_tp_exits(buy_dates, close, high, tp_ind, max_hold=10)
    results.append(summarize(
        [dict(gain=t['gain_pct'], hold_days=t['hold_days'],
              exit_type=t['exit_type']) for t in smart_trades],
        'Smart TP (all)'))

    # 2. Band exit only (bp > 0.90)
    band_trades = exit_band_only(buy_dates, close, bp_s, exit_sig, max_hold=10)
    results.append(summarize(band_trades, 'Band Exit only'))

    # 3. Band + Pullback
    bp_trades = exit_band_plus_pullback(
        buy_dates, close, high, exit_sig, max_hold=10)
    results.append(summarize(bp_trades, 'Band + Pullback'))

    # 4. Band + Pullback (looser: peak>1.5%, dd>1%)
    bpl_trades = exit_band_plus_pullback(
        buy_dates, close, high, exit_sig, max_hold=10,
        peak_thresh=1.5, dd_thresh=1.0)
    results.append(summarize(bpl_trades, 'Band + Pullback(loose)'))

    # 5. Fixed TP/SL variations
    for tp, sl in [(2, 3), (3, 3), (2, 5), (1.5, 2)]:
        ft = exit_fixed_tp_sl(buy_dates, close, high, low, tp, sl, max_hold=10)
        results.append(summarize(ft, f'Fixed TP{tp}/SL{sl}'))

    # 6. Timeout only (hold 5d)
    t5 = exit_fixed_tp_sl(buy_dates, close, high, low, 999, 999, max_hold=5)
    results.append(summarize(t5, 'Hold 5d'))

    # 7. Timeout only (hold 10d)
    t10 = exit_fixed_tp_sl(buy_dates, close, high, low, 999, 999, max_hold=10)
    results.append(summarize(t10, 'Hold 10d'))

    # 8. Smart TP but only MACD + Pullback (no RSI/MACDweak)
    # Simulate: find_tp_exits already returns exit_type, filter
    smart_mp = []
    for t in smart_trades:
        if t['exit_type'] in ('MACD', 'Pullback', 'Timeout'):
            smart_mp.append(dict(gain=t['gain_pct'], hold_days=t['hold_days'],
                                 exit_type=t['exit_type']))
        # else: would have been caught by MACD/Pullback first anyway,
        # so re-simulate without RSI/MACDweak
    # Actually need to re-simulate properly. Let me just report the breakdown.

    # ── Print results ──
    print(f"\n{'Strategy':<25} {'N':>4} {'Avg':>7} {'Med':>7} {'WR':>5} "
          f"{'Hold':>5} {'Worst':>7}")
    print("-" * 68)
    for r in results:
        print(f"{r['name']:<25} {r['n']:>4} {r['avg']:>+7.2f}% "
              f"{r.get('median',0):>+7.2f}% {r['wr']:>5.0f}% "
              f"{r['hold']:>5.1f}d {r.get('worst',0):>+7.2f}%")

    # ── Smart TP breakdown by exit type ──
    print(f"\n=== Smart TP Breakdown ===")
    stdf = pd.DataFrame([dict(gain=t['gain_pct'], hold_days=t['hold_days'],
                               exit_type=t['exit_type']) for t in smart_trades])
    for etype, grp in stdf.groupby('exit_type'):
        print(f"  {etype:<12} n={len(grp):>3} avg={grp['gain'].mean():>+.2f}% "
              f"wr={( grp['gain']>0).mean():.0%} hold={grp['hold_days'].mean():.1f}d")

    # ── Visualization ──
    rdf = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Avg gain
    colors = ['#4CAF50' if r['avg'] > 1.5 else '#FFC107' if r['avg'] > 0
              else '#F44336' for r in results]
    axes[0].barh(range(len(rdf)), rdf['avg'], color=colors)
    axes[0].set_yticks(range(len(rdf)))
    axes[0].set_yticklabels(rdf['name'], fontsize=9)
    for i, v in enumerate(rdf['avg']):
        axes[0].text(v + 0.05, i, f'{v:+.2f}%', va='center', fontsize=9)
    axes[0].set_xlabel('Avg Gain (%)')
    axes[0].set_title('Avg Gain', fontweight='bold')
    axes[0].axvline(0, color='black', lw=0.5)

    # Win rate
    axes[1].barh(range(len(rdf)), rdf['wr'], color='#2196F3')
    axes[1].set_yticks(range(len(rdf)))
    axes[1].set_yticklabels(rdf['name'], fontsize=9)
    for i, v in enumerate(rdf['wr']):
        axes[1].text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=9)
    axes[1].set_xlabel('Win Rate (%)')
    axes[1].set_title('Win Rate', fontweight='bold')

    # Hold days
    axes[2].barh(range(len(rdf)), rdf['hold'], color='#FF9800')
    axes[2].set_yticks(range(len(rdf)))
    axes[2].set_yticklabels(rdf['name'], fontsize=9)
    for i, v in enumerate(rdf['hold']):
        axes[2].text(v + 0.1, i, f'{v:.1f}d', va='center', fontsize=9)
    axes[2].set_xlabel('Avg Hold Days')
    axes[2].set_title('Avg Hold', fontweight='bold')

    plt.suptitle(f'Exit Strategy Comparison (n={len(buy_dates)} signals)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/exit_strategy_comparison.png', dpi=150, bbox_inches='tight')
    print('\nSaved: outputs/exit_strategy_comparison.png')


if __name__ == '__main__':
    main()
