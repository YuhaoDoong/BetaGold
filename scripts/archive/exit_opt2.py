"""Exit strategy optimization v2: test Band+MACDweak combinations."""
import pandas as pd, numpy as np, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.analysis_method_compare import (
    build_band, generate_v2_signals, compute_rv_pctile,
    compute_tp_indicators)


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

    ub, lb, bp = build_band(oos, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp_dates = bp.dropna().index
    bp_s = bp.reindex(bp_dates)
    rv_p = rv_pctile.reindex(bp_dates)
    is_bull = (regime.reindex(bp_dates) == 'Bull')
    buy_call, sell_put, exit_sig = generate_v2_signals(bp_s, rv_p, is_bull)

    buy_dates = pd.DatetimeIndex(
        [d for d in bp_dates
         if buy_call.get(d, False) or sell_put.get(d, False)])
    print(f"Buy signals: {len(buy_dates)}")

    tp_ind = compute_tp_indicators(close, high, low)
    macd_hist = tp_ind['macd_hist']
    rsi = tp_ind['rsi']
    all_dates = close.index

    def run(name, check_fn, max_hold=10):
        trades = []
        for ed in buy_dates:
            if ed not in all_dates:
                continue
            ep = close[ed]
            loc = all_dates.get_loc(ed)
            window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
            if len(window) == 0:
                continue
            exit_date, exit_type = None, 'Timeout'
            peak = ep
            for i, d in enumerate(window):
                c = close.get(d, ep)
                h = high.get(d, c)
                peak = max(peak, h)
                result = check_fn(d, ep, c, h, peak, i)
                if result:
                    exit_date, exit_type = d, result
                    break
            if exit_date is None:
                exit_date = window[-1]
            xp = close.get(exit_date, ep)
            g = (xp / ep - 1) * 100
            hd = all_dates.get_loc(exit_date) - loc
            trades.append(dict(gain=g, hold_days=hd, exit_type=exit_type))

        tdf = pd.DataFrame(trades)
        avg = tdf['gain'].mean()
        wr = (tdf['gain'] > 0).mean() * 100
        hold = tdf['hold_days'].mean()
        med = tdf['gain'].median()
        worst = tdf['gain'].min()
        ratio = avg / tdf['gain'].std() if tdf['gain'].std() > 0 else 0
        bd = tdf.groupby('exit_type')['gain'].agg(['count', 'mean'])
        return dict(name=name, n=len(tdf), avg=avg, wr=wr, hold=hold,
                    med=med, worst=worst, ratio=ratio, bd=bd)

    def _mh(d):
        d_loc = all_dates.get_loc(d)
        return (macd_hist.get(d, 0),
                macd_hist.get(all_dates[d_loc - 1], 0) if d_loc > 0 else 0,
                macd_hist.get(all_dates[d_loc - 2], 0) if d_loc > 1 else 0)

    def _rsi(d):
        d_loc = all_dates.get_loc(d)
        prev_d = all_dates[d_loc - 1] if d_loc > 0 else None
        return rsi.get(d, 50), rsi.get(prev_d, 50) if prev_d else 50

    # ── Strategies ──

    def band_only(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'

    def band_pb(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        ppct = (peak / ep - 1) * 100
        dd = (peak - c) / peak * 100
        if ppct > 2.0 and dd >= 1.5:
            return 'Pullback'

    def band_mw(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        gpct = (c / ep - 1) * 100
        mh, mhp, mhp2 = _mh(d)
        if (gpct > 1.0 and mh > 0 and mh < mhp and mhp < mhp2
                and mhp2 > 0):
            return 'MACDweak'

    def band_mw_pb(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        ppct = (peak / ep - 1) * 100
        dd = (peak - c) / peak * 100
        gpct = (c / ep - 1) * 100
        if ppct > 2.0 and dd >= 1.5:
            return 'Pullback'
        mh, mhp, mhp2 = _mh(d)
        if (gpct > 1.0 and mh > 0 and mh < mhp and mhp < mhp2
                and mhp2 > 0):
            return 'MACDweak'

    def band_smart_all(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        ppct = (peak / ep - 1) * 100
        dd = (peak - c) / peak * 100
        gpct = (c / ep - 1) * 100
        mh, mhp, mhp2 = _mh(d)
        rsi_cur, rsi_prev = _rsi(d)
        if ppct > 2.0 and dd >= 1.5:
            return 'Pullback'
        if mhp >= 0 and mh < 0 and gpct > 0.3:
            return 'MACD'
        if (gpct > 1.0 and mh > 0 and mh < mhp and mhp < mhp2
                and mhp2 > 0):
            return 'MACDweak'
        if rsi_prev > 70 and rsi_cur < 60 and gpct > 0:
            return 'RSI'

    strategies = [
        ('Band only',              band_only),
        ('Band+Pullback',          band_pb),
        ('Band+MACDweak',          band_mw),
        ('Band+MACDweak+PB',       band_mw_pb),
        ('Band+SmartTP(all)',      band_smart_all),
    ]

    results = []
    for name, fn in strategies:
        r = run(name, fn)
        results.append(r)

    print(f"\n{'Strategy':<22} {'N':>4} {'Avg':>7} {'Med':>7} {'WR':>5} "
          f"{'Hold':>5} {'Worst':>7} {'Avg/Std':>8}")
    print("=" * 72)
    for r in results:
        print(f"{r['name']:<22} {r['n']:>4} {r['avg']:>+7.2f}% "
              f"{r['med']:>+7.2f}% {r['wr']:>5.0f}% {r['hold']:>5.1f}d "
              f"{r['worst']:>+7.2f}% {r['ratio']:>8.3f}")
        for etype, row in r['bd'].iterrows():
            print(f"  {etype:<14} n={int(row['count']):>3}  "
                  f"avg={row['mean']:>+.2f}%")
    print()

    # Winner
    best = max(results, key=lambda r: r['ratio'])
    print(f">>> Best (risk-adjusted): {best['name']} "
          f"(avg={best['avg']:+.2f}%, wr={best['wr']:.0f}%, "
          f"avg/std={best['ratio']:.3f})")


if __name__ == '__main__':
    main()
