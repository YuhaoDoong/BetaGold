"""Visualize top 3 exit strategies side by side for comparison."""
import pandas as pd, numpy as np, sys, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config
from src.models.data_utils import load_dataset
from src.models.regime_classifier import RegimeClassifier
from src.models.analysis_method_compare import (
    build_band, generate_v2_signals, compute_rv_pctile,
    compute_tp_indicators)


def build_trades(close, high, dates_all, buy_call, sell_put, exit_sig,
                 check_fn, max_hold=10):
    entries = []
    for d in dates_all:
        if buy_call.get(d, False):
            entries.append((d, 'BUY CALL', close[d]))
        elif sell_put.get(d, False):
            entries.append((d, 'SELL PUT', close[d]))

    all_dates = close.index
    trades = []
    for entry_date, sig_type, entry_price in entries:
        loc = all_dates.get_loc(entry_date)
        window = all_dates[loc + 1: min(loc + max_hold + 1, len(all_dates))]
        if len(window) == 0:
            continue
        exit_date, exit_type = None, 'Timeout'
        peak = entry_price
        traj = [(entry_date, entry_price)]

        for i, fd in enumerate(window):
            fc = close.get(fd, entry_price)
            fh = high.get(fd, fc)
            peak = max(peak, fh)
            traj.append((fd, fc))
            result = check_fn(fd, entry_price, fc, fh, peak, i)
            if result:
                exit_date, exit_type = fd, result
                break

        if exit_date is None:
            exit_date, exit_type = window[-1], 'Timeout'

        exit_price = close.get(exit_date, entry_price)
        g = (exit_price / entry_price - 1) * 100
        hd = all_dates.get_loc(exit_date) - loc
        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date,
            sig_type=sig_type, exit_type=exit_type,
            entry_price=entry_price, exit_price=exit_price,
            gain=g, hold_days=hd, trajectory=traj))
    return trades


EXIT_MARKERS = {
    'BandExit': ('v', '#F44336'),
    'Pullback': ('s', '#FF6600'),
    'MACDweak': ('D', '#9C27B0'),
    'MACD':     ('d', '#9C27B0'),
    'RSI':      ('h', '#00BCD4'),
    'Timeout':  ('X', 'gray'),
}
SIG_COLORS = {'BUY CALL': '#2196F3', 'SELL PUT': '#FF9800'}


def draw_panel(ax, close, dates_all, upper_band, lower_band,
               trades, exit_sig, rv_pctile, regime, title):
    cl = close.loc[dates_all]
    entry_dates_set = set(t['entry_date'] for t in trades)
    ax.plot(dates_all, cl, 'k-', lw=1.5, alpha=0.85, zorder=3)

    ub = upper_band.reindex(dates_all).dropna()
    lb = lower_band.reindex(dates_all).dropna()
    ax.plot(ub.index, ub.values, color='green', lw=1, alpha=0.5)
    ax.plot(lb.index, lb.values, color='magenta', lw=1, alpha=0.5)
    cidx = ub.index.intersection(lb.index)
    ax.fill_between(cidx, lb.loc[cidx].values, ub.loc[cidx].values,
                     alpha=0.06, color='green')

    # Regime shading
    reg = regime.reindex(dates_all)
    bull = reg == 'Bull'
    starts = dates_all[bull & (~bull.shift(1, fill_value=False))]
    ends = dates_all[bull & (~bull.shift(-1, fill_value=False))]
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.04, color='green')

    # Exit signals (standalone)
    ex_dates = [d for d in dates_all if exit_sig.get(d, False)
                and d not in entry_dates_set]
    if ex_dates:
        ax.scatter(ex_dates, [cl.get(d) for d in ex_dates],
                   marker='v', s=120, color='#F44336', edgecolors='darkred',
                   linewidths=0.7, zorder=5)
        for d in ex_dates:
            ax.annotate(f"{d.strftime('%m/%d')}", xy=(d, cl.get(d)),
                        xytext=(0, 12), textcoords='offset points',
                        fontsize=6, ha='center', color='#F44336',
                        fontweight='bold')

    # Trade trajectories
    for t in trades:
        td = [x[0] for x in t['trajectory']]
        tp = [x[1] for x in t['trajectory']]
        c = SIG_COLORS[t['sig_type']]
        ax.plot(td, tp, '-', color=c, lw=2,
                alpha=0.85 if t['gain'] > 0 else 0.4, zorder=4)

        ax.scatter([t['entry_date']], [t['entry_price']], marker='^',
                   s=140, color=c, edgecolors='black', linewidths=0.7,
                   zorder=6)
        rv_val = rv_pctile.get(t['entry_date'], np.nan)
        rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
        ax.annotate(f"{t['entry_date'].strftime('%m/%d')}{rv_txt}",
                    xy=(t['entry_date'], t['entry_price']),
                    xytext=(0, -16), textcoords='offset points',
                    fontsize=6, ha='center', color=c, fontweight='bold')

        mk, mc = EXIT_MARKERS.get(t['exit_type'], ('o', 'gray'))
        if t['exit_date'] not in entry_dates_set:
            ax.scatter([t['exit_date']], [t['exit_price']], marker=mk,
                       s=100, color=mc, edgecolors='black', linewidths=0.5,
                       zorder=7)
        oy = (16 if t['exit_date'] in entry_dates_set
              else (12 if t['gain'] > 0 else -14))
        ax.annotate(f"{t['gain']:+.1f}% ({t['hold_days']}d)",
                    xy=(t['exit_date'], t['exit_price']),
                    xytext=(5, oy), textcoords='offset points',
                    fontsize=6.5, color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.8, ec='none'))

    # RV twin axis
    ax2 = ax.twinx()
    rv_plot = rv_pctile.reindex(dates_all).dropna()
    ax2.plot(rv_plot.index, rv_plot.values * 100, color='purple',
             lw=0.7, ls='--', alpha=0.3, zorder=1)
    ax2.axhline(85, color='purple', lw=0.5, ls=':', alpha=0.3)
    ax2.set_ylabel('RV%', fontsize=7, color='purple', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', colors='purple', labelsize=6)
    for lab in ax2.get_yticklabels():
        lab.set_alpha(0.4)

    # Summary
    if trades:
        tdf = pd.DataFrame(trades)
        tg = tdf['gain'].mean()
        wr = (tdf['gain'] > 0).mean()
        hd = tdf['hold_days'].mean()
        ratio = tg / tdf['gain'].std() if tdf['gain'].std() > 0 else 0
        n_bc = (tdf['sig_type'] == 'BUY CALL').sum()
        n_sp = (tdf['sig_type'] == 'SELL PUT').sum()
        n_ex = len(ex_dates)
        summary = (f"BC({n_bc})+SP({n_sp})+Exit({n_ex}) | "
                   f"Avg:{tg:+.1f}% WR:{wr:.0%} "
                   f"Hold:{hd:.1f}d Sharpe:{ratio:.2f}")
        ax.text(0.99, 0.02, summary, transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='gray', alpha=0.85))

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('GLD ($)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())


def main():
    print("Loading...")
    config = load_config()
    features, _ = load_dataset(config)
    gld = pd.read_csv('data/raw/market/gld.csv', index_col=0, parse_dates=True)
    common = features.index.intersection(gld.index)
    features, gld = features.loc[common], gld.loc[common]
    close, high = gld['Close'], gld['High']

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

    tp_ind = compute_tp_indicators(close, high, gld['Low'])
    macd_hist = tp_ind['macd_hist']
    all_dates = close.index

    def _mh(d):
        d_loc = all_dates.get_loc(d)
        return (macd_hist.get(d, 0),
                macd_hist.get(all_dates[d_loc - 1], 0) if d_loc > 0 else 0,
                macd_hist.get(all_dates[d_loc - 2], 0) if d_loc > 1 else 0)

    def check_band_pb(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        ppct = (peak / ep - 1) * 100
        dd = (peak - c) / peak * 100
        if ppct > 2.0 and dd >= 1.5:
            return 'Pullback'

    def check_band_mw(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        gpct = (c / ep - 1) * 100
        mh, mhp, mhp2 = _mh(d)
        if (gpct > 1.0 and mh > 0 and mh < mhp
                and mhp < mhp2 and mhp2 > 0):
            return 'MACDweak'

    def check_band_mw_pb(d, ep, c, h, peak, i):
        if exit_sig.get(d, False):
            return 'BandExit'
        ppct = (peak / ep - 1) * 100
        dd = (peak - c) / peak * 100
        gpct = (c / ep - 1) * 100
        if ppct > 2.0 and dd >= 1.5:
            return 'Pullback'
        mh, mhp, mhp2 = _mh(d)
        if (gpct > 1.0 and mh > 0 and mh < mhp
                and mhp < mhp2 and mhp2 > 0):
            return 'MACDweak'

    start = '2025-09-01'
    dates_all = close.index[close.index >= start]

    strategies = [
        ('Band + Pullback  (Avg/Std=0.508)', check_band_pb),
        ('Band + MACDweak  (Avg/Std=0.485)', check_band_mw),
        ('Band + MACDweak + Pullback  (Avg/Std=0.499)', check_band_mw_pb),
    ]

    print("Building trades...")
    all_trades = []
    for name, fn in strategies:
        trades = build_trades(close, high, dates_all, buy_call, sell_put,
                              exit_sig, fn)
        all_trades.append((name, trades))
        tdf = pd.DataFrame(trades)
        tg = tdf['gain'].mean()
        wr = (tdf['gain'] > 0).mean()
        print(f"  {name}: {len(trades)} trades, avg={tg:+.1f}%, wr={wr:.0%}")
        for t in trades:
            print(f"    {t['entry_date'].date()} {t['sig_type']:<10} -> "
                  f"{t['exit_date'].date()} {t['exit_type']:<10} "
                  f"{t['gain']:+.1f}% ({t['hold_days']}d)")

    # ── PDF: 3 panels stacked ──
    print("\nDrawing...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 18), sharex=True)
    fig.suptitle('Exit Strategy Comparison (RV 10d) — 2025-09 ~ 2026-03',
                 fontsize=16, fontweight='bold', y=0.98)

    # Legend (shared)
    legend_el = [
        Line2D([0], [0], color='black', lw=1.5, label='GLD'),
        Line2D([0], [0], color='green', lw=1, alpha=0.6, label='Upper Band'),
        Line2D([0], [0], color='magenta', lw=1, alpha=0.6, label='Lower Band'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2196F3',
               markersize=9, label='Buy Call'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF9800',
               markersize=9, label='Sell Put'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336',
               markersize=9, label='Exit(bp>0.90)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6600',
               markersize=8, label='Pullback'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#9C27B0',
               markersize=8, label='MACDweak'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markersize=9, label='Timeout'),
    ]

    for (name, trades), ax in zip(all_trades, axes):
        draw_panel(ax, close, dates_all, ub, lb, trades, exit_sig,
                   rv_pctile, regime, name)
        ax.legend(handles=legend_el, loc='upper left', fontsize=7, ncol=5,
                  framealpha=0.9)

    axes[-1].set_xlabel('Date', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = 'outputs/exit_strategy_comparison.pdf'
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    out_png = 'outputs/exit_strategy_comparison.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
