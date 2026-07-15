"""Trade analysis: RV(20d) vs RV(10d) comparison PDF.
Exit strategy: Band Exit (bp>0.90) + Pullback protection + 10d timeout.
Shows entry signals, exit signals (red triangles), and trade trajectories.
"""
import pandas as pd, numpy as np, sys, warnings, os
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
    build_band, generate_v2_signals, compute_rv_pctile)


def build_trades(close, high, dates_all, buy_call, sell_put, exit_sig,
                 max_hold=10, peak_thresh=2.0, dd_thresh=1.5):
    """Build trades with Band Exit + Pullback + Timeout."""
    entries = []
    for d in dates_all:
        if buy_call.get(d, False):
            entries.append((d, 'BUY CALL', close[d]))
        elif sell_put.get(d, False):
            entries.append((d, 'SELL PUT', close[d]))

    trades = []
    for entry_date, sig_type, entry_price in entries:
        future = dates_all[dates_all > entry_date][:max_hold]
        if len(future) == 0:
            continue
        exit_date, exit_type, exit_price = None, 'Timeout', None
        peak = entry_price
        traj = [(entry_date, entry_price)]

        for fd in future:
            fc, fh = close[fd], high.get(fd, close[fd])
            peak = max(peak, fh)
            traj.append((fd, fc))
            ppct = (peak / entry_price - 1) * 100
            dd = (peak - fc) / peak * 100

            if exit_sig.get(fd, False):
                exit_date, exit_type, exit_price = fd, 'Band Exit', fc
                break
            if ppct > peak_thresh and dd >= dd_thresh:
                exit_date, exit_type, exit_price = fd, 'Pullback', fc
                break

        if exit_date is None:
            exit_date, exit_price = future[-1], close[future[-1]]

        g = (exit_price / entry_price - 1) * 100
        hd = len([d for d in future if d <= exit_date])
        trades.append(dict(
            entry_date=entry_date, exit_date=exit_date,
            sig_type=sig_type, exit_type=exit_type,
            entry_price=entry_price, exit_price=exit_price,
            gain=g, hold_days=hd, trajectory=traj))
    return trades


def draw_panel(ax, close, dates_all, upper_band, lower_band,
               trades, exit_sig, rv_pctile, regime, title):
    """Draw one trade panel: price + band + entries + exits + trajectories."""
    cl = close.loc[dates_all]

    # GLD price
    ax.plot(dates_all, cl, 'k-', lw=1.5, alpha=0.85, zorder=3)

    # Hybrid Band (actual trading band)
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

    # Exit signals (red downward triangles) — standalone, not part of trades
    ex_dates = [d for d in dates_all if exit_sig.get(d, False)]
    if ex_dates:
        ax.scatter(ex_dates, [cl.get(d) for d in ex_dates],
                   marker='v', s=140, color='#F44336', edgecolors='darkred',
                   linewidths=0.8, zorder=5)
        for d in ex_dates:
            ax.annotate(f"{d.strftime('%m/%d')}", xy=(d, cl.get(d)),
                        xytext=(0, 12), textcoords='offset points',
                        fontsize=6.5, ha='center', color='#F44336',
                        fontweight='bold')

    # Trade trajectories
    for t in trades:
        td = [x[0] for x in t['trajectory']]
        tp = [x[1] for x in t['trajectory']]
        c = '#2196F3' if t['sig_type'] == 'BUY CALL' else '#FF9800'
        alpha = 0.85 if t['gain'] > 0 else 0.45
        ax.plot(td, tp, '-', color=c, lw=2.2, alpha=alpha, zorder=4)

        # Entry marker
        ax.scatter([t['entry_date']], [t['entry_price']], marker='^',
                   s=160, color=c, edgecolors='black', linewidths=0.8, zorder=6)
        # Entry label
        rv_val = rv_pctile.get(t['entry_date'], np.nan)
        rv_txt = f" RV{rv_val:.0%}" if not np.isnan(rv_val) else ""
        ax.annotate(f"{t['entry_date'].strftime('%m/%d')}{rv_txt}",
                    xy=(t['entry_date'], t['entry_price']),
                    xytext=(0, -18), textcoords='offset points',
                    fontsize=6.5, ha='center', color=c, fontweight='bold')

        # Exit marker (by type)
        if t['exit_type'] == 'Band Exit':
            mk, mc = 'v', '#F44336'
        elif t['exit_type'] == 'Pullback':
            mk, mc = 's', '#FF6600'
        else:
            mk, mc = 'X', 'gray'
        ax.scatter([t['exit_date']], [t['exit_price']], marker=mk,
                   s=120, color=mc, edgecolors='black', linewidths=0.5, zorder=7)

        # Gain annotation
        oy = 12 if t['gain'] > 0 else -16
        ax.annotate(f"{t['gain']:+.1f}% ({t['hold_days']}d)",
                    xy=(t['exit_date'], t['exit_price']),
                    xytext=(6, oy), textcoords='offset points',
                    fontsize=7, color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              alpha=0.8, ec='none'))

    # RV twin axis
    ax2 = ax.twinx()
    rv_plot = rv_pctile.reindex(dates_all).dropna()
    ax2.plot(rv_plot.index, rv_plot.values * 100, color='purple',
             lw=0.8, ls='--', alpha=0.3, zorder=1)
    ax2.axhline(85, color='purple', lw=0.5, ls=':', alpha=0.3)
    ax2.set_ylabel('RV%', fontsize=8, color='purple', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', colors='purple', labelsize=7)
    for lab in ax2.get_yticklabels():
        lab.set_alpha(0.4)

    # Summary
    if trades:
        tdf = pd.DataFrame(trades)
        n_bc = (tdf['sig_type'] == 'BUY CALL').sum()
        n_sp = (tdf['sig_type'] == 'SELL PUT').sum()
        n_ex = len(ex_dates)
        tg = tdf['gain'].mean()
        wr = (tdf['gain'] > 0).mean()
        summary = (f"Buy Call ({n_bc}) + Sell Put ({n_sp}) + Exit ({n_ex}) | "
                   f"Avg: {tg:+.1f}% | WR: {wr:.0%} | Hold: {tdf['hold_days'].mean():.1f}d")
        ax.text(0.99, 0.02, summary, transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='right', va='bottom',
                bbox=dict(fc='lightyellow', ec='gray', alpha=0.85))

    # Legend
    legend_el = [
        Line2D([0], [0], color='black', lw=1.5, label='GLD Close'),
        Line2D([0], [0], color='green', lw=1, alpha=0.6, label='Upper Band'),
        Line2D([0], [0], color='magenta', lw=1, alpha=0.6, label='Lower Band'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2196F3',
               markersize=10, label='Buy Call'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF9800',
               markersize=10, label='Sell Put'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F44336',
               markersize=10, label='Exit (bp>0.90)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6600',
               markersize=9, label='Pullback'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markersize=10, label='Timeout'),
    ]
    ax.legend(handles=legend_el, loc='upper left', fontsize=7.5, ncol=4,
              framealpha=0.9)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('GLD ($)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())


def main():
    print("Loading data...")
    config = load_config()
    features, _ = load_dataset(config)
    gld = pd.read_csv('data/raw/market/gld.csv', index_col=0, parse_dates=True)
    common = features.index.intersection(gld.index)
    features, gld = features.loc[common], gld.loc[common]
    close, high = gld['Close'], gld['High']

    feat_cols_r = [c for c in features.columns if not c.startswith('fwd_')]
    regime = RegimeClassifier().classify(features[feat_cols_r])['regime']

    start = '2025-09-01'
    dates_all = close.index[close.index >= start]

    # ── RV(20d) ──
    print("RV(20d)...")
    oos_20 = pd.read_parquet('data/models/dl_range_v2_oos_rv20d_backup.parquet')
    rv_20d = features['rv_20d']
    rvp_20 = compute_rv_pctile(rv_20d)
    ub20, lb20, bp20 = build_band(oos_20, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp20_s = bp20.reindex(bp20.dropna().index)
    rv20_p = rvp_20.reindex(bp20_s.index)
    ib20 = (regime.reindex(bp20_s.index) == 'Bull')
    bc20, sp20, ex20 = generate_v2_signals(bp20_s, rv20_p, ib20)
    trades_20 = build_trades(close, high, dates_all, bc20, sp20, ex20)

    # ── RV(10d) ──
    print("RV(10d)...")
    oos_10 = pd.read_parquet('data/models/dl_range_v2_oos.parquet')
    rv_10d = features['rv_10d']
    rvp_10 = compute_rv_pctile(rv_10d)
    ub10, lb10, bp10 = build_band(oos_10, close, upper_lags=(1,), lower_lags=(1, 2, 3))
    bp10_s = bp10.reindex(bp10.dropna().index)
    rv10_p = rvp_10.reindex(bp10_s.index)
    ib10 = (regime.reindex(bp10_s.index) == 'Bull')
    bc10, sp10, ex10 = generate_v2_signals(bp10_s, rv10_p, ib10)
    trades_10 = build_trades(close, high, dates_all, bc10, sp10, ex10)

    # Print
    for label, trades in [('RV(20d)', trades_20), ('RV(10d)', trades_10)]:
        print(f"\n=== {label}: {len(trades)} trades ===")
        for t in trades:
            print(f"  {t['entry_date'].date()} {t['sig_type']:<10} -> "
                  f"{t['exit_date'].date()} {t['exit_type']:<11} "
                  f"${t['entry_price']:.0f}->${t['exit_price']:.0f} "
                  f"{t['gain']:+.1f}% ({t['hold_days']}d)")

    # ── PDF ──
    print("\nDrawing...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
    fig.suptitle('GLD Trade Analysis: RV(20d) vs RV(10d) — 2025-09 ~ 2026-03',
                 fontsize=16, fontweight='bold', y=0.98)

    draw_panel(ax1, close, dates_all, ub20, lb20, trades_20, ex20,
               rvp_20, regime, 'RV(20d) — Old Model')
    draw_panel(ax2, close, dates_all, ub10, lb10, trades_10, ex10,
               rvp_10, regime, 'RV(10d) — New Model')
    ax2.set_xlabel('Date', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = 'outputs/trade_comparison_20d_vs_10d.pdf'
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    out_png = 'outputs/trade_comparison_20d_vs_10d.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
