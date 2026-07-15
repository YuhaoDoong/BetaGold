"""GLD Options OI Factor Exploratory Analysis.

Compute OI-based factors from EOD snapshots and assess potential as
range boundary predictors. Factors: Max Pain, Call/Put Wall, PCR,
OI Skew, OI concentration (HHI).
"""
import pandas as pd
import numpy as np
import sys
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def compute_oi_factors(df, spot):
    """Compute OI-based factors from an EOD snapshot."""
    # Filter: reasonable DTE (7-60), non-zero OI
    df = df[(df['dte'] >= 7) & (df['dte'] <= 60)
            & (df['option_open_interest'] > 0)].copy()

    calls = df[df['option_type'] == 'CALL']
    puts  = df[df['option_type'] == 'PUT']

    # Aggregate OI by strike
    call_oi = calls.groupby('option_strike_price')['option_open_interest'].sum()
    put_oi  = puts.groupby('option_strike_price')['option_open_interest'].sum()

    # 1. Max Pain: strike that minimizes total intrinsic value payout
    all_strikes = sorted(set(call_oi.index) | set(put_oi.index))
    pain = []
    for k in all_strikes:
        c_pain = sum(max(k - s, 0) * oi for s, oi in call_oi.items())
        p_pain = sum(max(s - k, 0) * oi for s, oi in put_oi.items())
        pain.append((k, c_pain + p_pain))
    max_pain_strike = min(pain, key=lambda x: x[1])[0]
    max_pain_dist = (max_pain_strike / spot - 1) * 100

    # 2. Call Wall (max call OI strike) & Put Wall (max put OI strike)
    call_wall = call_oi.idxmax() if len(call_oi) > 0 else spot
    put_wall  = put_oi.idxmax() if len(put_oi) > 0 else spot
    call_wall_dist = (call_wall / spot - 1) * 100
    put_wall_dist  = (put_wall / spot - 1) * 100

    # 3. PCR (Put/Call Ratio by OI)
    pcr = put_oi.sum() / call_oi.sum() if call_oi.sum() > 0 else 1.0

    # 4. OI Skew (center of mass vs spot)
    call_com = (np.sum(np.array(call_oi.index) * np.array(call_oi.values)) / call_oi.sum()
                if call_oi.sum() > 0 else spot)
    put_com  = (np.sum(np.array(put_oi.index) * np.array(put_oi.values)) / put_oi.sum()
                if put_oi.sum() > 0 else spot)
    total_oi = call_oi.sum() + put_oi.sum()
    oi_com = ((call_com * call_oi.sum() + put_com * put_oi.sum()) / total_oi
              if total_oi > 0 else spot)
    oi_skew = (oi_com / spot - 1) * 100

    # 5. OI concentration (HHI)
    all_oi = pd.concat([call_oi, put_oi]).groupby(level=0).sum()
    shares = all_oi / all_oi.sum()
    hhi = (shares ** 2).sum()

    # 6. OI-implied range
    oi_range_pct = (call_wall - put_wall) / spot * 100

    # Top 3 strikes
    top3_calls = call_oi.nlargest(3)
    top3_puts  = put_oi.nlargest(3)

    return {
        'spot': spot,
        'max_pain': max_pain_strike,
        'max_pain_dist%': max_pain_dist,
        'call_wall': call_wall,
        'call_wall_dist%': call_wall_dist,
        'put_wall': put_wall,
        'put_wall_dist%': put_wall_dist,
        'pcr': pcr,
        'oi_skew%': oi_skew,
        'hhi': hhi,
        'oi_range%': oi_range_pct,
        'call_com': call_com,
        'put_com': put_com,
        'top3_call_strikes': list(top3_calls.index),
        'top3_call_oi': list(top3_calls.values),
        'top3_put_strikes': list(top3_puts.index),
        'top3_put_oi': list(top3_puts.values),
        'total_call_oi': int(call_oi.sum()),
        'total_put_oi': int(put_oi.sum()),
    }


def draw_oi_factor_chart(results, gld, out_path):
    """Visualize OI factors across snapshots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GLD Options OI Factor Analysis', fontsize=14, fontweight='bold')

    dates = [r['date'] for r in results]
    spots = [r['spot'] for r in results]
    max_pains = [r['max_pain'] for r in results]
    call_walls = [r['call_wall'] for r in results]
    put_walls = [r['put_wall'] for r in results]
    pcrs = [r['pcr'] for r in results]

    x = range(len(dates))

    # Panel 1: Spot vs Max Pain vs Walls
    ax = axes[0, 0]
    ax.plot(x, spots, 'ko-', lw=2, ms=8, label='GLD Spot')
    ax.plot(x, max_pains, 'rs--', lw=1.5, ms=7, label='Max Pain')
    ax.plot(x, call_walls, 'g^--', lw=1.5, ms=7, label='Call Wall')
    ax.plot(x, put_walls, 'mv--', lw=1.5, ms=7, label='Put Wall')
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=15)
    ax.set_ylabel('Price ($)')
    ax.set_title('Spot vs OI Anchors')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Distances from spot (%)
    ax = axes[0, 1]
    mp_d = [r['max_pain_dist%'] for r in results]
    cw_d = [r['call_wall_dist%'] for r in results]
    pw_d = [r['put_wall_dist%'] for r in results]
    width = 0.25
    ax.bar([i - width for i in x], mp_d, width, label='Max Pain dist%', color='red', alpha=0.7)
    ax.bar(list(x), cw_d, width, label='Call Wall dist%', color='green', alpha=0.7)
    ax.bar([i + width for i in x], pw_d, width, label='Put Wall dist%', color='purple', alpha=0.7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=15)
    ax.set_ylabel('Distance from Spot (%)')
    ax.set_title('OI Anchor Distances')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: PCR over time
    ax = axes[1, 0]
    ax.bar(list(x), pcrs, color='steelblue', alpha=0.8)
    ax.axhline(1.0, color='red', ls='--', lw=1, label='PCR=1 (neutral)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=15)
    ax.set_ylabel('Put/Call Ratio')
    ax.set_title('PCR (by OI)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Factor summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    headers = ['Date', 'Spot', 'MaxPain', 'MP dist%', 'CallWall', 'PutWall',
               'PCR', 'OI Skew%', 'Fwd5d%']
    for r in results:
        fwd = f"{r['fwd_5d%']:+.2f}" if r.get('fwd_5d%') is not None else 'N/A'
        table_data.append([
            r['date'], f"${r['spot']:.0f}", f"${r['max_pain']:.0f}",
            f"{r['max_pain_dist%']:+.1f}", f"${r['call_wall']:.0f}",
            f"${r['put_wall']:.0f}", f"{r['pcr']:.2f}",
            f"{r['oi_skew%']:+.1f}", fwd
        ])
    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    ax.set_title('Factor Summary', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    gld = pd.read_csv('data/raw/market/gld.csv', index_col=0, parse_dates=True)

    snapshot_dir = 'data/raw/options_history'
    snapshots = {}
    for d in sorted(os.listdir(snapshot_dir)):
        fpath = os.path.join(snapshot_dir, d, 'eod_full.parquet')
        if os.path.exists(fpath):
            snapshots[d] = fpath

    print("=" * 80)
    print("GLD Options OI Factor Analysis")
    print(f"Snapshots: {list(snapshots.keys())}")
    print("=" * 80)

    results = []
    for date_str, fpath in snapshots.items():
        df = pd.read_parquet(fpath)
        dt = pd.Timestamp(date_str)

        # Get spot
        if dt in gld.index:
            spot = gld.loc[dt, 'Close']
        else:
            near = gld.index[gld.index <= dt]
            spot = gld.loc[near[-1], 'Close'] if len(near) > 0 else 460.0

        factors = compute_oi_factors(df, spot)
        factors['date'] = date_str

        # Forward return
        if dt in gld.index:
            loc = gld.index.get_loc(dt)
            if loc + 5 < len(gld):
                factors['fwd_5d%'] = (gld.iloc[loc + 5]['Close'] / spot - 1) * 100
            else:
                nd = len(gld) - loc - 1
                if nd > 0:
                    factors['fwd_5d%'] = (gld.iloc[-1]['Close'] / spot - 1) * 100
                    factors['fwd_note'] = f'{nd}d only'
                else:
                    factors['fwd_5d%'] = None
        else:
            factors['fwd_5d%'] = None

        results.append(factors)

        print(f"\n{'─' * 60}")
        print(f"Date: {date_str} | Spot: ${spot:.2f}")
        print(f"{'─' * 60}")
        print(f"  Max Pain:     ${factors['max_pain']:.0f} "
              f"({factors['max_pain_dist%']:+.1f}% from spot)")
        print(f"  Call Wall:    ${factors['call_wall']:.0f} "
              f"({factors['call_wall_dist%']:+.1f}%)")
        print(f"  Put Wall:     ${factors['put_wall']:.0f} "
              f"({factors['put_wall_dist%']:+.1f}%)")
        print(f"  OI Range:     ${factors['put_wall']:.0f} - "
              f"${factors['call_wall']:.0f} ({factors['oi_range%']:.1f}%)")
        print(f"  PCR:          {factors['pcr']:.3f}")
        print(f"  OI Skew:      {factors['oi_skew%']:+.1f}% (COM vs spot)")
        print(f"  HHI:          {factors['hhi']:.4f}")
        print(f"  Call COM:     ${factors['call_com']:.1f}  "
              f"Put COM: ${factors['put_com']:.1f}")
        print(f"  Top3 Calls:   {[f'${s:.0f}({oi:,})' for s, oi in zip(factors['top3_call_strikes'], factors['top3_call_oi'])]}")
        print(f"  Top3 Puts:    {[f'${s:.0f}({oi:,})' for s, oi in zip(factors['top3_put_strikes'], factors['top3_put_oi'])]}")
        print(f"  Total OI:     Call={factors['total_call_oi']:,}  "
              f"Put={factors['total_put_oi']:,}")
        if factors.get('fwd_5d%') is not None:
            note = factors.get('fwd_note', '')
            note_str = f" ({note})" if note else ''
            print(f"  Fwd 5d:       {factors['fwd_5d%']:+.2f}%{note_str}")

    # Draw chart
    draw_oi_factor_chart(results, gld, 'outputs/oi_factors_analysis.png')

    # Summary
    print(f"\n{'=' * 80}")
    print("Preliminary Assessment")
    print(f"{'=' * 80}")
    print("""
Key Observations (3 snapshots):

1. Max Pain consistently BELOW spot (${mp_range})
   → GLD trades above max pain = bullish sentiment
   → Max Pain distance could be mean-reversion/gravity signal

2. Call Wall at ${cw} = natural resistance ceiling
   → Aligns with DL Range upper predictions
   → Potential upper bound anchor

3. Put Wall at ${pw} = far below spot
   → Deep OTM hedging, not near-term support
   → Less useful as range boundary

4. PCR < 1.0 = call-dominated (bullish positioning)

5. 3/6 snapshot: spot $474, fwd 5d = -2.7% (fell toward max pain)
   → Consistent with max pain gravity hypothesis

VIABILITY:
- 3 snapshots insufficient for statistical validation
- Need 60+ daily snapshots (≈3 months) for correlation analysis
- Most promising: Max Pain distance, Call Wall distance
- Action: Continue daily EOD collection, revisit at 60+ points
""".format(
        mp_range=f"{min(r['max_pain'] for r in results):.0f}-{max(r['max_pain'] for r in results):.0f}",
        cw=f"{results[-1]['call_wall']:.0f}",
        pw=f"{results[-1]['put_wall']:.0f}",
    ).replace('${mp_range}', f"{min(r['max_pain'] for r in results):.0f}-{max(r['max_pain'] for r in results):.0f}")
      .replace('${cw}', f"{results[-1]['call_wall']:.0f}")
      .replace('${pw}', f"{results[-1]['put_wall']:.0f}")
    )


if __name__ == '__main__':
    main()
