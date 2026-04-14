#!/usr/bin/env python3
"""
Round 2 parameter sweep: fine-tune around Cancel=4min winner.

Groups:
  A – Fine-tune cancel window (3m..5m)
  B – Cancel=4m + VIX threshold
  C – Cancel=4m + TP price
  D – Cancel=4m + Stop Loss
  E – Best combo from A-D + extras (weekend, side, consec-loss halt, risk)
  F – Cancel=4m + session filter

Usage:
    python param_sweep_r2.py                  # full sweep
    python param_sweep_r2.py --markets 3000   # faster with fewer markets
"""
import sys
import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Engine

_dir = os.path.dirname(os.path.abspath(__file__))
MPLDIR = os.path.join(os.path.dirname(_dir), '.matplotlib')
os.environ['MPLCONFIGDIR'] = MPLDIR

NUM_MARKETS = 3000
for i, arg in enumerate(sys.argv):
    if arg == '--markets' and i + 1 < len(sys.argv):
        NUM_MARKETS = int(sys.argv[i + 1])


BASELINE = {
    'buy_price': 0.10,
    'sell_price': 0.33,
    'entry_cutoff_seconds': 360,
    'risk_pct': 0.02,
    'stop_loss_price': 0.0,
    'trailing_stop_trigger': 0.0,
    'trailing_stop_floor': 0.0,
    'vix_min': 0.0,
    'weekend_size_mult': 1.0,
    'single_side': 'both',
    'session_hours': (0, 24),
    'max_consecutive_losses_halt': 0,
}


def build_groups_a_to_d():
    """Groups A-D: cancel fine-tune, VIX, TP, SL around cancel=4m."""
    variants = [('BASELINE', {})]

    # Group A: cancel window fine-tuning
    for secs in [180, 210, 240, 270, 300]:
        label = f'Cancel={secs}s'
        variants.append((label, {'entry_cutoff_seconds': secs}))

    # Group B: Cancel=4m + VIX
    for vix in [16, 17, 18, 19, 20, 22]:
        variants.append((f'C4m+VIX>={vix}', {
            'entry_cutoff_seconds': 240, 'vix_min': float(vix),
        }))

    # Group C: Cancel=4m + TP
    for tp in [0.30, 0.33, 0.35, 0.38, 0.40]:
        variants.append((f'C4m+TP={tp}', {
            'entry_cutoff_seconds': 240, 'sell_price': tp,
        }))

    # Group D: Cancel=4m + SL
    for sl in [0.04, 0.05, 0.06]:
        variants.append((f'C4m+SL={sl}', {
            'entry_cutoff_seconds': 240, 'stop_loss_price': sl,
        }))

    return variants


def build_group_e(best_combo_overrides, best_combo_label):
    """Group E: best combo from A-D + extras."""
    base = {**best_combo_overrides}
    tag = best_combo_label.replace(' ', '')
    variants = []

    # Weekend: skip (0.0) or half-size (0.5)
    for wm in [0.0, 0.5]:
        variants.append((f'{tag}+Wknd={wm}', {**base, 'weekend_size_mult': wm}))

    # NO-only
    variants.append((f'{tag}+NO-only', {**base, 'single_side': 'no'}))

    # Consecutive-loss halt at 3
    variants.append((f'{tag}+ConsecHalt=3', {**base, 'max_consecutive_losses_halt': 3}))

    # Risk sizing
    for rp in [0.01, 0.015, 0.02]:
        variants.append((f'{tag}+Risk={rp*100:.1f}%', {**base, 'risk_pct': rp}))

    return variants


def build_group_f():
    """Group F: Cancel=4m + session filters."""
    c4 = {'entry_cutoff_seconds': 240}
    return [
        ('C4m+Asian(0-8UTC)', {**c4, 'session_hours': (0, 8)}),
        ('C4m+US(14-21UTC)', {**c4, 'session_hours': (14, 21)}),
        ('C4m+SkipLondon(14-8)', {**c4, 'session_hours': (14, 8)}),
        ('C4m+PeakUS(16-22UTC)', {**c4, 'session_hours': (16, 22)}),
    ]


def run_variant(label, overrides):
    """Run engine with parameter overrides. Returns metrics dict."""
    params = {**BASELINE, **overrides}

    engine = Engine(
        series_ticker='KXBTC15M',
        bankroll=1000.0,
        num_markets=NUM_MARKETS,
        volume_fill_pct=0.10,
        buy_price=params['buy_price'],
        sell_price=params['sell_price'],
        risk_pct=params['risk_pct'],
        tp_fill_rate=1.0,
        stop_loss_price=params['stop_loss_price'],
        trailing_stop_trigger=params['trailing_stop_trigger'],
        trailing_stop_floor=params['trailing_stop_floor'],
        vix_min=params['vix_min'],
        weekend_size_mult=params['weekend_size_mult'],
        single_side=params['single_side'],
        entry_cutoff_seconds=params['entry_cutoff_seconds'],
        session_hours=params['session_hours'],
        max_consecutive_losses_halt=params['max_consecutive_losses_halt'],
    )

    result = engine.run(silent=True)

    filled = result['markets_filled']
    tp = result['tp_count']
    settled = result['settled_count']

    fill_rate = (filled / result['markets_traded'] * 100) if result['markets_traded'] else 0
    win_rate = (tp / filled * 100) if filled else 0

    gross_wins = 0
    gross_losses = 0
    df = result.get('df')
    if df is not None and not df.empty and 'pnl' in df.columns:
        exits = df[df['action'].isin(['sell_fill', 'settlement', 'stop_loss'])]
        gross_wins = exits[exits['pnl'] > 0]['pnl'].sum()
        gross_losses = abs(exits[exits['pnl'] < 0]['pnl'].sum())

    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')

    return {
        'label': label,
        'net_pnl': result['net_pnl'],
        'total_return_pct': result['total_return_pct'],
        'sharpe': result['sharpe'],
        'max_dd_pct': result['max_drawdown_pct'],
        'filled': filled,
        'fill_rate': round(fill_rate, 1),
        'tp_count': tp,
        'win_rate': round(win_rate, 1),
        'settled': settled,
        'profit_factor': round(profit_factor, 2),
        'avg_win': result['avg_win'],
        'avg_loss': result['avg_loss'],
        'params': {**BASELINE, **overrides},
        'overrides': overrides,
        'trade_log': result.get('df'),
    }


def robustness_score(r):
    """Higher is better: blends P&L, Sharpe, low DD."""
    pnl_norm = r['net_pnl'] / 1000.0
    sharpe = max(r['sharpe'], 0)
    dd_penalty = max(0, r['max_dd_pct'] - 10) / 10
    return pnl_norm * 0.4 + sharpe * 0.4 - dd_penalty * 0.2


def print_row(i, total, r, baseline_pnl):
    tag = '★' if r['net_pnl'] > baseline_pnl else ' '
    pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞'
    print(f"  [{i:3d}/{total}] {tag} {r['label']:30s}  PnL=${r['net_pnl']:>+9,.2f}  "
          f"Ret={r['total_return_pct']:>+6.1f}%  Sh={r['sharpe']:>5.2f}  "
          f"DD={r['max_dd_pct']:>5.1f}%  PF={pf:>6}  Fill={r['filled']}")


def generate_dashboard(results, baseline_result, output_path, title_suffix=''):
    """Equity-curve overlay + stats table for baseline vs top variants."""
    ranked = sorted(
        [r for r in results if r['label'] != 'BASELINE'],
        key=lambda r: robustness_score(r), reverse=True,
    )
    top = ranked[:5]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#16213e')

    colors = ['#e94560', '#53d769', '#0f3460', '#f5a623', '#9b59b6', '#1abc9c']
    to_plot = [baseline_result] + top

    for i, r in enumerate(to_plot):
        df = r.get('trade_log')
        if df is None or df.empty or 'bankroll' not in df.columns:
            continue
        bankroll = pd.to_numeric(df['bankroll'], errors='coerce').dropna().values
        equity = np.concatenate([[1000.0], bankroll])
        lw = 2.8 if i == 0 else 1.6
        ax1.plot(np.arange(len(equity)), equity,
                 color=colors[i % len(colors)], linewidth=lw,
                 label=r['label'], alpha=0.9)

    ax1.set_title(f'Round 2 Sweep: Baseline vs Top 5{title_suffix}',
                  color='white', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Trade #', color='#aaa', fontsize=11)
    ax1.set_ylabel('Bankroll ($)', color='#aaa', fontsize=11)
    ax1.tick_params(colors='#aaa')
    ax1.legend(loc='upper left', fontsize=9, facecolor='#16213e',
               edgecolor='#444', labelcolor='white')
    ax1.grid(True, alpha=0.15, color='white')
    for spine in ax1.spines.values():
        spine.set_color('#333')

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    headers = ['Variant', 'Net P&L', 'Return%', 'Sharpe', 'MaxDD%',
               'FillRate%', 'WinRate%', 'PF', 'Robustness']
    table_data = []
    for r in to_plot:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞'
        table_data.append([
            r['label'],
            f"${r['net_pnl']:+,.2f}",
            f"{r['total_return_pct']:+.1f}%",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd_pct']:.1f}%",
            f"{r['fill_rate']:.1f}%",
            f"{r['win_rate']:.1f}%",
            pf,
            f"{robustness_score(r):.2f}",
        ])

    table = ax2.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#444')
        if row == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            ci = (row - 1) % len(colors)
            cell.set_facecolor(colors[ci] + '33')
            cell.set_text_props(color='white')

    ax2.set_title('Performance Comparison', color='white', fontsize=14,
                  fontweight='bold', pad=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nDashboard saved: {output_path}")


def main():
    print(f"=== Round 2 Sweep: Cancel=4min Combos ===")
    print(f"Markets: {NUM_MARKETS}")
    print()

    # ── Phase 1: Groups A-D ──
    ad_variants = build_groups_a_to_d()
    total_ad = len(ad_variants)
    print(f"Phase 1: Groups A-D ({total_ad} variants)")
    print('-' * 70)

    ad_results = []
    baseline_result = None
    start = time.time()

    for idx, (label, overrides) in enumerate(ad_variants, 1):
        t0 = time.time()
        r = run_variant(label, overrides)
        elapsed = time.time() - t0

        ad_results.append(r)
        if label == 'BASELINE':
            baseline_result = r

        print_row(idx, total_ad, r, (baseline_result or r)['net_pnl'])

    # Find best from A-D (exclude baseline)
    ad_non_base = [r for r in ad_results if r['label'] != 'BASELINE']
    best_ad = max(ad_non_base, key=robustness_score)
    print(f"\n  ► Best A-D: {best_ad['label']}  "
          f"(PnL=${best_ad['net_pnl']:+,.2f}  Sh={best_ad['sharpe']:.2f}  "
          f"DD={best_ad['max_dd_pct']:.1f}%  Score={robustness_score(best_ad):.2f})")

    # ── Phase 2: Groups E-F ──
    ef_variants = build_group_e(best_ad['overrides'], best_ad['label'])
    ef_variants += build_group_f()
    total_ef = len(ef_variants)
    print(f"\nPhase 2: Groups E-F ({total_ef} variants)")
    print('-' * 70)

    ef_results = []
    for idx, (label, overrides) in enumerate(ef_variants, 1):
        t0 = time.time()
        r = run_variant(label, overrides)
        elapsed = time.time() - t0

        ef_results.append(r)
        print_row(idx, total_ef, r, baseline_result['net_pnl'])

    total_time = time.time() - start

    # ── Combine all results ──
    all_results = ad_results + ef_results
    total = len(all_results)
    print(f"\nDone. {total} variants in {total_time:.0f}s ({total_time/total:.1f}s avg)")

    # ── Save CSV ──
    csv_path = os.path.join(_dir, 'param_sweep_r2_results.csv')
    csv_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ('params', 'trade_log', 'overrides')}
        row['robustness'] = round(robustness_score(r), 4)
        row.update({f'p_{k}': v for k, v in r['params'].items()})
        csv_rows.append(row)

    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # ── Top 10 ranking ──
    ranked = sorted(all_results, key=robustness_score, reverse=True)
    print(f"\n{'='*90}")
    print(f"  TOP 10 BY ROBUSTNESS SCORE")
    print(f"{'='*90}")
    print(f"  {'#':>3}  {'Variant':30s}  {'Net P&L':>11}  {'Ret%':>7}  "
          f"{'Sharpe':>6}  {'DD%':>5}  {'PF':>5}  {'Score':>6}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*11}  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*6}")
    for i, r in enumerate(ranked[:10], 1):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞'
        tag = ' ◀BASE' if r['label'] == 'BASELINE' else ''
        print(f"  {i:3d}  {r['label']:30s}  ${r['net_pnl']:>+9,.2f}  "
              f"{r['total_return_pct']:>+6.1f}%  {r['sharpe']:>5.2f}  "
              f"{r['max_dd_pct']:>4.1f}%  {pf:>5}  {robustness_score(r):>5.2f}{tag}")
    print(f"{'='*90}")

    # ── Dashboard ──
    chart_path = os.path.join(_dir, 'charts', 'MISC Charts', 'param_sweep_r2_dashboard.png')
    generate_dashboard(all_results, baseline_result, chart_path,
                       f'  ({NUM_MARKETS} markets)')


if __name__ == '__main__':
    main()
