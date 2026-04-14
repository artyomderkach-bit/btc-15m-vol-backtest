#!/usr/bin/env python3
"""
Parameter sweep for BTC 15-min strategy optimization.

Runs the Engine with different parameter combinations, collects metrics,
ranks them, and outputs:
  - param_sweep_results.csv        (full results table)
  - charts/MISC Charts/param_sweep_dashboard.png  (visual comparison)

Usage:
    python param_sweep.py                  # full sweep
    python param_sweep.py --markets 3000   # faster with fewer markets
"""
import sys
import os
import time
import math
import itertools
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


# ── Define parameter grid ──

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
}


def build_variants():
    """Build list of (label, params_dict) for each variant to test."""
    variants = [('BASELINE', {})]

    # Tier 1: Stop loss
    for sl in [0.04, 0.05, 0.06]:
        variants.append((f'SL={sl}', {'stop_loss_price': sl}))

    # Tier 1: VIX filter
    for vix in [18.0, 20.0, 22.0]:
        variants.append((f'VIX>={vix:.0f}', {'vix_min': vix}))

    # Tier 1: Take profit
    for tp in [0.28, 0.30, 0.35, 0.38, 0.40]:
        variants.append((f'TP={tp}', {'sell_price': tp}))

    # Tier 1: Combined stop loss + VIX
    for sl in [0.04, 0.05]:
        for vix in [18.0, 20.0]:
            variants.append((f'SL={sl}+VIX>={vix:.0f}', {
                'stop_loss_price': sl, 'vix_min': vix,
            }))

    # Tier 1: Combined VIX + TP
    for vix in [18.0, 20.0]:
        for tp in [0.30, 0.35, 0.38]:
            variants.append((f'VIX>={vix:.0f}+TP={tp}', {
                'vix_min': vix, 'sell_price': tp,
            }))

    # Tier 1: Combined SL + VIX + TP (best combos)
    for sl in [0.04, 0.05]:
        for vix in [18.0, 20.0]:
            for tp in [0.30, 0.35]:
                variants.append((f'SL={sl}+VIX>={vix:.0f}+TP={tp}', {
                    'stop_loss_price': sl, 'vix_min': vix, 'sell_price': tp,
                }))

    # Tier 2: Entry price
    for ep in [0.08, 0.12, 0.15]:
        variants.append((f'Entry={ep}', {'buy_price': ep}))

    # Tier 2: Cancel time
    for ct in [240, 300, 420, 480]:
        variants.append((f'Cancel={ct//60}min', {'entry_cutoff_seconds': ct}))

    # Tier 2: Risk pct
    for rp in [0.005, 0.01, 0.015, 0.03]:
        variants.append((f'Risk={rp*100:.1f}%', {'risk_pct': rp}))

    # Tier 3: Weekend
    variants.append(('Weekend=0.5x', {'weekend_size_mult': 0.5}))

    # Tier 3: Single side
    variants.append(('YES-only', {'single_side': 'yes'}))
    variants.append(('NO-only', {'single_side': 'no'}))

    # Tier 3: Trailing stop
    variants.append(('Trail=0.25/0.18', {
        'trailing_stop_trigger': 0.25, 'trailing_stop_floor': 0.18,
    }))
    variants.append(('Trail=0.25/0.20', {
        'trailing_stop_trigger': 0.25, 'trailing_stop_floor': 0.20,
    }))

    return variants


def run_variant(label, overrides):
    """Run the engine with given parameter overrides. Returns metrics dict."""
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
        'trade_log': result.get('df'),
    }


def generate_dashboard(results, baseline_result, output_path):
    """Create the comparison dashboard chart."""
    # Sort by net_pnl descending, pick top 3 non-baseline
    ranked = sorted(
        [r for r in results if r['label'] != 'BASELINE'],
        key=lambda r: r['net_pnl'], reverse=True,
    )
    top3 = ranked[:3]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#1a1a2e')
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.25)

    # ── Top: Equity curves ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#16213e')

    colors = ['#e94560', '#0f3460', '#53d769', '#f5a623']
    labels_to_plot = [baseline_result] + top3

    for i, r in enumerate(labels_to_plot):
        df = r.get('trade_log')
        if df is None or df.empty or 'bankroll' not in df.columns:
            continue
        df_eq = df.copy()
        df_eq['bankroll'] = pd.to_numeric(df_eq['bankroll'], errors='coerce')
        equity = np.concatenate([[1000.0], df_eq.dropna(subset=['bankroll'])['bankroll'].values])
        x = np.arange(len(equity))
        lw = 2.5 if i == 0 else 1.8
        ls = '-' if i == 0 else '-'
        ax1.plot(x, equity, color=colors[i], linewidth=lw, linestyle=ls,
                 label=r['label'], alpha=0.9)

    ax1.set_title('BTC 15-Min Strategy: Baseline vs Top 3 Optimized Variants',
                   color='white', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Trade #', color='#aaa', fontsize=11)
    ax1.set_ylabel('Bankroll ($)', color='#aaa', fontsize=11)
    ax1.tick_params(colors='#aaa')
    ax1.legend(loc='upper left', fontsize=10, facecolor='#16213e',
               edgecolor='#444', labelcolor='white')
    ax1.grid(True, alpha=0.15, color='white')
    for spine in ax1.spines.values():
        spine.set_color('#333')

    # ── Bottom: Stats table ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    headers = ['Variant', 'Net P&L', 'Return %', 'Sharpe', 'Max DD%',
               'Fill Rate%', 'Win Rate%', 'Profit Factor']

    table_data = []
    for r in labels_to_plot:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞'
        table_data.append([
            r['label'],
            f"${r['net_pnl']:+,.2f}",
            f"{r['total_return_pct']:+.1f}%",
            f"{r['sharpe']:.2f}",
            f"{r['max_dd_pct']:.1f}%",
            f"{r['fill_rate']:.1f}%",
            f"{r['win_rate']:.1f}%",
            pf_str,
        ])

    table = ax2.table(cellText=table_data, colLabels=headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#444')
        if row == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            color_idx = row - 1
            bg = colors[color_idx] if color_idx < len(colors) else '#16213e'
            cell.set_facecolor(bg + '33')
            cell.set_text_props(color='white')

    ax2.set_title('Performance Comparison', color='white', fontsize=14,
                   fontweight='bold', pad=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\nDashboard saved: {output_path}")


def main():
    variants = build_variants()
    total = len(variants)
    print(f"=== BTC 15-Min Parameter Sweep ===")
    print(f"Markets: {NUM_MARKETS}  |  Variants: {total}")
    print()

    results = []
    baseline_result = None
    start_time = time.time()

    for i, (label, overrides) in enumerate(variants):
        t0 = time.time()
        r = run_variant(label, overrides)
        elapsed = time.time() - t0

        results.append(r)
        if label == 'BASELINE':
            baseline_result = r

        tag = '★' if r['net_pnl'] > (baseline_result or r)['net_pnl'] else ' '
        print(f"  [{i+1:3d}/{total}] {tag} {label:35s}  PnL=${r['net_pnl']:>+9,.2f}  "
              f"Return={r['total_return_pct']:>+6.1f}%  Sharpe={r['sharpe']:>5.2f}  "
              f"DD={r['max_dd_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  ({elapsed:.1f}s)")

    total_time = time.time() - start_time
    print(f"\nDone. {total} variants in {total_time:.0f}s ({total_time/total:.1f}s avg)")

    # ── Save CSV ──
    csv_path = os.path.join(_dir, 'param_sweep_results.csv')
    csv_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ('params', 'trade_log')}
        row.update({f'p_{k}': v for k, v in r['params'].items()})
        csv_rows.append(row)

    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # ── Print ranking ──
    ranked = sorted(results, key=lambda r: r['net_pnl'], reverse=True)
    print(f"\n{'='*80}")
    print(f"  TOP 10 VARIANTS BY NET P&L")
    print(f"{'='*80}")
    print(f"  {'Rank':>4}  {'Variant':35s}  {'Net P&L':>12}  {'Return':>8}  {'Sharpe':>7}  {'DD%':>6}  {'PF':>6}")
    print(f"  {'----':>4}  {'-------':35s}  {'-------':>12}  {'------':>8}  {'------':>7}  {'---':>6}  {'--':>6}")
    for i, r in enumerate(ranked[:10]):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else '∞'
        marker = ' ◀ BASELINE' if r['label'] == 'BASELINE' else ''
        print(f"  {i+1:4d}  {r['label']:35s}  ${r['net_pnl']:>+10,.2f}  "
              f"{r['total_return_pct']:>+7.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['max_dd_pct']:>5.1f}%  {pf_str:>6}{marker}")
    print(f"{'='*80}")

    # ── Dashboard chart ──
    chart_path = os.path.join(_dir, 'charts', 'MISC Charts', 'param_sweep_dashboard.png')
    generate_dashboard(results, baseline_result, chart_path)


if __name__ == '__main__':
    main()
