#!/usr/bin/env python3
"""
Monte Carlo simulation: PnL vs execution (TP rate).
Varies 33c TP rate from 30% to 60%, runs N sims per rate, plots PnL line.

Usage:
    python monte_carlo.py                     # 100 sims per TP rate
    python monte_carlo.py 50                 # 50 sims per TP rate
    python monte_carlo.py 100 trades_KXETH15M.csv
"""
import sys
import os
import random
import pandas as pd
import numpy as np

INITIAL_BANKROLL = 1000.0


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    _dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_dir)
    csv_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_dir, "trades_KXBTC15M.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        print("Run a backtest first: python run_backtest.py KXBTC15M 7300")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)

    # Per-market: TP (sell_fill) vs settlement, and net PnL
    tp_pnls = []
    settle_pnls = []
    for market, grp in df.groupby("market"):
        has_tp = (grp["action"] == "sell_fill").any()
        closes = grp[grp["action"].isin(["sell_fill", "settlement"])]
        net_pnl = closes["pnl"].sum()
        buy_cost = -grp[grp["action"] == "buy_fill"]["pnl"].sum()
        if buy_cost <= 0:
            continue
        if has_tp:
            tp_pnls.append(net_pnl)
        else:
            settle_pnls.append(net_pnl)

    n_tp = len(tp_pnls)
    n_settle = len(settle_pnls)
    n_filled = n_tp + n_settle
    actual_tp_rate = n_tp / n_filled * 100 if n_filled else 0

    if not tp_pnls or not settle_pnls:
        print("Need both TP and settlement markets in trade log")
        sys.exit(1)

    print(f"Monte Carlo: PnL vs TP rate ({n_sims} sims per rate)")
    print(f"  Source: {os.path.basename(csv_path)}")
    print(f"  Actual: {n_tp} TP, {n_settle} settle -> {actual_tp_rate:.1f}% TP rate")
    print()

    # TP rates to sweep: 30% to 60% in 2% steps
    tp_rates_pct = list(range(30, 62, 2))
    mean_pnls = []
    median_pnls = []
    p5_pnls = []
    p95_pnls = []

    for tp_rate_pct in tp_rates_pct:
        tp_rate = tp_rate_pct / 100.0
        pnls = []
        for _ in range(n_sims):
            total = 0.0
            for _ in range(n_filled):
                if random.random() < tp_rate:
                    total += random.choice(tp_pnls)
                else:
                    total += random.choice(settle_pnls)
            pnls.append(total)
        mean_pnls.append(np.mean(pnls))
        median_pnls.append(np.median(pnls))
        p5_pnls.append(np.percentile(pnls, 5))
        p95_pnls.append(np.percentile(pnls, 95))

    # Print summary
    print("=" * 60)
    print("  PnL vs TP Rate (33c take-profit %)")
    print("=" * 60)
    print(f"  {'TP%':>6}  {'Mean PnL':>12}  {'Median PnL':>12}  {'5th %ile':>12}  {'95th %ile':>12}")
    for i, r in enumerate(tp_rates_pct):
        print(f"  {r:>5}%  ${mean_pnls[i]:>+10,.0f}  ${median_pnls[i]:>+10,.0f}  ${p5_pnls[i]:>+10,.0f}  ${p95_pnls[i]:>+10,.0f}")
    print("=" * 60)

    # Line graph: TP rate vs PnL
    try:
        import matplotlib
        matplotlib.use("Agg")
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tp_rates_pct, mean_pnls, "o-", color="#2563eb", linewidth=2, markersize=6, label="Mean PnL")
        ax.plot(tp_rates_pct, median_pnls, "s--", color="#059669", linewidth=1.5, markersize=5, label="Median PnL")
        ax.fill_between(tp_rates_pct, p5_pnls, p95_pnls, alpha=0.2, color="#2563eb")
        ax.axvline(actual_tp_rate, color="coral", linestyle=":", linewidth=2, label=f"Actual: {actual_tp_rate:.1f}%")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xlabel("33c TP Rate (%)")
        ax.set_ylabel("Net PnL ($)")
        ax.set_title(f"Monte Carlo: PnL vs Execution (TP Rate) - {n_sims} sims per point")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = csv_path.replace(".csv", "_monte_carlo.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\nInstall matplotlib for plot: pip install matplotlib")

    # Save data
    out_path = csv_path.replace(".csv", "_monte_carlo.csv")
    pd.DataFrame({
        "tp_rate_pct": tp_rates_pct,
        "mean_pnl": mean_pnls,
        "median_pnl": median_pnls,
        "p5_pnl": p5_pnls,
        "p95_pnl": p95_pnls,
    }).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
