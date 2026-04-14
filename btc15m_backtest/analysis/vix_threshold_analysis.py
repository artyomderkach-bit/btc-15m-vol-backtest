#!/usr/bin/env python3
"""
Find the VIX threshold where the 10c fill strategy has positive expected value.

For each market that got a 10c fill, look up the VIX on that day.
Then sweep VIX thresholds and compute: for all fills where VIX >= X,
what is the average PnL per market? Find the minimum X where avg PnL >= 0.

PnL is computed from sell_fill + settlement rows only (these already include
entry cost). buy_fill rows are excluded to avoid double-counting.
"""
import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_dir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    trades_path = os.path.join(_dir, "trades_KXBTC15M.csv")
    vix_path = os.path.join(_dir, "overlay_vix.csv")

    trades = pd.read_csv(trades_path)
    vix_df = pd.read_csv(vix_path)
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
    vix_df = vix_df.rename(columns={"value": "vix"})

    trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce")

    # ── 1. Build per-market summary ────────────────────────────────────
    # Entry date from first buy_fill timestamp
    filled = trades[trades["action"] == "buy_fill"]
    if filled.empty:
        print("No buy_fill rows found.")
        return

    market_entry_ts = filled.groupby("market")["ts"].min()
    market_entry_date = market_entry_ts.apply(
        lambda t: datetime.utcfromtimestamp(t).date() if pd.notna(t) else None
    )

    # Contracts filled per market
    market_qty = filled.groupby("market")["qty"].sum()

    # PnL: ONLY from sell_fill and settlement rows (net P&L, entry cost already subtracted)
    exit_rows = trades[trades["action"].isin(["sell_fill", "settlement"])]
    market_pnl = exit_rows.groupby("market")["pnl"].sum()

    # Outcome per market
    has_tp = trades[trades["action"] == "sell_fill"].groupby("market").size()

    # For markets with partial TP (some contracts sold, rest settled):
    # Count total sell_fill qty and settlement qty per market
    sell_qty_per_market = trades[trades["action"] == "sell_fill"].groupby("market")["qty"].sum()
    settle_qty_per_market = trades[trades["action"] == "settlement"].groupby("market")["qty"].sum()

    markets_df = pd.DataFrame({
        "date": market_entry_date,
        "pnl": market_pnl,
        "qty": market_qty,
    })
    markets_df["tp_qty"] = sell_qty_per_market.reindex(markets_df.index).fillna(0)
    markets_df["settle_qty"] = settle_qty_per_market.reindex(markets_df.index).fillna(0)
    markets_df["has_tp"] = markets_df.index.isin(has_tp.index)
    markets_df["outcome"] = markets_df.apply(
        lambda r: "full_tp" if r["tp_qty"] > 0 and r["settle_qty"] == 0
        else ("partial_tp" if r["tp_qty"] > 0 and r["settle_qty"] > 0
              else "settle_loss"), axis=1
    )

    # Normalize PnL: return per $1 risked (pnl / (qty * 0.10))
    markets_df["risk_dollars"] = markets_df["qty"] * 0.10
    markets_df["pnl_per_dollar"] = markets_df["pnl"] / markets_df["risk_dollars"]

    markets_df = markets_df[markets_df["qty"] > 0].copy()
    markets_df = markets_df.dropna(subset=["date"])

    # ── 2. Join with VIX ───────────────────────────────────────────────
    markets_df = markets_df.merge(vix_df, on="date", how="left")

    matched = markets_df.dropna(subset=["vix"]).copy()
    unmatched = markets_df[markets_df["vix"].isna()]

    print(f"Total markets with fills: {len(markets_df)}")
    print(f"  Matched to VIX:   {len(matched)}")
    print(f"  No VIX data:      {len(unmatched)} (weekends/holidays)")
    print()

    if len(matched) == 0:
        print("No trades could be matched to VIX data.")
        return

    # ── 3. Weekend check ───────────────────────────────────────────────
    weekend_pnl = unmatched["pnl"].sum()
    weekday_pnl = matched["pnl"].sum()
    print("=" * 70)
    print("  WEEKDAY vs WEEKEND")
    print("=" * 70)
    print(f"  Weekday markets: {len(matched):>6}  Total PnL: ${weekday_pnl:>+10,.2f}  Avg: ${matched['pnl'].mean():>+8.2f}")
    print(f"  Weekend markets: {len(unmatched):>6}  Total PnL: ${weekend_pnl:>+10,.2f}  Avg: ${unmatched['pnl'].mean():>+8.2f}")
    print(f"  (Weekend = Sat+Sun when VIX is not published)")
    print()

    # ── 4. Overall stats ───────────────────────────────────────────────
    print("=" * 70)
    print("  OVERALL STATS (VIX-matched weekday fills)")
    print("=" * 70)
    total_pnl = matched["pnl"].sum()
    avg_pnl = matched["pnl"].mean()
    avg_pnl_norm = matched["pnl_per_dollar"].mean()
    tp_count = matched["has_tp"].sum()
    tp_rate = tp_count / len(matched) * 100
    print(f"  Markets:          {len(matched)}")
    print(f"  TP hits:          {tp_count} ({tp_rate:.1f}%)")
    print(f"  Total PnL:        ${total_pnl:+,.2f}")
    print(f"  Avg PnL/market:   ${avg_pnl:+,.2f}")
    print(f"  Avg return/$risk: {avg_pnl_norm:+.2%}")
    print(f"  VIX range:        {matched['vix'].min():.1f} – {matched['vix'].max():.1f}")
    print()

    # ── 5. VIX threshold sweep (raw PnL) ──────────────────────────────
    vix_min = matched["vix"].min()
    vix_max = matched["vix"].max()
    thresholds = np.arange(np.floor(vix_min), np.ceil(vix_max) + 0.5, 0.5)

    results = []
    for thresh in thresholds:
        subset = matched[matched["vix"] >= thresh]
        if len(subset) < 5:
            continue
        avg = subset["pnl"].mean()
        avg_norm = subset["pnl_per_dollar"].mean()
        total = subset["pnl"].sum()
        tp = subset["has_tp"].sum()
        n = len(subset)
        tp_pct = tp / n * 100
        results.append({
            "vix_threshold": thresh,
            "markets": n,
            "tp_count": tp,
            "tp_pct": tp_pct,
            "avg_pnl": avg,
            "avg_return_per_dollar": avg_norm,
            "total_pnl": total,
            "median_pnl": subset["pnl"].median(),
        })

    res_df = pd.DataFrame(results)

    print("=" * 70)
    print("  VIX THRESHOLD SWEEP: avg PnL per market when VIX >= threshold")
    print("=" * 70)
    print(f"  {'VIX >=':>8}  {'Mkts':>6}  {'TP%':>7}  {'Avg PnL':>10}  {'Ret/Risk':>10}  {'Total PnL':>12}  {'Median':>9}")
    print(f"  {'------':>8}  {'----':>6}  {'---':>7}  {'-------':>10}  {'--------':>10}  {'---------':>12}  {'------':>9}")
    for _, r in res_df.iterrows():
        marker = "  *" if r["avg_pnl"] >= 0 else ""
        print(f"  {r['vix_threshold']:>8.1f}  {r['markets']:>6.0f}  {r['tp_pct']:>6.1f}%  ${r['avg_pnl']:>+9.2f}"
              f"  {r['avg_return_per_dollar']:>+9.1%}  ${r['total_pnl']:>+11.2f}  ${r['median_pnl']:>+8.2f}{marker}")
    print()

    # ── 6. Find crossover ─────────────────────────────────────────────
    positive_rows = res_df[res_df["avg_pnl"] >= 0]
    if len(positive_rows) == 0:
        best_idx = res_df["avg_pnl"].idxmax()
        best = res_df.loc[best_idx]
        print("=" * 70)
        print(f"  RESULT: Strategy is NEGATIVE EV at all VIX levels (weekdays)")
        print("=" * 70)
        print(f"    Closest to breakeven: VIX >= {best['vix_threshold']:.1f}")
        print(f"    Avg PnL:  ${best['avg_pnl']:+.2f}  ({best['markets']:.0f} markets)")
        print()
    else:
        min_vix = positive_rows["vix_threshold"].min()
        row = positive_rows[positive_rows["vix_threshold"] == min_vix].iloc[0]
        print("=" * 70)
        print(f"  RESULT: Strategy has +EV when VIX >= {min_vix:.1f}")
        print("=" * 70)
        print(f"    Markets in sample:  {row['markets']:.0f}")
        print(f"    TP rate:            {row['tp_pct']:.1f}%")
        print(f"    Avg PnL/market:     ${row['avg_pnl']:+.2f}")
        print(f"    Avg return/$risk:   {row['avg_return_per_dollar']:+.1%}")
        print(f"    Total PnL:          ${row['total_pnl']:+,.2f}")
        print()

    # ── 7. VIX BUCKET analysis ─────────────────────────────────────────
    print("=" * 70)
    print("  VIX BUCKET ANALYSIS (performance in each VIX range)")
    print("=" * 70)
    buckets = [(0, 15), (15, 17), (17, 19), (19, 21), (21, 23), (23, 25), (25, 28), (28, 100)]
    print(f"  {'VIX Range':>12}  {'Mkts':>6}  {'TP':>4}  {'Settle':>7}  {'TP%':>7}  {'Avg PnL':>10}  {'Ret/$':>8}  {'Total PnL':>12}")
    print(f"  {'----------':>12}  {'----':>6}  {'--':>4}  {'------':>7}  {'---':>7}  {'-------':>10}  {'-----':>8}  {'---------':>12}")
    for lo, hi in buckets:
        bucket = matched[(matched["vix"] >= lo) & (matched["vix"] < hi)]
        if len(bucket) == 0:
            continue
        tp = bucket["has_tp"].sum()
        settle = len(bucket) - tp
        tp_pct = tp / len(bucket) * 100
        avg = bucket["pnl"].mean()
        avg_norm = bucket["pnl_per_dollar"].mean()
        total = bucket["pnl"].sum()
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        tag = " +" if avg >= 0 else " -"
        print(f"  {label:>12}  {len(bucket):>6}  {tp:>4}  {settle:>7}  {tp_pct:>6.1f}%  ${avg:>+9.2f}  {avg_norm:>+7.1%}  ${total:>+11.2f}{tag}")
    print()

    # ── 8. Statistical tests ──────────────────────────────────────────
    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    # Test high vs low VIX
    vix_median = matched["vix"].median()
    above_med = matched[matched["vix"] >= vix_median]
    below_med = matched[matched["vix"] < vix_median]

    from scipy import stats

    t_stat, p_val = stats.ttest_ind(above_med["pnl"], below_med["pnl"], equal_var=False)
    print(f"  Welch t-test: VIX >= {vix_median:.1f} (median) vs VIX < {vix_median:.1f}")
    print(f"    Avg PnL (above): ${above_med['pnl'].mean():>+.2f}  (n={len(above_med)})")
    print(f"    Avg PnL (below): ${below_med['pnl'].mean():>+.2f}  (n={len(below_med)})")
    print(f"    t = {t_stat:.3f},  p = {p_val:.4f}  {'(sig at 5%)' if p_val < 0.05 else '(NOT sig at 5%)'}")
    print()

    # Spearman correlation: VIX vs PnL
    corr, pval_corr = stats.spearmanr(matched["vix"], matched["pnl"])
    print(f"  Spearman correlation (VIX vs PnL): r = {corr:.3f}, p = {pval_corr:.4f}")
    print(f"    {'Significant' if pval_corr < 0.05 else 'Not significant'} at 5% level")
    print()

    # Bootstrap CI for best bucket
    if len(positive_rows) > 0:
        thresh = positive_rows["vix_threshold"].min()
        group = matched[matched["vix"] >= thresh]["pnl"].values
    else:
        thresh = res_df.loc[res_df["avg_pnl"].idxmax(), "vix_threshold"]
        group = matched[matched["vix"] >= thresh]["pnl"].values

    n_boot = 10000
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(group, size=len(group), replace=True).mean() for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for avg PnL when VIX >= {thresh:.1f} (n={len(group)}):")
    print(f"    [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    print(f"    {'CI is fully positive — robust' if ci_lo > 0 else 'CI includes zero — noisy'}")
    print()

    # ── 9. Plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("VIX Threshold Analysis: When Is the 10¢ Fill Strategy +EV?", fontsize=14, fontweight="bold")

    # Plot 1: Avg PnL vs VIX threshold
    ax = axes[0, 0]
    ax.plot(res_df["vix_threshold"], res_df["avg_pnl"], color="#2563eb", linewidth=2, marker="o", markersize=3)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    if len(positive_rows) > 0:
        bv = positive_rows["vix_threshold"].min()
        ax.axvline(bv, color="green", linestyle=":", alpha=0.8, linewidth=2,
                   label=f"Breakeven: VIX ≥ {bv:.1f}")
        ax.legend(fontsize=10)
    ax.set_xlabel("VIX Threshold (≥)")
    ax.set_ylabel("Avg PnL per Market ($)")
    ax.set_title("Avg PnL vs VIX Threshold")
    ax.grid(True, alpha=0.3)

    # Plot 2: TP rate vs VIX threshold
    ax = axes[0, 1]
    ax.plot(res_df["vix_threshold"], res_df["tp_pct"], color="#22c55e", linewidth=2, marker="o", markersize=3)
    if len(positive_rows) > 0:
        ax.axvline(bv, color="green", linestyle=":", alpha=0.8, linewidth=2)
    ax.set_xlabel("VIX Threshold (≥)")
    ax.set_ylabel("TP Rate (%)")
    ax.set_title("Take-Profit Rate vs VIX Threshold")
    ax.grid(True, alpha=0.3)

    # Plot 3: Bucket bar chart
    ax = axes[1, 0]
    bucket_labels, bucket_avgs, bucket_counts = [], [], []
    for lo, hi in buckets:
        bucket = matched[(matched["vix"] >= lo) & (matched["vix"] < hi)]
        if len(bucket) == 0:
            continue
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        bucket_labels.append(label)
        bucket_avgs.append(bucket["pnl"].mean())
        bucket_counts.append(len(bucket))
    bar_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in bucket_avgs]
    bars = ax.bar(bucket_labels, bucket_avgs, color=bar_colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, bucket_counts):
        yp = bar.get_height()
        va = "bottom" if yp >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yp, f"n={count}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("VIX Range")
    ax.set_ylabel("Avg PnL per Market ($)")
    ax.set_title("Avg PnL by VIX Bucket")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Normalized return per dollar risked
    ax = axes[1, 1]
    norm_labels, norm_avgs, norm_counts = [], [], []
    for lo, hi in buckets:
        bucket = matched[(matched["vix"] >= lo) & (matched["vix"] < hi)]
        if len(bucket) == 0:
            continue
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        norm_labels.append(label)
        norm_avgs.append(bucket["pnl_per_dollar"].mean() * 100)
        norm_counts.append(len(bucket))
    bar_colors_n = ["#22c55e" if v >= 0 else "#ef4444" for v in norm_avgs]
    bars = ax.bar(norm_labels, norm_avgs, color=bar_colors_n, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, norm_counts):
        yp = bar.get_height()
        va = "bottom" if yp >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, yp, f"n={count}", ha="center", va=va, fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("VIX Range")
    ax.set_ylabel("Avg Return per $1 Risked (%)")
    ax.set_title("Normalized Return by VIX Bucket (controls for position sizing)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _misc = os.path.join(_project_root, "charts", "MISC Charts")
    os.makedirs(_misc, exist_ok=True)
    out_path = os.path.join(_misc, "vix_threshold_analysis.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Chart saved: {out_path}")


if __name__ == "__main__":
    main()
