#!/usr/bin/env python3
"""
Comprehensive strategy performance report.

Computes: Sharpe, Sortino, Calmar, profit factor, expectancy, Kelly criterion,
max drawdown analysis, streak analysis, time-based returns, risk-adjusted metrics,
and Monte Carlo forward projections.
"""
import os
import sys
import math

_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_dir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    if len(sys.argv) > 1:
        csv_arg = sys.argv[1]
        csv_path = csv_arg if os.path.isabs(csv_arg) else os.path.join(_dir, csv_arg)
    else:
        csv_path = os.path.join(_dir, "trades_KXBTC15M.csv")
    trades = pd.read_csv(csv_path)
    trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce")
    trades["bankroll"] = pd.to_numeric(trades["bankroll"], errors="coerce")

    # ── Per-market PnL (exit rows only: sell_fill + settlement + time_stop) ──
    exit_rows = trades[trades["action"].isin(["sell_fill", "settlement", "time_stop"])]
    filled = trades[trades["action"] == "buy_fill"]

    market_pnl = exit_rows.groupby("market")["pnl"].sum()
    market_entry_ts = filled.groupby("market")["ts"].min()
    market_qty = filled.groupby("market")["qty"].sum()
    market_bankroll = exit_rows.groupby("market")["bankroll"].last()

    has_tp = trades[trades["action"] == "sell_fill"].groupby("market").size()

    mkt = pd.DataFrame({
        "pnl": market_pnl,
        "entry_ts": market_entry_ts,
        "qty": market_qty,
        "bankroll_after": market_bankroll,
    }).dropna(subset=["entry_ts", "pnl"])

    mkt["risk_dollars"] = mkt["qty"] * 0.10
    mkt["return_pct"] = mkt["pnl"] / mkt["risk_dollars"]
    mkt["has_tp"] = mkt.index.isin(has_tp.index)
    # Convert to EST (UTC-5). EDT (UTC-4) is not applied — Kalshi uses fixed EST offset.
    EST_OFFSET = timedelta(hours=-5)
    mkt["date"] = mkt["entry_ts"].apply(lambda t: (datetime.utcfromtimestamp(t) + EST_OFFSET).date())
    mkt["datetime"] = mkt["entry_ts"].apply(lambda t: datetime.utcfromtimestamp(t) + EST_OFFSET)
    mkt = mkt.sort_values("entry_ts")

    n_markets = len(mkt)
    n_wins = (mkt["pnl"] > 0).sum()
    n_losses = (mkt["pnl"] <= 0).sum()
    win_rate = n_wins / n_markets

    initial_bankroll = 1000.0
    final_bankroll = mkt["bankroll_after"].iloc[-1]
    total_pnl = final_bankroll - initial_bankroll
    total_return_pct = total_pnl / initial_bankroll * 100

    avg_win = mkt.loc[mkt["pnl"] > 0, "pnl"].mean() if n_wins > 0 else 0
    avg_loss = abs(mkt.loc[mkt["pnl"] <= 0, "pnl"].mean()) if n_losses > 0 else 0
    median_win = mkt.loc[mkt["pnl"] > 0, "pnl"].median() if n_wins > 0 else 0
    median_loss = abs(mkt.loc[mkt["pnl"] <= 0, "pnl"].median()) if n_losses > 0 else 0
    largest_win = mkt["pnl"].max()
    largest_loss = mkt["pnl"].min()

    gross_profit = mkt.loc[mkt["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(mkt.loc[mkt["pnl"] <= 0, "pnl"].sum())

    # ── Daily returns ──
    daily = mkt.groupby("date").agg(
        pnl=("pnl", "sum"),
        trades=("pnl", "count"),
        bankroll=("bankroll_after", "last"),
    )
    daily = daily.sort_index()

    date_range = pd.date_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(date_range.date, fill_value=0)
    daily["bankroll"] = daily["bankroll"].replace(0, np.nan).ffill()
    daily.loc[daily.index[0], "bankroll"] = daily.loc[daily.index[0], "bankroll"] or initial_bankroll

    daily_returns = daily["pnl"].values
    daily_bankroll = daily["bankroll"].values

    n_days = len(daily)
    start_date = daily.index[0]
    end_date = daily.index[-1]

    # Bankroll series for drawdown
    bankroll_series = [initial_bankroll]
    for p in daily["pnl"].values:
        bankroll_series.append(bankroll_series[-1] + p)
    bankroll_arr = np.array(bankroll_series[1:])

    # ── Drawdown analysis ──
    peak = np.maximum.accumulate(bankroll_arr)
    drawdown_dollar = peak - bankroll_arr
    drawdown_pct = drawdown_dollar / peak * 100

    max_dd_dollar = drawdown_dollar.max()
    max_dd_pct = drawdown_pct.max()
    max_dd_idx = np.argmax(drawdown_dollar)
    max_dd_date = daily.index[max_dd_idx]
    peak_before_dd = peak[max_dd_idx]

    # Recovery: days from max DD to new high
    recovery_idx = None
    for i in range(max_dd_idx, len(bankroll_arr)):
        if bankroll_arr[i] >= peak_before_dd:
            recovery_idx = i
            break
    recovery_days = (recovery_idx - max_dd_idx) if recovery_idx else None

    # Average drawdown
    in_dd = drawdown_pct[drawdown_pct > 0]
    avg_dd_pct = in_dd.mean() if len(in_dd) > 0 else 0

    # ── Risk metrics ──
    daily_mean = np.mean(daily_returns)
    daily_std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1
    daily_downside = daily_returns[daily_returns < 0]
    downside_std = np.std(daily_downside, ddof=1) if len(daily_downside) > 1 else 1

    ann_factor = np.sqrt(365)  # crypto trades 365 days

    sharpe = (daily_mean / daily_std) * ann_factor if daily_std > 0 else 0
    sortino = (daily_mean / downside_std) * ann_factor if downside_std > 0 else 0

    ann_return = total_pnl / n_days * 365
    calmar = (ann_return / max_dd_dollar) if max_dd_dollar > 0 else 0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy per trade
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    expectancy_r = (expectancy / avg_loss) if avg_loss > 0 else 0

    # Kelly criterion
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    kelly_pct = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio if payoff_ratio > 0 else 0

    # ── Streak analysis ──
    wins_losses = (mkt["pnl"] > 0).astype(int).values
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    all_win_streaks = []
    all_loss_streaks = []
    for wl in wins_losses:
        if wl == 1:
            cur_win += 1
            if cur_loss > 0:
                all_loss_streaks.append(cur_loss)
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            if cur_win > 0:
                all_win_streaks.append(cur_win)
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
    if cur_win > 0:
        all_win_streaks.append(cur_win)
    if cur_loss > 0:
        all_loss_streaks.append(cur_loss)

    avg_win_streak = np.mean(all_win_streaks) if all_win_streaks else 0
    avg_loss_streak = np.mean(all_loss_streaks) if all_loss_streaks else 0

    # ── Time-in-market ──
    tp_rows = trades[trades["action"] == "sell_fill"]
    settle_rows = trades[trades["action"].isin(["settlement", "time_stop"])]

    # Avg hold time for TP (from entry to first sell)
    if len(tp_rows) > 0:
        first_sell_ts = tp_rows.groupby("market")["ts"].min()
        tp_markets = first_sell_ts.index.intersection(market_entry_ts.index)
        hold_tp = (first_sell_ts.loc[tp_markets] - market_entry_ts.loc[tp_markets])
        avg_hold_tp_min = hold_tp.mean() / 60 if len(hold_tp) > 0 else 0
    else:
        avg_hold_tp_min = 0

    # ── Normalized return stats ──
    norm_returns = mkt["return_pct"].values
    norm_mean = norm_returns.mean()
    norm_std = norm_returns.std(ddof=1)
    norm_skew = float(stats.skew(norm_returns))
    norm_kurt = float(stats.kurtosis(norm_returns))

    # ── Monte Carlo (forward projection) ──
    n_sims = 10000
    sim_trades = 500
    rng = np.random.default_rng(42)
    mc_finals = []
    mc_dd = []
    for _ in range(n_sims):
        sim_pnl = rng.choice(mkt["pnl"].values, size=sim_trades, replace=True)
        cum = np.cumsum(sim_pnl)
        equity = initial_bankroll + cum
        mc_finals.append(equity[-1])
        peak_eq = np.maximum.accumulate(equity)
        dd = (peak_eq - equity) / peak_eq * 100
        mc_dd.append(dd.max())

    mc_median = np.median(mc_finals)
    mc_5th = np.percentile(mc_finals, 5)
    mc_95th = np.percentile(mc_finals, 95)
    mc_ruin_pct = np.mean(np.array(mc_finals) <= 0) * 100
    mc_dd_median = np.median(mc_dd)
    mc_dd_95th = np.percentile(mc_dd, 95)

    # ── Per-market return distribution (for risk of ruin) ──
    prob_profitable_day = (daily["pnl"] > 0).mean()

    # ══════════════════════════════════════════════════════════════════
    #  PRINT REPORT
    # ══════════════════════════════════════════════════════════════════
    W = 72
    def header(title):
        print("=" * W)
        print(f"  {title}")
        print("=" * W)

    report_title = f"STRATEGY PERFORMANCE REPORT — {os.path.splitext(os.path.basename(csv_path))[0].replace('trades_', '').upper()}"
    header(report_title)
    print(f"  Period:        {start_date} to {end_date} ({n_days} calendar days)")
    print(f"  Markets filled: {n_markets}")
    print(f"  Initial BR:    ${initial_bankroll:,.2f}")
    print(f"  Final BR:      ${final_bankroll:,.2f}")
    print(f"  Net P&L:       ${total_pnl:+,.2f}  ({total_return_pct:+.1f}%)")
    print()

    header("1. RETURNS")
    print(f"  Total Return:          {total_return_pct:+.1f}%")
    print(f"  Annualized Return:     ${ann_return:+,.0f}  ({ann_return/initial_bankroll*100:+.0f}%)")
    print(f"  Daily Avg P&L:         ${daily_mean:+,.2f}")
    print(f"  Daily Std Dev:         ${daily_std:,.2f}")
    print(f"  Best Day:              ${daily_returns.max():+,.2f}")
    print(f"  Worst Day:             ${daily_returns.min():+,.2f}")
    print(f"  Profitable Days:       {prob_profitable_day*100:.1f}%")
    print()

    header("2. RISK-ADJUSTED RETURNS")
    print(f"  Sharpe Ratio (ann.):   {sharpe:.2f}")
    print(f"  Sortino Ratio (ann.):  {sortino:.2f}")
    print(f"  Calmar Ratio:          {calmar:.2f}")
    print(f"  Profit Factor:         {profit_factor:.2f}")
    print()
    print(f"  Interpretation:")
    if sharpe > 2:
        print(f"    Sharpe {sharpe:.2f} — Excellent. Hedge fund quality (>2).")
    elif sharpe > 1:
        print(f"    Sharpe {sharpe:.2f} — Good. Solid risk-adjusted returns (>1).")
    elif sharpe > 0.5:
        print(f"    Sharpe {sharpe:.2f} — Moderate. Acceptable but room to improve.")
    else:
        print(f"    Sharpe {sharpe:.2f} — Below average. High variance relative to return.")
    if sortino > sharpe:
        print(f"    Sortino ({sortino:.2f}) > Sharpe ({sharpe:.2f}) — upside vol > downside vol (good).")
    if profit_factor > 1.5:
        print(f"    Profit Factor {profit_factor:.2f} — Every $1 lost generates ${profit_factor:.2f} in gains.")
    else:
        print(f"    Profit Factor {profit_factor:.2f} — Thin edge, tread carefully.")
    print()

    header("3. DRAWDOWN ANALYSIS")
    print(f"  Max Drawdown ($):      ${max_dd_dollar:,.2f}")
    print(f"  Max Drawdown (%):      {max_dd_pct:.1f}%")
    print(f"  Max DD Date:           {max_dd_date}")
    print(f"  Peak Before DD:        ${peak_before_dd:,.2f}")
    print(f"  Recovery Time:         {f'{recovery_days} days' if recovery_days else 'Not recovered'}")
    print(f"  Avg Drawdown (%):      {avg_dd_pct:.1f}%")
    print()
    if max_dd_pct > 50:
        print(f"  ⚠ Max DD of {max_dd_pct:.0f}% is severe — could you stomach a 50% drop?")
    elif max_dd_pct > 30:
        print(f"  Moderate max DD of {max_dd_pct:.0f}% — manageable with proper risk sizing.")
    else:
        print(f"  Max DD of {max_dd_pct:.0f}% — conservative drawdown profile.")
    print()

    header("4. TRADE STATISTICS")
    print(f"  Win Rate:              {win_rate*100:.1f}%  ({n_wins}W / {n_losses}L)")
    print(f"  TP Rate:               {mkt['has_tp'].mean()*100:.1f}%")
    print(f"  Avg Win:               ${avg_win:+,.2f}")
    print(f"  Avg Loss:              ${avg_loss:,.2f}")
    print(f"  Median Win:            ${median_win:+,.2f}")
    print(f"  Median Loss:           ${median_loss:,.2f}")
    print(f"  Largest Win:           ${largest_win:+,.2f}")
    print(f"  Largest Loss:          ${largest_loss:,.2f}")
    print(f"  Payoff Ratio:          {payoff_ratio:.2f}x  (avg win / avg loss)")
    print(f"  Gross Profit:          ${gross_profit:+,.2f}")
    print(f"  Gross Loss:            ${gross_loss:,.2f}")
    print()

    header("5. EXPECTANCY & KELLY CRITERION")
    print(f"  Expectancy/trade:      ${expectancy:+,.2f}")
    print(f"  Expectancy (R):        {expectancy_r:+.2f}R  (multiples of avg loss)")
    print(f"  Kelly Optimal %:       {kelly_pct*100:.1f}% of bankroll per trade")
    print(f"  Half-Kelly (safer):    {kelly_pct*50:.1f}% of bankroll per trade")
    print()
    print(f"  What this means:")
    print(f"    Each trade has an expected profit of ${expectancy:+.2f}.")
    if expectancy > 0:
        print(f"    Over {n_markets} trades that compounds to ${total_pnl:+,.0f}.")
        print(f"    Kelly says risk {kelly_pct*100:.1f}% per trade to maximize growth.")
        print(f"    Your current risk (3% tiered) {'is near Kelly — good' if abs(kelly_pct - 0.03) < 0.02 else f'vs Kelly {kelly_pct*100:.1f}% — consider adjusting'}.")
    else:
        print(f"    Negative expectancy — strategy loses money long-term.")
    print()

    header("6. STREAK ANALYSIS")
    print(f"  Max Winning Streak:    {max_win_streak}")
    print(f"  Max Losing Streak:     {max_loss_streak}")
    print(f"  Avg Winning Streak:    {avg_win_streak:.1f}")
    print(f"  Avg Losing Streak:     {avg_loss_streak:.1f}")
    print()

    header("7. DISTRIBUTION OF RETURNS")
    print(f"  Per-trade normalized return (PnL / $ risked):")
    print(f"    Mean:                {norm_mean:+.2%}")
    print(f"    Std Dev:             {norm_std:.2%}")
    print(f"    Skewness:            {norm_skew:+.2f}  {'(right tail — big wins)' if norm_skew > 0 else '(left tail — big losses)'}")
    print(f"    Kurtosis:            {norm_kurt:+.2f}  {'(fat tails)' if norm_kurt > 0 else '(thin tails)'}")
    print(f"    5th percentile:      {np.percentile(norm_returns, 5):+.2%}")
    print(f"    25th percentile:     {np.percentile(norm_returns, 25):+.2%}")
    print(f"    Median:              {np.median(norm_returns):+.2%}")
    print(f"    75th percentile:     {np.percentile(norm_returns, 75):+.2%}")
    print(f"    95th percentile:     {np.percentile(norm_returns, 95):+.2%}")
    print()

    header("8. TIMING")
    print(f"  Avg Hold Time (TP):    {avg_hold_tp_min:.1f} min")
    print(f"  Markets/day (avg):     {n_markets / n_days:.1f}")
    print(f"  Capital Efficiency:    {total_return_pct / n_days:.2f}% per day deployed")
    print()

    header("9. MONTE CARLO PROJECTION (next 500 trades)")
    print(f"  Simulations:           {n_sims:,}")
    print(f"  Median Final BR:       ${mc_median:,.0f}")
    print(f"  5th Percentile:        ${mc_5th:,.0f}  (worst 5% of scenarios)")
    print(f"  95th Percentile:       ${mc_95th:,.0f}  (best 5% of scenarios)")
    print(f"  Prob of Ruin (BR<=0):  {mc_ruin_pct:.2f}%")
    print(f"  Median Max DD:         {mc_dd_median:.1f}%")
    print(f"  95th pctile Max DD:    {mc_dd_95th:.1f}%")
    print()
    if mc_ruin_pct > 5:
        print(f"  WARNING: {mc_ruin_pct:.1f}% chance of ruin — reduce position size!")
    elif mc_ruin_pct > 1:
        print(f"  Caution: {mc_ruin_pct:.1f}% ruin risk — acceptable but monitor.")
    else:
        print(f"  Ruin probability near zero — position sizing is safe.")
    print()

    header("10. OVERALL ASSESSMENT")
    score = 0
    notes = []
    if sharpe > 1:
        score += 2
        notes.append("Strong Sharpe ratio")
    elif sharpe > 0.5:
        score += 1
        notes.append("Decent Sharpe")
    else:
        notes.append("Weak Sharpe")

    if profit_factor > 1.5:
        score += 2
        notes.append("Healthy profit factor")
    elif profit_factor > 1.2:
        score += 1
        notes.append("OK profit factor")
    else:
        notes.append("Thin profit factor")

    if max_dd_pct < 30:
        score += 2
        notes.append("Controlled drawdowns")
    elif max_dd_pct < 50:
        score += 1
        notes.append("Moderate drawdowns")
    else:
        notes.append("Dangerous drawdowns")

    if win_rate > 0.45:
        score += 1
        notes.append(f"Good win rate ({win_rate*100:.0f}%)")
    else:
        notes.append(f"Low win rate ({win_rate*100:.0f}%)")

    if expectancy > 0:
        score += 1
        notes.append("Positive expectancy")
    else:
        notes.append("Negative expectancy")

    if mc_ruin_pct < 1:
        score += 1
        notes.append("Negligible ruin risk")
    else:
        notes.append(f"Ruin risk {mc_ruin_pct:.1f}%")

    if calmar > 1:
        score += 1
        notes.append("Good risk/return (Calmar)")

    grade = "A+" if score >= 9 else "A" if score >= 7 else "B" if score >= 5 else "C" if score >= 3 else "D"
    print(f"  Overall Grade: {grade}  ({score}/10)")
    for n in notes:
        print(f"    - {n}")
    print()

    # ══════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f"Strategy Performance Report — {os.path.splitext(os.path.basename(csv_path))[0].replace('trades_', '')}", fontsize=15, fontweight="bold", y=0.98)

    # 1. Equity curve with drawdown
    ax1 = fig.add_subplot(gs[0, :])
    dates = list(daily.index)
    ax1.plot(dates, bankroll_arr, color="#2563eb", linewidth=1.5, label="Bankroll")
    ax1.fill_between(dates, bankroll_arr, peak, color="#ef4444", alpha=0.25, label="Drawdown")
    ax1.axhline(initial_bankroll, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Bankroll ($)")
    ax1.set_title(f"Equity Curve + Drawdown  |  ${initial_bankroll:,.0f} → ${final_bankroll:,.0f}  ({total_return_pct:+.0f}%)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Daily P&L bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    bar_colors = ["#22c55e" if x > 0 else "#ef4444" for x in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=bar_colors, width=1, edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Daily P&L ($)")
    ax2.set_title(f"Daily P&L  |  {prob_profitable_day*100:.0f}% profitable days")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Per-trade return distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(norm_returns, bins=50, color="#8b5cf6", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    ax3.axvline(norm_mean, color="green", linestyle="-", linewidth=2, label=f"Mean: {norm_mean:+.1%}")
    ax3.set_xlabel("Return per $1 Risked")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Return Distribution  |  Skew: {norm_skew:+.1f}  Kurt: {norm_kurt:+.1f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Drawdown % over time
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.fill_between(dates, 0, -drawdown_pct, color="#ef4444", alpha=0.5)
    ax4.plot(dates, -drawdown_pct, color="#ef4444", linewidth=1)
    ax4.set_ylabel("Drawdown (%)")
    ax4.set_title(f"Drawdown Profile  |  Max: {max_dd_pct:.1f}%  Avg: {avg_dd_pct:.1f}%")
    ax4.grid(True, alpha=0.3)

    # 5. Monte Carlo fan
    ax5 = fig.add_subplot(gs[2, 1])
    mc_paths = []
    for _ in range(200):
        sim = rng.choice(mkt["pnl"].values, size=sim_trades, replace=True)
        mc_paths.append(initial_bankroll + np.cumsum(sim))
    for path in mc_paths:
        ax5.plot(range(sim_trades), path, color="#2563eb", alpha=0.03, linewidth=0.5)
    mc_all = np.array(mc_paths)
    ax5.plot(range(sim_trades), np.median(mc_all, axis=0), color="#f97316", linewidth=2, label="Median")
    ax5.plot(range(sim_trades), np.percentile(mc_all, 5, axis=0), color="#ef4444", linewidth=1, linestyle="--", label="5th pctile")
    ax5.plot(range(sim_trades), np.percentile(mc_all, 95, axis=0), color="#22c55e", linewidth=1, linestyle="--", label="95th pctile")
    ax5.axhline(initial_bankroll, color="gray", linestyle="--", alpha=0.5)
    ax5.axhline(0, color="red", linewidth=1, alpha=0.5)
    ax5.set_xlabel("Trades")
    ax5.set_ylabel("Bankroll ($)")
    ax5.set_title(f"Monte Carlo (next {sim_trades} trades)  |  Ruin: {mc_ruin_pct:.1f}%")
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Rolling Sharpe (30-day window)
    ax6 = fig.add_subplot(gs[3, 0])
    window = 30
    if len(daily_returns) > window:
        roll_mean = pd.Series(daily_returns).rolling(window).mean()
        roll_std = pd.Series(daily_returns).rolling(window).std()
        roll_sharpe = (roll_mean / roll_std) * ann_factor
        ax6.plot(range(len(roll_sharpe)), roll_sharpe, color="#2563eb", linewidth=1)
        ax6.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax6.axhline(1, color="green", linestyle=":", alpha=0.5, label="Sharpe = 1")
        ax6.axhline(2, color="green", linestyle=":", alpha=0.3, label="Sharpe = 2")
        ax6.set_xlabel("Day")
        ax6.set_ylabel("Rolling Sharpe")
        ax6.set_title(f"Rolling {window}-Day Sharpe  |  Overall: {sharpe:.2f}")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    # 7. Cumulative P&L by hour
    ax7 = fig.add_subplot(gs[3, 1])
    mkt_copy = mkt.copy()
    mkt_copy["hour"] = mkt_copy["datetime"].dt.hour
    hour_pnl = mkt_copy.groupby("hour")["pnl"].agg(["sum", "mean", "count"])
    bar_colors_h = ["#22c55e" if x > 0 else "#ef4444" for x in hour_pnl["sum"]]
    ax7.bar(hour_pnl.index, hour_pnl["sum"], color=bar_colors_h, edgecolor="black", alpha=0.85)
    for h, row in hour_pnl.iterrows():
        ax7.text(h, row["sum"], f"{row['count']:.0f}", ha="center",
                 va="bottom" if row["sum"] >= 0 else "top", fontsize=6)
    ax7.axhline(0, color="black", linewidth=0.5)
    ax7.set_xlabel("Hour (EST)")
    ax7.set_ylabel("Total P&L ($)")
    ax7.set_title("Cumulative P&L by Hour (EST)")
    ax7.grid(True, alpha=0.3, axis="y")

    base = os.path.splitext(os.path.basename(csv_path))[0].replace("trades_", "")
    out_name = f"strategy_report_{base}.png" if base != "KXBTC15M" else "strategy_report.png"
    out_path = os.path.join(_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Charts saved: {out_path}")


if __name__ == "__main__":
    main()
