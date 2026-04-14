#!/usr/bin/env python3
"""
Overlay external data with equity curve and analyze performance by regime.

1. BTC price – Performance in trending vs ranging regimes
2. VIX – Correlation with volatility
3. ETF flows – Match equity curve to inflow/outflow (proxy: IBIT volume/price)
4. Macro/risk regime – SPY (risk-on/off), DXY (dollar), TNX (10Y yield)
5. Economic calendar – FOMC, CPI, NFP performance around event days

Fetches data via yfinance. Saves overlay CSVs and generates combined plots + metrics.

Usage:
    python overlay_analysis.py
    python overlay_analysis.py trades_KXBTC15M.csv
    python overlay_analysis.py --no-fetch   # use cached overlay CSVs only
"""
import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_dir = _script_dir


def parse_args():
    p = argparse.ArgumentParser(description="Overlay external data with equity curve")
    p.add_argument("trades_csv", nargs="?", default=os.path.join(_dir, "trades_KXBTC15M.csv"),
                   help="Trade log CSV")
    p.add_argument("--no-fetch", action="store_true", help="Skip fetching; use cached overlay CSVs")
    p.add_argument("--out-dir", default=_dir, help="Output directory for overlay CSVs")
    p.add_argument("--plot-path", default=None,
                   help="Path for overlay_analysis.png (default: <out-dir>/overlay_analysis.png)")
    p.add_argument("--use-sample", action="store_true", help="Also use overlay_sample.csv if it covers the range")
    return p.parse_args()


def load_equity_series(csv_path):
    """Load equity curve (date, bankroll) from trades CSV."""
    df = pd.read_csv(csv_path)
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    market_max_ts = df.groupby("market")["ts"].transform("max")
    df["plot_ts"] = df["ts"].fillna(market_max_ts + 60)
    df = df.sort_values("plot_ts").dropna(subset=["bankroll"])
    initial = 1000.0
    times = [df["plot_ts"].iloc[0] - 3600, *df["plot_ts"].tolist()]
    bankrolls = [initial, *df["bankroll"].tolist()]
    dates = [datetime.utcfromtimestamp(t) for t in times]
    return pd.Series(bankrolls, index=pd.DatetimeIndex(dates)), initial, min(dates), max(dates)


def fetch_yf(ticker, start, end):
    """Fetch daily data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("Install yfinance: pip install yfinance")
        sys.exit(1)
    # If end is in future, use period instead
    now = datetime.utcnow()
    if end > now:
        # Use last 1 year as proxy
        end = now
        start = end - timedelta(days=365)
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or len(data) < 2:
        return None
    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else data.columns[0]
    close_col = "Close" if "Close" in data.columns else [c for c in data.columns if c != date_col][0]
    out = data[[date_col, close_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out


def classify_regime(btc_df, window=5, threshold_pct=2.0):
    """
    Classify each day as trending (up/down) or ranging.
    Uses N-day return: |return| > threshold_pct => trending.
    """
    df = btc_df.copy()
    df = df.sort_values("date").drop_duplicates("date")
    df["ret"] = df["value"].pct_change(window)
    df["regime"] = "ranging"
    df.loc[df["ret"] > threshold_pct / 100, "regime"] = "trending_up"
    df.loc[df["ret"] < -threshold_pct / 100, "regime"] = "trending_down"
    return df


def compute_daily_pnl(equity_series, initial):
    """Daily PnL from equity curve."""
    eq = equity_series.resample("D").last().ffill()
    pnl = eq.diff()
    pnl.iloc[0] = eq.iloc[0] - initial
    return pnl


def get_economic_event_dates(date_min, date_max):
    """
    Generate FOMC, CPI, NFP event dates for the given range.
    Returns DataFrame with columns: date, event_type
    """
    # FOMC meeting dates (decision day, 2nd day of meeting) - 2024-2026
    fomc_dates = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09",
    ]
    # CPI release dates (BLS schedule) - 2024-2026
    cpi_dates = [
        "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
        "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-12", "2024-12-11",
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
        "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-11-13", "2025-12-10",
        "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-11",
        "2026-07-15", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-12", "2026-12-10",
    ]
    # NFP: first Friday of each month
    nfp_dates = []
    for year in range(date_min.year, date_max.year + 1):
        for month in range(1, 13):
            _, ndays = monthrange(year, month)
            for d in range(1, min(8, ndays + 1)):
                dt = datetime(year, month, d)
                if dt.weekday() == 4:  # Friday
                    nfp_dates.append(dt.strftime("%Y-%m-%d"))
                    break

    rows = []
    for d in fomc_dates:
        rows.append({"date": pd.Timestamp(d), "event_type": "FOMC"})
    for d in cpi_dates:
        rows.append({"date": pd.Timestamp(d), "event_type": "CPI"})
    for d in nfp_dates:
        rows.append({"date": pd.Timestamp(d), "event_type": "NFP"})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
    return df


def align_overlay_to_equity(overlay_df, equity_index, date_min, date_max):
    """
    If overlay doesn't overlap equity dates, align by position (use last N days of overlay).
    Returns overlay with dates matching equity's daily index.
    """
    ov = overlay_df[(overlay_df["date"] >= date_min) & (overlay_df["date"] <= date_max)]
    if len(ov) >= 10:
        return ov
    # No overlap: align by position
    eq_dates = equity_index.resample("D").last().dropna().index
    n = len(eq_dates)
    ov_sorted = overlay_df.sort_values("date").drop_duplicates("date")
    if len(ov_sorted) < n:
        return None
    vals = ov_sorted["value"].iloc[-n:].values
    return pd.DataFrame({"date": eq_dates, "value": vals})


def main():
    args = parse_args()
    csv_path = args.trades_csv
    out_dir = args.out_dir

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    equity, initial, date_min, date_max = load_equity_series(csv_path)
    eq_daily = equity.resample("D").last().ffill()
    daily_pnl = compute_daily_pnl(equity, initial)

    # Extend range for fetching (add buffer)
    start = date_min - timedelta(days=30)
    end = date_max + timedelta(days=5)

    overlays = {}
    overlay_paths = {
        "btc": os.path.join(out_dir, "overlay_btc.csv"),
        "vix": os.path.join(out_dir, "overlay_vix.csv"),
        "etf": os.path.join(out_dir, "overlay_etf.csv"),
        "spy": os.path.join(out_dir, "overlay_spy.csv"),
        "dxy": os.path.join(out_dir, "overlay_dxy.csv"),
        "tnx": os.path.join(out_dir, "overlay_tnx.csv"),
    }

    def has_overlap(df, dmin, dmax):
        if df is None or len(df) == 0:
            return False
        d = pd.to_datetime(df["date"])
        return (d.min() <= dmax) and (d.max() >= dmin)

    if not args.no_fetch:
        print("Fetching external data...")
        btc = fetch_yf("BTC-USD", start, end)
        if btc is not None:
            btc.to_csv(overlay_paths["btc"], index=False)
            overlays["btc"] = btc
            if has_overlap(btc, date_min, date_max):
                print(f"  BTC: {len(btc)} days -> {overlay_paths['btc']}")
            else:
                print(f"  BTC: {len(btc)} days (proxy: backtest range in future)")

        vix = fetch_yf("^VIX", start, end)
        if vix is not None:
            vix.to_csv(overlay_paths["vix"], index=False)
            overlays["vix"] = vix
            if has_overlap(vix, date_min, date_max):
                print(f"  VIX: {len(vix)} days -> {overlay_paths['vix']}")
            else:
                print(f"  VIX: {len(vix)} days (proxy)")

        etf = fetch_yf("IBIT", start, end)
        if etf is not None:
            etf.to_csv(overlay_paths["etf"], index=False)
            overlays["etf"] = etf
            if has_overlap(etf, date_min, date_max):
                print(f"  ETF (IBIT proxy): {len(etf)} days -> {overlay_paths['etf']}")
            else:
                print(f"  ETF: {len(etf)} days (proxy)")

        # Macro / risk regime overlays
        spy = fetch_yf("SPY", start, end)
        if spy is not None:
            spy.to_csv(overlay_paths["spy"], index=False)
            overlays["spy"] = spy
            if has_overlap(spy, date_min, date_max):
                print(f"  SPY: {len(spy)} days -> {overlay_paths['spy']}")
            else:
                print(f"  SPY: {len(spy)} days (proxy)")

        dxy = fetch_yf("DX-Y.NYB", start, end)
        if dxy is None:
            dxy = fetch_yf("DX=F", start, end)  # fallback to futures
        if dxy is not None:
            dxy.to_csv(overlay_paths["dxy"], index=False)
            overlays["dxy"] = dxy
            if has_overlap(dxy, date_min, date_max):
                print(f"  DXY: {len(dxy)} days -> {overlay_paths['dxy']}")
            else:
                print(f"  DXY: {len(dxy)} days (proxy)")

        tnx = fetch_yf("^TNX", start, end)
        if tnx is not None:
            tnx.to_csv(overlay_paths["tnx"], index=False)
            overlays["tnx"] = tnx
            if has_overlap(tnx, date_min, date_max):
                print(f"  TNX (10Y yield): {len(tnx)} days -> {overlay_paths['tnx']}")
            else:
                print(f"  TNX: {len(tnx)} days (proxy)")

        if args.use_sample:
            sample_path = os.path.join(_dir, "overlay_sample.csv")
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                df["date"] = pd.to_datetime(df["date"])
                if has_overlap(df, date_min, date_max):
                    overlays["sample"] = df
                    overlay_paths["sample"] = sample_path
                    print(f"  overlay_sample.csv: using")
    else:
        for name, path in overlay_paths.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"])
                overlays[name] = df
                print(f"  Loaded {name} from {path}")
        if args.use_sample:
            sample_path = os.path.join(_dir, "overlay_sample.csv")
            if os.path.exists(sample_path) and "sample" not in overlays:
                df = pd.read_csv(sample_path)
                df["date"] = pd.to_datetime(df["date"])
                if has_overlap(df, date_min, date_max):
                    overlays["sample"] = df
                    overlay_paths["sample"] = sample_path
                    print(f"  Loaded sample from {sample_path}")

    if not overlays and os.path.exists(os.path.join(_dir, "overlay_sample.csv")):
        df = pd.read_csv(os.path.join(_dir, "overlay_sample.csv"))
        df["date"] = pd.to_datetime(df["date"])
        overlays["sample"] = df
        overlay_paths["sample"] = os.path.join(_dir, "overlay_sample.csv")
        print("  Using overlay_sample.csv (no other overlays)")

    # --- Regime analysis (BTC or sample as proxy) ---
    regime_stats = None
    btc_key = "btc" if "btc" in overlays else ("sample" if "sample" in overlays else None)
    if btc_key:
        btc_df = overlays[btc_key]
        btc_df = classify_regime(btc_df, window=5, threshold_pct=2.0)
        btc_daily = btc_df.set_index("date")["regime"]

        common = eq_daily.index.intersection(btc_daily.index)
        if len(common) < 5:
            # Align by position when no overlap (e.g. backtest in future)
            btc_aligned = align_overlay_to_equity(btc_df, equity, date_min, date_max)
            if btc_aligned is not None:
                btc_df2 = classify_regime(btc_aligned, window=5, threshold_pct=2.0)
                btc_daily = btc_df2.set_index("date")["regime"]
                common = eq_daily.index.intersection(btc_daily.index)
        if len(common) >= 5:
            eq_on_common = eq_daily.loc[common]
            pnl_on_common = daily_pnl.loc[common].fillna(0)
            regime_on_common = btc_daily.loc[common]

            regime_stats = []
            for reg in ["trending_up", "trending_down", "ranging"]:
                mask = regime_on_common == reg
                if mask.sum() >= 3:
                    days = mask.sum()
                    tot_pnl = pnl_on_common[mask].sum()
                    avg_pnl = pnl_on_common[mask].mean()
                    regime_stats.append((reg, days, tot_pnl, avg_pnl))

    # --- Correlations ---
    correlations = {}
    for name, ov_df in overlays.items():
        ov = ov_df[(ov_df["date"] >= date_min) & (ov_df["date"] <= date_max)]
        if len(ov) < 10:
            ov = align_overlay_to_equity(ov_df, equity, date_min, date_max)
        if ov is None or len(ov) < 10:
            continue
        ov_daily = ov.set_index("date")["value"].resample("D").last().ffill()
        common = eq_daily.index.intersection(ov_daily.index)
        if len(common) >= 10:
            a = eq_daily.loc[common].values
            b = ov_daily.loc[common].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() >= 10:
                correlations[name] = np.corrcoef(a[valid], b[valid])[0, 1]

    # --- Print report ---
    print("\n--- Overlay Analysis Report ---")
    print(f"Equity range: {date_min.date()} to {date_max.date()}")
    print(f"Initial: ${initial:.0f}, Final: ${equity.iloc[-1]:.0f}")

    if regime_stats:
        src = "BTC" if btc_key == "btc" else "Overlay"
        print(f"\nPerformance by {src} regime (5d return threshold ±2%):")
        for reg, days, tot_pnl, avg_pnl in regime_stats:
            label = reg.replace("_", " ").title()
            print(f"  {label}: {days} days, total PnL ${tot_pnl:.1f}, avg daily ${avg_pnl:.1f}")

    # --- VIX regime breakdown ---
    vix_regime_stats = None
    if "vix" in overlays:
        vix_df = overlays["vix"].copy()
        vix_df = vix_df.sort_values("date").drop_duplicates("date")
        # Classify: low (<15), medium (15-25), high (>25)
        vix_df["regime"] = "medium"
        vix_df.loc[vix_df["value"] < 15, "regime"] = "low"
        vix_df.loc[vix_df["value"] > 25, "regime"] = "high"
        vix_daily = vix_df.set_index("date")["regime"]

        common = eq_daily.index.intersection(vix_daily.index)
        if len(common) < 5:
            vix_aligned = align_overlay_to_equity(vix_df, equity, date_min, date_max)
            if vix_aligned is not None:
                vix_aligned["regime"] = "medium"
                vix_aligned.loc[vix_aligned["value"] < 15, "regime"] = "low"
                vix_aligned.loc[vix_aligned["value"] > 25, "regime"] = "high"
                vix_daily = vix_aligned.set_index("date")["regime"]
                common = eq_daily.index.intersection(vix_daily.index)
        if len(common) >= 5:
            pnl_on_common = daily_pnl.loc[common].fillna(0)
            regime_on_common = vix_daily.loc[common]
            vix_regime_stats = []
            for reg in ["low", "medium", "high"]:
                mask = regime_on_common == reg
                if mask.sum() >= 3:
                    days = mask.sum()
                    tot_pnl = pnl_on_common[mask].sum()
                    avg_pnl = pnl_on_common[mask].mean()
                    vix_regime_stats.append((reg, days, tot_pnl, avg_pnl))

    if vix_regime_stats:
        print("\nPerformance by VIX regime (low <15, medium 15-25, high >25):")
        for reg, days, tot_pnl, avg_pnl in vix_regime_stats:
            print(f"  VIX {reg}: {days} days, total PnL ${tot_pnl:.1f}, avg daily ${avg_pnl:.1f}")

    # --- Economic calendar: performance on event days vs non-event ---
    events_df = get_economic_event_dates(date_min, date_max)
    econ_stats = []
    if len(events_df) >= 1:
        event_dates = set(events_df["date"].dt.date)
        index_dates = pd.Series(eq_daily.index).dt.date
        pnl_all = daily_pnl.fillna(0)

        # Event days (any FOMC/CPI/NFP)
        event_mask = index_dates.isin(event_dates).values
        if event_mask.sum() >= 2:
            days = int(event_mask.sum())
            tot = pnl_all.iloc[event_mask].sum()
            avg = pnl_all.iloc[event_mask].mean()
            econ_stats.append(("Event days (any)", days, tot, avg))

        # Non-event days
        non_event_mask = ~index_dates.isin(event_dates).values
        if non_event_mask.sum() >= 5:
            days = int(non_event_mask.sum())
            tot = pnl_all.iloc[non_event_mask].sum()
            avg = pnl_all.iloc[non_event_mask].mean()
            econ_stats.append(("Non-event days", days, tot, avg))

        # By event type
        for evt in ["FOMC", "CPI", "NFP"]:
            evt_dates = set(events_df[events_df["event_type"] == evt]["date"].dt.date)
            evt_mask = index_dates.isin(evt_dates).values
            if evt_mask.sum() >= 2:
                days = int(evt_mask.sum())
                tot = pnl_all.iloc[evt_mask].sum()
                avg = pnl_all.iloc[evt_mask].mean()
                econ_stats.append((f"  {evt} only", days, tot, avg))

    if econ_stats:
        print("\nPerformance by economic calendar (FOMC/CPI/NFP):")
        for label, days, tot_pnl, avg_pnl in econ_stats:
            print(f"  {label}: {days} days, total PnL ${tot_pnl:.1f}, avg daily ${avg_pnl:.1f}")

    if correlations:
        print("\nCorrelation (equity vs overlay):")
        for name, corr in correlations.items():
            label = {
                "btc": "BTC price", "vix": "VIX", "etf": "ETF (IBIT)", "sample": "Sample",
                "spy": "SPY", "dxy": "DXY", "tnx": "TNX",
            }.get(name, name)
            print(f"  {label}: {corr:.3f}")

    # --- Plots ---
    n_overlays = len(overlays)
    if n_overlays == 0:
        print("\nNo overlay data. Run without --no-fetch to download.")
        return

    fig, axes = plt.subplots(n_overlays + 1, 1, figsize=(14, 4 * (n_overlays + 1)), sharex=True)
    if n_overlays == 0:
        axes = [axes]
    ax0 = axes[0]

    ax0.plot(equity.index, equity.values, linewidth=1, color="#2563eb", label="Bankroll")
    ax0.fill_between(equity.index, initial, equity.values, alpha=0.2, color="#2563eb")
    ax0.axhline(initial, color="gray", linestyle="--", alpha=0.7)

    # Economic event markers on equity curve
    from matplotlib.lines import Line2D
    if len(events_df) >= 1:
        for evt, color in [("FOMC", "#dc2626"), ("CPI", "#ea580c"), ("NFP", "#ca8a04")]:
            evt_dates = events_df[events_df["event_type"] == evt]["date"]
            for d in evt_dates:
                ax0.axvline(d, color=color, alpha=0.4, linewidth=0.8, linestyle=":")
        ax0.legend(loc="upper left", handles=[
            Line2D([0], [0], color="#2563eb", label="Bankroll"),
            Line2D([0], [0], color="#dc2626", alpha=0.6, linestyle=":", label="FOMC"),
            Line2D([0], [0], color="#ea580c", alpha=0.6, linestyle=":", label="CPI"),
            Line2D([0], [0], color="#ca8a04", alpha=0.6, linestyle=":", label="NFP"),
        ])
    else:
        ax0.legend(loc="upper left")
    ax0.set_ylabel("Bankroll ($)")
    ax0.set_title("Equity Curve")
    ax0.grid(True, alpha=0.3)

    colors = ["#eab308", "#22c55e", "#f97316", "#8b5cf6", "#ec4899", "#06b6d4"]
    labels_map = {
        "btc": "BTC Price", "vix": "VIX", "etf": "ETF (IBIT)", "sample": "Overlay",
        "spy": "SPY (S&P 500)", "dxy": "DXY (Dollar)", "tnx": "TNX (10Y Yield)",
    }
    for i, (name, ov_df) in enumerate(overlays.items()):
        ax = axes[i + 1]
        ov = ov_df[(ov_df["date"] >= date_min) & (ov_df["date"] <= date_max)]
        if len(ov) < 5:
            ov = align_overlay_to_equity(ov_df, equity, date_min, date_max)
        if ov is None or len(ov) == 0:
            continue
        color = colors[i % len(colors)]
        ax.plot(ov["date"], ov["value"], linewidth=1, color=color, alpha=0.9)
        ax.set_ylabel(labels_map.get(name, name))
        ax.set_title(labels_map.get(name, name) + (f" (corr: {correlations.get(name, 0):.3f})" if name in correlations else ""))
        ax.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()
    out_plot = args.plot_path or os.path.join(out_dir, "overlay_analysis.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or ".", exist_ok=True)
    plt.savefig(out_plot, dpi=150)
    print(f"\nSaved: {out_plot}")

    # --- Single overlay plots (for plot_equity compatibility) ---
    for name, ov_df in overlays.items():
        path = overlay_paths.get(name)
        if path:
            print(f"\nTo overlay {labels_map.get(name, name)} with plot_equity.py:")
            print(f"  python plot_equity.py -o {path} --overlay-label \"{labels_map.get(name, name)}\"")


if __name__ == "__main__":
    main()
