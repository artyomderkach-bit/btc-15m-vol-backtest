#!/usr/bin/env python3
"""
Plot equity curve (time vs profit/bankroll) from backtest trade log.
Optionally overlay external time series (BTC, VIX, ETF, SPY, DXY, TNX) and economic events.

Usage:
    python plot_equity.py                           # uses trades_KXBTC15M.csv + all overlays
    python plot_equity.py trades_KXETH15M.csv
    python plot_equity.py --overlay overlay.csv     # custom overlay(s)
    python plot_equity.py --no-default-overlays     # skip default overlays (BTC, VIX, etc.)

Default overlays (when available): BTC, VIX, ETF, SPY, DXY, TNX + FOMC/CPI/NFP markers.
"""
import argparse
import sys
import os

# Set matplotlib config dir before import (avoids permission issues)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange
import matplotlib
matplotlib.use("Agg")
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser(description="Plot equity curve with optional overlay")
    p.add_argument("trades_csv", nargs="?", default=os.path.join(_dir, "trades_KXBTC15M.csv"),
                   help="Trade log CSV (default: trades_KXBTC15M.csv)")
    p.add_argument("--overlay", "-o", metavar="CSV", action="append", help="Overlay CSV (can repeat). Use with --overlay-label for each.")
    p.add_argument("--overlay-label", action="append", help="Label for overlay (one per --overlay)")
    p.add_argument("--no-default-overlays", action="store_true", help="Skip default overlays (BTC, VIX, ETF, SPY, DXY, TNX)")
    p.add_argument("--output", "-O", metavar="PATH", help="Output PNG path")
    p.add_argument("--bankroll", type=float, default=None, help="Starting bankroll (auto-detected from CSV if omitted)")
    return p.parse_args()

DEFAULT_OVERLAYS = [
    ("overlay_btc.csv", "BTC"),
    ("overlay_vix.csv", "VIX"),
    ("overlay_etf.csv", "ETF"),
    ("overlay_spy.csv", "SPY"),
    ("overlay_dxy.csv", "DXY"),
    ("overlay_tnx.csv", "TNX"),
]


def get_economic_event_dates(date_min, date_max):
    """FOMC, CPI, NFP dates for the given range."""
    fomc_dates = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09",
    ]
    cpi_dates = [
        "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12",
        "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-12", "2024-12-11",
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11",
        "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-24", "2025-11-13", "2025-12-10",
        "2026-01-13", "2026-02-11", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-11",
        "2026-07-15", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-12", "2026-12-10",
    ]
    nfp_dates = []
    for year in range(date_min.year, date_max.year + 1):
        for month in range(1, 13):
            _, ndays = monthrange(year, month)
            for d in range(1, min(8, ndays + 1)):
                dt = datetime(year, month, d)
                if dt.weekday() == 4:
                    nfp_dates.append(datetime(year, month, d))
                    break
    rows = []
    for d in fomc_dates:
        t = pd.Timestamp(d)
        if date_min <= t <= date_max:
            rows.append((t, "FOMC"))
    for d in cpi_dates:
        t = pd.Timestamp(d)
        if date_min <= t <= date_max:
            rows.append((t, "CPI"))
    for d in nfp_dates:
        if date_min <= d <= date_max:
            rows.append((d, "NFP"))
    return rows


LABEL_TO_TICKER = {
    "BTC Price": "BTC-USD",
    "BTC": "BTC-USD",
    "VIX": "^VIX",
    "ETF (IBIT)": "IBIT",
    "ETF": "IBIT",
    "SPY": "SPY",
    "DXY": "DX-Y.NYB",
    "TNX": "^TNX",
}


def fetch_intraday(ticker, start, end):
    """Fetch hourly data from yfinance for short time ranges."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(start=start.strftime("%Y-%m-%d"),
                         end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         interval="1h")
        if hist.empty:
            return None
        hist = hist.reset_index()
        date_col = "Datetime" if "Datetime" in hist.columns else hist.columns[0]
        df = pd.DataFrame({"date": pd.to_datetime(hist[date_col]).dt.tz_localize(None),
                           "value": hist["Close"].values})
        return df.dropna().sort_values("date")
    except Exception as e:
        print(f"  [warn] Could not fetch intraday {ticker}: {e}")
        return None


def load_overlay(path):
    """Load overlay CSV. Expects date column + value column."""
    df = pd.read_csv(path)
    date_col = None
    for c in ["date", "Date", "datetime", "timestamp", "time"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    value_col = None
    for c in ["value", "Value", "close", "Close", "price", "Price"]:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        for c in df.columns:
            if c != date_col and pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
    if value_col is None:
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df = df[[date_col, value_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")
    return df

def main():
    args = parse_args()
    csv_path = args.trades_csv

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")

    # Build (timestamp, bankroll) for each row
    market_max_ts = df.groupby("market")["ts"].transform("max")
    df["plot_ts"] = df["ts"].fillna(market_max_ts + 60)
    df = df.sort_values("plot_ts").dropna(subset=["bankroll"])

    if args.bankroll is not None:
        initial = args.bankroll
    elif len(df) > 0:
        initial = df["bankroll"].iloc[0]
    else:
        initial = 1000.0
    times = [df["plot_ts"].iloc[0] - 3600, *df["plot_ts"].tolist()]
    bankrolls = [initial, *df["bankroll"].tolist()]

    EST_OFFSET = timedelta(hours=-5)
    dates = [datetime.utcfromtimestamp(t) + EST_OFFSET for t in times]

    EVENTS = [
        (datetime(2026, 1, 2), "Jan 2: BTC ETFs +$500M inflows"),
        (datetime(2026, 1, 15), "Jan 15: Crypto bill cancelled, BTC <$96k"),
        (datetime(2026, 2, 4), "Feb 4: Treasury rejects gov BTC"),
        (datetime(2026, 2, 5), "Feb 5: $500k flash crash"),
        (datetime(2026, 2, 21), "Feb 21: Trump tariff 15%"),
        (datetime(2026, 2, 23), "Feb 23: BTC 5% tariffs"),
        (datetime(2026, 3, 1), "Mar 1-2: Iran strikes, crypto +700%"),
        (datetime(2026, 3, 2), "Mar 2: First ETF inflow $834M"),
    ]

    overlays = []
    if args.overlay:
        labels = args.overlay_label or []
        for i, path in enumerate(args.overlay):
            if os.path.exists(path):
                odf = load_overlay(path)
                odf["date"] = odf["date"].dt.tz_localize(None)
                label = labels[i] if i < len(labels) else os.path.basename(path).replace(".csv", "")
                overlays.append((odf, label))
    elif not args.no_default_overlays:
        for fname, label in DEFAULT_OVERLAYS:
            path = os.path.join(_dir, fname)
            if os.path.exists(path):
                odf = load_overlay(path)
                odf["date"] = odf["date"].dt.tz_localize(None)
                overlays.append((odf, label))

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(dates, bankrolls, linewidth=1, color="#2563eb", label="Bankroll")
    ax1.fill_between(dates, initial, bankrolls, alpha=0.2, color="#2563eb")
    ax1.axhline(initial, color="gray", linestyle="--", alpha=0.7)

    date_min, date_max = min(dates), max(dates)
    span_days = (date_max - date_min).total_seconds() / 86400
    y_max = max(bankrolls)

    resample_freq = "h" if span_days < 7 else "D"
    eq_series = pd.Series(bankrolls, index=pd.DatetimeIndex(dates))
    eq_resampled = eq_series.resample(resample_freq).last().ffill()

    if span_days < 7:
        print(f"  Short date range ({span_days:.1f} days) — fetching hourly overlay data...")
        upgraded = []
        for odf, label in overlays:
            ov_in_range = odf[(odf["date"] >= date_min) & (odf["date"] <= date_max)]
            if len(ov_in_range) < 5:
                yf_ticker = LABEL_TO_TICKER.get(label)
                if yf_ticker:
                    intra = fetch_intraday(yf_ticker, pd.Timestamp(date_min), pd.Timestamp(date_max))
                    if intra is not None and len(intra) > 0:
                        upgraded.append((intra, label))
                        continue
            upgraded.append((odf, label))
        overlays = upgraded

    colors = ["#eab308", "#22c55e", "#f97316", "#8b5cf6", "#ec4899", "#06b6d4"]
    corr_text = []
    for i, (overlay_df, label) in enumerate(overlays):
        ov = overlay_df[(overlay_df["date"] >= date_min) & (overlay_df["date"] <= date_max)]
        if len(ov) == 0:
            continue
        ax2 = ax1.twinx()
        ax2.spines["right"].set_position(("outward", 60 * i))
        color = colors[i % len(colors)]
        ax2.plot(ov["date"], ov["value"], linewidth=1, color=color, alpha=0.9, label=label)
        ax2.set_ylabel(label, color=color, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=8)
        ov_resampled = ov.set_index("date")["value"].resample(resample_freq).last().ffill()
        common = eq_resampled.index.intersection(ov_resampled.index)
        min_pts = 5 if span_days < 7 else 10
        if len(common) >= min_pts:
            a, b = eq_resampled.loc[common].values, ov_resampled.loc[common].values
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() >= min_pts:
                corr = np.corrcoef(a[valid], b[valid])[0, 1]
                corr_text.append(f"{label}: {corr:.2f}")
    if corr_text:
        ax1.text(0.02, 0.98, "Correlation:\n" + "\n".join(corr_text), transform=ax1.transAxes,
                 fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    for i, (evt_date, label) in enumerate(EVENTS):
        if date_min <= evt_date <= date_max:
            ax1.axvline(evt_date, color="coral", linestyle=":", alpha=0.7, linewidth=1)
            y_pos = y_max * (0.92 - (i % 4) * 0.02)
            ax1.annotate(label, (evt_date, y_pos), fontsize=6, rotation=90,
                         va="top", ha="right", color="#8b2500")

    # Economic calendar (FOMC, CPI, NFP) - only when using default overlays
    if not args.no_default_overlays and not args.overlay:
        econ_events = get_economic_event_dates(date_min, date_max)
        for evt_date, evt_type in econ_events:
            d = evt_date if hasattr(evt_date, "to_pydatetime") else evt_date
            color = {"FOMC": "#dc2626", "CPI": "#ea580c", "NFP": "#ca8a04"}.get(evt_type, "gray")
            ax1.axvline(d, color=color, alpha=0.4, linewidth=0.8, linestyle=":")

    ax1.set_xlabel("Time (EST)")
    ax1.set_ylabel("Bankroll ($)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_title(f"Equity Curve: {os.path.basename(csv_path)}")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = (os.path.join(_dir, args.output) if args.output else csv_path.replace(".csv", "_equity.png"))
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if corr_text:
        print("Correlation (equity vs overlay):", ", ".join(corr_text))

if __name__ == "__main__":
    main()
