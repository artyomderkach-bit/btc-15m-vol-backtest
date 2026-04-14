#!/usr/bin/env python3
"""
Regenerate current_strat_full_pnl.png:
  - Bankroll vs calendar time (UTC)
  - Right axis: VIX daily close (level), fixed overlay.
  - Still screens all CANDIDATES and writes bankroll_overlay_screening.csv.
"""
import os
import sys
from typing import Any, Optional, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_project_root, ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

TRADELOG = os.path.join(_script_dir, "charts", "btc_main", "current_strat_full_tradelog.csv")
OUT_PNG = os.path.join(_script_dir, "charts", "btc_main", "current_strat_full_pnl.png")
OUT_CSV = os.path.join(_script_dir, "charts", "btc_main", "bankroll_overlay_screening.csv")

# Chart right axis (not the screening “winner”)
CHART_OVERLAY_TICKER = "^VIX"
CHART_OVERLAY_LABEL = "VIX"
CHART_OVERLAY_COLOR = "#f85149"

# Yahoo symbols: macro / risk / vol / crypto (same set we document in the report)
CANDIDATES = [
    ("^VIX", "VIX"),
    ("^VVIX", "VVIX"),
    ("^MOVE", "MOVE"),
    ("BTC-USD", "BTC-USD"),
    ("ETH-USD", "ETH-USD"),
    ("^GSPC", "S&P 500"),
    ("GLD", "GLD"),
    ("DX-Y.NYB", "DXY"),
    ("^TNX", "10Y yield"),
    ("HYG", "HYG"),
    ("IEF", "IEF"),
    ("UUP", "UUP"),
    ("IBIT", "IBIT"),
]


def _normalize_yf_hist(h):
    if h is None or h.empty:
        return None
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    h = h.reset_index()
    dcol = "Date" if "Date" in h.columns else h.columns[0]
    close = "Close" if "Close" in h.columns else h.columns[-1]
    out = pd.DataFrame({
        "date": pd.to_datetime(h[dcol]).dt.tz_localize(None),
        "value": pd.to_numeric(h[close], errors="coerce").values.ravel(),
    }).dropna().sort_values("date")
    return out


def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    return _normalize_yf_hist(h)


def daily_bankroll_series() -> pd.Series:
    df = pd.read_csv(TRADELOG)
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    mm = df.groupby("market")["ts"].transform("max")
    df["plot_ts"] = df["ts"].fillna(mm + 60)
    df = df.sort_values("plot_ts").dropna(subset=["bankroll"])
    df["day"] = pd.to_datetime(df["plot_ts"], unit="s", utc=True).dt.tz_localize(None).dt.floor("D")
    return df.groupby("day")["bankroll"].last().sort_index()


def screen_overlays(br: pd.Series) -> Tuple[pd.DataFrame, Optional[Tuple[Any, ...]]]:
    start = (br.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (br.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    rows = []
    best = None  # (abs_r_pct, ticker, label, r_pct, n, overlay_df)

    for ticker, label in CANDIDATES:
        try:
            od = fetch_daily(ticker, start, end)
            if od is None or len(od) < 10:
                rows.append({"ticker": ticker, "label": label, "n": 0, "r_pct": np.nan,
                             "r_level": np.nan, "r_dollar": np.nan, "note": "no_data"})
                continue
            m = br.rename("br").to_frame().join(od.set_index("date")["value"].rename("x"), how="inner")
            m = m.dropna()
            n = len(m)
            if n < 15:
                rows.append({"ticker": ticker, "label": label, "n": n, "r_pct": np.nan,
                             "r_level": np.nan, "r_dollar": np.nan, "note": "short_sample"})
                continue
            r_level = m["br"].corr(m["x"])
            m = m.copy()
            m["d_br"] = m["br"].diff()
            m["d_x"] = m["x"].diff()
            m["p_br"] = m["br"].pct_change()
            m["p_x"] = m["x"].pct_change()
            mm = m.dropna(subset=["p_br", "p_x", "d_br", "d_x"])
            r_pct = float(mm["p_br"].corr(mm["p_x"]))
            r_dollar = float(mm["d_br"].corr(mm["d_x"]))
            rows.append({
                "ticker": ticker, "label": label, "n": len(mm),
                "r_pct": r_pct, "r_level": r_level, "r_dollar": r_dollar, "note": "",
            })
            key = (abs(r_pct), len(mm))
            if best is None or key > (abs(best[3]), best[4]):
                best = (ticker, label, od, r_pct, len(mm), r_level, r_dollar)
        except Exception as e:
            rows.append({"ticker": ticker, "label": label, "n": 0, "r_pct": np.nan,
                         "r_level": np.nan, "r_dollar": np.nan, "note": str(e)[:50]})

    table = pd.DataFrame(rows).sort_values("r_pct", key=lambda s: s.abs(), ascending=False)
    return table, best


def overlay_stats_for_ticker(br: pd.Series, ticker: str, label: str):
    """Fetch series; return (ticker, label, od, r_pct, n, r_level, r_dollar) or None."""
    start = (br.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (br.index.max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    od = fetch_daily(ticker, start, end)
    if od is None or len(od) < 10:
        return None
    m = br.rename("br").to_frame().join(od.set_index("date")["value"].rename("x"), how="inner")
    m = m.dropna()
    if len(m) < 15:
        return None
    r_level = m["br"].corr(m["x"])
    m = m.copy()
    m["d_br"] = m["br"].diff()
    m["d_x"] = m["x"].diff()
    m["p_br"] = m["br"].pct_change()
    m["p_x"] = m["x"].pct_change()
    mm = m.dropna(subset=["p_br", "p_x", "d_br", "d_x"])
    r_pct = float(mm["p_br"].corr(mm["p_x"]))
    r_dollar = float(mm["d_br"].corr(mm["d_x"]))
    return (ticker, label, od, r_pct, len(mm), r_level, r_dollar)


def main():
    if not os.path.exists(TRADELOG):
        print(f"Missing: {TRADELOG}", file=sys.stderr)
        sys.exit(1)

    br = daily_bankroll_series()
    table, best = screen_overlays(br)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"Screening table: {OUT_CSV}")
    print(table.to_string(index=False))

    df = pd.read_csv(TRADELOG)
    df["bankroll"] = pd.to_numeric(df["bankroll"], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    market_max_ts = df.groupby("market")["ts"].transform("max")
    df["plot_ts"] = df["ts"].fillna(market_max_ts + 60)
    df = df.sort_values("plot_ts").dropna(subset=["bankroll"])

    initial = 1000.0
    times = np.array([df["plot_ts"].iloc[0] - 60, *df["plot_ts"].tolist()], dtype=float)
    bankrolls = np.array([initial, *df["bankroll"].tolist()], dtype=float)
    dates = pd.to_datetime(times, unit="s", utc=True)

    date_min = dates.min().tz_convert(None)
    date_max = dates.max().tz_convert(None)
    margin = pd.Timedelta(days=2)

    fig, ax1 = plt.subplots(figsize=(22, 10))
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#161b22")

    ax1.fill_between(dates, initial, bankrolls, where=bankrolls >= initial,
                     alpha=0.12, color="#53d769")
    ax1.fill_between(dates, initial, bankrolls, where=bankrolls < initial,
                     alpha=0.12, color="#e94560")
    ax1.plot(dates, bankrolls, color="#58a6ff", linewidth=1.6, label="Bankroll", zorder=3)
    ax1.axhline(y=initial, color="#8b949e", linewidth=0.8, linestyle="--", alpha=0.5)

    ax1.set_ylabel("Bankroll ($)", color="#58a6ff", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#58a6ff", colors="#8b949e")
    ax1.tick_params(axis="x", colors="#8b949e")
    ax1.set_xlabel("Date (UTC)", color="#8b949e", fontsize=12)
    ax1.grid(True, alpha=0.08, color="white")
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    final = float(bankrolls[-1])
    ret_pct = (final - initial) / initial * 100

    ax2 = ax1.twinx()
    subtitle = "No overlay (VIX fetch failed or insufficient overlap)"
    ann = ""

    plot_overlay = overlay_stats_for_ticker(br, CHART_OVERLAY_TICKER, CHART_OVERLAY_LABEL)
    if plot_overlay is not None:
        ticker, label, od, r_pct, n_ov, r_level, r_dollar = plot_overlay
        plot_df = od[(od["date"] >= date_min - margin) & (od["date"] <= date_max + margin)]
        color = CHART_OVERLAY_COLOR
        ax2.plot(plot_df["date"], plot_df["value"], color=color, linewidth=1.2, alpha=0.95,
                 label=f"{label} ({ticker})")
        ax2.set_ylabel(f"{label} (close)", color=color, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color, colors="#8b949e")
        subtitle = (
            f"Overlay: {label} — daily % correlation with bankroll "
            f"(r={r_pct:+.3f}, n={n_ov})"
        )
        best_ticker = table.iloc[0]["ticker"] if len(table) else "—"
        best_r = table.iloc[0]["r_pct"] if len(table) else float("nan")
        ann = (
            f"Chart overlay: {CHART_OVERLAY_TICKER} (close, level)  |  "
            f"Screened {len(CANDIDATES)} series; strongest |r(Δ%)|: {best_ticker} ({best_r:+.3f})\n"
            f"r(Δ% bankroll, Δ% {label}) = {r_pct:+.3f}  |  r(level) = {r_level:+.3f} (often spurious)\n"
            f"Full table: bankroll_overlay_screening.csv"
        )
    else:
        ax2.set_ylabel("", color="#8b949e")
        ax2.tick_params(axis="y", labelcolor="#8b949e")

    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    ax1.set_title(
        f"Current strategy P&L vs time  |  ${initial:,.0f} → ${final:,.0f} ({ret_pct:+.0f}%)\n{subtitle}",
        color="white", fontsize=14, fontweight="bold", pad=14,
    )
    if ann:
        ax1.text(0.99, 0.02, ann, transform=ax1.transAxes, ha="right", va="bottom", fontsize=9,
                 color="#8b949e",
                 bbox=dict(boxstyle="round", facecolor="#21262d", edgecolor="#30363d", alpha=0.95))

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=10,
               facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
