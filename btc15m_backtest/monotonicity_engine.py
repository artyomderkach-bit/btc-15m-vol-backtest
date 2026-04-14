"""
Grouped ladder backtest engine: multi-ticker aligned bars + monotonicity strategy.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_fetcher import DataFetcher
from engine import _to_unix_ts, expand_sparse_candles_to_minute_grid, parse_candle
from ladder_grouping import (
    LadderGroup,
    RejectedLadder,
    group_markets_into_ladders,
    validate_ladder_homogeneous,
)
from monotonicity_config import MonotonicityConfig
from monotonicity_strategy import MonotonicityStrategy
from models import CandleBar


def align_ladder_bars(
    fetcher: DataFetcher,
    ladder: LadderGroup,
    fill_minute_gaps: bool,
) -> Tuple[Optional[List[Dict[str, CandleBar]]], Optional[str]]:
    """
    Return list of per-timestamp bar maps aligned on the same minute grid,
    or (None, reason) if lengths mismatch or insufficient data.
    """
    if not ladder.markets:
        return None, "empty_ladder"

    open_ts = ladder.markets[0].open_ts
    close_ts = ladder.markets[0].close_ts
    tickers = [m.ticker for m in ladder.markets]

    series: Dict[str, List[CandleBar]] = {}
    for t in tickers:
        raw = fetcher.fetch_candles(t, open_ts, close_ts)
        if fill_minute_gaps:
            bars = expand_sparse_candles_to_minute_grid(raw, open_ts, close_ts)
        else:
            bars = []
            for rc in raw:
                b = parse_candle(rc)
                if b and open_ts < b.ts <= close_ts:
                    bars.append(b)
            bars.sort(key=lambda x: x.ts)
        if len(bars) < 6:
            return None, f"insufficient_bars:{t}"
        series[t] = bars

    lengths = {len(series[t]) for t in tickers}
    if len(lengths) > 1:
        return None, f"length_mismatch:{lengths}"

    n = len(series[tickers[0]])
    out: List[Dict[str, CandleBar]] = []
    for idx in range(n):
        ts_i = series[tickers[0]][idx].ts
        bm: Dict[str, CandleBar] = {}
        for t in tickers:
            b = series[t][idx]
            if b.ts != ts_i:
                return None, f"ts_mismatch:{t}:{b.ts}!={ts_i}"
            bm[t] = b
        out.append(bm)
    return out, None


def equity_metrics_from_trade_log(df: pd.DataFrame, initial_bankroll: float) -> float:
    """Annualized Sharpe from bankroll series in trade log."""
    if df.empty or "bankroll" not in df.columns:
        return 0.0
    df_sh = df.copy()
    df_sh["bankroll"] = pd.to_numeric(df_sh["bankroll"], errors="coerce")
    df_sh = df_sh.sort_values("ts").dropna(subset=["bankroll"])
    if len(df_sh) < 2:
        return 0.0
    equity = np.concatenate([[initial_bankroll], df_sh["bankroll"].values])
    ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], np.nan)
    ret = ret[~np.isnan(ret) & np.isfinite(ret)]
    if len(ret) < 2 or np.std(ret) == 0:
        return 0.0
    return float((np.mean(ret) / np.std(ret)) * math.sqrt(252))


class MonotonicityEngine:
    def __init__(
        self,
        series_ticker: str = "KXBTC15M",
        bankroll: float = 1000.0,
        num_markets: int = 500,
        refresh_markets: bool = False,
        max_close_exclusive_ts: Optional[int] = None,
        config: Optional[MonotonicityConfig] = None,
    ):
        self.series_ticker = series_ticker
        self.num_markets = num_markets
        self.refresh_markets = refresh_markets
        self.max_close_exclusive_ts = max_close_exclusive_ts
        self.config = config or MonotonicityConfig()
        self.fetcher = DataFetcher(series_ticker)
        self.strategy = MonotonicityStrategy(bankroll=bankroll, config=self.config)

        self.ladders_rejected_alignment: List[Tuple[str, str]] = []
        self.ladder_summaries: List[Dict[str, Any]] = []

    def run(self, silent: bool = False) -> Dict[str, Any]:
        strat = self.strategy
        markets = self.fetcher.fetch_markets(
            limit=self.num_markets,
            refresh=self.refresh_markets,
            max_close_exclusive_ts=self.max_close_exclusive_ts,
        )

        ladders, rejected = group_markets_into_ladders(
            markets,
            self.series_ticker,
            strict_grouping=self.config.strict_grouping,
        )

        rej_hist = Counter(r.reason for r in rejected)
        ladders_sorted = sorted(ladders, key=lambda L: L.close_ts)

        trades_entered = 0
        for ladder in ladders_sorted:
            rej = validate_ladder_homogeneous(ladder)
            if rej:
                self.ladder_summaries.append(
                    {
                        "ladder_id": ladder.ladder_id,
                        "size": len(ladder.markets),
                        "close_ts": ladder.close_ts,
                        "traded": False,
                        "pnl": 0.0,
                        "reject": rej.reason.value,
                    }
                )
                continue

            aligned, reason = align_ladder_bars(
                self.fetcher,
                ladder,
                fill_minute_gaps=self.config.fill_minute_gaps,
            )
            if aligned is None:
                self.ladders_rejected_alignment.append((ladder.ladder_id, reason or ""))
                self.ladder_summaries.append(
                    {
                        "ladder_id": ladder.ladder_id,
                        "size": len(ladder.markets),
                        "close_ts": ladder.close_ts,
                        "traded": False,
                        "pnl": 0.0,
                        "reject": f"align:{reason}",
                    }
                )
                continue

            open_ts = ladder.open_ts
            close_ts = ladder.close_ts
            results_map = {
                m.ticker: m.raw_market.get("result", "") for m in ladder.markets
            }

            pre_long = len(
                [
                    e
                    for e in strat.trade_log
                    if e.get("action") == "mono_entry" and e.get("leg") == "long_yes"
                ]
            )
            strat.on_ladder_open(ladder, open_ts)

            for bm in aligned:
                bar_ts = next(iter(bm.values())).ts
                strat.on_ladder_bar(ladder, bar_ts, bm, open_ts, close_ts)

            strat.on_ladder_settle(ladder, results_map)

            post_long = len(
                [
                    e
                    for e in strat.trade_log
                    if e.get("action") == "mono_entry" and e.get("leg") == "long_yes"
                ]
            )
            entered = post_long > pre_long
            if entered:
                trades_entered += 1

            settle_pnls = [
                e.get("pnl", 0)
                for e in strat.trade_log
                if e.get("action") == "mono_settlement" and e.get("ladder_id") == ladder.ladder_id
            ]
            ladder_pnl = float(sum(settle_pnls)) if settle_pnls else 0.0
            self.ladder_summaries.append(
                {
                    "ladder_id": ladder.ladder_id,
                    "size": len(ladder.markets),
                    "close_ts": ladder.close_ts,
                    "traded": entered,
                    "pnl": ladder_pnl,
                    "reject": "",
                }
            )

        self.fetcher.close()
        return self._build_report(
            markets_count=len(markets),
            ladders=len(ladders),
            rejected=rejected,
            rej_hist=rej_hist,
            trades_entered=trades_entered,
            silent=silent,
        )

    def _build_report(
        self,
        markets_count: int,
        ladders: int,
        rejected: List[RejectedLadder],
        rej_hist: Counter,
        trades_entered: int,
        silent: bool,
    ) -> Dict[str, Any]:
        strat = self.strategy
        df = pd.DataFrame(strat.trade_log) if strat.trade_log else pd.DataFrame()

        net_pnl = strat.bankroll - strat.initial_bankroll
        total_return = (net_pnl / strat.initial_bankroll) * 100 if strat.initial_bankroll else 0
        dd_pct = (strat.max_drawdown / strat.peak_bankroll) * 100 if strat.peak_bankroll else 0
        sharpe = equity_metrics_from_trade_log(df, strat.initial_bankroll)

        avg_raw = (
            float(np.mean(strat.raw_violation_samples)) if strat.raw_violation_samples else 0.0
        )
        avg_exe = (
            float(np.mean(strat.executable_violation_samples))
            if strat.executable_violation_samples
            else 0.0
        )
        avg_net = (
            float(np.mean(strat.net_violation_samples)) if strat.net_violation_samples else 0.0
        )

        analyzed = len(self.ladder_summaries)
        valid_ladders = len(
            [s for s in self.ladder_summaries if not s.get("reject")]
        )

        pnl_by_size: Dict[int, float] = defaultdict(float)
        pnl_by_gap: Dict[str, float] = defaultdict(float)
        hold_seconds: List[float] = []

        settle_rows = df[df["action"] == "mono_settlement"] if not df.empty else pd.DataFrame()
        if not settle_rows.empty and "ladder_id" in df.columns:
            for lid in settle_rows["ladder_id"].unique():
                sub = settle_rows[settle_rows["ladder_id"] == lid]
                pnl_l = float(sub["pnl"].sum())
                ls = next((x for x in self.ladder_summaries if x["ladder_id"] == lid), None)
                if ls:
                    sz = int(ls.get("size", 0))
                    pnl_by_size[sz] += pnl_l
                    ct = int(ls.get("close_ts", 0))
                    ts_first = sub["ts"].min()
                    if pd.notna(ts_first) and ct:
                        hold_seconds.append(float(ct - float(ts_first)))

        seen_ts = set()
        for e in strat.trade_log:
            if e.get("action") != "mono_settlement":
                continue
            if e.get("leg") != "long_yes":
                continue
            ts_key = (e.get("ladder_id"), e.get("ts"))
            if ts_key in seen_ts:
                continue
            seen_ts.add(ts_key)
            paired = e.get("paired_ticker")
            th = e.get("threshold")
            if not paired or th is None:
                continue
            oth = next(
                (
                    x.get("threshold")
                    for x in strat.trade_log
                    if x.get("market") == paired
                    and x.get("action") == "mono_settlement"
                    and x.get("ts") == e.get("ts")
                ),
                None,
            )
            if oth is None:
                continue
            gap = abs(float(th) - float(oth))
            key = f"{gap:g}"
            pnl_short = next(
                (
                    float(x.get("pnl", 0))
                    for x in strat.trade_log
                    if x.get("market") == paired
                    and x.get("action") == "mono_settlement"
                    and x.get("ts") == e.get("ts")
                ),
                0.0,
            )
            pnl_by_gap[key] += float(e.get("pnl", 0)) + pnl_short

        if not silent:
            print()
            print("=" * 60)
            print(f"  MONOTONICITY BACKTEST: {self.series_ticker}")
            print("=" * 60)
            print(f"  Markets fetched: {markets_count}")
            print(f"  Ladders formed:  {ladders}")
            print(f"  Valid / analyzed:{valid_ladders} / {analyzed}")
            print(f"  Rejected (grouping): {len(rejected)}")
            for reason, c in sorted(rej_hist.items(), key=lambda x: -x[1]):
                print(f"    {reason.value}: {c}")
            print(f"  Alignment rejects: {len(self.ladders_rejected_alignment)}")
            print(f"  Opportunity bars: {strat.opportunities_bars}")
            print(f"  Trades entered:   {trades_entered}")
            print(f"  Avg raw / exe / net violation: {avg_raw:.4f} / {avg_exe:.4f} / {avg_net:.4f}")
            print()
            print("  PERFORMANCE")
            print(f"    Initial: ${strat.initial_bankroll:,.2f}  Final: ${strat.bankroll:,.2f}")
            print(f"    Net P/L: ${net_pnl:+,.2f}  Return: {total_return:+.2f}%")
            print(f"    Max DD:  ${strat.max_drawdown:,.2f} ({dd_pct:.1f}%)  Sharpe: {sharpe:.2f}")
            print("=" * 60)
            if self.ladder_summaries:
                print()
                print("  LADDER SUMMARY (first 30)")
                print(f"  {'ladder_id':<40} {'sz':>3} {'close_ts':>12} {'traded':>7} {'pnl':>10} {'reject':<20}")
                for row in self.ladder_summaries[:30]:
                    print(
                        f"  {row['ladder_id'][:38]:<40} {row['size']:>3} {row['close_ts']:>12} "
                        f"{'Y' if row['traded'] else 'N':>7} {row['pnl']:>+10.2f} {str(row.get('reject','')):<20}"
                    )

        return {
            "df": df,
            "initial_bankroll": strat.initial_bankroll,
            "final_bankroll": strat.bankroll,
            "net_pnl": net_pnl,
            "total_return_pct": total_return,
            "markets_total": markets_count,
            "ladders_formed": ladders,
            "ladders_valid": valid_ladders,
            "ladders_analyzed": analyzed,
            "rejected_grouping": len(rejected),
            "rejection_by_reason": {k.value: v for k, v in rej_hist.items()},
            "alignment_rejects": len(self.ladders_rejected_alignment),
            "opportunities_bars": strat.opportunities_bars,
            "trades_entered": trades_entered,
            "avg_raw_violation": avg_raw,
            "avg_executable_violation": avg_exe,
            "avg_net_violation": avg_net,
            "max_drawdown": strat.max_drawdown,
            "max_drawdown_pct": dd_pct,
            "sharpe": sharpe,
            "ladder_summary_table": pd.DataFrame(self.ladder_summaries)
            if self.ladder_summaries
            else pd.DataFrame(),
            "pnl_by_ladder_size": dict(pnl_by_size),
            "pnl_by_threshold_distance": dict(pnl_by_gap),
            "holding_periods_sec": hold_seconds,
        }
