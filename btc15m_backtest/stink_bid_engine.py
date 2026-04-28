"""
Engine for BTC 15m stink-bid backtests.

This module reuses DataFetcher and existing candle parsing helpers while keeping
existing strategy engines untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data_fetcher import DataFetcher
from engine import _to_unix_ts, expand_sparse_candles_to_minute_grid, mid_from_bar, parse_candle
from stink_bid_config import StinkBidConfig
from stink_bid_strategy import StinkBidStrategy


class StinkBidEngine:
    def __init__(
        self,
        series_ticker: str = "KXBTC15M",
        bankroll: float = 1000.0,
        num_markets: int = 500,
        refresh_markets: bool = False,
        config: Optional[StinkBidConfig] = None,
        trade_cache: Optional[Dict[str, List[dict]]] = None,
    ):
        self.config = config or StinkBidConfig(series_ticker=series_ticker)
        self.series_ticker = series_ticker
        self.num_markets = int(num_markets)
        self.refresh_markets = bool(refresh_markets)
        self.fetcher = DataFetcher(series_ticker)
        self.strategy = StinkBidStrategy(bankroll=bankroll, config=self.config)
        self._trade_cache = trade_cache if trade_cache is not None else {}
        self._market_reports: List[Dict[str, Any]] = []

    def _dense_bars(self, ticker: str, open_ts: int, close_ts: int):
        try:
            raw_candles = self.fetcher.fetch_candles(ticker, open_ts, close_ts)
            bars = expand_sparse_candles_to_minute_grid(raw_candles, open_ts, close_ts)
            if bars:
                return bars
        except Exception:
            pass
        bars = []
        try:
            synthesized, _ = self.fetcher.fetch_minute_bars_from_trades(ticker, open_ts, close_ts)
            for rc in synthesized:
                b = parse_candle(rc)
                if b and open_ts < b.ts <= close_ts:
                    bars.append(b)
            bars.sort(key=lambda x: x.ts)
        except Exception:
            return []
        return bars

    def _get_trades(self, ticker: str, open_ts: int, close_ts: int) -> List[dict]:
        key = f"{ticker}:{open_ts}:{close_ts}"
        cached = self._trade_cache.get(key)
        if cached is not None:
            return cached
        out = self.fetcher.fetch_market_trades(ticker, open_ts, close_ts)
        self._trade_cache[key] = out
        return out

    @staticmethod
    def _ts_for_time_exit(open_ts: int, minutes: int) -> int:
        return int(open_ts + max(1, int(minutes)) * 60)

    @staticmethod
    def _first_mid_before_submit(bars, submit_ts: float) -> Optional[float]:
        candidate = None
        for b in bars:
            if b.ts <= submit_ts + 1e-9:
                candidate = mid_from_bar(b)
        return candidate

    @staticmethod
    def _bar_at_or_after_ts(bars, target_ts: int):
        for b in bars:
            if b.ts >= target_ts:
                return b
        return bars[-1] if bars else None

    def _annotate_adverse_selection(self, market: str, bars) -> None:
        if not bars:
            return
        fill_rows = [
            r for r in self.strategy.trade_log
            if r.get("market") == market and r.get("action") == "fill" and r.get("ts_fill") is not None
        ]
        for row in fill_rows:
            ts_fill = float(row["ts_fill"])
            side = row.get("side", "YES")
            entry = float(row.get("entry_price") or 0.0)
            bar_1m = self._bar_at_or_after_ts(bars, int(ts_fill + 60))
            bar_5m = self._bar_at_or_after_ts(bars, int(ts_fill + 300))
            mid_1m = mid_from_bar(bar_1m) if bar_1m is not None else None
            mid_5m = mid_from_bar(bar_5m) if bar_5m is not None else None

            def _adverse(mid):
                if mid is None:
                    return None
                if side == "YES":
                    return float(mid) - entry
                return (1.0 - float(mid)) - entry

            row["mid_1m"] = mid_1m
            row["mid_5m"] = mid_5m
            row["adverse_selection_1m"] = _adverse(mid_1m)
            row["adverse_selection_5m"] = _adverse(mid_5m)

    def run(
        self,
        config: Optional[StinkBidConfig] = None,
        silent: bool = False,
        progress_every: int = 200,
    ) -> Dict[str, Any]:
        if config is not None:
            self.config = config
            self.strategy = StinkBidStrategy(bankroll=self.strategy.initial_bankroll, config=self.config)
            self._market_reports = []
        mkts = self.fetcher.fetch_markets(limit=self.num_markets, refresh=self.refresh_markets)
        mkts = sorted(mkts, key=lambda m: _to_unix_ts(m.get("close_time")) or 0)
        used_trade_data_count = 0
        missing_trade_data_count = 0
        total_markets = len(mkts)
        import time as _time
        _t_start = _time.time()
        if not silent and total_markets > 0:
            print(f"Starting backtest over {total_markets} markets...", flush=True)

        for _idx, m in enumerate(mkts, start=1):
            ticker = m.get("ticker", "")
            result = (m.get("result") or "").strip().lower()
            close_ts = _to_unix_ts(m.get("close_time"))
            open_ts = _to_unix_ts(m.get("open_time"))
            if close_ts is None:
                continue
            if open_ts is None or open_ts >= close_ts:
                open_ts = int(close_ts - 15 * 60)

            bars = self._dense_bars(ticker, open_ts, close_ts)
            submit_ts = float(open_ts) + self.config.opening_delay_ms / 1000.0
            best_bid_for_percentile = self._first_mid_before_submit(bars, submit_ts)
            orders = self.strategy.submit_open_ladder(
                market=ticker,
                open_ts=open_ts,
                close_ts=close_ts,
                best_bid_yes=best_bid_for_percentile,
            )
            if not orders:
                self._market_reports.append(
                    {
                        "market": ticker,
                        "orders_submitted": 0,
                        "fills": 0,
                        "used_trade_data": False,
                        "note": "no_orders_after_caps_or_price_validation",
                    }
                )
                continue

            trades = self._get_trades(ticker, open_ts, close_ts)
            used_trade_data = len(trades) > 0
            if used_trade_data:
                used_trade_data_count += 1
            else:
                missing_trade_data_count += 1
            self.strategy.process_trade_window(
                market=ticker,
                open_ts=open_ts,
                close_ts=close_ts,
                trades=trades,
                used_trade_data=used_trade_data,
            )
            self._annotate_adverse_selection(ticker, bars)

            if self.config.exit_mode == "time_exit":
                exit_ts = self._ts_for_time_exit(open_ts, self.config.time_exit_minutes)
                exit_bar = self._bar_at_or_after_ts(bars, exit_ts)
                if exit_bar is not None:
                    self.strategy.exit_lots_time(
                        market=ticker,
                        ts_exit=float(exit_bar.ts),
                        yes_bid=exit_bar.bid_close,
                        yes_ask=exit_bar.ask_close,
                        yes_mid=mid_from_bar(exit_bar),
                    )
            self.strategy.settle_market(ticker, result=result, ts_settle=int(close_ts))

            fills_this_market = sum(
                1
                for r in self.strategy.trade_log
                if r.get("market") == ticker and r.get("action") == "fill"
            )
            submits_this_market = sum(
                1
                for r in self.strategy.trade_log
                if r.get("market") == ticker and r.get("action") == "submit"
            )
            cancels_this_market = sum(
                1
                for r in self.strategy.trade_log
                if r.get("market") == ticker and r.get("action") == "cancel"
            )
            self._market_reports.append(
                {
                    "market": ticker,
                    "orders_submitted": submits_this_market,
                    "fills": fills_this_market,
                    "cancels": cancels_this_market,
                    "used_trade_data": used_trade_data,
                }
            )

            if not silent and progress_every and (_idx % int(progress_every) == 0 or _idx == total_markets):
                total_fills_so_far = sum(
                    1 for r in self.strategy.trade_log if r.get("action") == "fill"
                )
                total_submits_so_far = sum(
                    1 for r in self.strategy.trade_log if r.get("action") == "submit"
                )
                elapsed = max(1e-6, _time.time() - _t_start)
                rate = _idx / elapsed
                remaining = (total_markets - _idx) / rate if rate > 0 else 0.0
                print(
                    f"  [{_idx}/{total_markets}] submits={total_submits_so_far} "
                    f"fills={total_fills_so_far} trade_data={used_trade_data_count}/{_idx} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                    flush=True,
                )

        trade_df = pd.DataFrame(self.strategy.trade_log) if self.strategy.trade_log else pd.DataFrame()
        summary = self.strategy.summarize()
        summary["markets_total"] = len(self._market_reports)
        summary["markets_with_trade_data"] = used_trade_data_count
        summary["markets_without_trade_data"] = missing_trade_data_count
        summary["limitations"] = (
            "No queue position model; partial-fill proxy via volume_fill_pct; "
            "sub-minute exits approximated using minute bars."
        )

        sharpe = 0.0
        if not trade_df.empty and "bankroll" in trade_df.columns:
            eq = pd.to_numeric(trade_df["bankroll"], errors="coerce").dropna().values
            if len(eq) >= 2:
                ret = np.diff(np.concatenate([[self.strategy.initial_bankroll], eq])) / np.clip(
                    np.concatenate([[self.strategy.initial_bankroll], eq])[:-1], 1e-9, None
                )
                if len(ret) >= 2 and np.std(ret) > 0:
                    sharpe = float((np.mean(ret) / np.std(ret)) * np.sqrt(252))
        summary["sharpe"] = sharpe
        summary_df = pd.DataFrame([summary])

        report_df = pd.DataFrame(self._market_reports) if self._market_reports else pd.DataFrame()
        if not silent:
            print("=" * 60)
            print(f"STINK BID BACKTEST: {self.series_ticker}")
            print("=" * 60)
            print(
                f"Markets: {summary['markets_total']} | Orders: {summary['total_orders_submitted']} | "
                f"Fills: {summary['total_fills']} | Fill rate: {summary['overall_fill_rate']:.2%}"
            )
            print(
                f"Net PnL: {summary['pnl_net_sum']:+.2f} | Final bankroll: {summary['final_bankroll']:.2f} | "
                f"Max DD: {summary['max_drawdown_pct']:.1f}%"
            )
        return {
            **summary,
            "summary": summary,
            "summary_df": summary_df,
            "df": trade_df,
            "market_df": report_df,
            "trade_cache": self._trade_cache,
        }
