"""
Backtest Engine: Event-driven simulation that ties DataFetcher and Strategy together.
Processes markets chronologically, feeds candles bar-by-bar, and produces results.
"""
import math
import datetime
import numpy as np
import pandas as pd
from dataclasses import replace
from typing import Optional, List, Dict
from collections import defaultdict

from data_fetcher import DataFetcher
from strategy import Strategy
from models import CandleBar, Side


def _to_unix_ts(v) -> Optional[int]:
    """Parse API/open-close timestamps in int(ms)/int(s)/ISO formats."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv // 1000 if iv > 1_000_000_000_000 else iv
    try:
        return int(pd.Timestamp(v).timestamp())
    except Exception:
        return None


def _cents_to_frac(v):
    """Convert API price (int cents or decimal-dollar string) to 0-1 fraction."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and ('.' in v):
            f = float(v)
            # Historical endpoints may return dollar-decimal strings like "0.5600".
            if 0.0 <= f <= 1.0:
                return f
            return f / 100.0
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v / 100.0


def _dollars_to_frac(v) -> Optional[float]:
    """Convert dollar string like '0.0700' to 0-1 fraction."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _extract_field(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def mid_from_bar(bar: CandleBar) -> Optional[float]:
    """YES midpoint at bar close (matches microstructure / mid filter)."""
    ac, bc = bar.ask_close, bar.bid_close
    if ac is not None and bc is not None and ac > bc:
        return (ac + bc) / 2.0
    if bar.price_close is not None:
        return bar.price_close
    return None


def parse_candle(raw) -> Optional[CandleBar]:
    """Parse a raw BTC-style candlestick dict.

    Auto-detects format: integer-cent keys (price['open']) vs dollar-string
    keys (price['open_dollars']) and handles both transparently.
    """
    ts = _extract_field(raw, 'end_period_ts')
    if ts is None:
        return None
    ts = int(ts)

    price = _extract_field(raw, 'price') or {}
    ask = _extract_field(raw, 'yes_ask') or {}
    bid = _extract_field(raw, 'yes_bid') or {}

    # Detect format: if *_dollars keys exist, use dollar-string parsing
    uses_dollars = ('open_dollars' in price or 'close_dollars' in price)

    if uses_dollars:
        def _p(key):
            return _dollars_to_frac(price.get(key + '_dollars'))
        def _a(key):
            return _dollars_to_frac(ask.get(key + '_dollars'))
        def _b(key):
            return _dollars_to_frac(bid.get(key + '_dollars'))

        vol_raw = _extract_field(raw, 'volume_fp')
        try:
            vol = int(float(vol_raw)) if vol_raw is not None else 0
        except (TypeError, ValueError):
            vol = 0
    else:
        def _p(key):
            return _cents_to_frac(_extract_field(price, key))
        def _a(key):
            return _cents_to_frac(_extract_field(ask, key))
        def _b(key):
            return _cents_to_frac(_extract_field(bid, key))

        vol = _extract_field(raw, 'volume')
        if vol is None:
            vol = _extract_field(raw, 'volume_fp')
        try:
            vol = int(float(vol)) if vol is not None else 0
        except (TypeError, ValueError):
            vol = 0

    return CandleBar(
        ts=ts,
        price_open=_p('open'),
        price_high=_p('high'),
        price_low=_p('low'),
        price_close=_p('close'),
        ask_low=_a('low'),
        ask_high=_a('high'),
        ask_close=_a('close'),
        bid_low=_b('low'),
        bid_high=_b('high'),
        bid_close=_b('close'),
        volume=vol,
    )


def expand_sparse_candles_to_minute_grid(
    raw_candles: list,
    open_ts: int,
    close_ts: int,
) -> List[CandleBar]:
    """
    Kalshi returns sparse candle rows (minutes with activity only). The trade-based
    path uses forward-filled minute OHLC (see aggregate_trades_to_minute_candles).
    This applies the same minute grid and forward-fill so API candles match that
    semantics and match the backtest bar count used for strategies.

    Minute ends are those with open_ts < end <= close_ts, aligned to the same
    first_end/last_end logic as data_fetcher.aggregate_trades_to_minute_candles.
    """
    parsed: List[CandleBar] = []
    for rc in raw_candles:
        b = parse_candle(rc)
        if b and open_ts < b.ts <= close_ts:
            parsed.append(b)
    parsed.sort(key=lambda x: x.ts)
    by_ts: Dict[int, CandleBar] = {}
    for b in parsed:
        by_ts[b.ts] = b

    first_end = (open_ts // 60) * 60 + 60
    last_end = ((close_ts - 1) // 60 + 1) * 60
    if first_end > last_end:
        return []

    out: List[CandleBar] = []
    last: Optional[CandleBar] = None
    for end in range(first_end, last_end + 1, 60):
        if end in by_ts:
            last = by_ts[end]
            out.append(last)
        elif last is not None:
            out.append(replace(last, ts=end, volume=0))
    return out


class Engine:
    def __init__(self, series_ticker='KXBTC15M', bankroll=1000.0,
                 risk_pct=0.01, num_markets=200, volume_fill_pct=0.10,
                 buy_price=0.10, sell_price=0.40, tp_fill_rate=1.0,
                 refresh_markets=False, start_after_ts=None,
                 stop_loss_price=0.0, trailing_stop_trigger=0.0,
                 trailing_stop_floor=0.0, vix_min=0.0,
                 weekend_size_mult=1.0, single_side='both',
                 entry_cutoff_seconds=240,
                 session_hours=(0, 24), max_consecutive_losses_halt=0,
                 mid_filter_min=None, mid_filter_max=None, mid_filter_bars=4,
                 mid_filter_mode='off', max_close_exclusive_ts=None,
                 fill_minute_gaps=False):
        self.series_ticker = series_ticker
        self.num_markets = num_markets
        self.fetcher = DataFetcher(series_ticker)
        self.max_close_exclusive_ts = max_close_exclusive_ts
        self.mid_filter_mode = (mid_filter_mode or 'off').lower()
        use_delayed = self.mid_filter_mode == 'delayed'
        self._mid_oracle_lo = mid_filter_min
        self._mid_oracle_hi = mid_filter_max
        self.strategy = Strategy(
            bankroll=bankroll, risk_pct=risk_pct,
            volume_fill_pct=volume_fill_pct,
            buy_price=buy_price, sell_price=sell_price,
            tp_fill_rate=tp_fill_rate,
            stop_loss_price=stop_loss_price,
            trailing_stop_trigger=trailing_stop_trigger,
            trailing_stop_floor=trailing_stop_floor,
            vix_min=vix_min,
            weekend_size_mult=weekend_size_mult,
            single_side=single_side,
            entry_cutoff_seconds=entry_cutoff_seconds,
            session_hours=session_hours,
            max_consecutive_losses_halt=max_consecutive_losses_halt,
            mid_filter_min=mid_filter_min if use_delayed else None,
            mid_filter_max=mid_filter_max if use_delayed else None,
            mid_filter_bars=mid_filter_bars,
        )
        self.refresh_markets = refresh_markets
        self.start_after_ts = start_after_ts
        # Sparse API candles (illiquid series): forward-fill to one bar per minute like trade aggregates.
        self.fill_minute_gaps = fill_minute_gaps

        # Load VIX daily data if vix_min is active
        if vix_min > 0:
            self._load_vix_data()

    def _load_vix_data(self):
        """Load overlay_vix.csv and inject it into the strategy as date_int -> vix_level."""
        import os
        vix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overlay_vix.csv')
        if not os.path.exists(vix_path):
            return
        try:
            vix_df = pd.read_csv(vix_path)
            vix_map = {}
            for _, row in vix_df.iterrows():
                d = pd.Timestamp(row['date'])
                date_int = int(d.strftime('%Y%m%d'))
                vix_map[date_int] = float(row['value'])
            self.strategy.set_vix_data(vix_map)
        except Exception:
            pass

    def run(self, silent=False):
        if not silent:
            tpfr = f"  |  TP fill: {self.strategy.tp_fill_rate*100:.0f}%" if self.strategy.tp_fill_rate < 1.0 else ""
            print(f"Starting backtest: {self.series_ticker}")
            print(f"  Bankroll: ${self.strategy.bankroll:.2f}  |  Risk: {self.strategy.risk_pct*100:.0f}%  |  Markets: {self.num_markets}  |  Vol fill: {self.strategy.volume_fill_pct*100:.0f}%{tpfr}")
            print()

        markets = self.fetcher.fetch_markets(
            limit=self.num_markets,
            refresh=self.refresh_markets,
            max_close_exclusive_ts=self.max_close_exclusive_ts,
        )

        # Sort oldest first for correct bankroll evolution
        def _close_ts(m):
            ts = _to_unix_ts(m.get('close_time'))
            return ts or 0
        markets = sorted(markets, key=_close_ts)

        if self.start_after_ts:
            markets = [m for m in markets if _close_ts(m) >= self.start_after_ts]

        total = len(markets)
        traded = 0
        filled_markets = 0
        tp_markets = 0
        settled_markets = 0
        settled_wins = 0
        settled_losses = 0
        tp_hold_times = []

        # Per-hour and per-day tracking
        hour_stats = defaultdict(lambda: {
            'traded': 0, 'filled': 0, 'tp': 0, 'settled': 0,
            'settled_wins': 0, 'settled_losses': 0, 'pnl': 0.0,
        })
        day_stats = defaultdict(lambda: {
            'traded': 0, 'filled': 0, 'tp': 0, 'settled': 0,
            'settled_wins': 0, 'settled_losses': 0, 'pnl': 0.0,
        })

        for i, market in enumerate(markets):
            if self.strategy.halted:
                if not silent:
                    print(f"  Halted at market {i+1}/{total}")
                break

            ticker = market.get('ticker', '')
            close_time = market.get('close_time')
            open_time = market.get('open_time')
            result = market.get('result', '')

            if close_time is None or result not in ('yes', 'no'):
                continue

            close_ts = _to_unix_ts(close_time)
            if close_ts is None:
                continue
            open_ts = _to_unix_ts(open_time)
            if open_ts is None or open_ts >= close_ts:
                # Backward-compatible fallback for legacy series without open_time.
                open_ts = close_ts - 15 * 60
            dt = datetime.datetime.utcfromtimestamp(open_ts)
            hour_utc = dt.hour
            day_utc = dt.weekday()  # 0=Mon, 6=Sun
            if not silent and ((i + 1) % 50 == 0 or i == 0):
                print(f"  Processing {i+1}/{total}: {ticker}  (bankroll: ${self.strategy.bankroll:.2f})")

            # Fetch candles
            try:
                raw_candles = self.fetcher.fetch_candles(ticker, open_ts, close_ts)
            except Exception as e:
                if not silent:
                    print(f"  Error fetching candles for {ticker}: {e}")
                continue

            if self.fill_minute_gaps:
                bars = expand_sparse_candles_to_minute_grid(raw_candles, open_ts, close_ts)
            else:
                bars = []
                for rc in raw_candles:
                    bar = parse_candle(rc)
                    if bar and open_ts < bar.ts <= close_ts:
                        bars.append(bar)
                bars.sort(key=lambda b: b.ts)

            if len(bars) < 6:
                continue

            traded += 1

            pre_log_start = len(self.strategy.trade_log)

            oracle_skip = False
            if (self.mid_filter_mode == 'oracle'
                    and self._mid_oracle_lo is not None
                    and self._mid_oracle_hi is not None
                    and self._mid_oracle_lo < self._mid_oracle_hi
                    and len(bars) >= self.strategy.mid_filter_bars):
                lo, hi = self._mid_oracle_lo, self._mid_oracle_hi
                nb = self.strategy.mid_filter_bars
                mid = mid_from_bar(bars[nb - 1])
                if mid is None or mid < lo or mid > hi:
                    oracle_skip = True
                    self.strategy.trade_log.append({
                        'market': ticker,
                        'side': '',
                        'action': 'skip_mid_oracle',
                        'qty': 0,
                        'entry_price': None,
                        'exit_price': None,
                        'pnl': 0.0,
                        'bankroll': self.strategy.bankroll,
                        'ts': bars[nb - 1].ts,
                        'mid': mid,
                        'mid_lo': lo,
                        'mid_hi': hi,
                    })

            had_fill = False
            had_tp = False
            held_to_settle = False

            if not oracle_skip:
                # Market open: place orders
                self.strategy.on_market_open(ticker, open_ts)

                # Feed bars one by one
                for bar in bars:
                    pre_log_len = len(self.strategy.trade_log)
                    self.strategy.on_candle(ticker, bar, open_ts, close_ts)
                    # Check new trade log entries
                    for entry in self.strategy.trade_log[pre_log_len:]:
                        if entry.get('action') == 'buy_fill':
                            had_fill = True
                        if entry.get('action') == 'sell_fill':
                            had_tp = True
                            hold_time = bar.ts - open_ts
                            tp_hold_times.append(hold_time)

                # Settle remaining positions
                pre_settle_len = len(self.strategy.trade_log)
                self.strategy.on_market_settle(ticker, result)
            else:
                pre_settle_len = len(self.strategy.trade_log)

            for entry in self.strategy.trade_log[pre_settle_len:]:
                if entry.get('action') == 'settlement':
                    held_to_settle = True
                    if entry.get('won'):
                        settled_wins += 1
                    else:
                        settled_losses += 1

            market_pnl = sum(
                e.get('pnl', 0) for e in self.strategy.trade_log[pre_log_start:]
                if e.get('action') in ('sell_fill', 'settlement', 'stop_loss')
            )

            hs = hour_stats[hour_utc]
            ds = day_stats[day_utc]
            hs['traded'] += 1
            ds['traded'] += 1
            if had_fill:
                filled_markets += 1
                hs['filled'] += 1
                ds['filled'] += 1
            if had_tp:
                tp_markets += 1
                hs['tp'] += 1
                ds['tp'] += 1
            if held_to_settle:
                settled_markets += 1
                hs['settled'] += 1
                ds['settled'] += 1
            hs['pnl'] += market_pnl
            ds['pnl'] += market_pnl

        self.fetcher.close()

        # Build results
        return self._build_report(
            total=total, traded=traded, filled=filled_markets,
            tp=tp_markets, settled=settled_markets,
            settled_wins=settled_wins, settled_losses=settled_losses,
            tp_hold_times=tp_hold_times, silent=silent,
            hour_stats=dict(hour_stats), day_stats=dict(day_stats),
        )

    def _build_report(self, total, traded, filled, tp, settled,
                      settled_wins, settled_losses, tp_hold_times, silent=False,
                      hour_stats=None, day_stats=None):
        strat = self.strategy
        df = pd.DataFrame(strat.trade_log) if strat.trade_log else pd.DataFrame()

        net_pnl = strat.bankroll - strat.initial_bankroll
        total_return = (net_pnl / strat.initial_bankroll) * 100
        dd_pct = (strat.max_drawdown / strat.peak_bankroll) * 100 if strat.peak_bankroll else 0

        # Win/loss averages
        if not df.empty and 'pnl' in df.columns:
            wins = df[(df['action'].isin(['sell_fill', 'settlement'])) & (df['pnl'] > 0)]
            losses = df[(df['action'].isin(['sell_fill', 'settlement'])) & (df['pnl'] < 0)]
            avg_win = wins['pnl'].mean() if len(wins) else 0
            avg_loss = losses['pnl'].mean() if len(losses) else 0
        else:
            avg_win = avg_loss = 0

        # Sharpe: equity-curve returns, annualized with sqrt(252)
        if not df.empty and 'bankroll' in df.columns:
            df_sharpe = df.copy()
            df_sharpe['bankroll'] = pd.to_numeric(df_sharpe['bankroll'], errors='coerce')
            market_max_ts = df_sharpe.groupby('market')['ts'].transform('max')
            df_sharpe['_pts'] = df_sharpe['ts'].fillna(market_max_ts + 60)
            df_sharpe = df_sharpe.sort_values('_pts').dropna(subset=['bankroll'])
            equity = np.concatenate([[strat.initial_bankroll], df_sharpe['bankroll'].values])
            ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], np.nan)
            ret = ret[~np.isnan(ret) & np.isfinite(ret)]
            if len(ret) > 1 and np.std(ret) > 0:
                sharpe = (np.mean(ret) / np.std(ret)) * math.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        avg_tp_hold = sum(tp_hold_times) / len(tp_hold_times) if tp_hold_times else 0
        tp_pct = (tp / filled * 100) if filled else 0
        settle_pct = (settled / filled * 100) if filled else 0
        settle_win_rate = (settled_wins / (settled_wins + settled_losses) * 100) if (settled_wins + settled_losses) else 0
        unfilled = traded - filled

        # Fill rates (from trade log)
        if not df.empty and 'action' in df.columns and 'qty' in df.columns:
            buy_fills = df[df['action'] == 'buy_fill']
            sell_fills = df[df['action'] == 'sell_fill']
            contracts_bought = int(buy_fills['qty'].sum()) if len(buy_fills) else 0
            contracts_sold = int(sell_fills['qty'].sum()) if len(sell_fills) else 0
        else:
            contracts_bought = contracts_sold = 0

        if not silent:
            print()
            print("=" * 60)
            print(f"  BACKTEST REPORT: {self.series_ticker}")
            print("=" * 60)
            print()
            print("  FILL RATES")
            pct_10c = (filled / traded * 100) if traded else 0
            tp_c = int(round(strat.sell_price * 100))
            pct_tp = (tp / filled * 100) if filled else 0
            pct_sold = (tp / filled * 100) if filled else 0
            pct_contracts = (contracts_sold / contracts_bought * 100) if contracts_bought else 0
            print(f"    10c buy:     {filled}/{traded} markets ({pct_10c:.1f}%)")
            print(f"    {tp_c}c TP sell: {tp}/{filled} markets ({pct_tp:.1f}%)")
            print(f"    Sold at {tp_c}c: {tp}/{filled} markets ({pct_sold:.1f}%)")
            print(f"    Contracts sold: {contracts_sold}/{contracts_bought} ({pct_contracts:.1f}%)")
            print()
            print("  OVERALL PERFORMANCE")
            print(f"    Initial Bankroll:       ${strat.initial_bankroll:,.2f}")
            print(f"    Final Bankroll:         ${strat.bankroll:,.2f}")
            print(f"    Net P/L:                ${net_pnl:+,.2f}")
            print(f"    Total Return:           {total_return:+.2f}%")
            print()
            print("  TRADE COUNTS")
            print(f"    Markets analyzed:       {total}")
            print(f"    Markets traded:         {traded}")
            print(f"    Markets with 10¢ fill:  {filled}")
            print(f"    Markets unfilled:       {unfilled}")
            print()
            print("  EXECUTION")
            print(f"    Hit {tp_c}¢ TP:             {tp}  ({tp_pct:.1f}% of fills)")
            print(f"    Went to settlement:     {settled}  ({settle_pct:.1f}% of fills)")
            print(f"    Avg hold time (TP):     {avg_tp_hold:.0f}s  ({avg_tp_hold/60:.1f}min)")
            print()
            print("  SETTLEMENT")
            print(f"    Settlement wins:        {settled_wins}")
            print(f"    Settlement losses:      {settled_losses}")
            print(f"    Settlement win rate:    {settle_win_rate:.1f}%")
            print()
            print("  RISK")
            print(f"    Max Drawdown:           ${strat.max_drawdown:,.2f}  ({dd_pct:.1f}%)")
            print(f"    Max Consecutive Losses: {strat.max_consecutive_losses}")
            print(f"    Sharpe Ratio (ann.):    {sharpe:.2f}")
            print()
            print("  AVERAGES")
            print(f"    Avg profit (winners):   ${avg_win:+,.2f}")
            print(f"    Avg loss (losers):      ${avg_loss:+,.2f}")
            print("=" * 60)

            if hour_stats:
                print()
                print("=" * 80)
                print("  PERFORMANCE BY HOUR (UTC)")
                print("=" * 80)
                print(f"  {'Hour':>6}  {'Traded':>7}  {'Filled':>7}  {'TP':>5}  {'Settle':>7}  {'TP%':>6}  {'PnL':>12}")
                print(f"  {'----':>6}  {'------':>7}  {'------':>7}  {'--':>5}  {'------':>7}  {'---':>6}  {'---':>12}")
                for h in range(24):
                    hs = hour_stats.get(h)
                    if not hs or hs['traded'] == 0:
                        continue
                    tp_r = (hs['tp'] / hs['filled'] * 100) if hs['filled'] else 0
                    print(f"  {h:02d}:00   {hs['traded']:>7}  {hs['filled']:>7}  {hs['tp']:>5}  {hs['settled']:>7}  {tp_r:>5.1f}%  ${hs['pnl']:>+10,.2f}")
                print("=" * 80)

            if day_stats:
                DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                print()
                print("=" * 80)
                print("  PERFORMANCE BY DAY OF WEEK (UTC)")
                print("=" * 80)
                print(f"  {'Day':>6}  {'Traded':>7}  {'Filled':>7}  {'TP':>5}  {'Settle':>7}  {'TP%':>6}  {'PnL':>12}")
                print(f"  {'---':>6}  {'------':>7}  {'------':>7}  {'--':>5}  {'------':>7}  {'---':>6}  {'---':>12}")
                for d in range(7):
                    ds = day_stats.get(d)
                    if not ds or ds['traded'] == 0:
                        continue
                    tp_r = (ds['tp'] / ds['filled'] * 100) if ds['filled'] else 0
                    print(f"  {DAY_NAMES[d]:>6}  {ds['traded']:>7}  {ds['filled']:>7}  {ds['tp']:>5}  {ds['settled']:>7}  {tp_r:>5.1f}%  ${ds['pnl']:>+10,.2f}")
                print("=" * 80)

        return {
            'df': df,
            'initial_bankroll': strat.initial_bankroll,
            'final_bankroll': strat.bankroll,
            'net_pnl': net_pnl,
            'total_return_pct': total_return,
            'markets_total': total,
            'markets_traded': traded,
            'markets_filled': filled,
            'tp_count': tp,
            'tp_pct': tp_pct,
            'settled_count': settled,
            'settle_pct': settle_pct,
            'settle_win_rate': settle_win_rate,
            'settled_wins': settled_wins,
            'settled_losses': settled_losses,
            'max_drawdown': strat.max_drawdown,
            'max_drawdown_pct': dd_pct,
            'max_consecutive_losses': strat.max_consecutive_losses,
            'sharpe': sharpe,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_tp_hold_sec': avg_tp_hold,
            'hour_stats': hour_stats or {},
            'day_stats': day_stats or {},
        }
