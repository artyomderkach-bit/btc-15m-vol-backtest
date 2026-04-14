"""
DataFetcher: Authenticated Kalshi API client with local SQLite caching.
Fetches markets, 1-minute candlesticks, and merged public trades for backtesting.
"""
import os
import sys
import time
import json
import base64
import sqlite3
import datetime
import warnings
from typing import Optional

warnings.filterwarnings('ignore', module='urllib3')

import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # parent dir: .env, cache.db, kalshi_private_key

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, '.env'), override=False)
except ImportError:
    pass

BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
API_KEY_ID = os.environ.get('KALSHI_API_KEY_ID', '')
_PRIVATE_KEY_PATH = os.environ.get('KALSHI_PRIVATE_KEY_PATH', 'kalshi_private_key')
_PRIVATE_KEY_PATH = os.path.join(_PROJECT_ROOT, _PRIVATE_KEY_PATH) if not os.path.isabs(_PRIVATE_KEY_PATH) else _PRIVATE_KEY_PATH

DB_PATH = os.path.join(_PROJECT_ROOT, 'cache.db')


def _trade_created_unix(tr: dict) -> Optional[int]:
    for key in ('created_time', 'created_ts', 'ts'):
        ct = tr.get(key)
        if ct is None:
            continue
        if isinstance(ct, (int, float)):
            ts = int(ct)
            return ts // 1000 if ts > 1_000_000_000_000 else ts
        try:
            return int(datetime.datetime.fromisoformat(str(ct).replace('Z', '+00:00')).timestamp())
        except Exception:
            continue
    return None


def _trade_yes_price_dollars(tr: dict) -> Optional[float]:
    """YES price in 0–1 dollars from API trade object (field names vary by version)."""
    yd = tr.get('yes_price_dollars')
    if yd is not None and yd != '':
        try:
            return float(yd)
        except (TypeError, ValueError):
            pass
    yc = tr.get('yes_price')
    if yc is not None:
        try:
            v = float(yc)
            if v > 1.0:
                return v / 100.0
            return v
        except (TypeError, ValueError):
            pass
    p = tr.get('price')
    if p is not None:
        try:
            v = float(p)
            if v > 1.0:
                return v / 100.0
            return v
        except (TypeError, ValueError):
            pass
    return None


def aggregate_trades_to_minute_candles(trades: list, open_ts: int, close_ts: int) -> list:
    """
    Build 1-minute synthetic candle dicts (dollar-string OHLC) from public trades.
    Compatible with engine.parse_candle. Uses trade YES prices; forward-fills
    minutes with no prints after the first trade so bar count matches the window.
    """
    parsed = []
    for tr in trades:
        ts = _trade_created_unix(tr)
        if ts is None:
            continue
        if not (open_ts <= ts <= close_ts):
            continue
        yp = _trade_yes_price_dollars(tr)
        if yp is None:
            continue
        try:
            v = float(tr.get('count_fp') or tr.get('count') or 0)
        except (TypeError, ValueError):
            v = 0.0
        parsed.append((ts, yp, v))
    parsed.sort(key=lambda x: x[0])

    from collections import defaultdict
    buckets = defaultdict(list)
    for ts, yp, v in parsed:
        end = (ts // 60) * 60 + 60
        buckets[end].append((ts, yp, v))

    # Bar end timestamps: first minute ending after open; last minute ending at/including close.
    first_end = (open_ts // 60) * 60 + 60
    last_end = ((close_ts - 1) // 60 + 1) * 60
    if first_end > last_end:
        return []

    def _px(x: float) -> str:
        return f"{x:.4f}"

    out = []
    last_yes = None
    for end in range(first_end, last_end + 1, 60):
        if end in buckets:
            pts = sorted(buckets[end], key=lambda x: x[0])
            o = pts[0][1]
            c = pts[-1][1]
            hi = max(p[1] for p in pts)
            lo = min(p[1] for p in pts)
            vol = int(sum(p[2] for p in pts))
            last_yes = c
        elif last_yes is not None:
            o = hi = lo = c = last_yes
            vol = 0
        else:
            continue
        px = {
            'open_dollars': _px(o),
            'high_dollars': _px(hi),
            'low_dollars': _px(lo),
            'close_dollars': _px(c),
        }
        raw = {
            'end_period_ts': end,
            'price': px,
            'yes_bid': dict(px),
            'yes_ask': dict(px),
            'volume_fp': f"{vol:.2f}",
        }
        out.append(raw)
    return out


class DataFetcher:
    def __init__(self, series_ticker='KXBTC15M'):
        self.series_ticker = series_ticker
        self._private_key = self._load_private_key()
        self._init_db()
        self._request_count = 0
        self._last_request_ts = 0.0
        self._cutoff_cache = None

    # ── Auth ──

    def _load_private_key(self):
        pem_env = os.environ.get('KALSHI_PRIVATE_KEY_PEM')
        if pem_env:
            key_data = pem_env.encode() if isinstance(pem_env, str) else pem_env
            if key_data.strip():
                return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())
        if not os.path.exists(_PRIVATE_KEY_PATH):
            print(f"ERROR: Private key not found: {_PRIVATE_KEY_PATH}", file=sys.stderr)
            sys.exit(1)
        with open(_PRIVATE_KEY_PATH, 'rb') as f:
            key_data = f.read()
        if not key_data.strip():
            print("ERROR: Private key file is empty.", file=sys.stderr)
            sys.exit(1)
        return serialization.load_pem_private_key(key_data, password=None, backend=default_backend())

    def _sign(self, timestamp, method, path):
        message = f"{timestamp}{method}{path}".encode('utf-8')
        sig = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256()
        )
        return base64.b64encode(sig).decode('utf-8')

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < 0.35:
            time.sleep(0.35 - elapsed)
        self._last_request_ts = time.time()
        self._request_count += 1

    def _api_get(self, path, params=None, retries=5, not_found_ok=False):
        """
        GET JSON from Trade API. If not_found_ok=True, returns None on HTTP 404
        (no retries) — used when probing live vs historical candlestick tiers.
        """
        for attempt in range(retries):
            self._rate_limit()
            timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
            full_path = f"/trade-api/v2{path}" if not path.startswith('/trade-api') else path
            signature = self._sign(timestamp, 'GET', full_path.split('?')[0])
            headers = {
                'KALSHI-ACCESS-KEY': API_KEY_ID,
                'KALSHI-ACCESS-SIGNATURE': signature,
                'KALSHI-ACCESS-TIMESTAMP': timestamp,
            }
            try:
                r = requests.get(BASE_URL + path, headers=headers, params=params, timeout=60)
                if r.status_code == 404 and not_found_ok:
                    return None
                if r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  API request retry {attempt + 1}/{retries}: {e}", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

    # ── SQLite cache ──

    def _init_db(self):
        self._conn = sqlite3.connect(DB_PATH)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                ticker TEXT PRIMARY KEY,
                series_ticker TEXT,
                data TEXT,
                fetched_at TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candlesticks (
                market_ticker TEXT,
                series_ticker TEXT,
                data TEXT,
                fetched_at TEXT,
                PRIMARY KEY (market_ticker, series_ticker)
            )
        """)
        # V2 candle cache includes time-range and source, preventing stale reuse.
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candlesticks_v2 (
                market_ticker TEXT,
                series_ticker TEXT,
                start_ts INTEGER,
                end_ts INTEGER,
                period_interval INTEGER,
                source TEXT,
                data TEXT,
                fetched_at TEXT,
                PRIMARY KEY (market_ticker, series_ticker, start_ts, end_ts, period_interval, source)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS market_trades_v2 (
                market_ticker TEXT,
                min_ts INTEGER,
                max_ts INTEGER,
                data TEXT,
                fetched_at TEXT,
                PRIMARY KEY (market_ticker, min_ts, max_ts)
            )
        """)
        self._conn.commit()

    def _get_cached_markets(self):
        rows = self._conn.execute(
            "SELECT ticker, data FROM markets WHERE series_ticker=?",
            (self.series_ticker,)
        ).fetchall()
        out = {}
        for r in rows:
            m = json.loads(r[1])
            if self._market_matches_series(m):
                out[r[0]] = m
        return out

    def _market_matches_series(self, market):
        st = str(market.get('series_ticker') or '')
        tk = str(market.get('ticker') or '')
        if st == self.series_ticker:
            return True
        # Some payloads may omit series_ticker; fallback to ticker prefix convention.
        return tk.startswith(self.series_ticker + '-')

    def _cache_market(self, market):
        ticker = market['ticker']
        series = market.get('series_ticker') or self.series_ticker
        self._conn.execute(
            "INSERT OR REPLACE INTO markets (ticker, series_ticker, data, fetched_at) VALUES (?,?,?,?)",
            (ticker, series, json.dumps(market), datetime.datetime.utcnow().isoformat())
        )

    def _get_cached_candles(self, market_ticker, start_ts, end_ts, period_interval, source):
        row = self._conn.execute(
            """
            SELECT data FROM candlesticks_v2
            WHERE market_ticker=? AND series_ticker=? AND start_ts=? AND end_ts=? AND period_interval=? AND source=?
            """,
            (market_ticker, self.series_ticker, int(start_ts), int(end_ts), int(period_interval), source)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def _cache_candles(self, market_ticker, start_ts, end_ts, period_interval, source, candles):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO candlesticks_v2
            (market_ticker, series_ticker, start_ts, end_ts, period_interval, source, data, fetched_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                market_ticker,
                self.series_ticker,
                int(start_ts),
                int(end_ts),
                int(period_interval),
                source,
                json.dumps(candles),
                datetime.datetime.utcnow().isoformat(),
            )
        )

    def _get_historical_cutoff(self):
        if self._cutoff_cache is not None:
            return self._cutoff_cache
        try:
            resp = self._api_get('/historical/cutoff')
            self._cutoff_cache = resp
        except Exception:
            self._cutoff_cache = {}
        return self._cutoff_cache

    def _market_settled_cutoff_ts(self):
        cutoff = self._get_historical_cutoff() or {}
        raw = cutoff.get('market_settled_ts')
        if raw is None and isinstance(cutoff.get('cutoff'), dict):
            raw = cutoff['cutoff'].get('market_settled_ts')
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            ts = int(raw)
            return ts // 1000 if ts > 1_000_000_000_000 else ts
        try:
            return int(datetime.datetime.fromisoformat(str(raw).replace('Z', '+00:00')).timestamp())
        except Exception:
            return None

    def _trades_created_cutoff_ts(self):
        cutoff = self._get_historical_cutoff()
        raw = cutoff.get('trades_created_ts')
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            ts = int(raw)
            return ts // 1000 if ts > 1_000_000_000_000 else ts
        try:
            return int(datetime.datetime.fromisoformat(str(raw).replace('Z', '+00:00')).timestamp())
        except Exception:
            return None

    def _get_cached_trades(self, market_ticker, min_ts, max_ts):
        row = self._conn.execute(
            """
            SELECT data FROM market_trades_v2
            WHERE market_ticker=? AND min_ts=? AND max_ts=?
            """,
            (market_ticker, int(min_ts), int(max_ts)),
        ).fetchone()
        if row:
            data = json.loads(row[0])
            # Do not treat empty cached lists as hits (legacy bad caches).
            if isinstance(data, list) and len(data) == 0:
                return None
            return data
        return None

    def _cache_trades(self, market_ticker, min_ts, max_ts, trades):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO market_trades_v2
            (market_ticker, min_ts, max_ts, data, fetched_at)
            VALUES (?,?,?,?,?)
            """,
            (
                market_ticker,
                int(min_ts),
                int(max_ts),
                json.dumps(trades),
                datetime.datetime.utcnow().isoformat(),
            )
        )

    def _fetch_trades_paginated(self, path, base_params):
        out = []
        cursor = None
        for _ in range(500):
            params = dict(base_params)
            if cursor:
                params['cursor'] = cursor
            resp = self._api_get(path, params=params)
            batch = resp.get('trades', [])
            out.extend(batch)
            cursor = resp.get('cursor')
            if not cursor or not batch:
                break
            time.sleep(0.05)
        return out

    def fetch_market_trades(self, market_ticker, min_ts, max_ts):
        """
        Public trades for a market in [min_ts, max_ts] (Unix seconds, inclusive),
        merged from /historical/trades (older than trades cutoff) and /markets/trades.
        Deduped by trade_id. Request bounds are widened slightly then filtered client-side.
        """
        min_ts = int(min_ts)
        max_ts = int(max_ts)
        cached = self._get_cached_trades(market_ticker, min_ts, max_ts)
        if cached is not None:
            return cached

        # Widen request window: Kalshi filters are often strict; we filter below.
        q_lo = max(0, min_ts - 60)
        q_hi = max_ts + 60

        C = self._trades_created_cutoff_ts()
        merged = []
        if C is not None:
            if q_lo < C:
                h_max = min(q_hi, C - 1)
                if h_max >= q_lo:
                    merged.extend(
                        self._fetch_trades_paginated(
                            '/historical/trades',
                            {'ticker': market_ticker, 'min_ts': q_lo, 'max_ts': h_max, 'limit': 1000},
                        )
                    )
            if q_hi >= C:
                live_min = max(q_lo, C)
                merged.extend(
                    self._fetch_trades_paginated(
                        '/markets/trades',
                        {'ticker': market_ticker, 'min_ts': live_min, 'max_ts': q_hi, 'limit': 1000},
                    )
                )
        else:
            merged.extend(
                self._fetch_trades_paginated(
                    '/markets/trades',
                    {'ticker': market_ticker, 'min_ts': q_lo, 'max_ts': q_hi, 'limit': 1000},
                )
            )
            if not merged:
                merged.extend(
                    self._fetch_trades_paginated(
                        '/historical/trades',
                        {'ticker': market_ticker, 'min_ts': q_lo, 'max_ts': q_hi, 'limit': 1000},
                    )
                )

        by_id = {}
        no_id = []
        for tr in merged:
            tid = tr.get('trade_id')
            if tid:
                by_id[tid] = tr
            else:
                no_id.append(tr)
        combined = list(by_id.values()) + no_id
        out = []
        for tr in combined:
            tt = _trade_created_unix(tr)
            if tt is None:
                continue
            # Inclusive window [open, close] for market lifetime
            if min_ts <= tt <= max_ts:
                out.append(tr)
        out.sort(key=lambda t: _trade_created_unix(t) or 0)
        if out:
            self._cache_trades(market_ticker, min_ts, max_ts, out)
            self._conn.commit()
        return out

    # ── Public API ──

    def _close_ts(self, m):
        """Extract close_time as Unix timestamp for sorting."""
        ct = m.get('close_time')
        if ct is None:
            return 0
        if isinstance(ct, (int, float)):
            return int(ct) if ct < 1e12 else int(ct) // 1000
        try:
            return int(datetime.datetime.fromisoformat(str(ct).replace('Z', '+00:00')).timestamp())
        except Exception:
            return 0

    def fetch_markets(self, limit=200, refresh=False, max_close_exclusive_ts=None, use_cache_only=False):
        """
        Fetch settled markets for the series.
        Returns the most recent `limit` markets (by close_time), optionally
        only those with close_time < max_close_exclusive_ts (UTC unix seconds).
        refresh=True: fetch fresh from API first to get latest settled markets.
        use_cache_only=True: never paginate the API just to grow the cache toward ``limit``;
        use whatever is already cached (for fixed ticker lists larger than cache size).
        """
        if refresh:
            print(f"  Refreshing markets from API for {self.series_ticker}...")
            cursor = None
            fetched = 0
            for _ in range(120):  # Enough pages to cover long history + latest settles
                params = {'status': 'settled', 'limit': 500, 'series_ticker': self.series_ticker}
                if cursor:
                    params['cursor'] = cursor
                resp = self._api_get('/markets', params=params)
                batch = resp.get('markets', [])
                for m in batch:
                    if self._market_matches_series(m):
                        self._cache_market(m)
                fetched += len(batch)
                cursor = resp.get('cursor')
                if not batch or not cursor:
                    break
                time.sleep(0.4)

            # Pull archived settled markets from historical endpoint as well.
            cursor = None
            for _ in range(120):
                params = {'limit': 500, 'series_ticker': self.series_ticker}
                if cursor:
                    params['cursor'] = cursor
                try:
                    resp = self._api_get('/historical/markets', params=params)
                except Exception:
                    break
                batch = resp.get('markets', [])
                for m in batch:
                    if self._market_matches_series(m):
                        self._cache_market(m)
                fetched += len(batch)
                cursor = resp.get('cursor')
                if not batch or not cursor:
                    break
                time.sleep(0.4)
            self._conn.commit()
            print(f"  Fetched {fetched} new/updated markets.")

        cached = self._get_cached_markets()
        need_api = (
            not cached
            or (refresh is False and len(cached) < limit and not use_cache_only)
        )
        if need_api:
            # Need to fetch from API
            if not refresh:
                print(f"  Fetching markets from API for {self.series_ticker}...")
            cursor = None
            for _ in range(20):
                params = {'status': 'settled', 'limit': min(1000, limit - len(cached) if cached else limit), 'series_ticker': self.series_ticker}
                if cursor:
                    params['cursor'] = cursor
                resp = self._api_get('/markets', params=params)
                batch = resp.get('markets', [])
                for m in batch:
                    if self._market_matches_series(m):
                        self._cache_market(m)
                cached = self._get_cached_markets()
                cursor = resp.get('cursor')
                if not batch or not cursor or len(cached) >= limit:
                    break
                time.sleep(0.4)
            self._conn.commit()
            cached = self._get_cached_markets()
        if not cached:
            return []

        # Sort by close_time descending (most recent first), take first limit
        all_markets = list(cached.values())
        if max_close_exclusive_ts is not None:
            before = len(all_markets)
            all_markets = [m for m in all_markets if self._close_ts(m) < max_close_exclusive_ts]
            print(f"  Date filter: {before} → {len(all_markets)} markets (close < {max_close_exclusive_ts} UTC).")
        all_markets.sort(key=self._close_ts, reverse=True)
        markets = all_markets[:limit]
        print(f"  Using {len(markets)} most recent markets (of {len(all_markets)} cached) for {self.series_ticker}.")
        return markets

    def fetch_candles(self, market_ticker, open_ts, close_ts, period_interval=1):
        """
        Fetch 1-min candles for a market. Uses cache.

        The API returns **sparse** rows: one object per period that had book/trade
        updates, not a dense minute grid. For backtests that need one bar per
        minute (aligned with trade-aggregated bars), use
        ``engine.expand_sparse_candles_to_minute_grid`` after fetching.

        Kalshi splits candle data: markets that settled before ``market_settled_ts``
        cutoff are served from ``/historical/markets/{ticker}/candlesticks``; newer
        markets from ``/series/{series}/markets/{ticker}/candlesticks``. A 404 on
        either tier means the market is not in that archive — we fall back to the
        other tier (handles cutoff skew, caching quirks, and API edge cases).
        See: https://docs.kalshi.com/getting_started/historical_data
        """
        start_ts = open_ts - 60
        end_ts = close_ts + 60
        period_interval = int(period_interval)
        params = {'start_ts': start_ts, 'end_ts': end_ts, 'period_interval': period_interval}

        settled_cutoff = self._market_settled_cutoff_ts()
        prefer_historical = bool(settled_cutoff and int(close_ts) < int(settled_cutoff))

        live_path = f'/series/{self.series_ticker}/markets/{market_ticker}/candlesticks'
        hist_path = f'/historical/markets/{market_ticker}/candlesticks'
        if prefer_historical:
            try_order = [('historical', hist_path), ('live', live_path)]
        else:
            try_order = [('live', live_path), ('historical', hist_path)]

        for source, path in try_order:
            cached = self._get_cached_candles(market_ticker, start_ts, end_ts, period_interval, source)
            if cached is not None:
                return cached
            resp = self._api_get(path, params=params, retries=3, not_found_ok=True)
            if resp is None:
                continue
            candles = resp.get('candlesticks') or []
            if not candles:
                continue
            self._cache_candles(market_ticker, start_ts, end_ts, period_interval, source, candles)
            self._conn.commit()
            return candles

        return []

    def fetch_minute_bars_from_trades(self, market_ticker, open_ts, close_ts):
        """
        Public trades merged from live + historical tiers, aggregated to 1-minute
        synthetic OHLC dicts (same shape as API candlesticks for parse_candle).

        Returns (candlestick_dicts, trades_list).
        """
        trades = self.fetch_market_trades(market_ticker, open_ts, close_ts)
        return aggregate_trades_to_minute_candles(trades, open_ts, close_ts), trades

    def close(self):
        self._conn.close()
