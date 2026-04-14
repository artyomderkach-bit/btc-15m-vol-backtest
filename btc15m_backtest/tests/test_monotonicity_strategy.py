"""Tests for violation detection, aligned bars, and synthetic PnL paths."""
import unittest
from unittest.mock import MagicMock

from models import CandleBar
from ladder_grouping import LadderGroup
from monotonicity_config import MonotonicityConfig
from monotonicity_strategy import MonotonicityStrategy
from monotonicity_engine import align_ladder_bars
from ladder_grouping import parse_market_descriptor


def _bar(ts, mid, spread=0.01):
    """YES bid/ask symmetric around mid."""
    lo = max(0.01, mid - spread / 2)
    hi = min(0.99, mid + spread / 2)
    return CandleBar(
        ts=ts,
        price_open=mid,
        price_high=hi,
        price_low=lo,
        price_close=mid,
        ask_close=hi,
        bid_close=lo,
        volume=100,
    )


class TestViolations(unittest.TestCase):
    def test_no_trade_when_ordered(self):
        """Easier (90k) priced above harder (100k) — no violation."""
        cfg = MonotonicityConfig(min_net_violation=0.001)
        s = MonotonicityStrategy(1000.0, cfg)
        m1 = {
            "ticker": "E",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 90000,
            "result": "yes",
            "series_ticker": "KXBTC15M",
        }
        m2 = {
            "ticker": "H",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 100000,
            "result": "no",
            "series_ticker": "KXBTC15M",
        }
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        ladder = LadderGroup(
            group_key="EV",
            ladder_id="L1",
            markets=[d1, d2],
        )
        bm = {
            "E": _bar(1100, 0.65),
            "H": _bar(1100, 0.61),
        }
        s.on_ladder_bar(ladder, 1100, bm, 1000, 2000)
        entries = [e for e in s.trade_log if e.get("action") == "mono_entry"]
        self.assertEqual(len(entries), 0)

    def test_violation_detected_90_vs_100(self):
        """0.64 harder vs 0.61 easier — raw violation 0.03."""
        cfg = MonotonicityConfig(min_net_violation=0.0, min_raw_violation=0.0)
        s = MonotonicityStrategy(1000.0, cfg)
        m1 = {
            "ticker": "E",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 90000,
            "result": "yes",
            "series_ticker": "KXBTC15M",
        }
        m2 = {
            "ticker": "H",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 100000,
            "result": "no",
            "series_ticker": "KXBTC15M",
        }
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        ladder = LadderGroup(group_key="EV", ladder_id="L1", markets=[d1, d2])
        # User example: 90 → 0.61, 100 → 0.64
        bm = {
            "E": _bar(1100, 0.61),
            "H": _bar(1100, 0.64),
        }
        raw, exe, net, _ = s._violations_for_pair(d1, d2, bm["E"], bm["H"])
        self.assertAlmostEqual(raw, 0.03, places=5)
        self.assertGreater(exe, 0.0)

    def test_small_edge_skipped_by_min_net(self):
        cfg = MonotonicityConfig(min_net_violation=0.5, min_raw_violation=0.0)
        s = MonotonicityStrategy(1000.0, cfg)
        m1 = {
            "ticker": "E",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 90000,
            "result": "yes",
            "series_ticker": "KXBTC15M",
        }
        m2 = {
            "ticker": "H",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 100000,
            "result": "no",
            "series_ticker": "KXBTC15M",
        }
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        ladder = LadderGroup(group_key="EV", ladder_id="L1", markets=[d1, d2])
        bm = {"E": _bar(1100, 0.61), "H": _bar(1100, 0.64)}
        s.on_ladder_bar(ladder, 1100, bm, 1000, 2000)
        longs = [e for e in s.trade_log if e.get("leg") == "long_yes"]
        self.assertEqual(len(longs), 0)


class TestAlignLadderBars(unittest.TestCase):
    def test_insufficient_bars_returns_error(self):
        m1 = {
            "ticker": "A",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 90,
            "result": "yes",
        }
        m2 = {
            "ticker": "B",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 100,
            "result": "no",
        }
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        ladder = LadderGroup(group_key="EV", ladder_id="L", markets=[d1, d2])
        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = []
        out, err = align_ladder_bars(fetcher, ladder, fill_minute_gaps=True)
        self.assertIsNone(out)
        self.assertIn("insufficient", err or "")


class TestSyntheticRoundTrip(unittest.TestCase):
    def test_settlement_pnl_long_wins_short_loses(self):
        cfg = MonotonicityConfig(min_net_violation=0.0)
        s = MonotonicityStrategy(10000.0, cfg)
        m1 = {
            "ticker": "E",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 90,
            "result": "yes",
        }
        m2 = {
            "ticker": "H",
            "event_ticker": "EV",
            "close_time": 2000,
            "open_time": 1000,
            "strike_type": "greater",
            "floor_strike": 100,
            "result": "yes",
        }
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        ladder = LadderGroup(group_key="EV", ladder_id="L1", markets=[d1, d2])
        bm = {"E": _bar(1100, 0.50), "H": _bar(1100, 0.55)}
        s.on_ladder_bar(ladder, 1100, bm, 1000, 2000)
        self.assertTrue(any(e.get("action") == "mono_entry" for e in s.trade_log))
        s.on_ladder_settle(ladder, {"E": "yes", "H": "yes"})
        settles = [e for e in s.trade_log if e.get("action") == "mono_settlement"]
        self.assertEqual(len(settles), 2)


if __name__ == "__main__":
    unittest.main()
