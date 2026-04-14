"""Tests for ladder grouping and threshold parsing."""
import unittest

from ladder_grouping import (
    LadderRejection,
    group_markets_into_ladders,
    parse_market_descriptor,
    validate_ladder_homogeneous,
)
from ladder_grouping import LadderGroup, MarketDescriptor, ThresholdDirection


def _m(
    ticker,
    event,
    close_ts,
    open_ts,
    floor_strike,
    result="yes",
    strike_type="greater",
):
    return {
        "ticker": ticker,
        "event_ticker": event,
        "close_time": close_ts,
        "open_time": open_ts,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "result": result,
        "title": f"BTC above {floor_strike}",
        "series_ticker": "KXBTC15M",
    }


class TestParseMarketDescriptor(unittest.TestCase):
    def test_parse_greater_from_floor_strike(self):
        m = _m("KXBTC15M-E1-T90", "EVT1", 1000, 100, 90000)
        d = parse_market_descriptor(m, "KXBTC15M", strict_grouping=True)
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.threshold, 90000.0)
        self.assertEqual(d.direction, ThresholdDirection.ABOVE)


class TestGroupMarkets(unittest.TestCase):
    def test_valid_three_rung_ladder(self):
        close_ts = 1_700_000_000
        open_ts = close_ts - 900
        markets = [
            _m("A-T90", "EVT1", close_ts, open_ts, 90000),
            _m("A-T100", "EVT1", close_ts, open_ts, 100000),
            _m("A-T110", "EVT1", close_ts, open_ts, 110000),
        ]
        ladders, rej = group_markets_into_ladders(markets, "KXBTC15M", strict_grouping=True)
        self.assertEqual(len(rej), 0)
        self.assertEqual(len(ladders), 1)
        self.assertEqual(len(ladders[0].markets), 3)
        self.assertEqual(ladders[0].markets[0].threshold, 90000.0)
        self.assertEqual(ladders[0].markets[2].threshold, 110000.0)

    def test_mixed_close_time_not_grouped(self):
        close_ts = 1_700_000_000
        markets = [
            _m("A-T90", "EVT1", close_ts, close_ts - 900, 90000),
            _m("A-T100", "EVT1", close_ts + 3600, close_ts - 900, 100000),
        ]
        ladders, rej = group_markets_into_ladders(markets, "KXBTC15M", strict_grouping=True)
        self.assertEqual(len(ladders), 0)
        self.assertTrue(len(rej) >= 1)

    def test_mixed_rule_strike_type(self):
        close_ts = 1_700_000_000
        open_ts = close_ts - 900
        m1 = _m("A-T90", "EVT1", close_ts, open_ts, 90000, strike_type="greater")
        m2 = {
            **m1,
            "ticker": "A-T100",
            "floor_strike": None,
            "cap_strike": 100000,
            "strike_type": "less",
        }
        ladders, rej = group_markets_into_ladders([m1, m2], "KXBTC15M", strict_grouping=True)
        self.assertEqual(len(ladders), 0)

    def test_non_monotone_duplicate_strike_rejected(self):
        close_ts = 1_700_000_000
        open_ts = close_ts - 900
        markets = [
            _m("A-T90a", "EVT1", close_ts, open_ts, 90000),
            _m("A-T90b", "EVT1", close_ts, open_ts, 90000),
        ]
        ladders, rej = group_markets_into_ladders(markets, "KXBTC15M", strict_grouping=True)
        self.assertEqual(len(ladders), 0)
        reasons = [r.reason for r in rej]
        self.assertIn(LadderRejection.non_monotone_thresholds, reasons)

    def test_validate_mixed_open(self):
        m1 = _m("a", "e", 1000, 100, 90000)
        m2 = _m("b", "e", 1000, 200, 100000)
        d1 = parse_market_descriptor(m1, "KXBTC15M", True)
        d2 = parse_market_descriptor(m2, "KXBTC15M", True)
        assert d1 and d2
        lg = LadderGroup(
            group_key="e",
            ladder_id="x",
            markets=[d1, d2],
        )
        r = validate_ladder_homogeneous(lg)
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.reason, LadderRejection.mixed_open_time)


if __name__ == "__main__":
    unittest.main()
