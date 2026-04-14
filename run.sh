#!/bin/bash
# Wrapper: run backtest from btc15m_backtest/
cd "$(dirname "$0")"
./btc15m_backtest/run.sh "$@"
