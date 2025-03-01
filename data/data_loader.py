# data_loader.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

def get_data_mt5(symbol: str, n_bars: int, timeframe, start_pos=None) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader 5.
    
    - `symbol`: Trading instrument (e.g., "BTCUSD").
    - `n_bars`: Number of bars to retrieve.
    - `timeframe`: MT5 timeframe (e.g., mt5.TIMEFRAME_H1).
    - `start_pos`: Offset from the most recent bar (default `None` for live trading).
    
    If `start_pos` is `None`, fetches the latest `n_bars` (useful for live trading).
    If `start_pos` is given, fetches `n_bars` from that historical position (useful for backtesting).
    """
    if start_pos is None:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)  # Latest n_bars for live trading
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, n_bars)  # Historical data for backtesting

    if rates is None:
        raise ValueError(f"Could not retrieve data for {symbol}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df
