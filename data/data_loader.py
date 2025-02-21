# data_loader.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

def get_data_mt5(symbol: str, n_bars: int, timeframe) -> pd.DataFrame:
    """
    Fetch historical data from MetaTrader 5 for a given symbol, timeframe, and number of bars.
    Returns a DataFrame with [open, high, low, close, tick_volume, ...] and a DateTime index.
    """
    # Initialize or assume MT5 is already initialized
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        raise ValueError(f"Could not retrieve data for {symbol}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# You can add other data-related functions here, e.g. CSV loading, cleaning, etc.
