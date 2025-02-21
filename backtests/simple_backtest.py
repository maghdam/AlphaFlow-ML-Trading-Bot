# simple_backtest.py

import numpy as np
import pandas as pd

# backtests/simple_backtest.py


def simulate_trading(signals, df, cost=0.0002):
    """
    A simple backtest function that simulates trading based on signals (+1/-1/0).
    
    Parameters
    ----------
    signals : array-like of int
        Sequence of +1, -1, or 0 indicating long, short, or flat.
    df : pd.DataFrame
        Must contain at least a 'close' column with the same length as 'signals'.
    cost : float
        Transaction cost fraction per position change (e.g. 0.0002 = 0.02%).

    Returns
    -------
    daily_returns : np.array
        The sequence of returns from the strategy for each bar.
    total_return : float
        The total percentage return (e.g., 10.0 = +10%).
    """
    if len(signals) != len(df):
        raise ValueError("Length of signals must match length of df.")
    if 'close' not in df.columns:
        raise ValueError("df must contain a 'close' column.")
    
    # 1) Calculate price returns bar to bar
    df['price_return'] = df['close'].pct_change().fillna(0)
    
    # 2) Strategy returns = signals * price_return
    #    But we must subtract cost each time we change position.
    #    If signals[i] != signals[i-1], we pay cost.
    daily_returns = np.zeros(len(signals))
    
    prev_signal = 0
    for i in range(len(signals)):
        # Base return from price movement
        daily_returns[i] = signals[i] * df['price_return'].iloc[i]
        
        # Check if position changed from previous bar
        if i > 0 and signals[i] != prev_signal:
            # Subtract cost
            daily_returns[i] -= cost
        prev_signal = signals[i]
    
    # 3) Compute total return in percent
    cumulative_return = (1 + daily_returns).prod() - 1
    total_return = cumulative_return * 100.0
    
    return daily_returns, total_return


def calculate_sharpe_ratio(returns, risk_free=0.0):
    """
    Calculates a simple Sharpe ratio for a series of returns.
    
    Parameters
    ----------
    returns : list or np.array
        A sequence of returns per bar/day.
    risk_free : float, optional
        Risk-free rate per bar/day, default is 0.0 (no risk-free rate).
    
    Returns
    -------
    float
        The Sharpe ratio = (mean(returns - risk_free)) / std(returns).
        If std is zero, returns np.nan.
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free
    avg_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return np.nan
    
    sharpe = avg_excess / std_excess
    return sharpe
