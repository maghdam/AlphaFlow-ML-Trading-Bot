# vectorbt_backtest.py

import numpy as np
import pandas as pd
import vectorbt as vbt

def run_vectorbt_backtest(
    model,
    X,
    selected_features,
    data,
    scaler,
    init_cash=10000,
    freq='4H',
    threshold=0.0
):
    """
    Runs a vectorbt backtest for a given pre-trained model.

    Parameters
    ----------
    model : fitted scikit-learn model
        Already fitted model (e.g. RandomForestRegressor).
    X : pd.DataFrame
        The full feature DataFrame (or the portion you want to backtest).
    selected_features : list
        List of feature names used by the model.
    data : pd.DataFrame
        Original DataFrame containing at least a 'close' column.
    scaler : fitted scaler
        The StandardScaler (or other) used to scale features.
    init_cash : float
        Starting capital for the backtest.
    freq : str
        Frequency for vectorbt (e.g. '4H', '1D').
    threshold : float
        Minimum absolute predicted return to place a trade (optional).

    Returns
    -------
    pf : vbt.Portfolio
        The resulting vectorbt portfolio object.
    """
    # 1) Subset X to the selected features
    X_sel = X[selected_features]

    # 2) Scale
    X_scaled = scaler.transform(X_sel)

    # 3) Generate predictions
    preds = model.predict(X_scaled)

    # 4) Convert predictions to signals
    #    Optionally use threshold to reduce whipsaws
    if threshold > 0.0:
        signals = np.where(preds > threshold, 1, np.where(preds < -threshold, -1, 0))
    else:
        signals = np.sign(preds)

    # 5) Align signals with close prices
    close_prices = data.loc[X_sel.index, "close"]
    # If signals is shorter or the same length
    if len(signals) < len(close_prices):
        # Pad signals with 0 if needed
        signals = np.append(signals, [0]*(len(close_prices)-len(signals)))

    signals_s = pd.Series(signals, index=close_prices.index)
    # Align if any missing indexes
    close_prices, signals_s = close_prices.align(signals_s, join="inner", axis=0)

    # 6) Run vectorbt Portfolio
    pf = vbt.Portfolio.from_signals(
        close_prices,
        entries=signals_s > 0,
        exits=signals_s < 0,
        init_cash=init_cash,
        freq=freq
    )

    return pf
