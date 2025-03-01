# feature_engineering.py

import numpy as np
import pandas as pd
import math
import ta
from statsmodels.tsa.stattools import adfuller
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler





# --------------------------------------------------------------------
# 1) TA-LIB FEATURES (add_all_ta_features)
# --------------------------------------------------------------------
def add_all_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a wide range of technical analysis indicators to the DataFrame
    using the 'ta' library. Modifies the DataFrame in place.
    """
    df = ta.add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True
    )
    return df



def create_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example custom feature. For instance, a rolling mean of the close price.
    """
    df["rolling_mean_10"] = df["close"].rolling(window=10).mean()
    return df

# --------------------------------------------------------------------
# 2) MISCELLANEOUS FEATURES
# --------------------------------------------------------------------
def spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the spread between 'high' and 'low' columns.
    """
    df_copy = df.copy()
    df_copy["spread"] = df_copy["high"] - df_copy["low"]
    return df_copy

def auto_corr_multi(df: pd.DataFrame, col: str, n: int = 50, lags: list = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Computes rolling autocorrelation for multiple lags.
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f"autocorr_{lag}"] = (
            df_copy[col]
            .rolling(window=n, min_periods=n)
            .apply(lambda x: x.autocorr(lag=lag), raw=False)
        )
    return df_copy


def candle_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds candle-specific features:
      - candle_way
      - fill
      - amplitude
    """
    df_copy = df.copy()
    df_copy["candle_way"] = 0
    df_copy.loc[df_copy["close"] > df_copy["open"], "candle_way"] = 1
    
    df_copy["fill"] = (
        np.abs(df_copy["close"] - df_copy["open"]) 
        / (df_copy["high"] - df_copy["low"] + 1e-5)
    )
    df_copy["amplitude"] = (
        np.abs(df_copy["close"] - df_copy["open"]) 
        / (df_copy["open"] + 1e-5)
    )
    return df_copy

def log_transform(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """
    Log-transform a column + compute % change over 'n' bars.
    """
    df_copy = df.copy()
    df_copy[f"log_{col}"] = np.log(df_copy[col])
    df_copy[f"ret_log_{n}"] = df_copy[f"log_{col}"].pct_change(periods=n)
    return df_copy

def mathematical_derivatives(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Adds 'velocity' and 'acceleration' for a given column.
    """
    df_copy = df.copy()
    df_copy["velocity"] = df_copy[col].diff()
    df_copy["acceleration"] = df_copy["velocity"].diff()
    return df_copy

# --------------------------------------------------------------------
# 3) VOLATILITY ESTIMATORS
# --------------------------------------------------------------------
def parkinson_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1:
        return np.nan
    sum_sq = np.sum(np.log(window['high'] / window['low']) ** 2)
    return math.sqrt(sum_sq / (4 * math.log(2) * n))

def moving_parkinson_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    df_copy = df.copy()
    rolling_vol = pd.Series(dtype="float64", index=df_copy.index)
    for i in range(window_size, len(df_copy)):
        w = df_copy.iloc[i - window_size : i]
        rolling_vol.iloc[i] = parkinson_estimator(w)
    df_copy["rolling_volatility_parkinson"] = rolling_vol
    return df_copy

def yang_zhang_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1:
        return np.nan
    term1 = np.log(window['high'] / window['low']) ** 2
    term2 = np.log(window['close'] / window['open']) ** 2
    return math.sqrt(np.mean(term1 + term2))

def moving_yang_zhang_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    df_copy = df.copy()
    rolling_vol = pd.Series(dtype="float64", index=df_copy.index)
    for i in range(window_size, len(df_copy)):
        w = df_copy.iloc[i - window_size : i]
        rolling_vol.iloc[i] = yang_zhang_estimator(w)
    df_copy["rolling_volatility_yang_zhang"] = rolling_vol
    return df_copy

# --------------------------------------------------------------------
# 4) MARKET REGIME / DC EVENTS
# --------------------------------------------------------------------
def dc_event(P: float, Pext: float, threshold: float) -> int:
    dc = 0
    var = (P - Pext) / Pext
    if var >= threshold:
        dc = 1
    elif var <= -threshold:
        dc = -1
    return dc

def calculate_dc(df: pd.DataFrame, threshold: float = 0.01) -> tuple:
    df_copy = df.copy()
    prices = df_copy['close'].values
    dc_events_up, dc_events_down = [], []
    Pext = prices[0]
    direction = 0
    for i in range(1, len(prices)):
        P = prices[i]
        dc_flag = dc_event(P, Pext, threshold)
        if dc_flag == 1:
            dc_events_up.append(i)
            direction = 1
            Pext = P
        elif dc_flag == -1:
            dc_events_down.append(i)
            direction = -1
            Pext = P
        else:
            if direction == 1 and P > Pext:
                Pext = P
            elif direction == -1 and P < Pext:
                Pext = P
    return dc_events_up, dc_events_down

def calculate_trend(dc_events_up: list, dc_events_down: list, df: pd.DataFrame):
    trend_events_down = []
    trend_events_up = []
    trend_events_down.extend(sorted(dc_events_down))
    trend_events_up.extend(sorted(dc_events_up))
    return trend_events_down, trend_events_up

def market_regime_dc(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    df_copy = df.copy()
    dc_up, dc_down = calculate_dc(df_copy, threshold=threshold)
    t_down, t_up = calculate_trend(dc_up, dc_down, df_copy)
    df_copy['market_regime'] = np.nan
    df_copy.loc[t_up, 'market_regime'] = 1
    df_copy.loc[t_down, 'market_regime'] = 0
    df_copy['market_regime'] = df_copy['market_regime'].ffill().bfill()
    return df_copy

def kama_market_regime(df: pd.DataFrame, col: str = 'close', n1: int = 10, n2: int = 30) -> pd.DataFrame:
    df_copy = df.copy()
    short_kama = df_copy[col].ewm(span=n1, adjust=False).mean()
    long_kama  = df_copy[col].ewm(span=n2, adjust=False).mean()
    df_copy['kama_diff'] = short_kama - long_kama
    df_copy['kama_trend'] = (df_copy['kama_diff'] >= 0).astype(int)
    return df_copy

# --------------------------------------------------------------------
# 5) GAP & DISPLACEMENT
# --------------------------------------------------------------------
def gap_detection(df: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['Bullish_gap_inf'] = np.nan
    df_copy['Bullish_gap_sup'] = np.nan
    df_copy['Bullish_gap_size'] = np.nan
    df_copy['Bearish_gap_inf'] = np.nan
    df_copy['Bearish_gap_sup'] = np.nan
    df_copy['Bearish_gap_size'] = np.nan
    for i in range(lookback, len(df_copy)):
        prev_high = df_copy['high'].iloc[i - lookback]
        prev_low = df_copy['low'].iloc[i - lookback]
        curr_high = df_copy['high'].iloc[i]
        curr_low = df_copy['low'].iloc[i]
        if curr_low > prev_high:
            df_copy.at[df_copy.index[i], 'Bullish_gap_inf'] = prev_high
            df_copy.at[df_copy.index[i], 'Bullish_gap_sup'] = curr_low
            df_copy.at[df_copy.index[i], 'Bullish_gap_size'] = curr_low - prev_high
        if curr_high < prev_low:
            df_copy.at[df_copy.index[i], 'Bearish_gap_inf'] = curr_high
            df_copy.at[df_copy.index[i], 'Bearish_gap_sup'] = prev_low
            df_copy.at[df_copy.index[i], 'Bearish_gap_size'] = prev_low - curr_high
    return df_copy

def displacement_detection(
    df: pd.DataFrame, 
    type_range: str = 'standard', 
    strenght: float = 3.0, 
    period: int = 20
) -> pd.DataFrame:
    df_copy = df.copy()
    if type_range == 'standard':
        df_copy['candle_range'] = np.abs(df_copy['close'] - df_copy['open'])
    elif type_range == 'extrem':
        df_copy['candle_range'] = np.abs(df_copy['high'] - df_copy['low'])
    else:
        raise ValueError("Invalid 'type_range'. Use 'standard' or 'extrem'.")

    df_copy['Variation'] = np.abs(df_copy['close'] / df_copy['open'] - 1)
    df_copy['STD'] = df_copy['candle_range'].rolling(period).std()
    df_copy['displacement'] = 0
    mask = df_copy['candle_range'] > strenght * df_copy['STD']
    df_copy.loc[mask, 'displacement'] = 1
    df_copy['red_displacement'] = (
        df_copy['displacement'] & df_copy['displacement'].shift(1).fillna(0)
    ).astype(int)
    return df_copy

# --------------------------------------------------------------------
# 6) ROLLING ADF (Stationarity)
# --------------------------------------------------------------------
def rolling_adf_with_flag(df: pd.DataFrame, col: str = 'close', window_size: int = 50, p_value_threshold=0.05) -> pd.DataFrame:
    """
    Computes rolling ADF test and adds a stationarity flag (1=stationary, 0=non-stationary).
    """
    df_copy = df.copy()
    adf_stat = pd.Series(dtype="float64", index=df_copy.index)
    adf_pval = pd.Series(dtype="float64", index=df_copy.index)
    stationarity_flag = pd.Series(dtype="int", index=df_copy.index)

    for i in range(window_size, len(df_copy)):
        slice_data = df_copy[col].iloc[i - window_size : i].values
        try:
            result = adfuller(slice_data, autolag='AIC')
            adf_stat.iloc[i] = result[0]
            adf_pval.iloc[i] = result[1]
            stationarity_flag.iloc[i] = 1 if result[1] < p_value_threshold else 0
        except:
            adf_stat.iloc[i] = np.nan
            adf_pval.iloc[i] = np.nan
            stationarity_flag.iloc[i] = np.nan

    df_copy['rolling_adf_stat'] = adf_stat
    df_copy['rolling_adf_pval'] = adf_pval
    df_copy['stationary_flag'] = stationarity_flag  # 1 = stationary, 0 = non-stationary

    return df_copy


# --------------------------------------------------------------------
# 7) DOUBLE-BARRIER LABEL
# --------------------------------------------------------------------
def set_double_barrier_label(
    df: pd.DataFrame, 
    up: float = 0.005, 
    down: float = 0.005, 
    horizon: int = 50
) -> pd.DataFrame:
    df_copy = df.copy()
    closes = df_copy["close"].values
    labels = np.full(len(closes), np.nan)

    for i in range(len(closes)):
        current_price = closes[i]
        upper_barrier = current_price * (1 + up)
        lower_barrier = current_price * (1 - down)
        end = min(i + horizon, len(closes))
        for forward_i in range(i + 1, end):
            if closes[forward_i] >= upper_barrier:
                labels[i] = 1
                break
            elif closes[forward_i] <= lower_barrier:
                labels[i] = 0
                break
    df_copy["barrier_label"] = labels
    df_copy.dropna(subset=["barrier_label"], inplace=True)
    return df_copy

# --------------------------------------------------------------------
# 8) FUTURE MARKET REGIME (Directional-Change Example)
# --------------------------------------------------------------------
def future_DC_market_regime(df: pd.DataFrame, threshold: float = 0.03, horizon: int = 10) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['future_return'] = df_copy['close'].shift(-horizon) / df_copy['close'] - 1.0
    df_copy['future_market_regime'] = np.nan
    df_copy.loc[df_copy['future_return'] >= threshold, 'future_market_regime'] = 1
    df_copy.loc[df_copy['future_return'] <= -threshold, 'future_market_regime'] = 0
    df_copy.dropna(subset=['future_market_regime'], inplace=True)
    return df_copy



# --------------------------------------------------------------------
# 9) Introduce Fourier & Wavelet Features for Cyclical Pattern Recognition
# --------------------------------------------------------------------

def add_fourier_features(df: pd.DataFrame, col: str = "close", n_components: int = 5) -> pd.DataFrame:
    """
    Extracts the top 'n_components' Fourier coefficients from price data.
    """
    fft_vals = np.abs(fft(df[col].values))
    for i in range(1, n_components + 1):
        df[f'fft_comp_{i}'] = fft_vals[i]
    return df


# --------------------------------------------------------------------
# 10) Optimize ADF Test for Model Selection
# --------------------------------------------------------------------

def apply_differencing_if_needed(df: pd.DataFrame, col: str = "close", threshold: float = 0.05) -> pd.DataFrame:
    """
    If ADF p-value > threshold (non-stationary), apply first differencing.
    """
    if df['rolling_adf_pval'].iloc[-1] > threshold:  # Check last rolling p-value
        df[f"{col}_diff"] = df[col] - df[col].shift(1)  # First differencing
    return df.dropna()



# --------------------------------------------------------------------
# 11) Normalize Feature Distributions (Scaling)
# --------------------------------------------------------------------
def scale_features(df: pd.DataFrame, cols_to_scale: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df



# 12) SINGLE PIPELINE EXAMPLE
# --------------------------------------------------------------------
def create_features(df: pd.DataFrame, col: str = "close", window_size: int = 30) -> pd.DataFrame:
    """
    Optimized pipeline integrating TA, autocorrelation, stationarity, Fourier transform, and normalization.
    """
    df = add_all_ta_features(df)          # Adds TA indicators
    df = spread(df)                        # Adds 'spread'
    df = auto_corr_multi(df, col='close')  # Multi-lag autocorrelation
    df = rolling_adf_with_flag(df)         # ADF with stationarity flag
    
    df = log_transform(df, col, 5)         # Log transform
    df = moving_yang_zhang_estimator(df, window_size)
    df = moving_parkinson_estimator(df, window_size)
    
    df = add_fourier_features(df, col="close")  # Fourier Transform for cyclic detection
    df = apply_differencing_if_needed(df, col="close")  # Ensure stationarity

    # Normalize all numeric features
    df = scale_features(df, df.select_dtypes(include=[np.number]).columns.tolist())

    return df
