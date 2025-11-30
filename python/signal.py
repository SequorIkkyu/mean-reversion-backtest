import numpy as np
import pandas as pd


def zscore_signal(price_series, window=20, z_entry=2.0):
    """
    Generate mean-reversion trading signals using z-score.

    Parameters
    ----------
    price_series : array-like
        Price series (close or mid prices).
    window : int
        Rolling window length for mean/std.
    z_entry : float
        Z-score threshold for long/short entries.

    Returns
    -------
    signals : np.ndarray (int)
        Trading signals: -1 = short, 0 = flat, +1 = long
    """
    s = pd.Series(price_series)

    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()

    z = (s - rolling_mean) / rolling_std

    # Long: z < -threshold
    # Short: z > +threshold
    long_signal = (z < -z_entry).astype(int)
    short_signal = (z > z_entry).astype(int) * -1

    signals = long_signal + short_signal
    signals = signals.fillna(0).astype(int)

    return signals.values


def load_price_csv(path, column="close"):
    """
    Helper to load price series from a CSV file.
    Column must exist (e.g. 'close').

    Returns numpy array of prices.
    """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV.")
    return df[column].values
