import os
import numpy as np
import pandas as pd

RAW_CSV = "data/raw/spy_2025_Jul_Dec.csv"
OUT_DIR = "data/derived"
os.makedirs(OUT_DIR, exist_ok=True)


def load_close(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df["Close"].astype(float)


def zscore_position(price: pd.Series, window: int, z_entry: float = 2.0, z_exit: float = 0.5) -> pd.Series:
    s = price.astype(float)
    mu = s.rolling(window).mean()
    sig = s.rolling(window).std(ddof=0)
    z = (s - mu) / sig

    pos = pd.Series(0, index=s.index, dtype=float)
    current = 0
    for i in range(len(s)):
        zi = z.iat[i]
        if np.isnan(zi):
            current = 0
        elif abs(zi) < z_exit:
            current = 0
        elif current == 0:
            if zi > z_entry:
                current = -1
            elif zi < -z_entry:
                current = 1
        pos.iat[i] = current
    return pos


def backtest_returns(price: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    pos = position.reindex(price.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)

    gross = pos_lag * r
    turnover = (pos - pos.shift(1)).abs().fillna(0.0)
    cost = (cost_bps / 10000.0) * turnover

    return gross - cost


def rolling_sharpe(returns: pd.Series, window: int, annualization: int = 252) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    sig = returns.rolling(window, min_periods=window).std(ddof=0)
    sr = np.sqrt(annualization) * (mu / sig)
    return sr.replace([np.inf, -np.inf], np.nan)


def main():
    price = load_close(RAW_CSV)

    signal_windows = [10, 20, 40, 80]
    sharpe_windows = [10, 20, 60]
    z_entry, z_exit = 2.0, 0.5
    cost_bps = 1.0

    rows = []
    snap_rows = []

    for sw in signal_windows:
        pos = zscore_position(price, window=sw, z_entry=z_entry, z_exit=z_exit)
        ret = backtest_returns(price, pos, cost_bps=cost_bps)

        for rw in sharpe_windows:
            sr = rolling_sharpe(ret, window=rw)
            tmp = pd.DataFrame(
                {
                    "Date": sr.index,
                    "signal_window": sw,
                    "sharpe_window": rw,
                    "rolling_sharpe": sr.values,
                }
            )
            rows.append(tmp)

        # latest snapshot for this signal window
        latest = {rw: rolling_sharpe(ret, rw).iloc[-1] for rw in sharpe_windows}
        latest["signal_window"] = sw
        snap_rows.append(latest)

    panel = pd.concat(rows, ignore_index=True).dropna(subset=["rolling_sharpe"])
    panel_path = f"{OUT_DIR}/spy_rolling_sharpe_panel.csv"
    panel.to_csv(panel_path, index=False)

    snap = pd.DataFrame(snap_rows).set_index("signal_window")
    snap_path = f"{OUT_DIR}/spy_latest_sharpe_snapshot.csv"
    snap.to_csv(snap_path)

    print("saved:", panel_path)
    print("saved:", snap_path)
    print("latest snapshot:")
    print(snap.round(2))


if __name__ == "__main__":
    main()
