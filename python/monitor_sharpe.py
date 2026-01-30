import os
from datetime import datetime
import numpy as np
import pandas as pd


# =========================
# Paths (robust)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "spy_2025_Jul_Dec.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "derived")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Data loading
# =========================
def load_close(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    # Case 1: explicit Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").set_index("Date")
    else:
        # Case 2: first column is date index (yfinance default)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.sort_values(first_col).set_index(first_col)

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found. Columns: {list(df.columns)}")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_index()

    return df["Close"].astype(float)


# =========================
# Strategy components
# =========================
def zscore_position(
    price: pd.Series,
    window: int,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
) -> pd.Series:
    mu = price.rolling(window).mean()
    sig = price.rolling(window).std(ddof=0)
    z = (price - mu) / sig

    pos = pd.Series(0.0, index=price.index)
    current = 0.0

    for i in range(len(price)):
        zi = z.iat[i]
        if np.isnan(zi):
            current = 0.0
        elif abs(zi) < z_exit:
            current = 0.0
        elif current == 0.0:
            if zi > z_entry:
                current = -1.0
            elif zi < -z_entry:
                current = 1.0
        pos.iat[i] = current

    return pos


def backtest_returns(
    price: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    pos = position.reindex(price.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)

    gross = pos_lag * r
    turnover = (pos - pos.shift(1)).abs().fillna(0.0)
    cost = (cost_bps / 10000.0) * turnover

    return gross - cost


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    annualization: int = 252,
) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    sig = returns.rolling(window, min_periods=window).std(ddof=0)
    sr = np.sqrt(annualization) * mu / sig
    return sr.replace([np.inf, -np.inf], np.nan)


# =========================
# Decision layer
# =========================
def make_decision(
    sharpe_snapshot: pd.DataFrame,
    trade_window: int,
    warn_level: float = 0.0,
    stop_level: float = -0.5,
    warn_frac: float = 0.5,
    stop_frac: float = 0.75,
) -> dict:
    scores = sharpe_snapshot.median(axis=1, skipna=True)

    frac_warn = (scores < warn_level).mean()
    frac_stop = (scores < stop_level).mean()

    if frac_stop >= stop_frac:
        mode = "STOP"
        multiplier = 0.0
    elif frac_warn >= warn_frac:
        mode = "REDUCE"
        multiplier = 0.5
    else:
        mode = "NORMAL"
        multiplier = 1.0

    return {
        "risk_mode": mode,
        "position_multiplier": multiplier,
        "trade_window": trade_window,
        "frac_below_warn": float(frac_warn),
        "frac_below_stop": float(frac_stop),
    }


# =========================
# Main
# =========================
def main():
    print("RAW_CSV =", RAW_CSV)
    print("OUT_DIR =", OUT_DIR)

    price = load_close(RAW_CSV)

    # --- core research idea ---
    # trade / monitor multiple alternative windows
    signal_windows = [10, 20, 40, 80]
    sharpe_windows = [10, 20, 60]

    z_entry, z_exit = 2.0, 0.5
    cost_bps = 1.0

    panel_rows = []
    snapshot_rows = []

    for w in signal_windows:
        pos = zscore_position(price, window=w, z_entry=z_entry, z_exit=z_exit)
        ret = backtest_returns(price, pos, cost_bps)

        # rolling sharpe for monitoring
        for sw in sharpe_windows:
            sr = rolling_sharpe(ret, sw)
            tmp = pd.DataFrame(
                {
                    "Date": sr.index,
                    "signal_window": w,
                    "sharpe_window": sw,
                    "rolling_sharpe": sr.values,
                }
            )
            panel_rows.append(tmp)

        # latest snapshot
        latest = {sw: rolling_sharpe(ret, sw).iloc[-1] for sw in sharpe_windows}
        latest["signal_window"] = w
        snapshot_rows.append(latest)

    panel = pd.concat(panel_rows, ignore_index=True).dropna()
    snapshot = pd.DataFrame(snapshot_rows).set_index("signal_window").sort_index()

    # --- time-stamped outputs (NO file lock issues) ---
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    panel_path = os.path.join(OUT_DIR, f"spy_rolling_sharpe_panel_{run_tag}.csv")
    snapshot_path = os.path.join(OUT_DIR, f"spy_latest_sharpe_snapshot_{run_tag}.csv")

    panel.to_csv(panel_path, index=False)
    snapshot.to_csv(snapshot_path)

    decision = make_decision(snapshot, trade_window=40)
    decision_path = os.path.join(OUT_DIR, f"spy_decision_today_{run_tag}.csv")
    pd.DataFrame([decision]).to_csv(decision_path, index=False)

    print("saved:", panel_path)
    print("saved:", snapshot_path)
    print("saved:", decision_path)
    print("decision:", decision)
    print("snapshot (rounded):")
    print(snapshot.round(2))


if __name__ == "__main__":
    main()
