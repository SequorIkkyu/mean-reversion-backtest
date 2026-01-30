import os
import numpy as np
import pandas as pd

# --- Paths (robust to running from any working directory) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "spy_2025_Jul_Dec.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "derived")
os.makedirs(OUT_DIR, exist_ok=True)


def load_close(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    # Determine datetime index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").set_index("Date")
    else:
        # yfinance to_csv often stores date as the first column (index)
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.sort_values(first_col).set_index(first_col)

    # Ensure numeric close
    if "Close" not in df.columns:
        raise ValueError(f"'Close' not found in CSV columns: {list(df.columns)}")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_index()

    return df["Close"].astype(float)


def zscore_position(
    price: pd.Series, window: int, z_entry: float = 2.0, z_exit: float = 0.5
) -> pd.Series:
    s = price.astype(float)
    mu = s.rolling(window).mean()
    sig = s.rolling(window).std(ddof=0)
    z = (s - mu) / sig

    pos = pd.Series(0.0, index=s.index)
    current = 0.0

    for i in range(len(s)):
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
    price: pd.Series, position: pd.Series, cost_bps: float = 1.0
) -> pd.Series:
    r = price.pct_change().fillna(0.0)

    pos = position.reindex(price.index).fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)  # no lookahead

    gross = pos_lag * r

    turnover = (pos - pos.shift(1)).abs().fillna(0.0)
    cost = (cost_bps / 10000.0) * turnover

    return gross - cost


def rolling_sharpe(
    returns: pd.Series, window: int, annualization: int = 252
) -> pd.Series:
    mu = returns.rolling(window, min_periods=window).mean()
    sig = returns.rolling(window, min_periods=window).std(ddof=0)
    sr = np.sqrt(annualization) * (mu / sig)
    return sr.replace([np.inf, -np.inf], np.nan)


def compute_decision(
    latest_sharpe_snapshot: pd.DataFrame,
    trade_window: int,
    warn_sharpe: float = 0.0,
    stop_sharpe: float = -0.5,
    frac_warn: float = 0.5,
    frac_stop: float = 0.75,
) -> dict:
    snap = latest_sharpe_snapshot.copy()
    scores = snap.median(axis=1, skipna=True)

    bad_warn = (scores < warn_sharpe).mean()
    bad_stop = (scores < stop_sharpe).mean()

    trade_score = (
        float(scores.loc[trade_window]) if trade_window in scores.index else np.nan
    )

    if bad_stop >= frac_stop:
        risk_mode = "STOP"
        position_multiplier = 0.0
    elif bad_warn >= frac_warn:
        risk_mode = "REDUCE"
        position_multiplier = 0.5
    else:
        risk_mode = "NORMAL"
        position_multiplier = 1.0

    return {
        "risk_mode": risk_mode,
        "position_multiplier": position_multiplier,
        "trade_window": trade_window,
        "trade_score": trade_score,
        "bad_warn_frac": float(bad_warn),
        "bad_stop_frac": float(bad_stop),
        "scores": scores.sort_index(),
    }


def main():
    print("RAW_CSV =", RAW_CSV)
    print("OUT_DIR =", OUT_DIR)

    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    price = load_close(RAW_CSV)

    signal_windows = [
        10,
        20,
        40,
        80,
    ]  # different trading specs (counterfactual strategies)
    sharpe_windows = [10, 20, 60]  # monitoring horizons
    z_entry, z_exit = 2.0, 0.5
    cost_bps = 1.0

    rows = []
    snap_rows = []

    for sig_w in signal_windows:
        pos = zscore_position(price, window=sig_w, z_entry=z_entry, z_exit=z_exit)
        ret = backtest_returns(price, pos, cost_bps=cost_bps)

        # rolling sharpe for each monitoring horizon
        for sh_w in sharpe_windows:
            sr = rolling_sharpe(ret, window=sh_w)
            tmp = pd.DataFrame(
                {
                    "Date": sr.index,
                    "signal_window": sig_w,
                    "sharpe_window": sh_w,
                    "rolling_sharpe": sr.values,
                }
            )
            rows.append(tmp)

        # latest snapshot for this signal window (for decision layer)
        latest = {sh_w: rolling_sharpe(ret, sh_w).iloc[-1] for sh_w in sharpe_windows}
        latest["signal_window"] = sig_w
        snap_rows.append(latest)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["rolling_sharpe"])

    panel_path = os.path.join(OUT_DIR, "spy_rolling_sharpe_panel.csv")
    panel.to_csv(panel_path, index=False)

    snap = pd.DataFrame(snap_rows).set_index("signal_window").sort_index()
    snap_path = os.path.join(OUT_DIR, "spy_latest_sharpe_snapshot.csv")
    snap.to_csv(snap_path)

    # decision layer: trade one window, monitor many
    trade_window = 40
    decision = compute_decision(
        latest_sharpe_snapshot=snap,
        trade_window=trade_window,
        warn_sharpe=0.0,
        stop_sharpe=-0.5,
        frac_warn=0.5,
        frac_stop=0.75,
    )

    decision_path = os.path.join(OUT_DIR, "spy_decision_today.csv")
    pd.DataFrame(
        [
            {
                "risk_mode": decision["risk_mode"],
                "position_multiplier": decision["position_multiplier"],
                "trade_window": decision["trade_window"],
                "trade_score": decision["trade_score"],
                "bad_warn_frac": decision["bad_warn_frac"],
                "bad_stop_frac": decision["bad_stop_frac"],
            }
        ]
    ).to_csv(decision_path, index=False)

    print("saved:", panel_path)
    print("saved:", snap_path)
    print("saved:", decision_path)
    print(
        "decision:",
        decision["risk_mode"],
        "| multiplier:",
        decision["position_multiplier"],
    )
    print("snapshot (rounded):")
    print(snap.round(2))


if __name__ == "__main__":
    main()
