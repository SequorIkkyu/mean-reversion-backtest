import numpy as np
import pandas as pd


def compute_decision(
    latest_sharpe_snapshot: pd.DataFrame,
    trade_window: int,
    warn_sharpe: float = 0.0,
    stop_sharpe: float = -0.5,
    frac_warn: float = 0.5,
    frac_stop: float = 0.75,
) -> dict:
    """
    latest_sharpe_snapshot:
        index = signal_window (int)
        columns = sharpe_window (int)
        values = latest rolling sharpe (float)

    Logic (simple + defensible):
      - Compute each signal_window's score = median Sharpe across sharpe_windows
      - Cross-window degradation:
          * warn if >= frac_warn of windows have score < warn_sharpe
          * stop if >= frac_stop of windows have score < stop_sharpe
      - Trade window chosen separately (single production spec)
      - Output risk_mode + position_multiplier + brief diagnostics
    """
    snap = latest_sharpe_snapshot.copy()

    # score per signal window: robust to a noisy sharpe_window
    scores = snap.median(axis=1, skipna=True)

    # degradation counts across signal windows
    bad_warn = (scores < warn_sharpe).mean()
    bad_stop = (scores < stop_sharpe).mean()

    # trade window score
    trade_score = (
        float(scores.loc[trade_window]) if trade_window in scores.index else np.nan
    )

    # decide risk mode
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


def apply_position_multiplier(position: pd.Series, multiplier: float) -> pd.Series:
    return position.astype(float) * float(multiplier)
