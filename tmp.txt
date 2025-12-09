#pragma once

#include <vector>
#include <cstddef>

// Simple structure to hold backtest outputs for a single-asset strategy.
struct BacktestResult {
    // Account equity over time (cumulative, including PnL and costs).
    std::vector<double> equity_curve;

    // Per-step PnL (change in equity due to price move and position).
    std::vector<double> pnl;

    // Position over time: -1 = short, 0 = flat, +1 = long.
    std::vector<int> position;

    // Final performance statistics.
    double total_return = 0.0;   // (final_equity / initial_equity - 1)
    double max_drawdown = 0.0;   // Max peak-to-trough drawdown (as fraction)
    double sharpe_ratio = 0.0;   // Simple Sharpe ratio (annualised)
};

// Core backtesting engine for a single-asset mean-reversion strategy.
// Prices and signals are assumed to be aligned time series.
// Signal convention: -1 = short, 0 = flat, +1 = long.
class BacktestEngine {
public:
    // Constructor
    //
    //  initial_capital:        starting equity
    //  transaction_cost_pct:   proportional transaction cost
    //                          e.g. 0.001 = 10 bps per traded notional
    //  risk_free_rate:         annual risk-free rate used in Sharpe calc (can be 0.0)
    BacktestEngine(double initial_capital,
                   double transaction_cost_pct,
                   double risk_free_rate = 0.0);

    // Run the backtest on a series of prices and pre-computed trading signals.
    //
    //  prices:          close or mid prices for each time step (size N)
    //  signals:         trading signal at each step (-1, 0, +1) (size N)
    //  dt_in_years:     time step in years (e.g. 1.0/252 for daily data)
    //
    // Returns:
    //  BacktestResult containing equity curve, PnL, positions and statistics.
    BacktestResult run_backtest(const std::vector<double>& prices,
                                const std::vector<int>& signals,
                                double dt_in_years);

private:
    double initial_capital_;
    double transaction_cost_pct_;
    double risk_free_rate_;

    // Compute max peak-to-trough drawdown for a given equity curve.
    double compute_max_drawdown(const std::vector<double>& equity) const;

    // Compute annualised Sharpe ratio based on per-step PnL (equity differences).
    double compute_sharpe(const std::vector<double>& pnl,
                          double dt_in_years) const;
};
