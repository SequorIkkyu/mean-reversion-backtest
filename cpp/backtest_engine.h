#pragma once

#include <vector>

// Simple structure to hold backtest outputs
struct BacktestResult {
    std::vector<double> equity_curve;   // account equity over time
    std::vector<double> pnl;            // per-step PnL
    std::vector<int>    position;       // position over time: -1, 0, +1
    double total_return = 0.0;          // (final_equity / initial_equity - 1)
    double max_drawdown = 0.0;          // max peak-to-trough drawdown
    double sharpe_ratio = 0.0;          // simple Sharpe (excess return / stdev of returns)
};

// Core backtesting engine for a single-asset mean-reversion strategy.
// Prices and signals are assumed to be aligned time series.
// Signal convention: -1 = short, 0 = flat, +1 = long.
class BacktestEngine {
public:
    // Constructor
    //  initial_capital: starting equity
    //  transaction_cost_pct: proportional transaction cost (e.g. 0.001 = 10 bps per traded notional)
    //  risk_free_rate: annual risk-free rate used in Sharpe calculation (can be 0.0)
    BacktestEngine(double initial_capital,
                   double transaction_cost_pct,
                   double risk_free_rate = 0.0);

    // Run the backtest on a series of prices and pre-computed trading signals.
    //  prices: close prices (or mid prices) for each time step
    //  signals: trading signal at each time step (-1, 0, +1)
    //  dt_in_years: time step in years (e.g. 1/252 for daily data)
    BacktestResult run_backtest(const std::vector<double>& prices,
                                const std::vector<int>& signals,
                                double dt_in_years);

private:
    double initial_capital_;
    double transaction_cost_pct_;
    double risk_free_rate_;

    // Helper methods (implemented in the .cpp file)
    double compute_max_drawdown(const std::vector<double>& equity) const;
    double compute_sharpe(const std::vector<double>& equity,
                          double dt_in_years) const;
};
