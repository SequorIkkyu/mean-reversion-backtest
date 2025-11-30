#include "BacktestEngine.hpp"

#include <algorithm>    // std::max
#include <cmath>        // std::sqrt
#include <numeric>      // std::accumulate

BacktestEngine::BacktestEngine(double initial_capital,
                               double transaction_cost_pct,
                               double risk_free_rate)
    : initial_capital_(initial_capital),
      transaction_cost_pct_(transaction_cost_pct),
      risk_free_rate_(risk_free_rate) {}


BacktestResult BacktestEngine::run_backtest(const std::vector<double>& prices,
                                            const std::vector<int>& signals,
                                            double dt_in_years)
{
    BacktestResult result;

    // Basic sanity checks
    std::size_t n = prices.size();
    if (n == 0 || signals.size() != n || n == 1) {
        // Return default result (everything zero / empty)
        return result;
    }

    result.equity_curve.resize(n);
    result.pnl.resize(n);
    result.position.resize(n);

    double equity = initial_capital_;
    int current_pos = 0;

    // Initialise at t = 0
    result.equity_curve[0] = equity;
    result.pnl[0] = 0.0;
    result.position[0] = current_pos;

    for (std::size_t i = 1; i < n; ++i) {
        int desired_pos = signals[i];

        // If position changes, pay transaction cost
        if (desired_pos != current_pos) {
            double traded_notional = std::abs(desired_pos - current_pos) * prices[i];
            double cost = traded_notional * transaction_cost_pct_;
            equity -= cost;
            result.pnl[i] -= cost;
            current_pos = desired_pos;
        }

        // PnL from price movement
        double price_change = prices[i] - prices[i - 1];
        double step_pnl = current_pos * price_change;
        result.pnl[i] += step_pnl;
        equity += step_pnl;

        // Store results
        result.position[i] = current_pos;
        result.equity_curve[i] = equity;
    }

    // Total return
    result.total_return = (equity / initial_capital_) - 1.0;

    // Max drawdown based on equity curve
    result.max_drawdown = compute_max_drawdown(result.equity_curve);

    // Sharpe ratio based on PnL series
    result.sharpe_ratio = compute_sharpe(result.pnl, dt_in_years);

    return result;
}

double BacktestEngine::compute_max_drawdown(const std::vector<double>& equity) const
{
    if (equity.empty()) {
        return 0.0;
    }

    double max_peak = equity[0];
    double max_dd = 0.0;

    for (double e : equity) {
        if (e > max_peak) {
            max_peak = e;
        }
        double dd = (max_peak - e) / max_peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}

double BacktestEngine::compute_sharpe(const std::vector<double>& pnl,
                                      double dt_in_years) const
{
    if (pnl.size() <= 1 || dt_in_years <= 0.0) {
        return 0.0;
    }

    // We ignore the first element (often zero) for statistics,
    // but it's not critical — you can choose either convention.
    std::size_t n = pnl.size();

    // Compute mean
    double sum = std::accumulate(pnl.begin(), pnl.end(), 0.0);
    double mean = sum / static_cast<double>(n);

    // Compute variance
    double var = 0.0;
    for (double x : pnl) {
        double diff = x - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(n);

    double std_dev = std::sqrt(var);
    if (std_dev == 0.0) {
        return 0.0;
    }

    // Per-step Sharpe, annualised via sqrt(1/dt)
    double sharpe = mean / std_dev * std::sqrt(1.0 / dt_in_years);

    // 这里可以减去 risk_free_rate 的影响，但目前我们假设 pnl 已经是 excess return
    // 如需更严格，可以把 risk_free_rate 转换成每 step 的无风险收益再减。

    return sharpe;
}
