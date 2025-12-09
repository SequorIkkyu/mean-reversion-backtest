#include <iostream>
#include <vector>
#include "BacktestEngine.hpp"

int main()
{
    // =========================
    // 1️⃣ 准备测试价格数据（示例）
    // =========================
    std::vector<double> prices = {
        100, 101, 102, 101, 100,
        99, 98, 99, 100, 102,
        101, 100, 99, 98, 97,
        98, 99, 100, 101, 103};

    // =========================
    // 2️⃣ 准备交易信号（-1, 0, +1）
    //    简单示例：低位做多，高位做空
    // =========================
    std::vector<int> signals = {
        0, 0, -1, 0, 0,
        1, 1, 0, 0, -1,
        0, 0, 1, 1, 0,
        0, 0, 0, 0, -1};

    // =========================
    // 3️⃣ 创建回测引擎
    // =========================
    double initial_capital = 100000.0; // 初始资金 10 万
    double transaction_cost = 0.001;   // 单边 0.1% 手续费
    double risk_free_rate = 0.0;       // 无风险利率

    BacktestEngine engine(
        initial_capital,
        transaction_cost,
        risk_free_rate);

    // =========================
    // 4️⃣ 运行回测（假设是日频）
    // =========================
    double dt = 1.0 / 252.0; // 每一步 = 1 个交易日

    BacktestResult result = engine.run_backtest(
        prices,
        signals,
        dt);

    // =========================
    // 5️⃣ 输出结果
    // =========================
    std::cout << "======== Backtest Result ========" << std::endl;
    std::cout << "Total Return: " << result.total_return * 100.0 << " %" << std::endl;
    std::cout << "Max Drawdown: " << result.max_drawdown * 100.0 << " %" << std::endl;
    std::cout << "Sharpe Ratio: " << result.sharpe_ratio << std::endl;

    std::cout << "\nFinal Equity: "
              << result.equity_curve.back()
              << std::endl;

    return 0;
}
