#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/BacktestEngine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(backtest, m) {
    m.doc() = "C++ backtesting engine exposed to Python via pybind11";

    // BacktestResult binding
    py::class_<BacktestResult>(m, "BacktestResult")
        .def_readonly("equity_curve", &BacktestResult::equity_curve)
        .def_readonly("pnl", &BacktestResult::pnl)
        .def_readonly("position", &BacktestResult::position)
        .def_readonly("total_return", &BacktestResult::total_return)
        .def_readonly("max_drawdown", &BacktestResult::max_drawdown)
        .def_readonly("sharpe_ratio", &BacktestResult::sharpe_ratio);

    // BacktestEngine binding
    py::class_<BacktestEngine>(m, "BacktestEngine")
        .def(py::init<double, double, double>(),
             py::arg("initial_capital"),
             py::arg("transaction_cost_pct"),
             py::arg("risk_free_rate") = 0.0)

        .def("run_backtest",
             &BacktestEngine::run_backtest,
             py::arg("prices"),
             py::arg("signals"),
             py::arg("dt_in_years"),
             "Run the backtest and return a BacktestResult");
}
