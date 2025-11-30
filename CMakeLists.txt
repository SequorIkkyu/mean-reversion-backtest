cmake_minimum_required(VERSION 3.12)
project(mean_reversion_backtester)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories (header files)
include_directories(include)

# Build Python extension module
pybind11_add_module(backtest
    python/bindings.cpp
    src/BacktestEngine.cpp
)
