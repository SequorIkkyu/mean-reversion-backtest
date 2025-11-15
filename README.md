# Mean-Reversion Strategy Backtesting Framework (Python/C++)

This repository implements a lightweight quantitative research framework for designing and evaluating mean-reversion trading signals on equity time-series data. The project combines **Python** for data handling, analysis, and visualisation with a **C++ backtesting engine** for efficient execution of trading logic.

---

## Overview

The objective of this project is to investigate the behaviour and robustness of simple mean-reversion signals constructed using rolling z-scores, and to evaluate their performance under different market regimes. The backtesting engine is written in C++ to ensure deterministic execution and improved computational performance compared with a pure Python implementation.

---

## Features

### **1. Signal Construction**
- Rolling mean and standard deviation over configurable lookback windows  
- Z-score signal computation  
- Long/short entry and exit thresholds  
- Optional volatility filters

### **2. C++ Backtesting Engine**
- Core logic implemented in C++ for speed and determinism:
  - Position management (long/short/flat)
  - Transaction cost modelling
  - P&L and returns computation
  - Daily mark-to-market evaluation
- Exposed to Python via a binding layer (pybind11 or ctypes)

### **3. Performance Evaluation**
- Standard risk and performance metrics:
  - Sharpe ratio
  - Maximum drawdown
  - Hit-rate
  - Return distribution statistics
- Regime-based performance comparison  
  (range-bound vs trending regimes)

### **4. Research Tools**
- Parameter sweeps for lookback windows, thresholds, and filters
- Sensitivity analysis to examine overfitting risk
- Clean plotting utilities for:
  - Equity curve
  - Drawdown curve
  - Signal vs price overlays

---

## Repository Structure

