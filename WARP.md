# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This is a reinforcement learning project that trains an agent to discover optimal trading strategies. The system uses a custom Gymnasium environment (`StrategyEnv`) where an RL agent learns to optimize technical indicator parameters and generate trading signals that are backtested against historical market data.

## Project Architecture

### Core Structure
- **`rl_agent/`** - Contains the RL environment and agent logic
  - `StrategyEnv.py` - Custom Gymnasium environment for strategy optimization
- **`framework/`** - Contains the existing trading framework
  - `trading_framework.ipynb` - Jupyter notebook with backtesting functions
- **Virtual Environment** - `.venv/` contains isolated Python dependencies

### Key Components

**StrategyEnv Class**: The main RL environment that:
- Manages 30+ technical indicators from pandas-ta (RSI, MACD, Bollinger Bands, etc.)
- Provides complex action/observation spaces for indicator parameter tuning
- Integrates with legacy trading framework functions for backtesting
- Uses Calmar ratio (CAGR/Max Drawdown) as primary reward function

**Trading Framework Functions** (defined in notebook):
- `dameSistema()` - Generates buy/sell signals based on indicator thresholds
- `damePosition()` - Converts signals to position tracking
- `dameSalidaVelas()` - Handles candle-based exit rules
- `dameSalidaPnl()` - Manages P&L calculations with stop-loss/take-profit
- `calculaCurvas()` - Computes performance curves
- `backSistemaList()` - Returns comprehensive backtest metrics list

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Components
```bash
# Launch Jupyter for framework development
jupyter notebook framework/trading_framework.ipynb

# Test the RL environment (when train.py is implemented)
python rl_agent/train.py

# Run single backtest evaluation (when evaluate.py is implemented)
python rl_agent/evaluate.py
```

### Development Workflow
```bash
# Install additional RL/ML packages as needed
pip install stable-baselines3[extra] gymnasium pandas-ta

# Update requirements
pip freeze > requirements.txt

# Run basic environment test
python -c "from rl_agent.StrategyEnv import StrategyEnv; print('Environment imports successfully')"
```

## Implementation Status

### Completed (Phase 1)
- [x] Project structure setup
- [x] Core dependencies installation  
- [x] StrategyEnv environment with comprehensive action/observation spaces
- [x] Technical indicator integration via pandas-ta
- [x] Framework integration (imports existing backtesting functions)

### Pending Implementation (Phases 2-3)
- [ ] Complete `_compute_all_indicators()` method in StrategyEnv
- [ ] Create `trading_framework.py` module (extract from notebook)
- [ ] Implement `train.py` - RL training script using stable-baselines3
- [ ] Implement `evaluate.py` - Model evaluation and strategy validation
- [ ] Create `garch_tester.py` - GARCH-based stress testing module
- [ ] Add GARCH synthetic data generation for robustness testing

## Key Technical Details

### Environment Specifications
- **Action Space**: Dict with 5 categories (overlap, momentum, volatility, trend, cycle actions)
- **Observation Space**: Complex dict with 40+ fields including indicator values, performance metrics, market regimes
- **Reward Function**: Calmar ratio + profit factor bonus, with heavy penalties for poor strategies

### Integration Points
The environment expects these functions from `trading_framework`:
- `damePosition`, `dameSalidaVelas`, `dameSalidaPnl`, `calculaCurvas`, `backSistemaList`

### Performance Metrics
The system tracks comprehensive trading metrics:
- CAGR, Max Drawdown, Sharpe Ratio, Win Rate
- Number of trades, Profit Factor, Expectancy
- Risk-adjusted returns via Calmar ratio optimization

## Development Notes

### Missing Components
- The `_compute_all_indicators()` method is referenced but not implemented
- Framework functions need extraction from Jupyter notebook to `.py` module
- Training and evaluation scripts are planned but not yet created

### Data Requirements
- Historical market data (OHLCV format)
- Currently configured for Excel input (`dfIS.xlsx`) 
- Market data should include sufficient history for indicator calculations (200+ periods recommended)

### Testing Strategy
The project implements a multi-stage validation approach:
1. In-sample (IS) training and optimization
2. Out-of-sample (OS) validation
3. GARCH stress testing with synthetic market scenarios
