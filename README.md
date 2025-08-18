Phase 1: Environment Setup & Integration
[x] Create Project Structure

[x] Initialize the main project directory.

[x] Create a framework subdirectory and move your existing backtesting and validation scripts into it.

[x] Create an rl_agent subdirectory for all new code.

[x] Install Dependencies

[x] Run pip install gymnasium stable-baselines3 pandas-ta numpy pandas.

[x] Build the Environment Wrapper (StrategyEnv.py)

[x] Create the StrategyEnv.py file with a class StrategyEnv that inherits from gymnasium.Env.

[x] Implement the __init__ method to load your historical data and define the action_space (e.g., Discrete(20) for 20 possible actions) and observation_space (e.g., Dict to describe the current strategy).

Phase 2: Defining the RL Logic & Core Components
[x] Implement the step(action) Method

[x] Translate the agent's action (an integer) into a specific modification of the current trading strategy.

[x] When the "run_backtest" action is triggered, call your existing framework functions (dameSistema, calculaCurvas, etc.).

[x] Return the standard (observation, reward, terminated, truncated, info) tuple.

[x] Design the _calculate_reward() Function

[x] Create a specific, non-trivial reward function that balances risk and return.

[x] Implement a formula (e.g., reward = (CAGR / Max_Drawdown)) inside the environment.

[x] Add heavy negative penalties for bad outcomes (e.g., drawdown > 30%, fewer than 10 trades).

[x] Implement the reset() Method

[x] Code the reset() function to clear the current strategy and prepare for a new episode.

[x] Ensure it returns the initial observation.

Phase 3: Training, Stress-Testing & Final Evaluation
[x] Create the Configuration Module (config.py)

[x] Centralize all project configurations including paths, model parameters, and trading settings.

[x] Define RISK_CONFIG with parameters like risk-free rate, drawdown limits, and risk parameters.

[x] Configure PLOT_CONFIG for consistent visualization styles and parameters.

[x] Create the Training Script (train.py)

[x] Instantiate your StrategyEnv with appropriate wrappers (VecEnv, VecNormalize).

[x] Configure and initialize PPO model from stable-baselines3 with appropriate hyperparameters.

[x] Implement comprehensive callbacks system for evaluation, early stopping, and checkpointing.

[x] Set up robust logging and experiment tracking (optional Weights & Biases integration).

[x] Create the Final Evaluation Script (evaluate.py)

[x] Implement the TradingMetricsCalculator for comprehensive performance metrics.

[x] Create BenchmarkStrategies for comparing against baseline trading approaches.

[x] Build StrategyEvaluator with model loading, backtesting, and statistical significance testing.

[x] Add ResultsVisualizer for generating plots and comprehensive HTML reports.

[x] Build the GARCH Stress-Test Module (garch_tester.py)

[x] Implement GARCHModel class with parameter fitting and return simulation capabilities.

[x] Create StressScenario to define different market volatility regimes for testing.

[x] Build GARCHStressTester with Monte Carlo simulation and strategy evaluation across scenarios.

[x] Add StressTestVisualizer for creating comparative visualizations and stress test reports.

[x] Update Documentation

[x] Document Phase 3 components in the README.

[x] Add usage examples and command-line options for each script.

[x] Ensure code is properly commented for maintainability.
