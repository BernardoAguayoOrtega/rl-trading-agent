# RL Trading Agent Framework

## Overview

`rl-trading-agent` is a **modular, extensible** Python framework that combines a solid back‑testing engine with reinforcement‑learning (RL) or hyper‑parameter search agents to automatically discover profitable trading strategies.  The architecture is built around **small, single‑responsibility agents** that can be developed, tested, and swapped independently.

---

## Project Structure

```
rl-trading-agent/
│
├─ data/                     # Data handling utilities (load CSV, API, DB)
│   ├─ __init__.py
│   ├─ loader.py            # load_market_data(ticker, start, end)
│   └─ utils.py
│
├─ features/                 # Feature‑Builder agent
│   ├─ __init__.py
│   └─ builder.py           # FeatureBuilder class (pandas‑ta, custom feats)
│
├─ strategies/               # Strategy pool – one file per strategy
│   ├─ __init__.py
│   ├─ base.py              # BaseStrategy abstract class
│   ├─ mean_reversion.py
│   ├─ momentum.py
│   └─ mlp_strategy.py
│
├─ risk/                     # Risk‑Management agent
│   ├─ __init__.py
│   └─ manager.py           # BasicRisk (max‑DD, position sizing, stop‑loss)
│
├─ agents/                   # The six tiny agents that compose the env
│   ├─ __init__.py
│   ├─ observation.py       # ObservationAgent – returns feature vector
│   ├─ action.py            # ActionAgent – wraps RL policy or optimiser
│   ├─ signal.py            # SignalAgent – maps action → strategy
│   ├─ risk.py              # RiskAgent – calls risk.manager
│   ├─ portfolio.py         # PortfolioAgent – updates equity, P&L
│   └─ reward.py            # RewardAgent – pnl – λ·drawdown
│
├─ env/                      # ModularTradingEnv orchestrator (gym.Env)
│   ├─ __init__.py
│   └─ trading_env.py        # class ModularTradingEnv(gym.Env)
│
├─ training/                 # Scripts that launch RL / search training
│   ├─ __init__.py
│   ├─ train_ppo.py         # PPO training loop (uses env)
│   └─ optuna_search.py     # Optional hyper‑parameter optimisation
│
├─ evaluation/               # Evaluation & metric calculation
│   ├─ __init__.py
│   ├─ evaluator.py         # Walk‑forward back‑test, metric aggregation
│   └─ metrics.py           # Sharpe, Sortino, max‑DD, trade‑count helpers
│
├─ reporting/                # Reporting agent – creates markdown / plots
│   ├─ __init__.py
│   ├─ reporter.py          # generate_report(metrics, logs)
│   └─ plots.py             # equity, draw‑down, trade‑frequency, heat‑map
│
├─ notebooks/                # Jupyter notebooks that glue everything together
│   ├─ 01_overview.ipynb
│   ├─ 02_data_and_features.ipynb
│   ├─ 03_training.ipynb
│   ├─ 04_evaluation.ipynb
│   └─ 05_report.ipynb
│
├─ tests/                    # Unit tests for each agent / module
│   ├─ __init__.py
│   ├─ test_feature_builder.py
│   ├─ test_strategies.py
│   ├─ test_risk_manager.py
│   ├─ test_agents.py
│   └─ test_env.py
│
├─ configs/                  # YAML / JSON configuration files
│   ├─ feature_cfg.yaml
│   ├─ risk_cfg.yaml
│   └─ training_cfg.yaml
│
├─ requirements.txt          # pip packages (pandas, pandas‑ta, gymnasium,
│                              stable‑baselines3, torch, optuna, matplotlib, etc.)
├─ pyproject.toml            # optional – build metadata
└─ README.md                 # **this file**
```

---

## How the Agents Work Together (Step‑by‑Step)

1. **Data Agent** – `data.loader.load_market_data()` pulls raw OHLCV data for the chosen ticker and period.
2. **Feature‑Builder Agent** – `features.builder.FeatureBuilder` transforms the raw DataFrame into a deterministic feature matrix using the indicator list defined in `configs/feature_cfg.yaml`.
3. **Strategy Pool** – All concrete strategies are instantiated (e.g., `MeanReversion`, `Momentum`, `MLPStrategy`). The pool is simply a list that the environment can index into.
4. **Risk‑Manager Agent** – `risk.manager.BasicRisk` enforces a maximum draw‑down (e.g., 15 %) and handles position sizing / stop‑loss logic.
5. **ModularTradingEnv** – This Gym environment orchestrates six sub‑agents for each time step:
   * **ObservationAgent** – returns the current feature vector.
   * **ActionAgent** – queries the RL policy (or any optimiser) and returns an integer action = index of a strategy.
   * **SignalAgent** – calls the selected strategy’s `generate_signals` method → raw signal.
   * **RiskAgent** – filters the raw signal through the Risk‑Manager.
   * **PortfolioAgent** – executes the final signal on the current price, updates equity, cash, holdings, daily P&L, and draw‑down.
   * **RewardAgent** – computes `reward = daily_pnl – λ·drawdown` (λ is a risk‑aversion coefficient).
6. **RL / Search Agent** – Wrapped by `agents/action.py`. It can be a PPO policy (`stable‑baselines3`), an A2C policy, or an Optuna hyper‑parameter optimiser. The agent learns to pick the most profitable strategy while respecting the draw‑down penalty.
7. **Training** – Run `training/train_ppo.py` (or `optuna_search.py`). The script creates the environment, wraps it with a vectorised gym wrapper, and calls `model.learn(total_timesteps=…)`. The trained policy is saved as `trained_policy.zip`.
8. **Evaluation** – Load a hold‑out period (e.g., the most recent 12‑24 months), rebuild the feature matrix, instantiate a fresh environment, and run a deterministic episode with the saved policy. `evaluation/evaluator.py` aggregates the trade log and computes:
   * Sharpe, Sortino, total return
   * **Maximum draw‑down** (must stay below the user‑defined ceiling)
   * Trade count, turnover, win‑rate, average holding period
9. **Reporting** – `reporting/reporter.py` consumes the metrics and trade log, creates a markdown summary and a set of plots (equity curve, draw‑down curve, trade‑frequency histogram, strategy‑usage heat‑map). The report is inserted into `notebooks/05_report.ipynb` and can also be exported as HTML or PDF.
10. **Deployment (optional)** – Package the trained policy, the feature‑builder config, and the risk‑manager config. A lightweight live‑trading script streams new data, builds the observation, queries the policy, applies the same risk logic, and sends orders to your execution layer.

---

## Quick Start Guide

```bash
# 1️⃣ Clone the repo and create a virtual environment
git clone https://github.com/BernardoAguayoOrtega/rl-trading-agent.git
cd rl-trading-agent
python -m venv .venv && source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Prepare configuration files (edit the YAMLs in configs/)
#    - feature_cfg.yaml: list of pandas‑ta indicators and parameters
#    - risk_cfg.yaml:    max_dd, position sizing rules
#    - training_cfg.yaml: learning_rate, lambda_dd, total_timesteps, etc.

# 4️⃣ Run the data + feature pipeline (notebook 02)
python -c "from data.loader import load_market_data; from features.builder import FeatureBuilder; \
    df = load_market_data('SPY', '2015-01-01', '2024-12-31'); \
    fb = FeatureBuilder(); X = fb.fit_transform(df); X.to_pickle('features/sp500_features.pkl')"

# 5️⃣ Train the RL agent (notebook 03 or script)
python training/train_ppo.py   # uses configs/training_cfg.yaml

# 6️⃣ Evaluate on out‑of‑sample data (notebook 04)
python evaluation/evaluator.py --policy trained_policy.zip --holdout-start 2023-01-01

# 7️⃣ Generate the final report (notebook 05)
python reporting/reporter.py --metrics results/metrics.json --log results/trade_log.csv

# 8️⃣ (Optional) Deploy live
python live/trade.py --policy trained_policy.zip
```

---

## Detailed Description of Each Agent

### Data Agent (`data/`)
* **Purpose:** Centralised data ingestion – CSV, SQL, or external APIs.
* **Key function:** `load_market_data(ticker, start, end) → pd.DataFrame`.
* **Extensibility:** Add new loaders (e.g., `yfinance_loader.py`) and expose them via `__all__`.

### Feature‑Builder Agent (`features/builder.py`)
* **Purpose:** Transform raw market data into a feature matrix **and** search for the best feature set.
* **New architecture:**
  + **DataAgent** loads OHLCV data.
  + **FeatureSearch** (`AutoFeatureEngine` in `search/auto_engineer.py`) samples indicator configurations, applies them via the plugin‑based `FeatureBuilder`, and evaluates each set with a fast back‑test (`ModularTradingEnv`).
  + **StrategyGen** (`LLMStrategyGenerator` in `search/llm_generator.py`) creates a strategy specification from an LLM based on the generated features.
  + **Optimiser** (Optuna or a genetic algorithm) searches over feature‑strategy combinations, persisting the best configuration to `configs/*.yaml` and `strategies/specs/`.
  + **RL Trainer (optional)** can then train a reinforcement‑learning policy on the selected pipeline.
* **Implementation details:**
  - `features/builder.py` remains a lightweight plugin loader that applies feature plugins defined in `configs/feature_cfg.yaml`.
  - `search/auto_engineer.py` orchestrates the search loop, calling the builder for each candidate configuration.
  - The resulting best feature list is saved and later used by the standard training/evaluation scripts.
* **Config‑driven:** The YAML config now lists plugin ``name`` and optional ``params`` entries for each feature.

### Strategy Pool (`strategies/`)
* **Base class:** `BaseStrategy` defines `generate_signals(state) → pd.Series`.
* **Concrete examples:**
  * `MeanReversion` – z‑score based entry/exit.
  * `Momentum` – price‑difference over a window.
  * `MLPStrategy` – a pre‑trained neural net that outputs a probability of “up”.
* **How it is used:** The pool is a simple list; the environment indexes it with the action integer.

### Risk‑Manager Agent (`risk/manager.py`)
* **Purpose:** Enforce a **reasonable max draw‑down** and apply position sizing.
* **Methods:**
  * `apply(raw_signal, portfolio) → final_signal` – may block a trade if the projected draw‑down would exceed the limit.
  * `update_portfolio(portfolio, final_signal, price) → new_portfolio` – updates equity, cash, holdings, daily P&L, and recomputes draw‑down.
* **Configurable:** Parameters are read from `configs/risk_cfg.yaml`.

### Sub‑Agents (`agents/`)
| File | Class | Responsibility |
|------|-------|----------------|
| `observation.py` | `ObservationAgent` | Returns the feature vector for the current timestep. |
| `action.py` | `ActionAgent` | Wraps the RL policy (or Optuna trial) and returns an integer action. |
| `signal.py` | `SignalAgent` | Calls `strategy_pool[action].generate_signals(state_row)`. |
| `risk.py` | `RiskAgent` | Calls `risk_manager.apply`. |
| `portfolio.py` | `PortfolioAgent` | Calls `risk_manager.update_portfolio`. |
| `reward.py` | `RewardAgent` | Computes `reward = pnl – λ·drawdown`. |

These agents contain **no business logic** beyond delegating to the underlying modules, making them trivial to unit‑test.

### ModularTradingEnv (`env/trading_env.py`)
* Inherits from `gymnasium.Env`.
* Implements `reset()` and `step(action)` by invoking the six sub‑agents in the exact order shown above.
* Returns the standard Gym tuple `(obs, reward, done, info)` where `info` contains useful debugging fields (selected strategy, raw/final signals, portfolio snapshot).

### Training Scripts (`training/`)
* **`train_ppo.py`** – builds the environment, creates a `PPO` policy, runs `model.learn()`, and saves the policy.
* **`optuna_search.py`** – demonstrates how to replace `ActionAgent` with an Optuna trial that suggests a strategy index or hyper‑parameters; the same reward from `RewardAgent` is fed back to Optuna.

### Evaluation (`evaluation/`)
* **`evaluator.py`** – loads a hold‑out feature matrix, runs a deterministic episode with the saved policy, writes a trade log CSV and a JSON metrics file.
* **`metrics.py`** – pure‑Python functions that compute Sharpe, Sortino, max‑draw‑down, trade count, turnover, win‑rate, etc.

### Reporting (`reporting/`)
* **`reporter.py`** – reads the metrics JSON and trade log, builds a markdown summary, and calls `plots.py` to generate PNG/HTML figures.
* **`plots.py`** – Matplotlib/Plotly helpers for equity curve, draw‑down curve, trade‑frequency histogram, and a heat‑map of strategy usage over time.

---

## Automated Feature & Strategy Search

The framework now includes a **self‑contained optimisation layer** that can automatically discover profitable feature sets and trading strategies without manual coding.  This layer lives in the new `search/` package and consists of:

- **`search/auto_engineer.py`** – `AutoFeatureEngine` samples indicator configurations (periods, types) using random search, Optuna or a simple genetic algorithm.  It returns a feature matrix compatible with the existing `FeatureBuilder` API.
- **`search/llm_generator.py`** – `LLMStrategyGenerator` prompts an LLM (e.g., OpenAI GPT) with a templated request and receives a JSON `StrategySpec`.  The spec is deserialized into a `DynamicStrategy` (sub‑class of `BaseStrategy`).
- **`search/feature_strategy_search_agent.py`** – orchestrates data loading, feature generation, LLM‑driven strategy creation, fast back‑testing via `evaluation/evaluator.py`, and optimisation of the combined pipeline using Optuna.  The best configuration is persisted to `configs/feature_cfg.yaml` and `strategies/specs/`.
- **`search/utils.py`** – helper functions such as `save_best_config` for writing the winning feature list.
- **`run_search.py`** – a one‑liner entry script to launch the autonomous search:
  ```bash
  python run_search.py --n-trials 200 --target-sharpe 1.5
  ```

This addition keeps the original modular architecture intact: the generated feature matrix and strategy JSON are consumed by the existing `ModularTradingEnv`, allowing you to continue with the standard RL training (`training/train_ppo.py`) or evaluation pipelines.

---

## TODO List – Build the System Step by Step (Detailed)

### Chunk 1 – Project Setup
- [x] Initialize repository & environment – git, venv, dependencies, .gitignore.
  - Create a new GitHub repo and push the initial commit.
  - Set up a Python virtual environment (`python -m venv .venv`).
  - Install required packages from `requirements.txt`.
  - Add a comprehensive `.gitignore` (venv, __pycache__, *.ipynb_checkpoints, data/*.csv, etc.).
- [x] Data Agent – `data/loader.py` & `data/utils.py` (already implemented).
  - Verify loader works for multiple tickers and date ranges.
  - Add unit tests for CSV and API loading paths.
  - Document usage examples in the README.

### Chunk 2 – Search‑Driven Core Pipeline

#### Chunk 2.1 – AutoFeatureEngine
- [ ] Implement `search/auto_engineer.py`.
  - Design a configuration schema for indicator search (list of possible TA functions, parameter ranges).
  - Implement a random‑search baseline and an Optuna sampler.
  - Integrate with `FeatureBuilder` to generate a feature matrix for each trial.
  - Run a quick back‑test using a lightweight version of `ModularTradingEnv` (e.g., only equity tracking, no RL).
  - Log trial results (Sharpe, max‑DD) to a CSV for analysis.

#### Chunk 2.2 – LLMStrategyGenerator
- [ ] Implement `search/llm_generator.py`.
  - Define a prompt template that describes the feature set and asks the LLM to suggest a strategy.
  - Use OpenAI's API (or a mock) to obtain the JSON response.
  - Validate the JSON against a Pydantic model (`StrategySpec`).
  - Convert the spec into a concrete `DynamicStrategy` class that inherits from `BaseStrategy`.
  - Add error handling for malformed LLM output.

#### Chunk 2.3 – Optimiser
- [ ] Implement `search/feature_strategy_search_agent.py` (or a thin wrapper).
  - Set up an Optuna study with objectives: maximize Sharpe, minimize max‑DD.
  - In each trial, call `AutoFeatureEngine` → `FeatureBuilder` → `LLMStrategyGenerator` → fast back‑test.
  - Store the best trial’s feature config and strategy spec.
  - Provide a CLI interface to specify number of trials, target metrics, and random seed.

#### Chunk 2.4 – PersistBest
- [ ] Add helper `search/utils.py` (`save_best_config`).
  - Implement YAML serialization for the winning feature list.
  - Write the strategy JSON to a version‑controlled `strategies/specs/` folder.
  - Ensure the function is idempotent and creates backups of previous configs.

#### Chunk 2.5 – RL Trainer (optional)
- [ ] Use existing `training/train_ppo.py`.
  - Update the training script to read the latest `feature_cfg.yaml` and strategy spec.
  - Add a command‑line flag to skip training if only the search is desired.
  - Document how to resume training from a saved checkpoint.

### Chunk 3 – Supporting Components & Infrastructure

#### Chunk 3.1 – Feature‑Builder Agent
- [ ] Plugin‑based implementation already in `features/builder.py`.
  - Write unit tests for each plugin (e.g., moving_average, rsi).
  - Add documentation on how to create new plugins.

#### Chunk 3.2 – Strategy Pool
- [ ] Create abstract `strategies/base.py` and concrete strategies (`mean_reversion.py`, `momentum.py`, `mlp_strategy.py`).
  - Define `BaseStrategy` with a clear `generate_signals` contract.
  - Implement three baseline strategies with configurable parameters.
  - Add unit tests for signal generation logic.
  - Register strategies in `strategies/__init__.py` for easy import.

#### Chunk 3.3 – Risk‑Manager Agent
- [ ] Implement `risk/manager.py` and config `configs/risk_cfg.yaml`.
  - Implement `BasicRisk` with max‑draw‑down enforcement and position sizing.
  - Add unit tests covering edge cases (e.g., trade blocked due to DD).
  - Provide example YAML with typical risk parameters.

#### Chunk 3.4 – Sub‑Agents (agents/)
- [ ] Observation, action, signal, risk, portfolio, reward modules.
  - Create thin wrapper classes that delegate to the core modules.
  - Ensure each sub‑agent has a single public method and is fully typed.
  - Write integration tests that chain all six agents in a mock environment.

#### Chunk 3.5 – ModularTradingEnv
- [ ] Implement `env/trading_env.py`.
  - Follow the Gymnasium `Env` API (`reset`, `step`, `render`).
  - Include `info` dict with debug fields (selected strategy, portfolio snapshot).
  - Add a simple deterministic mode for fast back‑testing during search.

#### Chunk 3.6 – Training Scripts
- [ ] `training/train_ppo.py` (already present) and optional `training/optuna_search.py`.
  - Verify script works with both CPU and GPU.
  - Add argparse options for learning rate, total timesteps, and lambda‑dd.
  - Include checkpoint saving every N steps.

#### Chunk 3.7 – Evaluation Module
- [ ] `evaluation/evaluator.py` and `evaluation/metrics.py` (already present).
  - Extend evaluator to accept a custom `ModularTradingEnv` instance.
  - Add CSV export of trade‑log and JSON export of metrics.
  - Write unit tests for metric calculations.

#### Chunk 3.8 – Reporting Tools
- [ ] `reporting/plots.py` and `reporting/reporter.py` (already present).
  - Create reusable Matplotlib style sheet for consistent visuals.
  - Add functions to export plots as PNG and SVG.
  - Ensure reporter can be called from CLI with paths to metrics and log.

#### Chunk 3.9 – Unit Tests
- [ ] Ensure >80 % coverage for all new modules.
  - Set up `pytest` with coverage plugin.
  - Add CI step to enforce coverage threshold.

#### Chunk 3.10 – Notebooks
- [ ] Update the five notebooks to reflect the new search‑driven workflow.
  - Include a notebook that runs `run_search.py` and visualises trial results.
  - Add explanatory markdown cells for each pipeline stage.

#### Chunk 3.11 – End‑to‑End Demo
- [ ] Write a shell script `demo.sh` that executes all steps sequentially.
  - Capture timings and resource usage.

#### Chunk 3.12 – Documentation & CI
- [ ] Update README, add GitHub Actions workflow for linting, testing, and a quick training sanity check.
  - Create `.github/workflows/ci.yml` with jobs: lint (ruff/flake8), test (pytest), train‑sanity (short PPO run).
  - Add badges for build status and coverage.

#### Chunk 3.13 – Optional Extensions
- [ ] Live‑trading script, Dockerfile, Sphinx docs.
  - Implement `live/trade.py` that streams live data, builds observations, queries the saved policy, and sends orders via a broker API.
  - Write a minimal `Dockerfile` that installs dependencies and copies the repo.
  - Set up Sphinx with autodoc to generate API documentation.

Once each bullet is checked off, the repository will contain a fully‑automated, search‑driven feature and strategy pipeline together with optional RL training, ready for research and production use.
