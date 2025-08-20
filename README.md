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
* **Purpose:** Deterministic transformation of raw OHLCV into a state vector.
* **Implementation:** Uses `pandas‑ta` internally; can also load custom Python functions.
* **Config‑driven:** Indicator list lives in `configs/feature_cfg.yaml` – you can add/remove indicators without touching code.

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

## Testing

All modules have corresponding unit tests in the `tests/` folder. Run them with:

```bash
pytest tests/
```

The tests cover:
* FeatureBuilder output shape and NaN handling.
* Each concrete strategy’s signal logic.
* RiskManager’s draw‑down enforcement.
* The six sub‑agents (they should return the expected types).
* The full `ModularTradingEnv` (step returns correct dimensions and reward sign).

---

## Extending the Framework

* **Add a new indicator** – edit `configs/feature_cfg.yaml` and, if needed, add a helper function in `features/custom.py`.
* **Add a new strategy** – create a new file in `strategies/` that inherits from `BaseStrategy` and implement `generate_signals`. Then add it to the pool in `notebooks/02_data_and_features.ipynb`.
* **Swap the learning algorithm** – replace `PPO` with `SAC` or `A2C` in `training/train_ppo.py`. No other code changes are required.
* **Replace RL with a genetic algorithm** – implement a new `ActionAgent` that receives a chromosome, decodes it into a strategy index/parameters, and returns the action. The rest of the environment stays unchanged.

---

## TODO List – Build the System Step by Step (Detailed)

- [x] **Initialize repository & environment**
  - `git init` and create a remote on GitHub.
  - `python -m venv .venv && source .venv/bin/activate`.
  - `pip install -r requirements.txt`.
  - Add a `.gitignore` (venv, __pycache__, *.ipynb_checkpoints, data/*.csv, etc.).

- [x] **Data Agent**
  - **Create `data/loader.py`**
    - Define `load_market_data(ticker: str, start: str, end: str) → pd.DataFrame`.
    - Use `yfinance` as a fallback, and allow CSV loading via a `source` argument.
    - Return a DataFrame with columns `['open','high','low','close','volume']` and a datetime index.
  - **Create `data/utils.py`**
    - Helper `validate_dataframe(df)` to ensure required columns exist.
    - Function `resample_to_daily(df, freq='1D')`.
  - Add unit tests in `tests/test_agent.py` (covers loader, utils, and error cases).

- [ ] **Feature‑Builder Agent**
  - **Create `features/builder.py`**
    - Class `FeatureBuilder` with `__init__(self, cfg_path: str = None)`.
    - Method `fit_transform(self, raw_df: pd.DataFrame) → pd.DataFrame`.
    - Load indicator configuration from `configs/feature_cfg.yaml` (list of dicts: name, params).
    - Loop over config and call `getattr(raw_df.ta, name)(**params, append=True)`.
    - Drop rows with NaNs after the longest indicator warm‑up.
  - **Create `features/custom.py`** (optional) for user‑defined functions.
  - Add unit tests `tests/test_feature_builder.py` covering:
    - Correct columns are added.
    - No NaNs remain.
    - Config parsing works.

- [ ] **Strategy Pool**
  - **Create `strategies/base.py`**
    - Abstract class `BaseStrategy` with `def generate_signals(self, state: pd.DataFrame) -> pd.Series`.
  - **Implement concrete strategies**
    - `strategies/mean_reversion.py`
      - Params: `window`, `z_thresh`.
      - Compute rolling mean/std, z‑score, return signals.
    - `strategies/momentum.py`
      - Params: `window`.
      - Use price diff over window.
    - `strategies/mlp_strategy.py`
      - Params: `model_path`.
      - Load a scikit‑learn/torch model and predict probability.
  - **Update `strategies/__init__.py`** to expose the classes.
  - Add unit tests `tests/test_strategies.py` for each strategy.

- [ ] **Risk‑Manager Agent**
  - **Create `risk/manager.py`**
    - Class `BasicRisk` with `__init__(self, max_dd: float = 0.15, position_size: float = 0.01)`.
    - Method `apply(self, raw_signal: int, portfolio: dict) -> int` – block trade if projected draw‑down > `max_dd`.
    - Method `update_portfolio(self, portfolio: dict, signal: int, price: float) -> dict` – update cash, holdings, equity, daily P&L, compute new draw‑down.
    - Helper `initial_portfolio(self) -> dict` returning cash, holdings, equity = 1.0, max_dd = 0.
  - Add config file `configs/risk_cfg.yaml` with `max_dd` and `position_size`.
  - Unit tests `tests/test_risk_manager.py`.

- [ ] **Sub‑Agents (agents/)**
  - **`agents/observation.py`**
    - Class `ObservationAgent` with `def get(self, step: int) -> np.ndarray`.
  - **`agents/action.py`**
    - Class `ActionAgent` wrapping a policy object.
    - Method `select(self, obs: np.ndarray) -> int`.
    - Provide a dummy random policy for early testing.
  - **`agents/signal.py`**
    - Class `SignalAgent` with `def generate(self, action: int, state_row: pd.DataFrame) -> int`.
  - **`agents/risk.py`**
    - Class `RiskAgent` delegating to `BasicRisk.apply`.
  - **`agents/portfolio.py`**
    - Class `PortfolioAgent` delegating to `BasicRisk.update_portfolio`.
  - **`agents/reward.py`**
    - Class `RewardAgent` with `def compute(self, portfolio: dict) -> float` using a configurable `lambda_dd` (read from `configs/training_cfg.yaml`).
  - Add a combined test `tests/test_agents.py` that mocks a simple environment and checks the data flow.

- [ ] **ModularTradingEnv**
  - **Create `env/trading_env.py`**
    - Inherit from `gymnasium.Env`.
    - `__init__(self, features: pd.DataFrame, strategy_pool: List[BaseStrategy], risk_manager: BasicRisk, policy, lambda_dd: float)`.
    - Instantiate the six sub‑agents.
    - Implement `reset(self)` returning first observation.
    - Implement `step(self, action)` following the exact order:
      1. Get raw signal via `SignalAgent`.
      2. Apply risk via `RiskAgent`.
      3. Update portfolio via `PortfolioAgent`.
      4. Compute reward via `RewardAgent`.
      5. Advance step, return next observation (or zeros), reward, done, info dict.
  - Write integration test `tests/test_env.py` that runs a short episode (e.g., 10 steps) with a dummy policy.

- [ ] **Training Scripts**
  - **`training/train_ppo.py`**
    - Load config from `configs/training_cfg.yaml` (learning_rate, total_timesteps, lambda_dd).
    - Build feature matrix (`features/sp500_features.pkl`).
    - Instantiate strategy pool and risk manager.
    - Create the environment.
    - Wrap with `gymnasium.vector.SyncVectorEnv`.
    - Initialise `stable_baselines3.PPO` with `MlpPolicy`.
    - Call `model.learn(total_timesteps=…)`.
    - Save model to `trained_policy.zip`.
  - **`training/optuna_search.py`** (optional)
    - Define an Optuna objective that creates a temporary policy, runs a short episode, returns cumulative reward.
    - Store best hyper‑parameters in `configs/training_cfg.yaml`.

- [ ] **Evaluation Module**
  - **`evaluation/evaluator.py`**
    - CLI arguments: `--policy`, `--holdout-start`, `--holdout-end`.
    - Load hold‑out raw data, build features, create env with same pool & risk manager.
    - Load policy (`PPO.load`).
    - Run deterministic episode, collect trade log (timestamp, price, signal, equity, drawdown).
    - Compute metrics via `evaluation/metrics.py`.
    - Save `results/trade_log.csv` and `results/metrics.json`.
  - **`evaluation/metrics.py`**
    - Functions: `sharpe(returns)`, `sortino(returns)`, `max_drawdown(equity)`, `trade_count(log)`, `turnover(log)`, `win_rate(log)`.

- [ ] **Reporting Tools**
  - **`reporting/plots.py`**
    - Functions returning Matplotlib figures: `plot_equity(equity)`, `plot_drawdown(drawdown)`, `plot_trade_frequency(log)`, `plot_strategy_heatmap(log, strategy_names)`.
  - **`reporting/reporter.py`**
    - Load metrics JSON and trade log CSV.
    - Generate markdown string with key metrics.
    - Call plot functions, save PNGs to `reports/`.
    - Append markdown + image links to `notebooks/05_report.ipynb` (or export as HTML).

- [ ] **Unit Tests** (already mentioned per component) – ensure 80 %+ coverage.

- [ ] **Notebooks**
  - **`notebooks/01_overview.ipynb`** – project intro, architecture diagram.
  - **`notebooks/02_data_and_features.ipynb`** – load data, run `FeatureBuilder`, visualise a few features.
  - **`notebooks/03_training.ipynb`** – instantiate env, train PPO (show TensorBoard screenshots).
  - **`notebooks/04_evaluation.ipynb`** – run evaluator, display metrics table and plots.
  - **`notebooks/05_report.ipynb`** – final markdown report generated by `reporting/reporter.py`.

- [ ] **End‑to‑End Demo**
  - Follow the Quick Start Guide to train on SPY 2015‑2024, evaluate on 2023‑2024, and generate the final report.
  - Verify that the report contains:
    * Equity curve PNG.
    * Draw‑down curve PNG.
    * Trade‑frequency histogram.
    * Strategy usage heat‑map.
    * Table with Sharpe, Sortino, max‑DD, trade count.

- [ ] **Documentation & CI**
  - Update this README with usage examples for each script.
  - Add a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
    * Sets up Python 3.11.
    * Installs dependencies.
    * Runs `pytest`.
    * Optionally runs a short training run (`total_timesteps=1000`) to ensure the pipeline works.

- [ ] **Optional Extensions**
  - Implement a **live‑trading script** (`live/trade.py`) that streams real‑time data, builds observations, queries the saved policy, and sends orders via a broker API.
  - Add **Dockerfile** for reproducible container builds.
  - Create **Sphinx docs** under `docs/`.

---

Once each bullet is checked off, you will have a fully functional, test‑covered, and documented RL‑driven trading framework ready for research and production use.
````
