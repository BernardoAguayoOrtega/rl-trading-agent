# Reinforcement Learning Trading Agent

## 📋 Checklist

- [ ] **Project Structure** – `/data`, `/notebooks`, `/src`, `/models`, `/results`
- [ ] **Core Dependencies** – `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `gymnasium`, `stable-baselines3`, `tensorflow` **or** `torch`
- **Feature Engineering**
  - [ ] Trend indicators (SMAs/EMAs, MACD, ADX)
  - [ ] Momentum indicators (RSI, Stochastic, ROC)
  - [ ] Volatility indicators (ATR, Bollinger Bands)
  - [ ] Volume indicators (OBV, Volume MA)
  - [ ] Data normalization (e.g., `MinMaxScaler`)
  - [ ] Final master dataset (`dfAll.xlsx`) with clear train / out‑of‑sample splits
- **RL Environment (Gymnasium)**
  - [ ] Create `trading_env.py`
  - [ ] Implement `TradingEnv` class (compatible with `gymnasium.Env`)
  - [ ] Define state & action spaces
  - [ ] Implement `__init__`, `reset`, `step` (including reward function)
  - [ ] Validate with `gymnasium.utils.env_checker`
- **Agent Training**
  - [ ] Create `train.py`
  - [ ] Baseline algorithm (e.g., PPO)
  - [ ] Hyper‑parameter configuration
  - [ ] Optional callbacks for evaluation & checkpointing
  - [ ] TensorBoard logging
- **Validation & Analysis**
  - [ ] Create `evaluate.py`
  - [ ] Adapt existing performance metrics for RL agent
  - [ ] Out‑of‑sample testing
  - [ ] Compare with original RSI system and Buy‑and‑Hold baseline
  - [ ] Plot trades and analyze behavior
  - [ ] (Optional) Walk‑forward validation

---

## 📈 Updated Project Plan (including Trading Framework)

### Phase 0 – Review Existing Trading Framework
1. **Read and understand** `trading_framework.ipynb` to capture existing backtesting, data loading, and feature‑engineering logic.
2. **Extract and modularize** the data import (`dfIS.xlsx` loading) and preprocessing steps (SMA, RSI functions) into reusable modules under `/src`.
3. **Adapt the backtesting functions** (`backSistemaList`, `calculaCurvas`, `crearDfBacktesting`, etc.) into a reusable Python package (`src/backtesting.py`).
4. **Create a wrapper** that can be called from the RL environment to evaluate episode performance using the same metrics.
5. **Validate** the extracted functions with unit tests to ensure they produce identical results to the notebook.

### Phase 1 – Foundation & Feature Engineering
1. Set up folder structure and initialize a Python virtual environment.
2. Install core dependencies (`pip install -r requirements.txt`).
3. Expand feature pipeline with trend, momentum, volatility, and volume indicators.
4. Implement data normalization using `MinMaxScaler`.
5. Prepare final dataset (`dfAll.xlsx`) and split into training / OS validation sets.
6. **Integrate** the modularized data loading and feature‑engineering code from Phase 0.

### Phase 2 – RL Environment (Gymnasium)
1. Create `trading_env.py` with a `TradingEnv` class.
2. Define observation (state) and action spaces.
3. Implement `__init__`, `reset`, and `step` methods, including a reward function (starting with daily returns).
4. **Incorporate** the backtesting evaluation logic to compute episode rewards and performance metrics.
5. Validate the environment with `gymnasium.utils.env_checker`.

### Phase 3 – Agent Training & Experimentation
1. Write `train.py` to instantiate the environment and a PPO agent.
2. Configure hyper‑parameters (learning rate, gamma, batch size, etc.).
3. Add optional callbacks for evaluation and model checkpointing.
4. Run an initial training run to ensure the pipeline works.
5. Integrate TensorBoard for visualizing training progress.

### Phase 4 – Validation & Analysis
1. Write `evaluate.py` to load a trained model and run it on the OS validation set.
2. Use the backtesting functions (`backSistemaList`, `calculaCurvas`) to assess the RL agent’s performance.
3. Perform out‑of‑sample testing and generate performance reports.
4. Compare results against the original RSI system and a Buy‑and‑Hold baseline.
5. Visualize trades on price charts and analyze decision patterns.
6. (Optional) Implement walk‑forward validation: retrain on expanding/rolling windows and re‑evaluate.

---

*Happy coding!* 🚀
