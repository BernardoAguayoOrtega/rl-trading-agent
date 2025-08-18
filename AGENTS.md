# Repository Guidelines

## Project Structure & Module Organization
- `rl_agent/StrategyEnv.py`: Gymnasium environment for strategy construction and evaluation.
- `framework/trading_framework.ipynb`: Backtesting/validation notebook and legacy utilities.
- `requirements.txt`: Runtime dependencies (Gymnasium, SB3, pandas-ta, numpy, pandas, yfinance).
- `.venv/`: Local virtual environment (not committed); prefer per-developer venvs.

## Build, Test, and Development Commands
- Setup env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Quick sanity check:
  - Python: `python -c "from rl_agent.StrategyEnv import StrategyEnv; print('Loaded')"`
  - Notebook: open `framework/trading_framework.ipynb` in Jupyter/VS Code to run backtests.
- Example data load (interactive): use `yfinance` to pull OHLCV and pass a DataFrame to `StrategyEnv`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; add type hints where practical.
- Names: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Docstrings for public classes/methods; keep functions small and testable.
- No linters configured; if adding, prefer `black` (line length 88), `isort`, `ruff` via a separate PR.

## Testing Guidelines
- Framework: prefer `pytest` with tests under `tests/` (e.g., `tests/test_strategy_env.py`).
- Focus: `reset/step` contract, action/observation space shapes, indicator computation boundaries.
- Conventions: name tests `test_*`, mark slow training as `@pytest.mark.slow` and run `pytest -m "not slow"` for CI.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood; emoji prefixes are common in history (e.g., ✨, ♻️, 📝). Group related changes.
- PRs: clear description, motivation, before/after notes or logs, linked issues, and reproduction steps (commands, seeds, data ranges). Include screenshots for notebook outputs when relevant.

## Agent-Specific Notes
- `StrategyEnv` expects a price DataFrame (OHLCV). Actions adjust indicator parameters; observations include technical metrics and performance features.
- Keep heavy backtesting logic in `framework/`; keep the environment pure and deterministic where possible (seed RNGs).
