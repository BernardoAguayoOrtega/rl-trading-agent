# Repository Guidelines

## Project Structure & Module Organization
- `rl_agent/StrategyEnv.py`: Gymnasium environment for strategy construction/evaluation.
- `framework/trading_framework.ipynb`: Backtesting/validation notebook and legacy helpers.
- `requirements.txt`: Runtime deps (Gymnasium, SB3, pandas-ta, numpy, pandas, yfinance).
- `tests/`: Pytest suite (e.g., `tests/test_strategy_env.py`).
- `.venv/`: Local virtual env (not committed); prefer per-developer venvs.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Sanity check import: `python -c "from rl_agent.StrategyEnv import StrategyEnv; print('Loaded')"`
- Run tests: `pytest -m "not slow"`
- Open notebook: launch Jupyter/VS Code and run `framework/trading_framework.ipynb`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation; add type hints where practical.
- Names: `snake_case` (functions/modules), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- Docstrings for public classes/methods; keep functions small and testable.
- Linters/formatters: none enforced; if adding, prefer `black` (88), `isort`, `ruff` in a separate PR.

## Testing Guidelines
- Framework: `pytest` under `tests/` with files named `test_*.py`.
- Focus: `reset`/`step` contract, action/observation shapes, indicator boundary conditions.
- Mark slow training: `@pytest.mark.slow`; CI: `pytest -m "not slow"`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood; emoji prefixes common (e.g., ✨ feature, ♻️ refactor, 📝 docs).
- PRs: include motivation, before/after notes or logs, linked issues, and reproduction steps (commands, seeds, data ranges). Add screenshots for notebook outputs when relevant.

## Agent-Specific Notes
- `StrategyEnv` expects an OHLCV `pandas.DataFrame`. Actions adjust indicator parameters; observations include technical metrics and performance features.
- Keep the environment deterministic (seed RNGs). Heavy backtesting stays in `framework/`.

## Security & Configuration Tips
- Do not commit credentials or `.venv/`. Use per-developer environments.
- Prefer reproducible runs: pin seeds, record data ranges and symbols in PR descriptions.
