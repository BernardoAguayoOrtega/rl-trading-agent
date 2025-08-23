# RL Trading Agent

AI-powered quantitative trading research agent built on top of a robust backtesting framework.

## Features

- **Dynamic Strategy Framework**: Create strategies using configurable indicators and rules
- **AI Agent Integration**: OpenAI-powered strategy generation and optimization
- **Comprehensive Backtesting**: Advanced validation and risk management
- **Multiple Validation Methods**: Monte Carlo, Walk Forward, Cross Validation

## Quick Start

1. Install dependencies:
   ```bash
   uv venv rl-trading-env
   source rl-trading-env/bin/activate
   uv pip install -e .
   ```

2. Set your OpenAI API key:
   ```python
   import os
   os.environ['OPENAI_API_KEY'] = 'your-api-key'
   ```

3. Start generating strategies:
   ```python
   agent = QuantitativeResearchAgent()
   results = await agent.research_strategies(data)
   ```

## Requirements

- Python 3.9+
- OpenAI API key
- Financial data (Excel/CSV format)

## License

MIT
