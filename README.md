# AI Trading Agent 🚀

An intelligent algorithmic trading system that leverages OpenAI's GPT models to make sophisticated trading decisions based on technical analysis, market conditions, and risk assessment.

## 🎯 Overview

This project integrates professional-grade backtesting frameworks with AI-powered decision making to create a comprehensive trading system that can:

- **🧠 Analyze market conditions** using advanced AI
- **📊 Process technical indicators** from your existing framework
- **⚡ Make real-time trading decisions** with confidence scoring
- **🔍 Validate strategies** using multiple testing methods
- **📈 Adapt to market regimes** dynamically

## 🏗️ Project Structure

```
ai-trading-agent/
├── src/ai_trading_agent/     # Main source code
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── secrets.py           # Secure secrets handling
│   └── ai_client.py         # OpenAI integration
├── tests/                   # Test suites
│   ├── test_phase1.py       # Phase 1 environment tests
│   └── test_ai_live.py      # Live AI functionality tests
├── notebooks/               # Jupyter notebooks
│   ├── ai_trading_agent_plan.ipynb  # Complete implementation plan
│   ├── framework.ipynb      # Your existing backtesting framework
│   └── indicators.ipynb     # Technical indicators library
├── scripts/                 # Utility scripts
├── docs/                    # Documentation
├── .env.example            # Environment variables template
└── pyproject.toml          # Project configuration
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter the project
cd ai-trading-agent

# Install dependencies with uv
uv sync

# Set up your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Test Installation

```bash
# Test Phase 1 setup
uv run python tests/test_phase1.py

# Test AI integration
uv run python tests/test_ai_live.py
```

### 3. Run the Application

```bash
# Start the main application
uv run ai-trading-agent

# Or start Jupyter Lab for development
uv run jupyter lab
```

## 📋 Development Phases

### ✅ Phase 1: Environment Setup (COMPLETE)
- [x] Project structure with uv package manager
- [x] OpenAI API integration
- [x] Configuration management
- [x] Secrets handling
- [x] Testing framework

### 🚧 Phase 2: Intelligent Data Management (NEXT)
- [ ] Yahoo Finance integration
- [ ] Real-time data pipeline
- [ ] Data validation and cleaning
- [ ] Technical indicator calculation

### 📅 Future Phases
- [ ] Advanced decision engine
- [ ] Backtesting integration
- [ ] Performance validation
- [ ] Continuous learning
- [ ] Production deployment

## 🔧 Configuration

The system uses a comprehensive configuration system with these main sections:

- **OpenAI**: Model settings, temperature, token limits
- **Trading**: Position sizing, risk management, stop-losses
- **Data**: Yahoo Finance settings, data validation
- **Validation**: Testing parameters, performance thresholds
- **AI**: Learning parameters, decision making settings

## 🔒 Security

- API keys are managed securely through environment variables
- `.env` files are excluded from git
- Secrets are validated and masked in logs
- Configuration validation ensures safe parameters

## 🧪 Testing

```bash
# Run all tests
uv run python -m pytest tests/

# Test specific components
uv run python tests/test_phase1.py
uv run python tests/test_ai_live.py
```

## 📚 Documentation

- **Planning Notebook**: `notebooks/ai_trading_agent_plan.ipynb` - Complete implementation roadmap
- **Framework**: `notebooks/framework.ipynb` - Existing backtesting system
- **Indicators**: `notebooks/indicators.ipynb` - Technical analysis library

## 🤝 Contributing

This is a personal trading project, but the architecture follows professional standards:

1. Use `uv` for dependency management
2. Follow the existing code structure
3. Add tests for new features
4. Update documentation

## 📈 Performance

The system is designed for:
- **Low latency**: Efficient data processing and AI inference
- **High accuracy**: Professional-grade technical analysis
- **Risk management**: Multiple validation and safety mechanisms
- **Scalability**: Modular architecture for easy extension

## ⚠️ Disclaimer

This software is for educational and research purposes. Always:
- Test thoroughly before live trading
- Start with paper trading
- Understand the risks involved
- Never risk more than you can afford to lose

## 📄 License

Private project - All rights reserved.

---

**Ready to revolutionize your trading with AI! 🎯**
