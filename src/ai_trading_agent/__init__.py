"""
AI Trading Agent Package
========================

An intelligent algorithmic trading system that leverages OpenAI's GPT models
to make sophisticated trading decisions based on technical analysis, market
conditions, and risk assessment.
"""

__version__ = "0.3.0"
__author__ = "Bernardo Aguayo Ortega"

# Import main components
from .config import Config, OpenAIConfig, TradingConfig, DataConfig, ValidationConfig, AIConfig
from .secrets import SecretsManager, get_secret
from .ai_client import AITradingClient

# Phase 2 components - Intelligent Data Management
from .data_manager import IntelligentDataManager, DataValidationError
from .indicators import TechnicalIndicatorFactory, MarketDataProcessor

# Phase 3 components - Framework Integration
from .trading_engine import AITradingEngine, TradingDecision, Position
from .trading_signals import AITradingSignalGenerator, AISignal
from .backtesting_integration import AIBacktestingIntegration

# Main entry point
def main():
    """Main entry point for the AI Trading Agent"""
    print(f"🚀 AI Trading Agent v{__version__}")
    print("Ready to revolutionize algorithmic trading with AI!")

__all__ = [
    "Config",
    "OpenAIConfig", 
    "TradingConfig",
    "DataConfig",
    "ValidationConfig",
    "AIConfig",
    "SecretsManager",
    "get_secret",
    "AITradingClient",
    "IntelligentDataManager",
    "DataValidationError",
    "TechnicalIndicatorFactory",
    "MarketDataProcessor",
    "AITradingEngine",
    "TradingDecision",
    "Position",
    "AITradingSignalGenerator",
    "AISignal",
    "AIBacktestingIntegration",
    "main"
]
