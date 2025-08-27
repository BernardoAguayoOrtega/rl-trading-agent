"""
Configuration Management for AI Trading Agent
===========================================

This module handles all configuration settings including:
- OpenAI API configuration
- Trading parameters
- Data source settings
- Risk management parameters
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from .secrets import get_secret


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    validate_on_init: bool = True
    
    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.api_key is None:
            # Use secure secrets manager to get API key
            self.api_key = get_secret("OPENAI_API_KEY")
        
        if not self.api_key and self.validate_on_init:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or provide it directly in the configuration."
            )


@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.2  # Maximum 20% of capital per position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit (2:1 R:R)
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage
    
    # Risk management
    max_daily_trades: int = 10
    max_portfolio_risk: float = 0.15  # Maximum 15% portfolio at risk
    confidence_threshold: float = 0.7  # Minimum AI confidence to trade


@dataclass
class DataConfig:
    """Data source configuration"""
    default_symbol: str = "SPY"
    default_period: str = "2y"  # 2 years of data
    default_interval: str = "1d"  # Daily data
    
    # Yahoo Finance settings
    yf_auto_adjust: bool = True
    yf_prepost: bool = False
    
    # Data validation
    min_data_points: int = 50  # Minimum 50 data points (about 2 months daily data)


@dataclass
class ValidationConfig:
    """Validation and testing configuration"""
    train_test_split: float = 0.7  # 70% training, 30% testing
    walk_forward_periods: int = 10
    monte_carlo_simulations: int = 1000
    cross_validation_folds: int = 5
    
    # Performance thresholds
    min_sharpe_ratio: float = 1.0
    max_drawdown_threshold: float = 0.15  # 15%
    min_win_rate: float = 0.55  # 55%


@dataclass
class AIConfig:
    """AI-specific configuration"""
    confidence_calibration: bool = True
    learning_rate: float = 0.01
    memory_length: int = 100  # Remember last 100 decisions
    strategy_switching_threshold: float = 0.3  # Switch strategy if performance drops 30%
    
    # Decision making
    multi_timeframe_analysis: bool = True
    market_regime_detection: bool = True
    risk_assessment_enabled: bool = True


@dataclass
class Config:
    """Main configuration class combining all settings"""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        return cls(
            openai=OpenAIConfig(**config_dict.get('openai', {})),
            trading=TradingConfig(**config_dict.get('trading', {})),
            data=DataConfig(**config_dict.get('data', {})),
            validation=ValidationConfig(**config_dict.get('validation', {})),
            ai=AIConfig(**config_dict.get('ai', {}))
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate OpenAI configuration
        if not self.openai.api_key:
            errors.append("OpenAI API key is required")
        
        # Validate trading parameters
        if self.trading.max_position_size > 1.0:
            errors.append("Maximum position size cannot exceed 100%")
        
        if self.trading.stop_loss_pct <= 0:
            errors.append("Stop loss percentage must be positive")
        
        if self.trading.take_profit_pct <= self.trading.stop_loss_pct:
            errors.append("Take profit must be greater than stop loss")
        
        # Validate data configuration
        if self.data.min_data_points < 50:
            errors.append("Minimum data points should be at least 50")
        
        # Validate validation configuration
        if not (0 < self.validation.train_test_split < 1):
            errors.append("Train/test split must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation errors: {'; '.join(errors)}")
        
        return True


# Default configuration instance with validation disabled for testing
default_config = Config(
    openai=OpenAIConfig(validate_on_init=False)
)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    config.validate()
    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    # Remove API key for security
    if 'api_key' in config_dict.get('openai', {}):
        config_dict['openai']['api_key'] = "[REDACTED]"
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
