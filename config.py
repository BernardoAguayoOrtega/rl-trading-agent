"""
Configuration file for RL Trading Agent

This file contains all hyperparameters, paths, and configuration settings
for training, evaluation, and testing the reinforcement learning trading agent.
"""

import os
from pathlib import Path

# =====================================
# PROJECT PATHS
# =====================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

# =====================================
# DATA CONFIGURATION
# =====================================

# Sample data files (these should be replaced with your actual data files)
DEFAULT_DATA_FILES = {
    'train': DATA_DIR / 'train_data.csv',
    'validation': DATA_DIR / 'validation_data.csv', 
    'test': DATA_DIR / 'test_data.csv'
}

# Data preprocessing
DATA_CONFIG = {
    'date_column': 'Date',
    'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'min_data_points': 1000,  # Minimum rows required for training
    'train_split': 0.7,       # 70% for training
    'validation_split': 0.2,  # 20% for validation  
    'test_split': 0.1,        # 10% for testing
}

# =====================================
# TRAINING CONFIGURATION
# =====================================

# Stable-Baselines3 Algorithm Settings
SB3_CONFIG = {
    'algorithm': 'PPO',  # Proximal Policy Optimization
    'policy': 'MultiInputPolicy',  # For Dict observation spaces
    'learning_rate': 3e-4,
    'n_steps': 2048,      # Number of steps to collect before updating
    'batch_size': 64,     # Minibatch size
    'n_epochs': 10,       # Number of epochs when optimizing the surrogate loss
    'gamma': 0.99,        # Discount factor
    'gae_lambda': 0.95,   # GAE (Generalized Advantage Estimation) parameter
    'clip_range': 0.2,    # Clipping parameter
    'ent_coef': 0.0,      # Entropy coefficient for exploration
    'vf_coef': 0.5,       # Value function coefficient
    'max_grad_norm': 0.5, # Maximum gradient norm
    'verbose': 1,         # Verbosity level
}

# Training hyperparameters
TRAINING_CONFIG = {
    'total_timesteps': 100_000,    # Total training steps
    'eval_freq': 5_000,            # Evaluate every N steps
    'n_eval_episodes': 10,         # Number of episodes for evaluation
    'save_freq': 10_000,           # Save model every N steps
    'log_interval': 100,           # Log progress every N steps
    'seed': 42,                    # Random seed for reproducibility
}

# Environment settings
ENV_CONFIG = {
    'max_episode_steps': 100,      # Maximum steps per episode
    'reward_scaling': 1.0,         # Scale rewards by this factor
    'normalize_observations': True, # Whether to normalize observations
    'normalize_rewards': True,     # Whether to normalize rewards
}

# =====================================
# MODEL ARCHITECTURE
# =====================================

# Neural network architecture for the policy and value functions
NETWORK_CONFIG = {
    'policy_kwargs': {
        'net_arch': [256, 256, 128],  # Hidden layer sizes
        'activation_fn': 'relu',       # Activation function
        'features_extractor_kwargs': {
            'features_dim': 128,       # Feature extraction dimension
        }
    }
}

# =====================================
# EVALUATION CONFIGURATION  
# =====================================

EVALUATION_CONFIG = {
    'n_episodes': 50,              # Episodes for comprehensive evaluation
    'render_mode': 'rgb_array',    # Rendering mode for visualization
    'save_trajectories': True,     # Save episode trajectories
    'generate_plots': True,        # Generate performance plots
    'metrics': [                   # Metrics to calculate
        'cagr', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
        'win_rate', 'profit_factor', 'calmar_ratio', 'expectancy'
    ]
}

# Benchmarking
BENCHMARK_CONFIG = {
    'strategies': [
        'buy_and_hold',
        'simple_moving_average', 
        'rsi_mean_reversion',
        'random_strategy'
    ],
    'benchmark_period': 'same_as_test'  # or specific date range
}

# =====================================
# GARCH TESTING CONFIGURATION
# =====================================

GARCH_CONFIG = {
    'model_type': 'GARCH',         # GARCH, EGARCH, GJR-GARCH
    'p': 1,                        # GARCH order
    'q': 1,                        # ARCH order
    'vol_target': None,            # Target volatility (None for historical)
    'n_simulations': 1000,         # Monte Carlo simulations
    'forecast_horizon': 252,       # Days to forecast (1 year)
    'confidence_levels': [0.95, 0.99],  # VaR confidence levels
    'stress_scenarios': {
        'low_volatility': 0.5,     # 50% of historical volatility
        'normal_volatility': 1.0,  # Historical volatility  
        'high_volatility': 2.0,    # 200% of historical volatility
        'crisis_volatility': 3.0   # 300% of historical volatility (crisis)
    }
}

# =====================================
# LOGGING AND MONITORING
# =====================================

LOGGING_CONFIG = {
    'tensorboard_log': str(LOGS_DIR / "tensorboard"),
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'wandb_project': None,  # Set to project name to enable Weights & Biases logging
    'wandb_entity': None,   # Your wandb username/entity
}

# Callback configuration
CALLBACK_CONFIG = {
    'use_eval_callback': True,
    'use_checkpoint_callback': True,
    'use_stop_training_callback': False,
    'early_stopping': {
        'patience': 20,            # Stop if no improvement for N evaluations
        'min_improvement': 0.01,   # Minimum improvement threshold
    }
}

# =====================================
# VISUALIZATION SETTINGS
# =====================================

PLOT_CONFIG = {
    'figsize': (12, 8),
    'style': 'seaborn-v0_8',  # Matplotlib style
    'save_format': 'png',      # Format for saved plots
    'dpi': 300,               # Resolution for saved plots
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728'
    }
}

# =====================================
# EXPERIMENT TRACKING
# =====================================

EXPERIMENT_CONFIG = {
    'experiment_name': 'rl_trading_v1',
    'track_hyperparameters': True,
    'save_model_artifacts': True,
    'auto_version': True,      # Automatically version experiments
    'tags': ['reinforcement-learning', 'trading', 'pandas-ta'],
}

# =====================================
# HARDWARE CONFIGURATION
# =====================================

HARDWARE_CONFIG = {
    'device': 'auto',          # 'auto', 'cpu', 'cuda'
    'n_envs': 1,              # Number of parallel environments
    'n_jobs': -1,             # Number of parallel jobs (-1 = all cores)
}

# =====================================
# RISK MANAGEMENT
# =====================================

RISK_CONFIG = {
    'max_position_size': 1.0,     # Maximum position size
    'max_drawdown_stop': 0.20,    # Stop trading if drawdown > 20%
    'volatility_target': 0.15,    # Target annualized volatility
    'risk_free_rate': 0.02,       # Risk-free rate for Sharpe calculation
    'lookback_period': 252,       # Days for risk calculations
}

# =====================================
# DEBUGGING AND DEVELOPMENT
# =====================================

DEBUG_CONFIG = {
    'debug_mode': False,           # Enable debug mode
    'profile_training': False,     # Profile training performance
    'save_intermediate_models': False,  # Save models during training
    'detailed_logging': False,     # Enable detailed logging
    'render_during_training': False,  # Render environment during training
}

# =====================================
# UTILITY FUNCTIONS
# =====================================

def get_model_path(experiment_name=None, timestamp=None):
    """Get the path for saving/loading models"""
    if experiment_name is None:
        experiment_name = EXPERIMENT_CONFIG['experiment_name']
    
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return MODELS_DIR / f"{experiment_name}_{timestamp}.zip"

def get_log_path(experiment_name=None, timestamp=None):
    """Get the path for saving logs"""
    if experiment_name is None:
        experiment_name = EXPERIMENT_CONFIG['experiment_name']
    
    if timestamp is None:
        from datetime import datetime  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return LOGS_DIR / f"{experiment_name}_{timestamp}"

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check data configuration
    if DATA_CONFIG['train_split'] + DATA_CONFIG['validation_split'] + DATA_CONFIG['test_split'] != 1.0:
        errors.append("Data splits must sum to 1.0")
    
    # Check training configuration
    if TRAINING_CONFIG['total_timesteps'] <= 0:
        errors.append("total_timesteps must be positive")
    
    # Check if required directories exist
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
        if not directory.exists():
            directory.mkdir(exist_ok=True)
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    print("✅ Configuration validation passed")
    return True

# =====================================
# ENVIRONMENT VARIABLES OVERRIDE
# =====================================

# Allow environment variables to override configuration
def load_env_overrides():
    """Load environment variable overrides"""
    
    # Training overrides
    if os.getenv('RL_TOTAL_TIMESTEPS'):
        TRAINING_CONFIG['total_timesteps'] = int(os.getenv('RL_TOTAL_TIMESTEPS'))
    
    if os.getenv('RL_LEARNING_RATE'):
        SB3_CONFIG['learning_rate'] = float(os.getenv('RL_LEARNING_RATE'))
    
    if os.getenv('RL_SEED'):
        TRAINING_CONFIG['seed'] = int(os.getenv('RL_SEED'))
    
    # Logging overrides
    if os.getenv('WANDB_PROJECT'):
        LOGGING_CONFIG['wandb_project'] = os.getenv('WANDB_PROJECT')
    
    if os.getenv('WANDB_ENTITY'):
        LOGGING_CONFIG['wandb_entity'] = os.getenv('WANDB_ENTITY')
    
    # Debug overrides
    if os.getenv('RL_DEBUG'):
        DEBUG_CONFIG['debug_mode'] = os.getenv('RL_DEBUG').lower() == 'true'

# Load environment overrides at import time
load_env_overrides()

# Validate configuration at import time
if __name__ == "__main__":
    validate_config()
    print("🔧 Configuration loaded successfully")
    
    # Print key settings
    print(f"📊 Training timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"🧠 Algorithm: {SB3_CONFIG['algorithm']}")
    print(f"📁 Models directory: {MODELS_DIR}")
    print(f"📈 Logging directory: {LOGS_DIR}")
