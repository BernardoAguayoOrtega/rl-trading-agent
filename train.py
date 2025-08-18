#!/usr/bin/env python3
"""
RL Trading Agent Training Script

This script trains a reinforcement learning agent to learn optimal trading strategies
using the custom StrategyEnv environment and dynamic trading framework.

Usage:
    python train.py [--data_file path/to/data.csv] [--experiment_name my_experiment] [--timesteps 100000]

Features:
- Dynamic indicator selection and strategy construction
- PPO algorithm with optimized hyperparameters
- Comprehensive logging and monitoring
- Model checkpointing and evaluation callbacks
- Integration with TensorBoard and Weights & Biases
- Automatic hyperparameter tracking
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    CallbackList,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
import torch

# Local imports
from config import *
from rl_agent.StrategyEnv import StrategyEnv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_logging(experiment_name: str, timestamp: str) -> logging.Logger:
    """
    Set up comprehensive logging for the training process
    
    Args:
        experiment_name: Name of the experiment
        timestamp: Timestamp for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory for this experiment
    log_dir = LOGS_DIR / f"{experiment_name}_{timestamp}"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['log_level']),
        format=LOGGING_CONFIG['log_format'],
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting RL Trading Agent Training")
    logger.info(f"📝 Experiment: {experiment_name}")
    logger.info(f"⏰ Timestamp: {timestamp}")
    logger.info(f"📂 Log directory: {log_dir}")
    
    return logger

def load_and_prepare_data(data_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and prepare trading data for training
    
    Args:
        data_file: Path to the CSV data file
        logger: Logger instance
        
    Returns:
        Prepared DataFrame with OHLCV data
    """
    logger.info(f"📊 Loading data from: {data_file}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    logger.info(f"📈 Loaded {len(df):,} rows of data")
    
    # Validate required columns
    required_columns = DATA_CONFIG['required_columns'] + [DATA_CONFIG['date_column']]
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Set date index
    df[DATA_CONFIG['date_column']] = pd.to_datetime(df[DATA_CONFIG['date_column']])
    df.set_index(DATA_CONFIG['date_column'], inplace=True)
    df.index.name = 'Date'  # Ensure consistent naming
    
    # Sort by date
    df.sort_index(inplace=True)
    
    # Remove any NaN values
    initial_len = len(df)
    df.dropna(inplace=True)
    final_len = len(df)
    
    if initial_len != final_len:
        logger.warning(f"⚠️  Removed {initial_len - final_len} rows with NaN values")
    
    # Validate minimum data points
    if len(df) < DATA_CONFIG['min_data_points']:
        raise ValueError(f"Insufficient data: {len(df)} < {DATA_CONFIG['min_data_points']}")
    
    logger.info(f"✅ Data preparation complete: {len(df):,} rows")
    logger.info(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def create_training_environment(df: pd.DataFrame, logger: logging.Logger) -> gym.Env:
    """
    Create and configure the trading environment for training
    
    Args:
        df: Training data DataFrame
        logger: Logger instance
        
    Returns:
        Configured trading environment
    """
    logger.info("🏗️  Creating training environment")
    
    # Create base environment
    env = StrategyEnv(df)
    logger.info(f"✅ Created StrategyEnv with {len(df)} data points")
    
    # Wrap with Monitor for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor_path = LOGS_DIR / f"monitor_{timestamp}.csv"
    env = Monitor(env, str(monitor_path))
    logger.info(f"📊 Added Monitor wrapper, logging to: {monitor_path}")
    
    return env

def create_vectorized_environment(env: gym.Env, logger: logging.Logger) -> VecNormalize:
    """
    Create vectorized and normalized environment
    
    Args:
        env: Base environment
        logger: Logger instance
        
    Returns:
        Vectorized and normalized environment
    """
    logger.info("🔄 Creating vectorized environment")
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    logger.info("✅ Created DummyVecEnv")
    
    # Add normalization if enabled
    if ENV_CONFIG['normalize_observations'] or ENV_CONFIG['normalize_rewards']:
        vec_env = VecNormalize(
            vec_env,
            normalize_obs=ENV_CONFIG['normalize_observations'],
            normalize_reward=ENV_CONFIG['normalize_rewards'],
            gamma=SB3_CONFIG['gamma']
        )
        logger.info("🎯 Added VecNormalize wrapper")
    
    return vec_env

def setup_callbacks(eval_env: gym.Env, experiment_name: str, timestamp: str, logger: logging.Logger) -> CallbackList:
    """
    Set up training callbacks for monitoring and checkpointing
    
    Args:
        eval_env: Environment for evaluation
        experiment_name: Name of the experiment
        timestamp: Timestamp for file naming
        logger: Logger instance
        
    Returns:
        List of configured callbacks
    """
    logger.info("📋 Setting up training callbacks")
    
    callbacks = []
    
    # Evaluation callback
    if CALLBACK_CONFIG['use_eval_callback']:
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=TRAINING_CONFIG['eval_freq'],
            n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
            best_model_save_path=str(MODELS_DIR / f"best_model_{experiment_name}_{timestamp}"),
            log_path=str(LOGS_DIR / f"eval_{experiment_name}_{timestamp}"),
            deterministic=True,
            verbose=1
        )
        callbacks.append(eval_callback)
        logger.info("✅ Added EvalCallback")
    
    # Checkpoint callback
    if CALLBACK_CONFIG['use_checkpoint_callback']:
        checkpoint_callback = CheckpointCallback(
            save_freq=TRAINING_CONFIG['save_freq'],
            save_path=str(MODELS_DIR / f"checkpoints_{experiment_name}_{timestamp}"),
            name_prefix=f"model_{experiment_name}",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        logger.info("✅ Added CheckpointCallback")
    
    # Early stopping callback (optional)
    if CALLBACK_CONFIG['use_stop_training_callback']:
        early_stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=1000,  # Adjust based on your reward scale
            verbose=1
        )
        callbacks.append(early_stop_callback)
        logger.info("✅ Added StopTrainingOnRewardThreshold")
    
    return CallbackList(callbacks)

def create_ppo_model(env: gym.Env, experiment_name: str, timestamp: str, logger: logging.Logger) -> PPO:
    """
    Create and configure PPO model
    
    Args:
        env: Training environment
        experiment_name: Name of the experiment
        timestamp: Timestamp for logging
        logger: Logger instance
        
    Returns:
        Configured PPO model
    """
    logger.info("🧠 Creating PPO model")
    
    # Set device
    device = HARDWARE_CONFIG['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"💻 Using device: {device}")
    
    # Create tensorboard log directory
    tensorboard_log = Path(LOGGING_CONFIG['tensorboard_log']) / f"{experiment_name}_{timestamp}"
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    
    # Create PPO model
    model = PPO(
        policy=SB3_CONFIG['policy'],
        env=env,
        learning_rate=SB3_CONFIG['learning_rate'],
        n_steps=SB3_CONFIG['n_steps'],
        batch_size=SB3_CONFIG['batch_size'],
        n_epochs=SB3_CONFIG['n_epochs'],
        gamma=SB3_CONFIG['gamma'],
        gae_lambda=SB3_CONFIG['gae_lambda'],
        clip_range=SB3_CONFIG['clip_range'],
        ent_coef=SB3_CONFIG['ent_coef'],
        vf_coef=SB3_CONFIG['vf_coef'],
        max_grad_norm=SB3_CONFIG['max_grad_norm'],
        verbose=SB3_CONFIG['verbose'],
        seed=TRAINING_CONFIG['seed'],
        device=device,
        tensorboard_log=str(tensorboard_log),
        policy_kwargs=NETWORK_CONFIG['policy_kwargs']
    )
    
    logger.info("✅ PPO model created successfully")
    logger.info(f"📈 TensorBoard logs: {tensorboard_log}")
    
    # Log model architecture
    logger.info(f"🏗️  Model architecture: {NETWORK_CONFIG['policy_kwargs']['net_arch']}")
    logger.info(f"📊 Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    return model

def setup_wandb(experiment_name: str, timestamp: str, logger: logging.Logger) -> Optional[Any]:
    """
    Set up Weights & Biases logging (optional)
    
    Args:
        experiment_name: Name of the experiment
        timestamp: Timestamp for run naming
        logger: Logger instance
        
    Returns:
        Wandb run object or None
    """
    if not LOGGING_CONFIG['wandb_project']:
        logger.info("📊 W&B logging disabled (no project specified)")
        return None
    
    try:
        import wandb
        
        # Initialize wandb run
        run = wandb.init(
            project=LOGGING_CONFIG['wandb_project'],
            entity=LOGGING_CONFIG['wandb_entity'],
            name=f"{experiment_name}_{timestamp}",
            tags=EXPERIMENT_CONFIG['tags'],
            config={
                **SB3_CONFIG,
                **TRAINING_CONFIG,
                **ENV_CONFIG,
                **NETWORK_CONFIG,
                'experiment_name': experiment_name,
                'timestamp': timestamp
            }
        )
        
        logger.info(f"✅ W&B logging enabled: {run.url}")
        return run
        
    except ImportError:
        logger.warning("⚠️  wandb not installed, skipping W&B logging")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize W&B: {e}")
        return None

def train_model(
    model: PPO, 
    callbacks: CallbackList,
    experiment_name: str,
    timestamp: str,
    logger: logging.Logger,
    wandb_run: Optional[Any] = None
) -> PPO:
    """
    Train the PPO model
    
    Args:
        model: PPO model to train
        callbacks: Training callbacks
        experiment_name: Name of the experiment
        timestamp: Timestamp for naming
        logger: Logger instance
        wandb_run: Optional W&B run object
        
    Returns:
        Trained model
    """
    logger.info("🏋️  Starting model training")
    logger.info(f"⏱️  Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    
    start_time = datetime.now()
    
    try:
        # Train the model
        model.learn(
            total_timesteps=TRAINING_CONFIG['total_timesteps'],
            callback=callbacks,
            log_interval=TRAINING_CONFIG['log_interval'],
            tb_log_name=f"ppo_{experiment_name}_{timestamp}",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"⏱️  Training time: {training_time}")
        
        # Save final model
        final_model_path = get_model_path(experiment_name, timestamp)
        model.save(final_model_path)
        logger.info(f"💾 Final model saved: {final_model_path}")
        
        # Log training completion to W&B
        if wandb_run:
            wandb_run.log({
                "training_time_seconds": training_time.total_seconds(),
                "final_timesteps": TRAINING_CONFIG['total_timesteps']
            })
        
        return model
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        
        # Save current model
        interrupted_model_path = MODELS_DIR / f"interrupted_{experiment_name}_{timestamp}.zip"
        model.save(interrupted_model_path)
        logger.info(f"💾 Interrupted model saved: {interrupted_model_path}")
        
        raise
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def save_training_artifacts(
    model: PPO,
    env: gym.Env, 
    experiment_name: str,
    timestamp: str,
    config_dict: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """
    Save training artifacts and metadata
    
    Args:
        model: Trained model
        env: Training environment
        experiment_name: Name of the experiment
        timestamp: Timestamp for naming
        config_dict: Configuration dictionary
        logger: Logger instance
    """
    logger.info("💾 Saving training artifacts")
    
    # Create artifacts directory
    artifacts_dir = RESULTS_DIR / f"{experiment_name}_{timestamp}"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = artifacts_dir / "config.json"
    import json
    with open(config_path, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_config = {}
        for key, value in config_dict.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_config[key] = value
            except (TypeError, ValueError):
                serializable_config[key] = str(value)
        
        json.dump(serializable_config, f, indent=2, default=str)
    
    logger.info(f"⚙️  Configuration saved: {config_path}")
    
    # Save environment normalization stats if applicable
    if hasattr(env, 'save'):
        normalize_path = artifacts_dir / "vec_normalize.pkl"
        env.save(normalize_path)
        logger.info(f"📊 Normalization stats saved: {normalize_path}")
    
    # Save model summary
    summary_path = artifacts_dir / "model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Algorithm: {SB3_CONFIG['algorithm']}\n")
        f.write(f"Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}\n")
        f.write(f"Network Architecture: {NETWORK_CONFIG['policy_kwargs']['net_arch']}\n")
        f.write(f"Learning Rate: {SB3_CONFIG['learning_rate']}\n")
        f.write(f"Batch Size: {SB3_CONFIG['batch_size']}\n")
        f.write(f"Training Time: {datetime.now().isoformat()}\n")
    
    logger.info(f"📋 Model summary saved: {summary_path}")
    logger.info(f"🎯 All artifacts saved to: {artifacts_dir}")

def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL Trading Agent")
    parser.add_argument(
        "--data_file", 
        type=str,
        help="Path to CSV data file for training"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=EXPERIMENT_CONFIG['experiment_name'],
        help="Name for this training experiment"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAINING_CONFIG['total_timesteps'],
        help="Total training timesteps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAINING_CONFIG['seed'],
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.timesteps != TRAINING_CONFIG['total_timesteps']:
        TRAINING_CONFIG['total_timesteps'] = args.timesteps
    
    if args.seed != TRAINING_CONFIG['seed']:
        TRAINING_CONFIG['seed'] = args.seed
    
    if args.debug:
        DEBUG_CONFIG['debug_mode'] = True
        LOGGING_CONFIG['log_level'] = 'DEBUG'
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up logging
    logger = setup_logging(args.experiment_name, timestamp)
    
    try:
        # Validate configuration
        validate_config()
        
        # Determine data file
        if args.data_file:
            data_file = Path(args.data_file)
        else:
            # Use default or ask user to specify
            if DEFAULT_DATA_FILES['train'].exists():
                data_file = DEFAULT_DATA_FILES['train']
            else:
                logger.error("❌ No data file specified and default not found")
                logger.error(f"Please provide --data_file or place data at {DEFAULT_DATA_FILES['train']}")
                return 1
        
        # Load and prepare data
        df = load_and_prepare_data(data_file, logger)
        
        # Create training environment
        train_env = create_training_environment(df, logger)
        
        # Create vectorized environment
        vec_env = create_vectorized_environment(train_env, logger)
        
        # Create evaluation environment (using same data for now)
        eval_env = create_training_environment(df, logger)
        eval_vec_env = create_vectorized_environment(eval_env, logger)
        
        # Set up callbacks
        callbacks = setup_callbacks(eval_vec_env, args.experiment_name, timestamp, logger)
        
        # Create PPO model
        model = create_ppo_model(vec_env, args.experiment_name, timestamp, logger)
        
        # Set up W&B (optional)
        wandb_run = setup_wandb(args.experiment_name, timestamp, logger)
        
        # Train the model
        trained_model = train_model(
            model, callbacks, args.experiment_name, timestamp, logger, wandb_run
        )
        
        # Save training artifacts
        config_dict = {
            'sb3_config': SB3_CONFIG,
            'training_config': TRAINING_CONFIG,
            'env_config': ENV_CONFIG,
            'network_config': NETWORK_CONFIG,
            'data_file': str(data_file),
            'data_shape': df.shape
        }
        
        save_training_artifacts(
            trained_model, vec_env, args.experiment_name, 
            timestamp, config_dict, logger
        )
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("🚀 Ready for evaluation and deployment!")
        
        # Close W&B run
        if wandb_run:
            wandb_run.finish()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if DEBUG_CONFIG['debug_mode']:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1
        
    finally:
        # Clean up
        try:
            if 'vec_env' in locals():
                vec_env.close()
            if 'eval_vec_env' in locals():
                eval_vec_env.close()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
