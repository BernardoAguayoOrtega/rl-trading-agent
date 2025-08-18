import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta

class StrategyEnv(gym.Env):
    """
    Custom Environment for the RL agent to learn how to build trading strategies.
    Supports all pandas-ta indicators with flexible parameter ranges.
    """
    def __init__(self, df):
        super(StrategyEnv, self).__init__()
        
        # Store the historical price data
        self.df = df.copy()
        self.original_df = df.copy() 
        
        # Define action space: 20 discrete actions for comprehensive strategy optimization
        self.action_space = spaces.Discrete(20)
        
        # All pandas-ta indicators grouped by categories for easy access
        self.ta_categories = {
            'overlap': ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'mama', 't3', 'midpoint', 'midprice'],
            'momentum': ['rsi', 'stoch', 'stochrsi', 'ao', 'apo', 'macd', 'ppo', 'roc', 'rsi_stoch', 'ultimate_oscillator', 'williams_r', 'kst'],
            'volume': ['mfi', 'adi', 'obv', 'adosc', 'aobv', 'cmf', 'efi', 'eom', 'nvi', 'pvi', 'volume_sma', 'volume_ema'],
            'volatility': ['bb', 'atr', 'kc', 'dc', 'ui'],
            'trend': ['adx', 'aroon', 'cci', 'dpo', 'ema_fast', 'ema_slow', 'ichimoku', 'kst', 'stc', 'vi', 'wt'],
            'cycle': ['ht_dcperiod', 'ht_dcphase', 'ht_phlead', 'ht_quadralead', 'ht_sine', 'ht_trendline', 'ht_trendmode'],
            'patterns': ['cdl_doji', 'cdl_hammer', 'cdl_engulfing', 'cdl_harami', 'cdl_piercing', 'cdl_morning_star', 'cdl_evening_star', 'cdl_three_white_soldiers', 'cdl_three_black_crows']
        }
        
        # Key indicators with their parameter ranges for optimization
        self.strategy_params = {
            # Moving Averages - Overlap Studies
            'fast_sma': 10,                # SMA Fast (2-50 periods)
            'slow_sma': 20,                # SMA Slow (10-200 periods)  
            'fast_ema': 12,                # EMA Fast (2-50 periods)
            'slow_ema': 26,                # EMA Slow (10-200 periods)
            
            # Momentum Indicators
            'rsi_period': 14,              # RSI period (2-30 periods)
            'rsi_overbought': 70,          # RSI overbought (50-95)
            'rsi_oversold': 30,            # RSI oversold (5-50)
            
            'stoch_k': 14,                 # Stochastic K (5-30)
            'stoch_d': 3,                  # Stochastic D smoothing
            'stoch_overbought': 80,        # Stochastic overbought
            'stoch_oversold': 20,           # Stochastic oversold
            
            # MACD parameters
            'macd_fast': 12,               # MACD fast period (2-30)
            'macd_slow': 26,               # MACD slow period (10-50)
            'macd_signal': 9,              # MACD signal period (2-20)
            
            # Volatility-based Indicators
            'bb_period': 20,                 # Bollinger Bands period (2-50)
            'bb_std': 2.0,                  # Bollinger Bands std dev (0.5-4.0)
            'atr_period': 14,                # ATR period (2-30)
            
            # Volume Indicators
            'volume_sma': 20,                # Volume MA period (5-50)
            'obv_sma': 5,                   # OBV EMA smoothing (2-20)
            
            # ADX Trend Strength
            'adx_period': 14,                # ADX period (2-30)
            'adx_threshold': 25,              # ADX trend threshold (15-50)
            
            # Parabolic SAR
            'sar_start': 0.02,             # SAR acceleration start (0.001-0.2)
            'sar_increment': 0.02,           # SAR acceleration step (0.001-0.2)
            'sar_max': 0.2,                 # SAR acceleration max (0.1-2.0)
        }
        
        # Observation space including all pandas-ta indicators
        self.observation_space = spaces.Dict({
            # Strategy parameters - grouped by indicator types
            'ma_params': spaces.Box(low=2, high=200, shape=(16,), dtype=np.int32),  # Moving average periods
            'oscillator_params': spaces.Box(low=1, high=100, shape=(8,), dtype=np.int32),  # RSI, MACD, Stoch periods
            'volatility_params': spaces.Box(low=0.1, high=5.0, shape=(6,), dtype=np.float32),  # BB, ATR params
            'volume_params': spaces.Box(low=1, high=50, shape=(4,), dtype=np.int32),  # Volume indicators
            'trend_params': spaces.Box(low=0.5, high=2.0, shape=(4,), dtype=np.float32),  # ADX, SAR params
            
            # Current indicator values (latest computed values)
            'sma_fast': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'sma_slow': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'ema_fast': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'ema_slow': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'rsi': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'macd': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'macd_signal': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'stoch_k': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'stoch_d': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'bb_upper': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'bb_middle': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'bb_lower': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'atr': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            'adx': spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            'volume_sma': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            'obv': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            
            # Performance metrics (current state)
            'current_cagr': spaces.Box(low=-1.0, high=3.0, shape=(1,), dtype=np.float32),
            'current_max_drawdown': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_sharpe_ratio': spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32),
            'current_win_rate': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_trades': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            'current_profit_factor': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            'current_expectancy': spaces.Box(low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32),
            
            # Environment state
            'total_backtests': spaces.Box(low=0, high=100000, shape=(1,), dtype=np.int32),
            'consecutive_failures': spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
            'episode_step': spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            
            # Market regime indicators (computed from pandas-ta)
            'trend_strength': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'volatility_regime': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'momentum_regime': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'volume_regime': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Recent market data (last 50 bars normalized)
            'price_trend': spaces.Box(low=-1.0, high=1.0, shape=(50,), dtype=np.float32),
            'volume_trend': spaces.Box(low=-1.0, high=1.0, shape=(50,), dtype=np.float32),
            'rsi_history': spaces.Box(low=0.0, high=100.0, shape=(20,), dtype=np.float32),
            'macd_history': spaces.Box(low=-5.0, high=5.0, shape=(20,), dtype=np.float32),
            
            # Indicator relationships (derived from panda-ta outputs)
            'ma_crossover': spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
            'macd_crossover': spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32),
            'bb_position': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'rsi_position': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Candlestick patterns (from pandas-ta)
            'recent_doji': spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            'recent_engulfing': spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
            'recent_hammer': spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        })
        
        # Performance tracking
        self.performance_metrics = {
            'cagr': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'trades': 0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'consecutive_losses': 0,
            'consecutive_wins': 0
        }
        
        # Initialize technical indicators cache
        self.indicators = {}
        self._compute_all_indicators()
        
        # Environment metrics
        self.metrics = {
            'total_backtests': 0,
            'consecutive_failures': 0,
            'episode_step': 0,
            'max_episode_steps': 100,
            'best_cagr': -float('inf'),
            'best_drawdown': float('inf')
        }

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass