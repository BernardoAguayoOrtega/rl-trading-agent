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
        
        # Replace the current action_space with this more structured one
        self.action_space = spaces.Dict({
            'overlap_actions': spaces.Box(low=2, high=200, shape=(4,), dtype=np.int32),  # For MA periods
            'momentum_actions': spaces.Box(low=1, high=100, shape=(4,), dtype=np.int32),  # For momentum indicator periods
            'volatility_actions': spaces.Box(low=0.1, high=5.0, shape=(3,), dtype=np.float32),  # For volatility parameters
            'trend_actions': spaces.Box(low=0.5, high=2.0, shape=(2,), dtype=np.float32),  # For trend indicator parameters
            'cycle_actions': spaces.Box(low=0.1, high=2.0, shape=(2,), dtype=np.float32)  # For cycle indicator parameters
        })
        
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
        """
        Executes one time step within the environment.
        """
        self.metrics['episode_step'] += 1
        terminated = False
        
        # --- 1. APPLY ACTION: Update strategy parameters based on agent's action ---
        # The agent's action is a dictionary with new parameter values.
        # We'll update our internal strategy_params dictionary.
        param_keys = list(self.strategy_params.keys())
        
        # Unpack actions and update the corresponding parameters
        ma_action_values = action['overlap_actions']
        # ... (similar unpacking for other action groups) ...

        # Note: A more complete implementation would map each action value
        # to a specific parameter. For now, we'll assume a direct mapping.
        # This part will become more complex as you refine the agent's control.
        self.strategy_params['fast_sma'] = ma_action_values[0]
        self.strategy_params['slow_sma'] = ma_action_values[1]
        
        # --- 2. RUN BACKTEST: Execute your original trading framework ---
        # For simplicity, we'll make every step a backtest.
        terminated = True 
        
        try:
            # Re-compute indicators with new parameters
            self._compute_all_indicators()

            # Create a clean copy of the data for the backtest
            data_copy = self.df.copy()

            # A simplified example of how you would call your framework:
            # This part needs to be adapted to create a general-purpose signal generator
            # based on the agent's chosen parameters.
            perSma_fast = self.strategy_params['fast_sma']
            perSma_slow = self.strategy_params['slow_sma']
            
            # This is a placeholder for a much more complex signal generation logic
            # that the agent will eventually learn to create.
            data_copy['signal'] = np.where(data_copy[f'SMA_{perSma_fast}'] > data_copy[f'SMA_{perSma_slow}'], 'P', '')
            data_copy['signal'] = np.where(data_copy[f'SMA_{perSma_fast}'] < data_copy[f'SMA_{perSma_slow}'], 'cP', data_copy.signal)
            data_copy['position'] = data_copy.signal.shift()
            
            # Call your existing framework functions for backtesting and performance calculation
            from trading_framework import damePosition, dameSalidaVelas, dameSalidaPnl, calculaCurvas, backSistemaList
            
            data_copy = damePosition(data_copy)
            data_copy = dameSalidaVelas(data_copy, 0) # No candle exit limit
            data_copy = dameSalidaPnl(data_copy, 'long', 0, 0, 0, 0) # No TP/SL
            data_copy = calculaCurvas(data_copy, size=1)
            results = backSistemaList(data_copy)

            # --- 3. CALCULATE REWARD ---
            reward = self._calculate_reward(results)
            
            # --- 4. UPDATE OBSERVATION with the new results ---
            # This would involve creating the full observation dictionary as in reset()
            # but with the new performance metrics from the 'results' list.
            observation = self.reset(seed=None)[0] # Placeholder: reset for next episode
            
        except Exception as e:
            reward = -200  # Penalize heavily if the backtest fails
            observation = self.reset(seed=None)[0] # Reset on failure

        return observation, reward, terminated, False, {}

    def _calculate_reward(self, results):
        """
        Calculates a reward score based on the backtest results list.
        """
        try:
            # Extract metrics from your backSistemaList's output
            num_ops = results[2]
            cagr = results[12]
            max_dd = abs(results[21])
            profit_factor = results[18]

            # Rule-based penalties for poor strategies
            if num_ops < 20 or cagr < 1.0 or max_dd > 80.0:
                return -100.0

            # The core reward function: Calmar Ratio (CAGR / Max Drawdown)
            # We add a small bonus for a good profit factor.
            calmar_ratio = cagr / max_dd if max_dd > 0 else cagr
            reward = (calmar_ratio * 100) + (profit_factor * 10)
            
            return reward
        
        except (IndexError, TypeError, ZeroDivisionError):
            return -200.0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state for a new episode.
        """
        super().reset(seed=seed)

        # Reset episode-specific metrics
        self.metrics['episode_step'] = 0
        self.metrics['consecutive_failures'] = 0
        
        # --- Create the Initial Observation ---
        # This dictionary MUST match the structure of self.observation_space
        
        # Start with default strategy parameters
        initial_params = self.strategy_params.copy()

        # Helper function to safely get slices of historical data
        def get_initial_history(key, length, default_val=0.0):
            if key in self.df.columns and len(self.df) >= length:
                return self.df[key].iloc[:length].values.astype(np.float32)
            return np.full(length, default_val, dtype=np.float32)

        observation = {
            # Strategy parameters are set to their defaults
            'ma_params': np.array(list(initial_params.values())[:16], dtype=np.int32),
            'oscillator_params': np.array(list(initial_params.values())[16:24], dtype=np.int32),
            'volatility_params': np.array(list(initial_params.values())[24:30], dtype=np.float32),
            'volume_params': np.array(list(initial_params.values())[30:34], dtype=np.int32),
            'trend_params': np.array(list(initial_params.values())[34:38], dtype=np.float32),
            
            # Current indicator values are initialized to 0 or a safe value
            'sma_fast': np.array([0.0], dtype=np.float32), 'sma_slow': np.array([0.0], dtype=np.float32),
            'ema_fast': np.array([0.0], dtype=np.float32), 'ema_slow': np.array([0.0], dtype=np.float32),
            'rsi': np.array([50.0], dtype=np.float32), 'macd': np.array([0.0], dtype=np.float32),
            'macd_signal': np.array([0.0], dtype=np.float32), 'stoch_k': np.array([50.0], dtype=np.float32),
            'stoch_d': np.array([50.0], dtype=np.float32), 'bb_upper': np.array([0.0], dtype=np.float32),
            'bb_middle': np.array([0.0], dtype=np.float32), 'bb_lower': np.array([0.0], dtype=np.float32),
            'atr': np.array([0.0], dtype=np.float32), 'adx': np.array([0.0], dtype=np.float32),
            'volume_sma': np.array([0.0], dtype=np.float32), 'obv': np.array([0.0], dtype=np.float32),

            # Performance metrics start at 0
            'current_cagr': np.array([0.0], dtype=np.float32), 'current_max_drawdown': np.array([0.0], dtype=np.float32),
            'current_sharpe_ratio': np.array([0.0], dtype=np.float32), 'current_win_rate': np.array([0.0], dtype=np.float32),
            'current_trades': np.array([0], dtype=np.int32), 'current_profit_factor': np.array([0.0], dtype=np.float32),
            'current_expectancy': np.array([0.0], dtype=np.float32),
            
            # Environment state starts at 0
            'total_backtests': np.array([self.metrics['total_backtests']], dtype=np.int32),
            'consecutive_failures': np.array([0], dtype=np.int32),
            'episode_step': np.array([0], dtype=np.int32),
            
            # Market regime and history are taken from the beginning of the dataset
            'trend_strength': np.array([0.0], dtype=np.float32), 'volatility_regime': np.array([0.0], dtype=np.float32),
            'momentum_regime': np.array([0.0], dtype=np.float32), 'volume_regime': np.array([0.0], dtype=np.float32),
            'price_trend': get_initial_history('Close', 50), 'volume_trend': get_initial_history('Volume', 50),
            'rsi_history': get_initial_history(f"RSI_{self.strategy_params['rsi_period']}", 20, 50.0),
            'macd_history': get_initial_history(f"MACDh_{self.strategy_params['macd_fast']}_{self.strategy_params['macd_slow']}_{self.strategy_params['macd_signal']}", 20),
            
            # Relationships and patterns start at neutral values
            'ma_crossover': np.array([0.0], dtype=np.float32), 'macd_crossover': np.array([0.0], dtype=np.float32),
            'bb_position': np.array([0.5], dtype=np.float32), 'rsi_position': np.array([0.5], dtype=np.float32),
            'recent_doji': np.zeros(5, dtype=np.float32), 'recent_engulfing': np.zeros(5, dtype=np.float32),
            'recent_hammer': np.zeros(5, dtype=np.float32)
        }
        
        # Return the initial observation and an empty info dictionary
        return observation, {}

    def render(self, mode='human'):
        """
        Render the environment for visualization
        
        Args:
            mode (str): Rendering mode
                - 'human': Display in a matplotlib window
                - 'rgb_array': Return image array
                - 'ansi': Print text representation
        
        Returns:
            None or rgb array depending on mode
        """
        if mode == 'ansi':
            # Text-based rendering
            self._render_text()
        elif mode in ['human', 'rgb_array']:
            # Graphical rendering
            return self._render_plot(mode)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_text(self):
        """Render environment state as text"""
        print("\n" + "="*60)
        print(f"STRATEGY ENVIRONMENT - Episode Step: {self.metrics['episode_step']}")
        print("="*60)
        
        # Current strategy parameters
        print("\nCURRENT STRATEGY PARAMETERS:")
        print(f"  Fast SMA: {self.strategy_params['fast_sma']}")
        print(f"  Slow SMA: {self.strategy_params['slow_sma']}")
        print(f"  RSI Period: {self.strategy_params['rsi_period']}")
        print(f"  MACD: ({self.strategy_params['macd_fast']}, {self.strategy_params['macd_slow']}, {self.strategy_params['macd_signal']})")
        
        # Current indicator values
        print("\nCURRENT INDICATORS:")
        if hasattr(self, 'current_indicators'):
            for key, value in self.current_indicators.items():
                print(f"  {key.upper()}: {value:.4f}")
        
        # Performance metrics
        print("\nPERFORMANCE METRICS:")
        for key, value in self.performance_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        
        # Environment metrics
        print("\nENVIRONMENT STATE:")
        print(f"  Total Backtests: {self.metrics['total_backtests']}")
        print(f"  Consecutive Failures: {self.metrics['consecutive_failures']}")
        print(f"  Best CAGR: {self.metrics['best_cagr']:.4f}")
        print(f"  Best Drawdown: {self.metrics['best_drawdown']:.4f}")
        print("="*60)
    
    def _render_plot(self, mode='human'):
        """Render environment state as plots"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'RL Trading Environment - Step {self.metrics["episode_step"]}', fontsize=16)
            
            # Subplot 1: Price and Moving Averages
            ax1 = axes[0, 0]
            if len(self.df) > 0:
                # Plot recent price data (last 100 periods)
                recent_data = self.df.tail(100).copy()
                ax1.plot(recent_data.index, recent_data['Close'], 'k-', label='Close', linewidth=1.5)
                
                # Plot moving averages if they exist
                fast_sma_col = f'SMA_{self.strategy_params["fast_sma"]}'
                slow_sma_col = f'SMA_{self.strategy_params["slow_sma"]}'
                
                if fast_sma_col in recent_data.columns:
                    ax1.plot(recent_data.index, recent_data[fast_sma_col], 'b--', 
                            label=f'SMA{self.strategy_params["fast_sma"]}', alpha=0.7)
                
                if slow_sma_col in recent_data.columns:
                    ax1.plot(recent_data.index, recent_data[slow_sma_col], 'r--', 
                            label=f'SMA{self.strategy_params["slow_sma"]}', alpha=0.7)
                
                ax1.set_title('Price & Moving Averages')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Subplot 2: RSI
            ax2 = axes[0, 1]
            rsi_col = f'RSI_{self.strategy_params["rsi_period"]}'
            if len(self.df) > 0 and rsi_col in self.df.columns:
                recent_data = self.df.tail(100).copy()
                ax2.plot(recent_data.index, recent_data[rsi_col], 'purple', linewidth=1.5)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                ax2.set_title(f'RSI ({self.strategy_params["rsi_period"]})')
                ax2.set_ylim(0, 100)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Subplot 3: MACD
            ax3 = axes[1, 0]
            macd_cols = [col for col in self.df.columns if 'MACD' in col]
            if len(self.df) > 0 and len(macd_cols) > 0:
                recent_data = self.df.tail(100).copy()
                for col in macd_cols[:2]:  # Plot first 2 MACD columns
                    ax3.plot(recent_data.index, recent_data[col], label=col, linewidth=1.5)
                ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax3.set_title('MACD')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Subplot 4: Performance Metrics Bar Chart
            ax4 = axes[1, 1]
            metrics_to_plot = ['cagr', 'max_drawdown', 'sharpe_ratio', 'win_rate']
            metric_values = [self.performance_metrics.get(m, 0) for m in metrics_to_plot]
            metric_labels = [m.replace('_', ' ').title() for m in metrics_to_plot]
            
            colors = ['green' if v > 0 else 'red' for v in metric_values]
            bars = ax4.bar(metric_labels, metric_values, color=colors, alpha=0.7)
            ax4.set_title('Performance Metrics')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Subplot 5: Strategy Parameters
            ax5 = axes[2, 0]
            param_names = ['Fast SMA', 'Slow SMA', 'RSI Period', 'MACD Fast', 'MACD Slow']
            param_values = [
                self.strategy_params['fast_sma'],
                self.strategy_params['slow_sma'],
                self.strategy_params['rsi_period'],
                self.strategy_params['macd_fast'],
                self.strategy_params['macd_slow']
            ]
            
            bars = ax5.barh(param_names, param_values, color='skyblue', alpha=0.7)
            ax5.set_title('Current Strategy Parameters')
            ax5.set_xlabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, param_values):
                width = bar.get_width()
                ax5.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{value}', ha='left', va='center', fontsize=9)
            
            # Subplot 6: Environment Stats
            ax6 = axes[2, 1]
            env_stats = [
                ('Total Backtests', self.metrics['total_backtests']),
                ('Episode Step', self.metrics['episode_step']),
                ('Consecutive Failures', self.metrics['consecutive_failures']),
                ('Best CAGR', f"{self.metrics['best_cagr']:.2f}"),
                ('Best Drawdown', f"{self.metrics['best_drawdown']:.2f}")
            ]
            
            ax6.axis('off')
            ax6.set_title('Environment Statistics', pad=20)
            
            # Create a table-like display
            table_text = []
            for stat_name, stat_value in env_stats:
                table_text.append([stat_name, str(stat_value)])
            
            table = ax6.table(cellText=table_text,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(env_stats) + 1):  # +1 for header
                for j in range(2):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            plt.tight_layout()
            
            if mode == 'human':
                plt.show(block=False)
                plt.pause(0.1)  # Brief pause to allow display
                return None
            elif mode == 'rgb_array':
                # Convert plot to RGB array
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return buf
                
        except ImportError:
            print("Warning: matplotlib not available for plotting. Install with: pip install matplotlib")
            self._render_text()
        except Exception as e:
            print(f"Warning: Error in rendering: {e}")
            self._render_text()

    def close(self):
        """
        Clean up environment resources
        
        This method is called when the environment is no longer needed.
        It should close any open resources, plots, files, etc.
        """
        try:
            # Close any matplotlib figures that might be open
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass  # matplotlib not available, nothing to close
        except Exception as e:
            print(f"Warning: Error closing matplotlib figures: {e}")
        
        # Clear large data structures to free memory
        if hasattr(self, 'df'):
            del self.df
        if hasattr(self, 'original_df'):
            del self.original_df
        if hasattr(self, 'indicators'):
            self.indicators.clear()
        if hasattr(self, 'current_indicators'):
            self.current_indicators.clear()
        
        # Reset metrics
        self.performance_metrics.clear()
        self.metrics.clear()
        
        print("Environment closed and resources cleaned up.")
    
    def _compute_all_indicators(self):
        """
        Compute all technical indicators using pandas-ta and custom functions
        """
        # Clear existing indicators
        self.indicators = {}
        
        # Make a copy of the data to avoid modifying original
        df_work = self.df.copy()
        
        try:
            # === OVERLAP INDICATORS (Moving Averages) ===
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df_work[f'SMA_{period}'] = ta.sma(df_work['Close'], length=period)
                self.indicators[f'SMA_{period}'] = df_work[f'SMA_{period}'].iloc[-1] if not df_work[f'SMA_{period}'].isna().all() else 0
            
            # Exponential Moving Averages
            for period in [5, 12, 21, 26, 50]:
                df_work[f'EMA_{period}'] = ta.ema(df_work['Close'], length=period)
                self.indicators[f'EMA_{period}'] = df_work[f'EMA_{period}'].iloc[-1] if not df_work[f'EMA_{period}'].isna().all() else 0
            
            # Bollinger Bands
            bb = ta.bbands(df_work['Close'], length=self.strategy_params['bb_period'], std=self.strategy_params['bb_std'])
            if bb is not None:
                df_work = df_work.join(bb)
                self.indicators['BB_Upper'] = bb.iloc[-1, 0] if len(bb.columns) > 0 else 0
                self.indicators['BB_Middle'] = bb.iloc[-1, 1] if len(bb.columns) > 1 else 0  
                self.indicators['BB_Lower'] = bb.iloc[-1, 2] if len(bb.columns) > 2 else 0
            
            # === MOMENTUM INDICATORS ===
            # RSI
            for period in [2, 14, 21]:
                rsi = ta.rsi(df_work['Close'], length=period)
                if rsi is not None:
                    df_work[f'RSI_{period}'] = rsi
                    self.indicators[f'RSI_{period}'] = rsi.iloc[-1] if not rsi.isna().all() else 50
            
            # MACD
            macd = ta.macd(df_work['Close'], 
                          fast=self.strategy_params['macd_fast'],
                          slow=self.strategy_params['macd_slow'], 
                          signal=self.strategy_params['macd_signal'])
            if macd is not None:
                df_work = df_work.join(macd)
                self.indicators['MACD'] = macd.iloc[-1, 0] if len(macd.columns) > 0 else 0
                self.indicators['MACD_Signal'] = macd.iloc[-1, 2] if len(macd.columns) > 2 else 0
                self.indicators['MACD_Histogram'] = macd.iloc[-1, 1] if len(macd.columns) > 1 else 0
            
            # Stochastic Oscillator
            stoch = ta.stoch(df_work['High'], df_work['Low'], df_work['Close'], 
                            k=self.strategy_params['stoch_k'], 
                            d=self.strategy_params['stoch_d'])
            if stoch is not None:
                df_work = df_work.join(stoch)
                self.indicators['STOCH_K'] = stoch.iloc[-1, 0] if len(stoch.columns) > 0 else 50
                self.indicators['STOCH_D'] = stoch.iloc[-1, 1] if len(stoch.columns) > 1 else 50
            
            # === VOLATILITY INDICATORS ===
            # Average True Range
            atr = ta.atr(df_work['High'], df_work['Low'], df_work['Close'], 
                        length=self.strategy_params['atr_period'])
            if atr is not None:
                df_work['ATR'] = atr
                self.indicators['ATR'] = atr.iloc[-1] if not atr.isna().all() else 0
            
            # === TREND INDICATORS ===
            # ADX (Average Directional Index)
            adx = ta.adx(df_work['High'], df_work['Low'], df_work['Close'], 
                        length=self.strategy_params['adx_period'])
            if adx is not None:
                df_work = df_work.join(adx)
                self.indicators['ADX'] = adx.iloc[-1, 0] if len(adx.columns) > 0 else 0
            
            # === VOLUME INDICATORS ===
            if 'Volume' in df_work.columns:
                # Volume SMA
                vol_sma = ta.sma(df_work['Volume'], length=self.strategy_params['volume_sma'])
                if vol_sma is not None:
                    df_work['Volume_SMA'] = vol_sma
                    self.indicators['Volume_SMA'] = vol_sma.iloc[-1] if not vol_sma.isna().all() else 0
                
                # On Balance Volume
                obv = ta.obv(df_work['Close'], df_work['Volume'])
                if obv is not None:
                    df_work['OBV'] = obv
                    self.indicators['OBV'] = obv.iloc[-1] if not obv.isna().all() else 0
            
            # === CUSTOM INDICATORS FROM FRAMEWORK ===
            # Use our custom SMA and RSI functions for consistency
            from trading_framework import ocpSma, ocpRsi
            
            # Add SMA periods that match strategy parameters
            df_work = ocpSma(df_work, self.strategy_params['fast_sma'])
            df_work = ocpSma(df_work, self.strategy_params['slow_sma'])
            
            # Add RSI that matches strategy parameters
            df_work = ocpRsi(df_work, self.strategy_params['rsi_period'])
            
            # Update self.df with computed indicators
            self.df = df_work.copy()
            
            # Store key current values for observation space
            self.current_indicators = {
                'sma_fast': self.indicators.get(f'SMA_{self.strategy_params["fast_sma"]}', 0),
                'sma_slow': self.indicators.get(f'SMA_{self.strategy_params["slow_sma"]}', 0),
                'rsi': self.indicators.get(f'RSI_{self.strategy_params["rsi_period"]}', 50),
                'macd': self.indicators.get('MACD', 0),
                'macd_signal': self.indicators.get('MACD_Signal', 0),
                'atr': self.indicators.get('ATR', 0),
                'adx': self.indicators.get('ADX', 0)
            }
            
        except Exception as e:
            print(f"Warning: Error computing indicators: {e}")
            # Initialize with default values if computation fails
            self.indicators = {}
            self.current_indicators = {
                'sma_fast': 0, 'sma_slow': 0, 'rsi': 50,
                'macd': 0, 'macd_signal': 0, 'atr': 0, 'adx': 0
            }
