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
        
        # Ensure index is named 'Date' for trading framework compatibility
        if self.df.index.name != 'Date':
            self.df.index.name = 'Date'
        if self.original_df.index.name != 'Date':
            self.original_df.index.name = 'Date'
        
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
        The agent chooses which indicators to use and how to combine them dynamically.
        """
        self.metrics['episode_step'] += 1
        terminated = True  # Each step is a complete backtest episode
        
        try:
            # --- 1. DECODE AGENT ACTION INTO DYNAMIC STRATEGY CONFIG ---
            strategy_config = self._decode_agent_action(action)
            
            # --- 2. RUN DYNAMIC BACKTEST ---
            from trading_framework_dynamic import DynamicTradingSystem
            
            trading_system = DynamicTradingSystem()
            data_processed, results = trading_system.backtest_strategy(self.df.copy(), strategy_config)
            
            # --- 3. UPDATE PERFORMANCE METRICS ---
            self._update_performance_metrics(results)

            # --- 4. CALCULATE REWARD ---
            reward = self._calculate_reward(results)
            
            # --- 5. UPDATE OBSERVATION ---
            observation = self._create_observation_from_results(data_processed, results)
            
        except Exception as e:
            print(f"Backtest failed: {e}")
            reward = -200  # Heavy penalty for failed backtests
            observation = self._create_default_observation()
            self.metrics['consecutive_failures'] += 1

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
    
    # === DYNAMIC STRATEGY METHODS ===
    
    def _decode_agent_action(self, action):
        """
        Convert agent action space into dynamic strategy configuration.
        The agent chooses indicators and rules dynamically.
        """
        # Decode which indicators to use based on action values
        overlap_actions = action['overlap_actions']
        momentum_actions = action['momentum_actions'] 
        volatility_actions = action['volatility_actions']
        trend_actions = action['trend_actions']
        cycle_actions = action['cycle_actions']
        
        # Build indicator configuration dynamically
        indicator_config = {}
        
        # Moving Averages (agent chooses periods)
        if overlap_actions[0] != overlap_actions[1]:  # Use MA crossover if periods are different
            indicator_config['sma'] = [int(overlap_actions[0]), int(overlap_actions[1])]
        
        # Momentum indicators (agent chooses to include)
        if momentum_actions[0] > 5:  # Threshold to include RSI
            indicator_config['rsi'] = [int(momentum_actions[0])]
            
        if momentum_actions[1] > 5 and momentum_actions[2] > 5:  # Include MACD
            indicator_config['macd'] = {
                'fast': int(momentum_actions[1]),
                'slow': int(momentum_actions[2]), 
                'signal': int(momentum_actions[3])
            }
        
        # Volatility indicators
        if volatility_actions[0] > 0.5:  # Include Bollinger Bands
            indicator_config['bb'] = {
                'period': 20,
                'std': float(volatility_actions[0])
            }
            
        if volatility_actions[1] > 0.5:  # Include ATR
            indicator_config['atr'] = {'period': 14}
        
        # Trend indicators
        if trend_actions[0] > 0.7:  # Include ADX
            indicator_config['adx'] = {'period': 14}
        
        # Build trading rules dynamically based on available indicators
        entry_conditions = []
        exit_conditions = []
        
        # SMA Crossover rules (if SMA is selected)
        if 'sma' in indicator_config and len(indicator_config['sma']) == 2:
            fast_sma, slow_sma = indicator_config['sma']
            entry_conditions.append({
                'indicator1': f'SMA_{fast_sma}',
                'operator': 'crossover',
                'indicator2': f'SMA_{slow_sma}'
            })
            exit_conditions.append({
                'indicator1': f'SMA_{fast_sma}',
                'operator': 'crossunder', 
                'indicator2': f'SMA_{slow_sma}'
            })
        
        # RSI rules (if RSI is selected)
        if 'rsi' in indicator_config:
            rsi_period = indicator_config['rsi'][0]
            # Agent controls RSI thresholds through momentum_actions
            rsi_oversold = 20 + (momentum_actions[0] % 30)  # Range 20-50
            rsi_overbought = 70 + (momentum_actions[0] % 25)  # Range 70-95
            
            entry_conditions.append({
                'indicator1': f'RSI_{rsi_period}',
                'operator': '<',
                'value': rsi_oversold
            })
            exit_conditions.append({
                'indicator1': f'RSI_{rsi_period}',
                'operator': '>',
                'value': rsi_overbought
            })
        
        # MACD rules (if MACD is selected)
        if 'macd' in indicator_config:
            macd_config = indicator_config['macd']
            fast, slow, signal = macd_config['fast'], macd_config['slow'], macd_config['signal']
            
            entry_conditions.append({
                'indicator1': f'MACD_{fast}_{slow}_{signal}',
                'operator': 'crossover',
                'indicator2': f'MACDs_{fast}_{slow}_{signal}'
            })
            exit_conditions.append({
                'indicator1': f'MACD_{fast}_{slow}_{signal}',
                'operator': 'crossunder',
                'indicator2': f'MACDs_{fast}_{slow}_{signal}'
            })
        
        # Bollinger Bands rules (if BB is selected)
        if 'bb' in indicator_config:
            entry_conditions.append({
                'indicator1': 'Close',
                'operator': '<',
                'indicator2': 'BBL_20_2.0'  # Bollinger Lower Band
            })
            exit_conditions.append({
                'indicator1': 'Close',
                'operator': '>',
                'indicator2': 'BBU_20_2.0'  # Bollinger Upper Band
            })
        
        # If no specific rules were created, use simple price momentum
        if not entry_conditions:
            entry_conditions.append({
                'indicator1': 'Close',
                'operator': '>',
                'indicator2': 'Open'
            })
            exit_conditions.append({
                'indicator1': 'Close',
                'operator': '<',
                'indicator2': 'Open'
            })
        
        # Construct final strategy configuration
        strategy_config = {
            'indicators': indicator_config,
            'rules': {
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions
            },
            'direction': 'long',
            'candle_limit': 0,
            'take_profit': 0,
            'stop_loss': 0,
            'commission': 0,
            'slippage': 0,
            'position_size': 1
        }
        
        return strategy_config
    
    def _create_observation_from_results(self, data_processed, results):
        """
        Create observation dictionary from backtest results
        """
        try:
            # Update current indicator values from processed data
            current_indicators = {}
            
            # Extract latest indicator values from processed data
            if not data_processed.empty:
                for col in data_processed.columns:
                    if any(indicator in col for indicator in ['SMA', 'RSI', 'MACD', 'ATR', 'ADX', 'BB', 'STOCH']):
                        latest_value = data_processed[col].iloc[-1] if not data_processed[col].isna().all() else 0
                        current_indicators[col] = latest_value
            
            # Helper function to safely get historical data
            def get_history_safe(data, col, length, default_val=0.0):
                if col in data.columns and len(data) >= length:
                    return data[col].tail(length).fillna(default_val).values.astype(np.float32)
                return np.full(length, default_val, dtype=np.float32)
            
            # Build observation dictionary
            observation = {
                # Strategy parameters (current values from last action)
                'ma_params': np.array(list(self.strategy_params.values())[:16], dtype=np.int32),
                'oscillator_params': np.array(list(self.strategy_params.values())[16:24], dtype=np.int32),
                'volatility_params': np.array(list(self.strategy_params.values())[24:30], dtype=np.float32),
                'volume_params': np.array(list(self.strategy_params.values())[30:34], dtype=np.int32),
                'trend_params': np.array(list(self.strategy_params.values())[34:38], dtype=np.float32),
                
                # Current indicator values (normalized)
                'sma_fast': np.array([current_indicators.get('SMA_10', 0.0)], dtype=np.float32),
                'sma_slow': np.array([current_indicators.get('SMA_20', 0.0)], dtype=np.float32),
                'ema_fast': np.array([current_indicators.get('EMA_12', 0.0)], dtype=np.float32),
                'ema_slow': np.array([current_indicators.get('EMA_26', 0.0)], dtype=np.float32),
                'rsi': np.array([current_indicators.get('RSI_14', 50.0)], dtype=np.float32),
                'macd': np.array([current_indicators.get('MACD_12_26_9', 0.0)], dtype=np.float32),
                'macd_signal': np.array([current_indicators.get('MACDs_12_26_9', 0.0)], dtype=np.float32),
                'stoch_k': np.array([current_indicators.get('STOCHk_14_3_3', 50.0)], dtype=np.float32),
                'stoch_d': np.array([current_indicators.get('STOCHd_14_3_3', 50.0)], dtype=np.float32),
                'bb_upper': np.array([current_indicators.get('BBU_20_2.0', 0.0)], dtype=np.float32),
                'bb_middle': np.array([current_indicators.get('BBM_20_2.0', 0.0)], dtype=np.float32),
                'bb_lower': np.array([current_indicators.get('BBL_20_2.0', 0.0)], dtype=np.float32),
                'atr': np.array([current_indicators.get('ATR_14', 0.0)], dtype=np.float32),
                'adx': np.array([current_indicators.get('ADX_14', 0.0)], dtype=np.float32),
                'volume_sma': np.array([current_indicators.get('Volume_SMA', 0.0)], dtype=np.float32),
                'obv': np.array([current_indicators.get('OBV', 0.0)], dtype=np.float32),
                
                # Performance metrics from current backtest
                'current_cagr': np.array([self.performance_metrics.get('cagr', 0.0)], dtype=np.float32),
                'current_max_drawdown': np.array([self.performance_metrics.get('max_drawdown', 0.0)], dtype=np.float32),
                'current_sharpe_ratio': np.array([self.performance_metrics.get('sharpe_ratio', 0.0)], dtype=np.float32),
                'current_win_rate': np.array([self.performance_metrics.get('win_rate', 0.0)], dtype=np.float32),
                'current_trades': np.array([self.performance_metrics.get('trades', 0)], dtype=np.int32),
                'current_profit_factor': np.array([self.performance_metrics.get('profit_factor', 0.0)], dtype=np.float32),
                'current_expectancy': np.array([self.performance_metrics.get('expectancy', 0.0)], dtype=np.float32),
                
                # Environment state
                'total_backtests': np.array([self.metrics['total_backtests']], dtype=np.int32),
                'consecutive_failures': np.array([self.metrics['consecutive_failures']], dtype=np.int32),
                'episode_step': np.array([self.metrics['episode_step']], dtype=np.int32),
                
                # Market regime indicators
                'trend_strength': np.array([self._calculate_trend_strength(data_processed)], dtype=np.float32),
                'volatility_regime': np.array([self._calculate_volatility_regime(data_processed)], dtype=np.float32),
                'momentum_regime': np.array([self._calculate_momentum_regime(data_processed)], dtype=np.float32),
                'volume_regime': np.array([self._calculate_volume_regime(data_processed)], dtype=np.float32),
                
                # Historical data (normalized)
                'price_trend': get_history_safe(data_processed, 'Close', 50),
                'volume_trend': get_history_safe(data_processed, 'Volume', 50) if 'Volume' in data_processed.columns else np.zeros(50, dtype=np.float32),
                'rsi_history': get_history_safe(data_processed, 'RSI_14', 20, 50.0),
                'macd_history': get_history_safe(data_processed, 'MACD_12_26_9', 20),
                
                # Indicator relationships
                'ma_crossover': np.array([self._calculate_ma_crossover(data_processed)], dtype=np.float32),
                'macd_crossover': np.array([self._calculate_macd_crossover(data_processed)], dtype=np.float32),
                'bb_position': np.array([self._calculate_bb_position(data_processed)], dtype=np.float32),
                'rsi_position': np.array([self._calculate_rsi_position(data_processed)], dtype=np.float32),
                
                # Candlestick patterns (simplified)
                'recent_doji': np.zeros(5, dtype=np.float32),
                'recent_engulfing': np.zeros(5, dtype=np.float32),
                'recent_hammer': np.zeros(5, dtype=np.float32)
            }
            
            return observation
            
        except Exception as e:
            print(f"Error creating observation: {e}")
            return self._create_default_observation()
    
    def _create_default_observation(self):
        """
        Create default observation when backtest fails
        """
        return {
            'ma_params': np.zeros(16, dtype=np.int32),
            'oscillator_params': np.zeros(8, dtype=np.int32),
            'volatility_params': np.zeros(6, dtype=np.float32),
            'volume_params': np.zeros(4, dtype=np.int32),
            'trend_params': np.zeros(4, dtype=np.float32),
            
            'sma_fast': np.array([0.0], dtype=np.float32),
            'sma_slow': np.array([0.0], dtype=np.float32),
            'ema_fast': np.array([0.0], dtype=np.float32),
            'ema_slow': np.array([0.0], dtype=np.float32),
            'rsi': np.array([50.0], dtype=np.float32),
            'macd': np.array([0.0], dtype=np.float32),
            'macd_signal': np.array([0.0], dtype=np.float32),
            'stoch_k': np.array([50.0], dtype=np.float32),
            'stoch_d': np.array([50.0], dtype=np.float32),
            'bb_upper': np.array([0.0], dtype=np.float32),
            'bb_middle': np.array([0.0], dtype=np.float32),
            'bb_lower': np.array([0.0], dtype=np.float32),
            'atr': np.array([0.0], dtype=np.float32),
            'adx': np.array([0.0], dtype=np.float32),
            'volume_sma': np.array([0.0], dtype=np.float32),
            'obv': np.array([0.0], dtype=np.float32),
            
            'current_cagr': np.array([0.0], dtype=np.float32),
            'current_max_drawdown': np.array([0.0], dtype=np.float32),
            'current_sharpe_ratio': np.array([0.0], dtype=np.float32),
            'current_win_rate': np.array([0.0], dtype=np.float32),
            'current_trades': np.array([0], dtype=np.int32),
            'current_profit_factor': np.array([0.0], dtype=np.float32),
            'current_expectancy': np.array([0.0], dtype=np.float32),
            
            'total_backtests': np.array([self.metrics['total_backtests']], dtype=np.int32),
            'consecutive_failures': np.array([self.metrics['consecutive_failures']], dtype=np.int32),
            'episode_step': np.array([self.metrics['episode_step']], dtype=np.int32),
            
            'trend_strength': np.array([0.0], dtype=np.float32),
            'volatility_regime': np.array([0.0], dtype=np.float32),
            'momentum_regime': np.array([0.0], dtype=np.float32),
            'volume_regime': np.array([0.0], dtype=np.float32),
            
            'price_trend': np.zeros(50, dtype=np.float32),
            'volume_trend': np.zeros(50, dtype=np.float32),
            'rsi_history': np.full(20, 50.0, dtype=np.float32),
            'macd_history': np.zeros(20, dtype=np.float32),
            
            'ma_crossover': np.array([0.0], dtype=np.float32),
            'macd_crossover': np.array([0.0], dtype=np.float32),
            'bb_position': np.array([0.5], dtype=np.float32),
            'rsi_position': np.array([0.5], dtype=np.float32),
            
            'recent_doji': np.zeros(5, dtype=np.float32),
            'recent_engulfing': np.zeros(5, dtype=np.float32),
            'recent_hammer': np.zeros(5, dtype=np.float32)
        }
    
    # === MARKET REGIME CALCULATION METHODS ===
    
    def _calculate_trend_strength(self, data):
        """Calculate overall trend strength from ADX and price movement"""
        try:
            if 'ADX_14' in data.columns and len(data) > 0:
                adx = data['ADX_14'].iloc[-1]
                return min(adx / 100.0, 1.0)  # Normalize to 0-1
            return 0.0
        except:
            return 0.0
    
    def _calculate_volatility_regime(self, data):
        """Calculate current volatility regime from ATR"""
        try:
            if 'ATR_14' in data.columns and len(data) > 20:
                current_atr = data['ATR_14'].iloc[-1]
                avg_atr = data['ATR_14'].tail(20).mean()
                return min(current_atr / avg_atr, 2.0) / 2.0  # Normalize to 0-1
            return 0.0
        except:
            return 0.0
    
    def _calculate_momentum_regime(self, data):
        """Calculate momentum regime from RSI and price changes"""
        try:
            if 'RSI_14' in data.columns and len(data) > 0:
                rsi = data['RSI_14'].iloc[-1]
                return (rsi - 50.0) / 50.0  # Normalize to -1 to 1
            return 0.0
        except:
            return 0.0
    
    def _calculate_volume_regime(self, data):
        """Calculate volume regime from volume indicators"""
        try:
            if 'Volume' in data.columns and len(data) > 20:
                current_vol = data['Volume'].iloc[-1]
                avg_vol = data['Volume'].tail(20).mean()
                return min(current_vol / avg_vol, 2.0) / 2.0  # Normalize to 0-1
            return 0.0
        except:
            return 0.0
    
    # === INDICATOR RELATIONSHIP CALCULATION METHODS ===
    
    def _calculate_ma_crossover(self, data):
        """Calculate moving average crossover signal"""
        try:
            sma_cols = [col for col in data.columns if col.startswith('SMA_')]
            if len(sma_cols) >= 2 and len(data) >= 2:
                fast_col = min(sma_cols, key=lambda x: int(x.split('_')[1]))
                slow_col = max(sma_cols, key=lambda x: int(x.split('_')[1]))
                
                current_diff = data[fast_col].iloc[-1] - data[slow_col].iloc[-1]
                prev_diff = data[fast_col].iloc[-2] - data[slow_col].iloc[-2]
                
                if current_diff > 0 and prev_diff <= 0:
                    return 1.0  # Golden cross
                elif current_diff < 0 and prev_diff >= 0:
                    return -1.0  # Death cross
                else:
                    return current_diff / data[slow_col].iloc[-1] * 10  # Normalized difference
            return 0.0
        except:
            return 0.0
    
    def _calculate_macd_crossover(self, data):
        """Calculate MACD crossover signal"""
        try:
            macd_cols = [col for col in data.columns if 'MACD_' in col and not col.startswith('MACDs')]
            signal_cols = [col for col in data.columns if col.startswith('MACDs_')]
            
            if macd_cols and signal_cols and len(data) >= 2:
                macd_col = macd_cols[0]
                signal_col = signal_cols[0]
                
                current_diff = data[macd_col].iloc[-1] - data[signal_col].iloc[-1]
                prev_diff = data[macd_col].iloc[-2] - data[signal_col].iloc[-2]
                
                if current_diff > 0 and prev_diff <= 0:
                    return 1.0  # Bullish crossover
                elif current_diff < 0 and prev_diff >= 0:
                    return -1.0  # Bearish crossover
                else:
                    return np.tanh(current_diff)  # Normalized difference
            return 0.0
        except:
            return 0.0
    
    def _calculate_bb_position(self, data):
        """Calculate price position within Bollinger Bands"""
        try:
            if all(col in data.columns for col in ['BBU_20_2.0', 'BBL_20_2.0', 'Close']) and len(data) > 0:
                close = data['Close'].iloc[-1]
                upper = data['BBU_20_2.0'].iloc[-1]
                lower = data['BBL_20_2.0'].iloc[-1]
                
                if upper > lower:
                    return (close - lower) / (upper - lower)  # 0 = at lower band, 1 = at upper band
                else:
                    return 0.5
            return 0.5
        except:
            return 0.5
    
    def _calculate_rsi_position(self, data):
        """Calculate RSI position (normalized)"""
        try:
            if 'RSI_14' in data.columns and len(data) > 0:
                rsi = data['RSI_14'].iloc[-1]
                return rsi / 100.0  # Normalize to 0-1
            return 0.5
        except:
            return 0.5
    
    def _update_performance_metrics(self, results):
        """
        Updates internal performance metrics from backtest results.
        """
        try:
            # Extract metrics from backSistemaList output
            self.performance_metrics['trades'] = results[2] if len(results) > 2 else 0
            self.performance_metrics['cagr'] = results[12] if len(results) > 12 else 0.0
            self.performance_metrics['max_drawdown'] = abs(results[21]) if len(results) > 21 else 0.0
            self.performance_metrics['profit_factor'] = results[18] if len(results) > 18 else 0.0
            self.performance_metrics['win_rate'] = results[5] if len(results) > 5 else 0.0
            self.performance_metrics['avg_trade'] = results[6] if len(results) > 6 else 0.0
            self.performance_metrics['avg_win'] = results[7] if len(results) > 7 else 0.0
            self.performance_metrics['avg_loss'] = results[8] if len(results) > 8 else 0.0
            
            # Calculate Sharpe ratio approximation
            if self.performance_metrics['max_drawdown'] > 0:
                self.performance_metrics['sharpe_ratio'] = self.performance_metrics['cagr'] / self.performance_metrics['max_drawdown']
            else:
                self.performance_metrics['sharpe_ratio'] = 0.0
            
            # Update environment metrics
            self.metrics['total_backtests'] += 1
            if self.performance_metrics['cagr'] > self.metrics['best_cagr']:
                self.metrics['best_cagr'] = self.performance_metrics['cagr']
            if self.performance_metrics['max_drawdown'] < self.metrics['best_drawdown']:
                self.metrics['best_drawdown'] = self.performance_metrics['max_drawdown']
                
        except (IndexError, TypeError) as e:
            print(f"Warning: Could not update performance metrics: {e}")
            # Reset to default values on error
            for key in self.performance_metrics:
                if key in ['trades']:
                    self.performance_metrics[key] = 0
                else:
                    self.performance_metrics[key] = 0.0
