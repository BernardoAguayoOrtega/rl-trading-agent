#!/usr/bin/env python3
"""
RL Trading Agent Evaluation Script

This script provides comprehensive evaluation and testing of trained RL trading agents,
including performance analysis, strategy comparison, backtesting, and visualization.

Usage:
    python evaluate.py --model_path models/best_model.zip --data_file data/test_data.csv
    python evaluate.py --model_path models/best_model.zip --benchmark --compare_strategies

Features:
- Load and evaluate trained RL models
- Compare against baseline strategies
- Generate comprehensive performance reports  
- Create detailed visualizations and plots
- Export results in multiple formats
- Statistical significance testing
- Risk-adjusted performance metrics
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import scipy.stats as stats

# Local imports
from config import *
from rl_agent.StrategyEnv import StrategyEnv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TradingMetricsCalculator:
    """Calculate comprehensive trading performance metrics"""
    
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate return-based performance metrics"""
        if returns.empty:
            return {}
        
        # Convert to decimal if percentage
        if returns.abs().mean() > 1:
            returns = returns / 100
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - RISK_CONFIG['risk_free_rate']) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Downside deviation and Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                metrics['sortino_ratio'] = (metrics['annualized_return'] - RISK_CONFIG['risk_free_rate']) / downside_deviation
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = float('inf')
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Additional risk metrics
        metrics['var_95'] = returns.quantile(0.05)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        # Win rate
        winning_periods = (returns > 0).sum()
        total_periods = len(returns)
        metrics['win_rate'] = winning_periods / total_periods if total_periods > 0 else 0
        
        # Benchmark comparison
        if benchmark_returns is not None and not benchmark_returns.empty:
            if benchmark_returns.abs().mean() > 1:
                benchmark_returns = benchmark_returns / 100
            
            # Information ratio
            excess_returns = returns - benchmark_returns
            if excess_returns.std() > 0:
                metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            else:
                metrics['information_ratio'] = 0
            
            # Beta
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            if benchmark_variance > 0:
                metrics['beta'] = covariance / benchmark_variance
                metrics['alpha'] = metrics['annualized_return'] - (RISK_CONFIG['risk_free_rate'] + metrics['beta'] * (benchmark_returns.mean() * 252 - RISK_CONFIG['risk_free_rate']))
            else:
                metrics['beta'] = 1.0
                metrics['alpha'] = 0.0
        
        return metrics

class BenchmarkStrategies:
    """Implement baseline trading strategies for comparison"""
    
    @staticmethod
    def buy_and_hold(data: pd.DataFrame) -> pd.Series:
        """Simple buy and hold strategy"""
        returns = data['Close'].pct_change().dropna()
        return returns
    
    @staticmethod
    def simple_moving_average(data: pd.DataFrame, fast_period: int = 10, slow_period: int = 20) -> pd.Series:
        """Simple moving average crossover strategy"""
        fast_ma = data['Close'].rolling(fast_period).mean()
        slow_ma = data['Close'].rolling(slow_period).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        # Calculate returns
        position = signals.shift(1).fillna(0)
        returns = position * data['Close'].pct_change()
        
        return returns.dropna()
    
    @staticmethod
    def rsi_mean_reversion(data: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """RSI mean reversion strategy"""
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[rsi < oversold] = 1   # Buy when oversold
        signals[rsi > overbought] = -1  # Sell when overbought
        
        # Calculate returns
        position = signals.shift(1).fillna(0)
        returns = position * data['Close'].pct_change()
        
        return returns.dropna()
    
    @staticmethod
    def random_strategy(data: pd.DataFrame, seed: int = 42) -> pd.Series:
        """Random trading strategy for baseline comparison"""
        np.random.seed(seed)
        signals = np.random.choice([-1, 0, 1], size=len(data), p=[0.1, 0.8, 0.1])
        signals = pd.Series(signals, index=data.index)
        
        position = signals.shift(1).fillna(0)
        returns = position * data['Close'].pct_change()
        
        return returns.dropna()

class StrategyEvaluator:
    """Main evaluation class for RL trading strategies"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics_calculator = TradingMetricsCalculator()
        self.benchmark_strategies = BenchmarkStrategies()
        
    def load_model_and_env(self, model_path: Path, data: pd.DataFrame) -> Tuple[PPO, Any]:
        """Load trained model and create evaluation environment"""
        self.logger.info(f"🔄 Loading model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create evaluation environment
        env = StrategyEnv(data)
        vec_env = DummyVecEnv([lambda: env])
        
        # Check for normalization stats
        normalize_path = model_path.parent / "vec_normalize.pkl"
        if normalize_path.exists():
            self.logger.info("📊 Loading normalization statistics")
            vec_env = VecNormalize.load(normalize_path, vec_env)
            vec_env.training = False  # Disable training mode for evaluation
        
        # Load model
        model = PPO.load(model_path, env=vec_env)
        self.logger.info("✅ Model loaded successfully")
        
        return model, vec_env
    
    def evaluate_rl_strategy(self, model: PPO, env: Any, n_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate the RL strategy"""
        self.logger.info(f"🎯 Evaluating RL strategy over {n_episodes} episodes")
        
        # Evaluate policy
        episode_rewards, episode_lengths = evaluate_policy(
            model, env, n_eval_episodes=n_episodes, 
            deterministic=True, return_episode_rewards=True
        )
        
        # Collect detailed episode information
        episode_data = []
        returns_series = []
        
        obs = env.reset()
        for episode in range(n_episodes):
            episode_returns = []
            episode_actions = []
            done = False
            step = 0
            
            while not done and step < ENV_CONFIG['max_episode_steps']:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_returns.append(reward)
                episode_actions.append(action)
                step += 1
            
            # Store episode data
            episode_info = {
                'episode': episode,
                'total_reward': sum(episode_returns),
                'length': len(episode_returns),
                'actions': episode_actions
            }
            episode_data.append(episode_info)
            returns_series.extend(episode_returns)
            
            obs = env.reset()
        
        # Calculate metrics
        returns_series = pd.Series(returns_series)
        metrics = self.metrics_calculator.calculate_returns_metrics(returns_series)
        
        # Add episode-specific metrics
        metrics['mean_episode_reward'] = np.mean(episode_rewards)
        metrics['std_episode_reward'] = np.std(episode_rewards)
        metrics['mean_episode_length'] = np.mean(episode_lengths)
        metrics['total_episodes'] = n_episodes
        
        results = {
            'metrics': metrics,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_data': episode_data,
            'returns_series': returns_series
        }
        
        self.logger.info(f"✅ RL evaluation complete - Mean reward: {metrics['mean_episode_reward']:.2f}")
        return results
    
    def evaluate_benchmark_strategies(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate all benchmark strategies"""
        self.logger.info("📊 Evaluating benchmark strategies")
        
        benchmark_results = {}
        
        # Buy and Hold
        try:
            bh_returns = self.benchmark_strategies.buy_and_hold(data)
            bh_metrics = self.metrics_calculator.calculate_returns_metrics(bh_returns)
            benchmark_results['buy_and_hold'] = {
                'metrics': bh_metrics,
                'returns': bh_returns,
                'name': 'Buy & Hold'
            }
            self.logger.info(f"✅ Buy & Hold: {bh_metrics.get('total_return', 0):.2%} total return")
        except Exception as e:
            self.logger.error(f"❌ Buy & Hold evaluation failed: {e}")
        
        # Simple Moving Average
        try:
            sma_returns = self.benchmark_strategies.simple_moving_average(data)
            sma_metrics = self.metrics_calculator.calculate_returns_metrics(sma_returns)
            benchmark_results['simple_ma'] = {
                'metrics': sma_metrics,
                'returns': sma_returns,
                'name': 'Simple MA Crossover'
            }
            self.logger.info(f"✅ Simple MA: {sma_metrics.get('total_return', 0):.2%} total return")
        except Exception as e:
            self.logger.error(f"❌ Simple MA evaluation failed: {e}")
        
        # RSI Mean Reversion
        try:
            rsi_returns = self.benchmark_strategies.rsi_mean_reversion(data)
            rsi_metrics = self.metrics_calculator.calculate_returns_metrics(rsi_returns)
            benchmark_results['rsi_mean_reversion'] = {
                'metrics': rsi_metrics,
                'returns': rsi_returns,
                'name': 'RSI Mean Reversion'
            }
            self.logger.info(f"✅ RSI Strategy: {rsi_metrics.get('total_return', 0):.2%} total return")
        except Exception as e:
            self.logger.error(f"❌ RSI evaluation failed: {e}")
        
        # Random Strategy
        try:
            random_returns = self.benchmark_strategies.random_strategy(data)
            random_metrics = self.metrics_calculator.calculate_returns_metrics(random_returns)
            benchmark_results['random'] = {
                'metrics': random_metrics,
                'returns': random_returns,
                'name': 'Random Strategy'
            }
            self.logger.info(f"✅ Random Strategy: {random_metrics.get('total_return', 0):.2%} total return")
        except Exception as e:
            self.logger.error(f"❌ Random strategy evaluation failed: {e}")
        
        return benchmark_results
    
    def statistical_significance_test(self, rl_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Test statistical significance of RL strategy vs benchmark"""
        try:
            # Paired t-test
            t_stat, t_p_value = stats.ttest_rel(rl_returns, benchmark_returns)
            
            # Wilcoxon signed-rank test (non-parametric)
            w_stat, w_p_value = stats.wilcoxon(rl_returns, benchmark_returns, alternative='two-sided')
            
            return {
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_p_value': w_p_value,
                'is_significant_t_test': t_p_value < 0.05,
                'is_significant_wilcoxon': w_p_value < 0.05
            }
        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")
            return {}

class ResultsVisualizer:
    """Create comprehensive visualizations and reports"""
    
    def __init__(self, output_dir: Path, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use(PLOT_CONFIG.get('style', 'default'))
        sns.set_palette("husl")
    
    def create_performance_comparison_plot(self, rl_results: Dict, benchmark_results: Dict) -> Path:
        """Create comprehensive performance comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL Strategy vs Benchmarks - Performance Comparison', fontsize=16, fontweight='bold')
        
        # Collect all strategies
        all_strategies = {'RL Strategy': rl_results}
        all_strategies.update(benchmark_results)
        
        # 1. Total Returns Comparison
        ax1 = axes[0, 0]
        strategy_names = []
        total_returns = []
        colors = []
        
        for name, results in all_strategies.items():
            if 'metrics' in results:
                strategy_names.append(results.get('name', name))
                total_returns.append(results['metrics'].get('total_return', 0) * 100)
                colors.append(PLOT_CONFIG['colors']['primary'] if name == 'RL Strategy' else PLOT_CONFIG['colors']['secondary'])
        
        bars = ax1.bar(strategy_names, total_returns, color=colors, alpha=0.7)
        ax1.set_title('Total Returns (%)', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, total_returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Risk-Adjusted Returns (Sharpe Ratio)
        ax2 = axes[0, 1]
        sharpe_ratios = []
        for name, results in all_strategies.items():
            if 'metrics' in results:
                sharpe_ratios.append(results['metrics'].get('sharpe_ratio', 0))
        
        bars = ax2.bar(strategy_names, sharpe_ratios, color=colors, alpha=0.7)
        ax2.set_title('Sharpe Ratio', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Maximum Drawdown
        ax3 = axes[1, 0]
        max_drawdowns = []
        for name, results in all_strategies.items():
            if 'metrics' in results:
                max_drawdowns.append(abs(results['metrics'].get('max_drawdown', 0)) * 100)
        
        bars = ax3.bar(strategy_names, max_drawdowns, color=colors, alpha=0.7)
        ax3.set_title('Maximum Drawdown (%)', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 4. Win Rate
        ax4 = axes[1, 1]
        win_rates = []
        for name, results in all_strategies.items():
            if 'metrics' in results:
                win_rates.append(results['metrics'].get('win_rate', 0) * 100)
        
        bars = ax4.bar(strategy_names, win_rates, color=colors, alpha=0.7)
        ax4.set_title('Win Rate (%)', fontweight='bold')
        ax4.set_ylabel('Win Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 100)
        
        for bar, value in zip(bars, win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Performance comparison plot saved: {plot_path}")
        return plot_path
    
    def create_returns_distribution_plot(self, rl_results: Dict, benchmark_results: Dict) -> Path:
        """Create returns distribution comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Returns Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Get returns data
        rl_returns = rl_results.get('returns_series', pd.Series())
        
        # 1. Returns Distribution Histogram
        ax1 = axes[0, 0]
        if not rl_returns.empty:
            ax1.hist(rl_returns, bins=50, alpha=0.7, color=PLOT_CONFIG['colors']['primary'], label='RL Strategy')
        
        # Add benchmark if available
        if 'buy_and_hold' in benchmark_results:
            bh_returns = benchmark_results['buy_and_hold']['returns']
            if not bh_returns.empty:
                ax1.hist(bh_returns, bins=50, alpha=0.5, color=PLOT_CONFIG['colors']['secondary'], label='Buy & Hold')
        
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        ax2 = axes[0, 1]
        if not rl_returns.empty:
            stats.probplot(rl_returns, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normality Test)')
        
        # 3. Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        if not rl_returns.empty and len(rl_returns) > 30:
            rolling_sharpe = rl_returns.rolling(30).mean() / rl_returns.rolling(30).std() * np.sqrt(252)
            ax3.plot(rolling_sharpe.index, rolling_sharpe, label='RL Strategy', color=PLOT_CONFIG['colors']['primary'])
        
        if 'buy_and_hold' in benchmark_results:
            bh_returns = benchmark_results['buy_and_hold']['returns']
            if not bh_returns.empty and len(bh_returns) > 30:
                bh_rolling_sharpe = bh_returns.rolling(30).mean() / bh_returns.rolling(30).std() * np.sqrt(252)
                ax3.plot(bh_rolling_sharpe.index, bh_rolling_sharpe, label='Buy & Hold', 
                        color=PLOT_CONFIG['colors']['secondary'], alpha=0.7)
        
        ax3.set_title('Rolling 30-Day Sharpe Ratio')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Returns
        ax4 = axes[1, 1]
        if not rl_returns.empty:
            rl_cumulative = (1 + rl_returns).cumprod()
            ax4.plot(rl_cumulative.index, rl_cumulative, label='RL Strategy', 
                    color=PLOT_CONFIG['colors']['primary'], linewidth=2)
        
        if 'buy_and_hold' in benchmark_results:
            bh_returns = benchmark_results['buy_and_hold']['returns']
            if not bh_returns.empty:
                bh_cumulative = (1 + bh_returns).cumprod()
                ax4.plot(bh_cumulative.index, bh_cumulative, label='Buy & Hold', 
                        color=PLOT_CONFIG['colors']['secondary'], alpha=0.7, linewidth=2)
        
        ax4.set_title('Cumulative Returns')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "returns_analysis.png"
        plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Returns analysis plot saved: {plot_path}")
        return plot_path
    
    def generate_performance_report(self, rl_results: Dict, benchmark_results: Dict, 
                                  statistical_tests: Dict = None) -> Path:
        """Generate comprehensive performance report"""
        report_path = self.output_dir / "performance_report.html"
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL Trading Strategy - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e6f3ff; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RL Trading Strategy Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Performance Summary Table
        html_content += """
        <div class="section">
            <h2>Performance Summary</h2>
            <table class="metrics-table">
                <tr><th>Strategy</th><th>Total Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Win Rate</th></tr>
        """
        
        # Add RL strategy
        rl_metrics = rl_results.get('metrics', {})
        html_content += f"""
        <tr class="highlight">
            <td><strong>RL Strategy</strong></td>
            <td class="{'positive' if rl_metrics.get('total_return', 0) > 0 else 'negative'}">{rl_metrics.get('total_return', 0):.2%}</td>
            <td>{rl_metrics.get('sharpe_ratio', 0):.2f}</td>
            <td class="negative">{rl_metrics.get('max_drawdown', 0):.2%}</td>
            <td>{rl_metrics.get('win_rate', 0):.2%}</td>
        </tr>
        """
        
        # Add benchmarks
        for name, results in benchmark_results.items():
            metrics = results.get('metrics', {})
            strategy_name = results.get('name', name)
            html_content += f"""
            <tr>
                <td>{strategy_name}</td>
                <td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                <td>{metrics.get('win_rate', 0):.2%}</td>
            </tr>
            """
        
        html_content += "</table></div>"
        
        # Detailed Metrics
        html_content += f"""
        <div class="section">
            <h2>Detailed RL Strategy Metrics</h2>
            <table class="metrics-table">
        """
        
        for metric, value in rl_metrics.items():
            if isinstance(value, (int, float)):
                if 'ratio' in metric.lower() or 'return' in metric.lower():
                    formatted_value = f"{value:.4f}"
                elif 'rate' in metric.lower():
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
                
                html_content += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
                """
        
        html_content += "</table></div>"
        
        # Statistical Tests
        if statistical_tests:
            html_content += f"""
            <div class="section">
                <h2>Statistical Significance Tests</h2>
                <p>Tests comparing RL strategy vs Buy & Hold:</p>
                <ul>
                    <li>T-test p-value: {statistical_tests.get('t_p_value', 'N/A'):.4f} {'(Significant)' if statistical_tests.get('is_significant_t_test', False) else '(Not Significant)'}</li>
                    <li>Wilcoxon test p-value: {statistical_tests.get('wilcoxon_p_value', 'N/A'):.4f} {'(Significant)' if statistical_tests.get('is_significant_wilcoxon', False) else '(Not Significant)'}</li>
                </ul>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"📋 Performance report saved: {report_path}")
        return report_path

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for evaluation"""
    log_file = output_dir / "evaluation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting RL Trading Strategy Evaluation")
    return logger

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate RL Trading Strategy")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=EVALUATION_CONFIG['n_episodes'],
        help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare against benchmark strategies"
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        default=True,
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Set up paths
        model_path = Path(args.model_path)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = RESULTS_DIR / f"evaluation_{timestamp}"
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logging
        logger = setup_logging(output_dir)
        
        # Load data
        if args.data_file:
            data_file = Path(args.data_file)
        else:
            data_file = DEFAULT_DATA_FILES.get('test', DEFAULT_DATA_FILES['train'])
        
        if not data_file.exists():
            logger.error(f"❌ Data file not found: {data_file}")
            return 1
        
        logger.info(f"📊 Loading test data from: {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        logger.info(f"📈 Loaded {len(data)} rows of test data")
        
        # Initialize evaluator
        evaluator = StrategyEvaluator(logger)
        
        # Load model and evaluate RL strategy
        model, env = evaluator.load_model_and_env(model_path, data)
        rl_results = evaluator.evaluate_rl_strategy(model, env, args.n_episodes)
        
        # Evaluate benchmarks if requested
        benchmark_results = {}
        if args.benchmark:
            benchmark_results = evaluator.evaluate_benchmark_strategies(data)
        
        # Statistical significance tests
        statistical_tests = {}
        if benchmark_results and 'buy_and_hold' in benchmark_results:
            if not rl_results['returns_series'].empty and not benchmark_results['buy_and_hold']['returns'].empty:
                # Align series for comparison
                min_length = min(len(rl_results['returns_series']), len(benchmark_results['buy_and_hold']['returns']))
                rl_aligned = rl_results['returns_series'].iloc[:min_length]
                bh_aligned = benchmark_results['buy_and_hold']['returns'].iloc[:min_length]
                
                statistical_tests = evaluator.statistical_significance_test(rl_aligned, bh_aligned)
        
        # Generate visualizations
        if args.generate_plots:
            visualizer = ResultsVisualizer(output_dir, logger)
            visualizer.create_performance_comparison_plot(rl_results, benchmark_results)
            visualizer.create_returns_distribution_plot(rl_results, benchmark_results)
            visualizer.generate_performance_report(rl_results, benchmark_results, statistical_tests)
        
        # Save results to JSON
        results_summary = {
            'rl_strategy': {
                'metrics': rl_results['metrics'],
                'mean_episode_reward': rl_results.get('mean_episode_reward', 0),
                'total_episodes': rl_results.get('total_episodes', 0)
            },
            'benchmark_strategies': {name: results['metrics'] for name, results in benchmark_results.items()},
            'statistical_tests': statistical_tests,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'data_file': str(data_file)
        }
        
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Print summary
        rl_metrics = rl_results['metrics']
        logger.info("🎉 Evaluation completed successfully!")
        logger.info(f"📊 RL Strategy Performance:")
        logger.info(f"   • Total Return: {rl_metrics.get('total_return', 0):.2%}")
        logger.info(f"   • Sharpe Ratio: {rl_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   • Max Drawdown: {rl_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"   • Win Rate: {rl_metrics.get('win_rate', 0):.2%}")
        
        if benchmark_results:
            logger.info("🏆 Benchmark Comparison:")
            for name, results in benchmark_results.items():
                metrics = results['metrics']
                logger.info(f"   • {results['name']}: {metrics.get('total_return', 0):.2%} return, {metrics.get('sharpe_ratio', 0):.2f} Sharpe")
        
        logger.info(f"📂 All results saved to: {output_dir}")
        
        # Close environments
        env.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
