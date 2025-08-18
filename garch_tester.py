#!/usr/bin/env python3
"""
GARCH-Based Strategy Stress Testing Tool

This module provides comprehensive stress testing capabilities for RL trading strategies
using GARCH models to simulate different volatility regimes and market conditions.

Features:
- GARCH(1,1) model fitting and parameter estimation
- Monte Carlo simulation of price paths under different volatility scenarios
- Stress testing of trained RL models across various market conditions
- Regime-specific performance analysis
- Risk metric calculation under extreme scenarios
- Comparative analysis across volatility environments
- Detailed reporting and visualization

Usage:
    python garch_tester.py --model_path models/best_model.zip --data_file data/test_data.csv
    python garch_tester.py --model_path models/best_model.zip --stress_scenarios high_vol,low_vol,crisis
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Local imports
from config import *
from rl_agent.StrategyEnv import StrategyEnv
from evaluate import TradingMetricsCalculator, setup_logging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class GARCHModel:
    """GARCH(1,1) model implementation for volatility modeling"""
    
    def __init__(self):
        self.params = None
        self.fitted = False
        self.log_likelihood = None
        
    def fit(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit GARCH(1,1) model to return series"""
        # Prepare returns
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 50:
            raise ValueError("Insufficient data points for GARCH fitting (minimum 50 required)")
        
        # Center the returns
        mean_return = np.mean(returns)
        centered_returns = returns - mean_return
        
        # Initial parameter estimates
        initial_variance = np.var(centered_returns)
        initial_params = [
            initial_variance * 0.1,  # omega (long-term variance component)
            0.1,                      # alpha (ARCH effect)
            0.8,                      # beta (GARCH effect)
            mean_return              # mu (mean return)
        ]
        
        # Parameter bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [
            (1e-6, None),    # omega
            (0, 0.99),       # alpha
            (0, 0.99),       # beta  
            (None, None)     # mu
        ]
        
        # Constraint: alpha + beta < 1 (stationarity condition)
        constraints = {'type': 'ineq', 'fun': lambda x: 0.99 - (x[1] + x[2])}
        
        # Maximize log-likelihood (minimize negative log-likelihood)
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(centered_returns,),
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            # Try different initial values
            for _ in range(3):
                initial_params[0] = np.random.uniform(1e-6, initial_variance * 0.5)
                initial_params[1] = np.random.uniform(0.05, 0.3)
                initial_params[2] = np.random.uniform(0.5, 0.9)
                
                result = minimize(
                    self._negative_log_likelihood,
                    initial_params,
                    args=(centered_returns,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    break
        
        if not result.success:
            raise RuntimeError(f"GARCH optimization failed: {result.message}")
        
        # Store fitted parameters
        self.params = {
            'omega': result.x[0],
            'alpha': result.x[1], 
            'beta': result.x[2],
            'mu': result.x[3]
        }
        
        self.fitted = True
        self.log_likelihood = -result.fun
        
        # Calculate additional statistics
        self.params['persistence'] = self.params['alpha'] + self.params['beta']
        self.params['unconditional_variance'] = self.params['omega'] / (1 - self.params['persistence'])
        
        return self.params
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Calculate negative log-likelihood for GARCH(1,1) model"""
        omega, alpha, beta, mu = params
        
        # Check parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e8
        
        n = len(returns)
        
        # Initialize conditional variance
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initial variance estimate
        
        # Calculate conditional variances
        for t in range(1, n):
            sigma2[t] = omega + alpha * (returns[t-1] - mu)**2 + beta * sigma2[t-1]
            
            # Ensure positive variance
            if sigma2[t] <= 0:
                return 1e8
        
        # Calculate log-likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma2)) - 0.5 * np.sum((returns - mu)**2 / sigma2)
        
        return -log_likelihood
    
    def predict_volatility(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Predict future volatility using fitted GARCH model"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        returns = np.asarray(returns).flatten()
        n = len(returns)
        
        # Calculate current conditional variance
        sigma2_current = self.params['omega']
        for t in range(n):
            if t == 0:
                sigma2_current = self.params['omega'] + self.params['alpha'] * (returns[0] - self.params['mu'])**2 + self.params['beta'] * np.var(returns)
            else:
                sigma2_current = self.params['omega'] + self.params['alpha'] * (returns[t] - self.params['mu'])**2 + self.params['beta'] * sigma2_current
        
        # Multi-step ahead prediction
        volatility_forecast = np.zeros(horizon)
        sigma2_forecast = sigma2_current
        
        for h in range(horizon):
            if h == 0:
                volatility_forecast[h] = np.sqrt(sigma2_forecast)
            else:
                # Long-term variance convergence
                sigma2_forecast = self.params['omega'] + (self.params['alpha'] + self.params['beta']) * sigma2_forecast
                volatility_forecast[h] = np.sqrt(sigma2_forecast)
        
        return volatility_forecast
    
    def simulate_returns(self, n_steps: int, n_simulations: int = 1000, 
                        volatility_multiplier: float = 1.0, random_seed: int = None) -> np.ndarray:
        """Simulate future returns using fitted GARCH parameters"""
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Scale parameters by volatility multiplier
        omega_scaled = self.params['omega'] * volatility_multiplier**2
        alpha_scaled = self.params['alpha']
        beta_scaled = self.params['beta']
        mu_scaled = self.params['mu']
        
        simulations = np.zeros((n_simulations, n_steps))
        
        for sim in range(n_simulations):
            returns_sim = np.zeros(n_steps)
            sigma2_sim = np.zeros(n_steps)
            
            # Initialize with unconditional variance
            sigma2_sim[0] = omega_scaled / (1 - alpha_scaled - beta_scaled)
            
            for t in range(n_steps):
                if t > 0:
                    sigma2_sim[t] = omega_scaled + alpha_scaled * (returns_sim[t-1] - mu_scaled)**2 + beta_scaled * sigma2_sim[t-1]
                
                # Generate return
                epsilon = np.random.standard_normal()
                returns_sim[t] = mu_scaled + np.sqrt(sigma2_sim[t]) * epsilon
            
            simulations[sim, :] = returns_sim
        
        return simulations

class StressScenario:
    """Define different stress testing scenarios"""
    
    SCENARIOS = {
        'normal': {
            'name': 'Normal Market',
            'volatility_multiplier': 1.0,
            'description': 'Normal market conditions based on historical data'
        },
        'high_vol': {
            'name': 'High Volatility',
            'volatility_multiplier': 2.0,
            'description': 'High volatility period (2x normal volatility)'
        },
        'low_vol': {
            'name': 'Low Volatility',
            'volatility_multiplier': 0.5,
            'description': 'Low volatility period (0.5x normal volatility)'
        },
        'crisis': {
            'name': 'Financial Crisis',
            'volatility_multiplier': 3.0,
            'description': 'Crisis-like conditions (3x normal volatility)'
        },
        'extreme_crisis': {
            'name': 'Extreme Crisis',
            'volatility_multiplier': 5.0,
            'description': 'Extreme crisis conditions (5x normal volatility)'
        }
    }
    
    @classmethod
    def get_scenario(cls, scenario_name: str) -> Dict[str, Any]:
        """Get scenario configuration"""
        if scenario_name not in cls.SCENARIOS:
            available = ', '.join(cls.SCENARIOS.keys())
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {available}")
        
        return cls.SCENARIOS[scenario_name]
    
    @classmethod
    def get_all_scenarios(cls) -> List[str]:
        """Get list of all available scenarios"""
        return list(cls.SCENARIOS.keys())

class GARCHStressTester:
    """Main class for GARCH-based strategy stress testing"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.garch_model = GARCHModel()
        self.metrics_calculator = TradingMetricsCalculator()
        self.fitted_data = None
        
    def fit_garch_model(self, data: pd.DataFrame) -> Dict[str, float]:
        """Fit GARCH model to historical data"""
        self.logger.info("📊 Fitting GARCH(1,1) model to historical data...")
        
        # Calculate returns
        if 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 100:
            raise ValueError("Insufficient data for GARCH fitting (minimum 100 observations required)")
        
        # Fit GARCH model
        params = self.garch_model.fit(returns.values)
        self.fitted_data = data
        
        self.logger.info("✅ GARCH model fitted successfully:")
        self.logger.info(f"   • Omega: {params['omega']:.6f}")
        self.logger.info(f"   • Alpha: {params['alpha']:.4f}")
        self.logger.info(f"   • Beta: {params['beta']:.4f}")
        self.logger.info(f"   • Mu: {params['mu']:.6f}")
        self.logger.info(f"   • Persistence: {params['persistence']:.4f}")
        self.logger.info(f"   • Unconditional Variance: {params['unconditional_variance']:.6f}")
        
        return params
    
    def generate_stress_scenarios(self, scenarios: List[str], n_steps: int = 252, 
                                n_simulations: int = 100, random_seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate price paths for stress testing scenarios"""
        if not self.garch_model.fitted:
            raise ValueError("GARCH model must be fitted before generating scenarios")
        
        self.logger.info(f"🎲 Generating {n_simulations} simulations for {len(scenarios)} scenarios...")
        
        scenario_data = {}
        
        for scenario_name in scenarios:
            scenario_config = StressScenario.get_scenario(scenario_name)
            self.logger.info(f"   Generating {scenario_config['name']} scenario...")
            
            # Generate returns using GARCH simulation
            simulated_returns = self.garch_model.simulate_returns(
                n_steps=n_steps,
                n_simulations=n_simulations,
                volatility_multiplier=scenario_config['volatility_multiplier'],
                random_seed=random_seed + hash(scenario_name) % 1000
            )
            
            # Convert returns to price paths
            initial_price = self.fitted_data['Close'].iloc[-1] if self.fitted_data is not None else 100.0
            price_paths = np.zeros((n_simulations, n_steps + 1))
            price_paths[:, 0] = initial_price
            
            for sim in range(n_simulations):
                for t in range(n_steps):
                    price_paths[sim, t + 1] = price_paths[sim, t] * (1 + simulated_returns[sim, t])
            
            scenario_data[scenario_name] = {
                'config': scenario_config,
                'returns': simulated_returns,
                'price_paths': price_paths,
                'stats': self._calculate_scenario_stats(simulated_returns)
            }
        
        self.logger.info("✅ Scenario generation completed")
        return scenario_data
    
    def _calculate_scenario_stats(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for a scenario's returns"""
        returns_flat = returns.flatten()
        
        return {
            'mean_return': np.mean(returns_flat),
            'volatility': np.std(returns_flat) * np.sqrt(252),
            'skewness': stats.skew(returns_flat),
            'kurtosis': stats.kurtosis(returns_flat),
            'var_95': np.percentile(returns_flat, 5),
            'var_99': np.percentile(returns_flat, 1),
            'max_drawdown': self._calculate_max_drawdown(returns)
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown across all simulations"""
        max_dd = 0
        
        for sim in range(returns.shape[0]):
            cumulative = np.cumprod(1 + returns[sim, :])
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_dd = min(max_dd, np.min(drawdown))
        
        return max_dd
    
    def stress_test_model(self, model: PPO, vec_env: Any, scenario_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Stress test RL model across different scenarios"""
        self.logger.info("🧪 Starting stress testing of RL model...")
        
        results = {}
        
        for scenario_name, scenario_info in scenario_data.items():
            self.logger.info(f"   Testing {scenario_info['config']['name']} scenario...")
            
            scenario_results = []
            price_paths = scenario_info['price_paths']
            
            # Test model on multiple price paths from this scenario
            n_paths_to_test = min(10, price_paths.shape[0])  # Test on subset for efficiency
            
            for path_idx in range(n_paths_to_test):
                # Convert price path to DataFrame format
                price_path = price_paths[path_idx]
                test_data = self._create_test_data_from_path(price_path)
                
                # Test model on this path
                path_results = self._evaluate_model_on_path(model, test_data, vec_env)
                scenario_results.append(path_results)
            
            # Aggregate results for this scenario
            aggregated_results = self._aggregate_scenario_results(scenario_results)
            aggregated_results['scenario_stats'] = scenario_info['stats']
            aggregated_results['scenario_config'] = scenario_info['config']
            
            results[scenario_name] = aggregated_results
            
            self.logger.info(f"   ✅ {scenario_info['config']['name']}: Mean reward = {aggregated_results['mean_total_reward']:.2f}")
        
        return results
    
    def _create_test_data_from_path(self, price_path: np.ndarray) -> pd.DataFrame:
        """Create test data DataFrame from price path"""
        n_steps = len(price_path)
        dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
        
        # Create basic OHLCV data from price path
        data = pd.DataFrame({
            'Open': price_path,
            'High': price_path * (1 + np.random.uniform(0, 0.02, n_steps)),  # Small random high
            'Low': price_path * (1 - np.random.uniform(0, 0.02, n_steps)),   # Small random low
            'Close': price_path,
            'Volume': np.random.randint(1000000, 10000000, n_steps)  # Random volume
        }, index=dates)
        
        # Ensure High >= Open, Close and Low <= Open, Close
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def _evaluate_model_on_path(self, model: PPO, test_data: pd.DataFrame, vec_env: Any) -> Dict[str, float]:
        """Evaluate model performance on a single price path"""
        try:
            # Create temporary environment for this path
            temp_env = StrategyEnv(test_data)
            temp_vec_env = DummyVecEnv([lambda: temp_env])
            
            # Apply same normalization if available
            if hasattr(vec_env, 'normalize_obs') and vec_env.normalize_obs:
                # Use the same normalization statistics
                temp_vec_env = VecNormalize(temp_vec_env, training=False)
                if hasattr(vec_env, 'obs_rms'):
                    temp_vec_env.obs_rms = vec_env.obs_rms
                if hasattr(vec_env, 'ret_rms'):
                    temp_vec_env.ret_rms = vec_env.ret_rms
            
            # Run single episode
            obs = temp_vec_env.reset()
            total_reward = 0
            episode_returns = []
            done = False
            step = 0
            max_steps = min(len(test_data) - 1, ENV_CONFIG.get('max_episode_steps', 1000))
            
            while not done and step < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = temp_vec_env.step(action)
                
                total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                episode_returns.append(reward[0] if isinstance(reward, (list, np.ndarray)) else reward)
                step += 1
            
            # Calculate metrics
            returns_series = pd.Series(episode_returns)
            metrics = self.metrics_calculator.calculate_returns_metrics(returns_series)
            
            # Add path-specific metrics
            metrics['total_reward'] = total_reward
            metrics['episode_length'] = step
            
            temp_vec_env.close()
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error evaluating path: {e}")
            return {'total_reward': 0, 'episode_length': 0}
    
    def _aggregate_scenario_results(self, scenario_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate results across multiple paths in a scenario"""
        if not scenario_results:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for result in scenario_results:
            for metric, value in result.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate aggregate statistics
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
                aggregated[f'median_{metric}'] = np.median(values)
        
        return aggregated
    
    def compare_scenarios(self, stress_test_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare performance across different stress scenarios"""
        self.logger.info("📈 Comparing performance across scenarios...")
        
        comparison_data = []
        
        for scenario_name, results in stress_test_results.items():
            scenario_config = results.get('scenario_config', {})
            
            row = {
                'Scenario': scenario_config.get('name', scenario_name),
                'Volatility_Multiplier': scenario_config.get('volatility_multiplier', 1.0),
                'Mean_Total_Reward': results.get('mean_total_reward', 0),
                'Std_Total_Reward': results.get('std_total_reward', 0),
                'Min_Total_Reward': results.get('min_total_reward', 0),
                'Max_Total_Reward': results.get('max_total_reward', 0),
                'Mean_Sharpe_Ratio': results.get('mean_sharpe_ratio', 0),
                'Mean_Max_Drawdown': results.get('mean_max_drawdown', 0),
                'Mean_Win_Rate': results.get('mean_win_rate', 0)
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('Volatility_Multiplier')

class StressTestVisualizer:
    """Create visualizations for stress test results"""
    
    def __init__(self, output_dir: Path, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use(PLOT_CONFIG.get('style', 'default'))
        sns.set_palette("viridis")
    
    def plot_scenario_comparison(self, comparison_df: pd.DataFrame) -> Path:
        """Create scenario comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stress Test Results - Scenario Comparison', fontsize=16, fontweight='bold')
        
        # 1. Mean Total Reward vs Volatility
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            comparison_df['Volatility_Multiplier'], 
            comparison_df['Mean_Total_Reward'],
            c=comparison_df['Volatility_Multiplier'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add scenario labels
        for i, row in comparison_df.iterrows():
            ax1.annotate(
                row['Scenario'], 
                (row['Volatility_Multiplier'], row['Mean_Total_Reward']),
                xytext=(5, 5), textcoords='offset points', fontsize=9
            )
        
        ax1.set_xlabel('Volatility Multiplier')
        ax1.set_ylabel('Mean Total Reward')
        ax1.set_title('Performance vs Volatility')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(
            comparison_df['Scenario'],
            comparison_df['Mean_Sharpe_Ratio'],
            color='lightblue',
            alpha=0.7
        )
        ax2.set_title('Sharpe Ratio by Scenario')
        ax2.set_ylabel('Mean Sharpe Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, comparison_df['Mean_Sharpe_Ratio']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Risk-Return Scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(
            comparison_df['Mean_Max_Drawdown'].abs() * 100,
            comparison_df['Mean_Total_Reward'],
            c=comparison_df['Volatility_Multiplier'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        for i, row in comparison_df.iterrows():
            ax3.annotate(
                row['Scenario'], 
                (abs(row['Mean_Max_Drawdown']) * 100, row['Mean_Total_Reward']),
                xytext=(5, 5), textcoords='offset points', fontsize=9
            )
        
        ax3.set_xlabel('Mean Max Drawdown (%)')
        ax3.set_ylabel('Mean Total Reward')
        ax3.set_title('Risk-Return Profile')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Stability (Std vs Mean)
        ax4 = axes[1, 1]
        ax4.scatter(
            comparison_df['Mean_Total_Reward'],
            comparison_df['Std_Total_Reward'],
            c=comparison_df['Volatility_Multiplier'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        for i, row in comparison_df.iterrows():
            ax4.annotate(
                row['Scenario'], 
                (row['Mean_Total_Reward'], row['Std_Total_Reward']),
                xytext=(5, 5), textcoords='offset points', fontsize=9
            )
        
        ax4.set_xlabel('Mean Total Reward')
        ax4.set_ylabel('Std Total Reward')
        ax4.set_title('Performance Stability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "stress_test_comparison.png"
        plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Stress test comparison plot saved: {plot_path}")
        return plot_path
    
    def plot_garch_diagnostics(self, garch_model: GARCHModel, historical_returns: pd.Series) -> Path:
        """Create GARCH model diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GARCH Model Diagnostics', fontsize=16, fontweight='bold')
        
        # 1. Historical returns and volatility
        ax1 = axes[0, 0]
        ax1.plot(historical_returns.index, historical_returns, alpha=0.7, linewidth=0.5)
        ax1.set_title('Historical Returns')
        ax1.set_ylabel('Return')
        ax1.grid(True, alpha=0.3)
        
        # 2. Return distribution
        ax2 = axes[0, 1]
        ax2.hist(historical_returns.dropna(), bins=50, alpha=0.7, density=True)
        
        # Overlay normal distribution
        mu, sigma = historical_returns.mean(), historical_returns.std()
        x = np.linspace(historical_returns.min(), historical_returns.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
        ax2.set_title('Return Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. GARCH parameters visualization
        ax3 = axes[1, 0]
        if garch_model.fitted:
            params = ['omega', 'alpha', 'beta', 'persistence']
            values = [
                garch_model.params['omega'],
                garch_model.params['alpha'],
                garch_model.params['beta'],
                garch_model.params['persistence']
            ]
            
            bars = ax3.bar(params, values, alpha=0.7)
            ax3.set_title('GARCH Parameters')
            ax3.set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Volatility forecast
        ax4 = axes[1, 1]
        if garch_model.fitted:
            forecast_horizon = min(30, len(historical_returns) // 4)
            volatility_forecast = garch_model.predict_volatility(historical_returns.values, forecast_horizon)
            
            ax4.plot(range(1, forecast_horizon + 1), volatility_forecast, 'b-', linewidth=2, marker='o')
            ax4.set_title(f'{forecast_horizon}-Day Volatility Forecast')
            ax4.set_xlabel('Days Ahead')
            ax4.set_ylabel('Volatility')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "garch_diagnostics.png"
        plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 GARCH diagnostics plot saved: {plot_path}")
        return plot_path
    
    def generate_stress_test_report(self, garch_params: Dict, comparison_df: pd.DataFrame, 
                                  stress_results: Dict) -> Path:
        """Generate comprehensive stress test report"""
        report_path = self.output_dir / "stress_test_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GARCH Stress Test Report</title>
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
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GARCH-Based Stress Test Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # GARCH Model Parameters
        html_content += f"""
        <div class="section">
            <h2>GARCH Model Parameters</h2>
            <table class="metrics-table">
                <tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr>
                <tr><td>Omega (ω)</td><td>{garch_params.get('omega', 0):.6f}</td><td>Long-term variance component</td></tr>
                <tr><td>Alpha (α)</td><td>{garch_params.get('alpha', 0):.4f}</td><td>ARCH effect (short-term volatility persistence)</td></tr>
                <tr><td>Beta (β)</td><td>{garch_params.get('beta', 0):.4f}</td><td>GARCH effect (long-term volatility persistence)</td></tr>
                <tr><td>Mu (μ)</td><td>{garch_params.get('mu', 0):.6f}</td><td>Mean return</td></tr>
                <tr class="highlight"><td>Persistence (α + β)</td><td>{garch_params.get('persistence', 0):.4f}</td><td>Overall volatility persistence</td></tr>
                <tr><td>Unconditional Variance</td><td>{garch_params.get('unconditional_variance', 0):.6f}</td><td>Long-term variance level</td></tr>
            </table>
        </div>
        """
        
        # Scenario Performance Summary
        html_content += """
        <div class="section">
            <h2>Stress Test Results Summary</h2>
            <table class="metrics-table">
                <tr><th>Scenario</th><th>Vol. Multiplier</th><th>Mean Reward</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Risk Assessment</th></tr>
        """
        
        for _, row in comparison_df.iterrows():
            # Determine risk level
            if row['Volatility_Multiplier'] <= 1.0:
                risk_level = "Low"
                risk_class = "positive"
            elif row['Volatility_Multiplier'] <= 2.0:
                risk_level = "Medium"
                risk_class = "warning"
            else:
                risk_level = "High"
                risk_class = "negative"
            
            html_content += f"""
            <tr>
                <td>{row['Scenario']}</td>
                <td>{row['Volatility_Multiplier']:.1f}x</td>
                <td class="{'positive' if row['Mean_Total_Reward'] > 0 else 'negative'}">{row['Mean_Total_Reward']:.2f}</td>
                <td>{row['Mean_Sharpe_Ratio']:.2f}</td>
                <td class="negative">{row['Mean_Max_Drawdown']:.2%}</td>
                <td class="{risk_class}"><strong>{risk_level}</strong></td>
            </tr>
            """
        
        html_content += "</table></div>"
        
        # Performance Analysis
        best_scenario = comparison_df.loc[comparison_df['Mean_Total_Reward'].idxmax()]
        worst_scenario = comparison_df.loc[comparison_df['Mean_Total_Reward'].idxmin()]
        
        html_content += f"""
        <div class="section">
            <h2>Performance Analysis</h2>
            <h3>Key Findings:</h3>
            <ul>
                <li><strong>Best Performance:</strong> {best_scenario['Scenario']} (Mean Reward: {best_scenario['Mean_Total_Reward']:.2f})</li>
                <li><strong>Worst Performance:</strong> {worst_scenario['Scenario']} (Mean Reward: {worst_scenario['Mean_Total_Reward']:.2f})</li>
                <li><strong>Volatility Sensitivity:</strong> {'High' if abs(best_scenario['Mean_Total_Reward'] - worst_scenario['Mean_Total_Reward']) > 10 else 'Moderate'}</li>
                <li><strong>Model Persistence:</strong> {garch_params.get('persistence', 0):.3f} {'(Stationary)' if garch_params.get('persistence', 0) < 1 else '(Non-stationary)'}</li>
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
        
        self.logger.info(f"📋 Stress test report saved: {report_path}")
        return report_path

def main():
    """Main stress testing function"""
    parser = argparse.ArgumentParser(description="GARCH-based Strategy Stress Testing")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="Path to historical data CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--stress_scenarios",
        type=str,
        default="normal,high_vol,low_vol,crisis",
        help="Comma-separated list of stress scenarios"
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations per scenario"
    )
    parser.add_argument(
        "--simulation_length",
        type=int,
        default=252,
        help="Length of each simulation (trading days)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        # Set up paths
        model_path = Path(args.model_path)
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = RESULTS_DIR / f"stress_test_{timestamp}"
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up logging
        logger = setup_logging(output_dir)
        logger.info("🧪 Starting GARCH-based stress testing")
        
        # Load data
        if args.data_file:
            data_file = Path(args.data_file)
        else:
            data_file = DEFAULT_DATA_FILES.get('test', DEFAULT_DATA_FILES['train'])
        
        if not data_file.exists():
            logger.error(f"❌ Data file not found: {data_file}")
            return 1
        
        logger.info(f"📊 Loading data from: {data_file}")
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        logger.info(f"📈 Loaded {len(data)} rows of data")
        
        # Initialize stress tester
        stress_tester = GARCHStressTester(logger)
        
        # Fit GARCH model
        garch_params = stress_tester.fit_garch_model(data)
        
        # Parse scenarios
        scenarios = [s.strip() for s in args.stress_scenarios.split(',')]
        available_scenarios = StressScenario.get_all_scenarios()
        invalid_scenarios = [s for s in scenarios if s not in available_scenarios]
        
        if invalid_scenarios:
            logger.error(f"❌ Invalid scenarios: {invalid_scenarios}")
            logger.error(f"Available scenarios: {', '.join(available_scenarios)}")
            return 1
        
        # Generate stress scenarios
        scenario_data = stress_tester.generate_stress_scenarios(
            scenarios=scenarios,
            n_steps=args.simulation_length,
            n_simulations=args.n_simulations,
            random_seed=args.random_seed
        )
        
        # Load model
        logger.info(f"🔄 Loading trained model from: {model_path}")
        if not model_path.exists():
            logger.error(f"❌ Model file not found: {model_path}")
            return 1
        
        # Load model and environment
        from evaluate import StrategyEvaluator
        evaluator = StrategyEvaluator(logger)
        model, vec_env = evaluator.load_model_and_env(model_path, data)
        
        # Run stress tests
        stress_results = stress_tester.stress_test_model(model, vec_env, scenario_data)
        
        # Compare scenarios
        comparison_df = stress_tester.compare_scenarios(stress_results)
        
        # Generate visualizations
        visualizer = StressTestVisualizer(output_dir, logger)
        visualizer.plot_scenario_comparison(comparison_df)
        visualizer.plot_garch_diagnostics(stress_tester.garch_model, data['Close'].pct_change().dropna())
        visualizer.generate_stress_test_report(garch_params, comparison_df, stress_results)
        
        # Save results to JSON
        results_summary = {
            'garch_parameters': garch_params,
            'stress_scenarios': {name: {
                'config': info['config'],
                'stats': info['stats']
            } for name, info in scenario_data.items()},
            'performance_results': comparison_df.to_dict('records'),
            'test_configuration': {
                'n_simulations': args.n_simulations,
                'simulation_length': args.simulation_length,
                'random_seed': args.random_seed,
                'scenarios_tested': scenarios
            },
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'data_file': str(data_file)
        }
        
        results_file = output_dir / "stress_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("🎉 Stress testing completed successfully!")
        logger.info("📊 Results Summary:")
        logger.info(f"   • GARCH Persistence: {garch_params.get('persistence', 0):.3f}")
        
        for _, row in comparison_df.iterrows():
            logger.info(f"   • {row['Scenario']}: Reward={row['Mean_Total_Reward']:.2f}, Sharpe={row['Mean_Sharpe_Ratio']:.2f}")
        
        logger.info(f"📂 All results saved to: {output_dir}")
        
        # Close environment
        vec_env.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Stress testing failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
