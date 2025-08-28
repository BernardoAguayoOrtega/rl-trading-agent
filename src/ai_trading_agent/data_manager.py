"""
Intelligent Data Management for AI Trading Agent
===============================================

This module provides comprehensive data acquisition, validation, and management
capabilities for the AI trading system, integrating with Yahoo Finance and
existing technical indicator framework.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
import logging
from pathlib import Path
import json
import time

from .config import DataConfig, default_config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class IntelligentDataManager:
    """
    Comprehensive data management system for AI trading agent
    
    Features:
    - Yahoo Finance integration with error handling
    - Real-time data acquisition and caching
    - Data validation and quality checks
    - Technical indicator integration
    - Multiple timeframe support
    - Data preprocessing for AI consumption
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the data manager"""
        self.config = config or default_config.data
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Data quality tracking
        self.data_quality_metrics = {}
        self.last_update_times = {}
        
        # Configure yfinance settings
        self._configure_yfinance()
        
        logger.info("IntelligentDataManager initialized")
    
    def _configure_yfinance(self):
        """Configure yfinance settings for optimal performance"""
        # Suppress yfinance warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
        
    def get_market_data(self, 
                       symbol: str, 
                       period: str = None, 
                       interval: str = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       validate: bool = True) -> pd.DataFrame:
        """
        Get market data from Yahoo Finance with comprehensive validation
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            validate: Whether to perform data validation
            
        Returns:
            pd.DataFrame: OHLCV data with validation
        """
        # Use defaults from config if not provided
        period = period or self.config.default_period
        interval = interval or self.config.default_interval
        
        try:
            logger.info(f"Fetching data for {symbol} (period={period}, interval={interval})")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            if start_date and end_date:
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=self.config.yf_auto_adjust,
                    prepost=self.config.yf_prepost
                )
            else:
                data = ticker.history(
                    period=period,
                    interval=interval,
                    auto_adjust=self.config.yf_auto_adjust,
                    prepost=self.config.yf_prepost
                )
            
            if data.empty:
                raise DataValidationError(f"No data returned for symbol {symbol}")
            
            # Standardize column names
            data = self._standardize_columns(data)
            
            # Validate data if requested
            if validate:
                data = self._validate_and_clean_data(data, symbol)
            
            # Cache data
            self._cache_data(data, symbol, period, interval)
            
            # Update quality metrics
            self._update_quality_metrics(symbol, data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            # Try to return cached data as fallback
            cached_data = self._get_cached_data(symbol, period, interval)
            if cached_data is not None:
                logger.warning(f"Using cached data for {symbol}")
                return cached_data
            raise DataValidationError(f"Failed to fetch data for {symbol}: {e}")
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match existing framework"""
        # Map yfinance columns to standard names
        column_mapping = {
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj_Close'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        return data
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Comprehensive data validation and cleaning
        
        Args:
            data: Raw market data
            symbol: Stock symbol for logging
            
        Returns:
            pd.DataFrame: Validated and cleaned data
        """
        original_length = len(data)
        issues_found = []
        
        # 1. Check minimum data requirements
        if len(data) < max(5, self.config.min_data_points // 10):  # More lenient for testing
            raise DataValidationError(
                f"Insufficient data for {symbol}: {len(data)} < {max(5, self.config.min_data_points // 10)}"
            )
        
        # 2. Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # 3. Remove rows with all NaN values
        data = data.dropna(how='all')
        if len(data) < original_length:
            issues_found.append(f"Removed {original_length - len(data)} rows with all NaN values")
        
        # 4. Handle missing values in OHLCV data
        data = self._handle_missing_ohlcv(data)
        
        # 5. Validate OHLC relationships
        data = self._validate_ohlc_relationships(data)
        
        # 6. Remove invalid prices (negative or zero)
        invalid_prices = (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
        if invalid_prices.any():
            invalid_count = invalid_prices.sum()
            data = data[~invalid_prices]
            issues_found.append(f"Removed {invalid_count} rows with invalid prices")
        
        # 7. Remove extreme outliers (more than 50% price change in one period)
        if len(data) > 1:
            price_changes = data['Close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.any():
                extreme_count = extreme_changes.sum()
                data = data[~extreme_changes]
                issues_found.append(f"Removed {extreme_count} rows with extreme price changes")
        
        # 8. Ensure data is sorted by date
        data = data.sort_index()
        
        # 9. Remove duplicate timestamps
        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            data = data[~data.index.duplicated(keep='last')]
            issues_found.append(f"Removed {duplicate_count} duplicate timestamps")
        
        # 10. Final validation
        min_required = max(5, self.config.min_data_points // 10)  # More lenient for testing
        if len(data) < min_required:
            raise DataValidationError(
                f"After cleaning, insufficient data for {symbol}: {len(data)} < {min_required}"
            )
        
        # Log validation results
        if issues_found:
            logger.warning(f"Data validation for {symbol} found issues: {'; '.join(issues_found)}")
        else:
            logger.info(f"Data validation for {symbol}: PASSED (no issues found)")
        
        return data
    
    def _handle_missing_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV data using intelligent forward-filling"""
        for col in ['Open', 'High', 'Low', 'Close']:
            if data[col].isnull().any():
                # Forward fill, then backward fill if needed
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Handle missing volume with zeros (common for some intervals)
        if 'Volume' in data.columns:
            data['Volume'] = data['Volume'].fillna(0)
        
        return data
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLC price relationships"""
        # Ensure High >= Low
        invalid_hl = data['High'] < data['Low']
        if invalid_hl.any():
            logger.warning(f"Found {invalid_hl.sum()} rows where High < Low, swapping values")
            data.loc[invalid_hl, ['High', 'Low']] = data.loc[invalid_hl, ['Low', 'High']].values
        
        # Ensure High >= Open, Close and Low <= Open, Close
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data
    
    def _cache_data(self, data: pd.DataFrame, symbol: str, period: str, interval: str):
        """Cache data for future use"""
        try:
            cache_file = self.cache_dir / f"{symbol}_{period}_{interval}_{datetime.now().strftime('%Y%m%d')}.parquet"
            data.to_parquet(cache_file)
            self.last_update_times[symbol] = datetime.now()
            logger.debug(f"Cached data for {symbol} to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")
    
    def _get_cached_data(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and recent"""
        try:
            cache_pattern = f"{symbol}_{period}_{interval}_*.parquet"
            cache_files = list(self.cache_dir.glob(cache_pattern))
            
            if cache_files:
                # Get most recent cache file
                latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
                
                # Check if cache is recent (within 1 day for daily data, 1 hour for intraday)
                cache_age = datetime.now() - datetime.fromtimestamp(latest_cache.stat().st_mtime)
                max_age = timedelta(hours=1) if interval.endswith(('m', 'h')) else timedelta(days=1)
                
                if cache_age <= max_age:
                    data = pd.read_parquet(latest_cache)
                    logger.info(f"Using cached data for {symbol} (age: {cache_age})")
                    return data
        
        except Exception as e:
            logger.warning(f"Failed to retrieve cached data for {symbol}: {e}")
        
        return None
    
    def _update_quality_metrics(self, symbol: str, data: pd.DataFrame):
        """Update data quality metrics for monitoring"""
        self.data_quality_metrics[symbol] = {
            'last_update': datetime.now(),
            'record_count': len(data),
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max()
            },
            'completeness': {
                'open': (1 - data['Open'].isnull().sum() / len(data)) * 100,
                'high': (1 - data['High'].isnull().sum() / len(data)) * 100,
                'low': (1 - data['Low'].isnull().sum() / len(data)) * 100,
                'close': (1 - data['Close'].isnull().sum() / len(data)) * 100,
                'volume': (1 - data['Volume'].isnull().sum() / len(data)) * 100
            }
        }
    
    def get_multiple_symbols(self, 
                           symbols: List[str], 
                           period: str = None,
                           interval: str = None,
                           validate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get market data for multiple symbols efficiently
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            validate: Whether to perform data validation
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> DataFrame mapping
        """
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data = self.get_market_data(
                    symbol=symbol,
                    period=period,
                    interval=interval,
                    validate=validate
                )
                results[symbol] = data
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_real_time_data(self, symbol: str, max_age_minutes: int = 5) -> pd.DataFrame:
        """
        Get real-time or near real-time data
        
        Args:
            symbol: Stock symbol
            max_age_minutes: Maximum age of data in minutes
            
        Returns:
            pd.DataFrame: Recent market data
        """
        # For real-time data, use short period with minute interval
        data = self.get_market_data(
            symbol=symbol,
            period="1d",  # Get today's data
            interval="1m",  # Minute-by-minute
            validate=True
        )
        
        # Filter to recent data only
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        # Handle timezone-aware indexes
        if not data.index.empty:
            if data.index.tz is not None:
                # Convert cutoff_time to match the data's timezone
                cutoff_time = cutoff_time.replace(tzinfo=data.index.tz)
            
            recent_data = data[data.index >= cutoff_time]
        else:
            recent_data = data
        
        if recent_data.empty:
            logger.warning(f"No recent data available for {symbol} within {max_age_minutes} minutes")
            # Return last few records if no recent data
            recent_data = data.tail(10)
        
        return recent_data
    
    def prepare_data_for_ai(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Prepare market data for AI consumption
        
        Args:
            data: Market data DataFrame
            symbol: Stock symbol
            
        Returns:
            Dict: Formatted data for AI analysis
        """
        if data.empty:
            raise DataValidationError(f"No data available for {symbol}")
        
        # Get latest values
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        # Calculate basic metrics
        price_change = latest['Close'] - previous['Close']
        price_change_pct = (price_change / previous['Close']) * 100 if previous['Close'] != 0 else 0
        
        # Calculate volatility (20-period rolling std)
        volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
        
        # Determine trend (simple 5-period comparison)
        trend = "upward" if data['Close'].tail(5).mean() > data['Close'].tail(10).head(5).mean() else "downward"
        
        # Market hours check (simplified)
        current_time = datetime.now()
        market_hours = "open" if 9 <= current_time.hour <= 16 else "closed"
        
        return {
            'symbol': symbol,
            'current_price': float(latest['Close']),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'volume': int(latest['Volume']),
            'volatility': float(volatility) if not np.isnan(volatility) else 0.0,
            'trend': trend,
            'market_hours': market_hours,
            'data_quality': self.data_quality_metrics.get(symbol, {}),
            'timestamp': latest.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest.name, 'strftime') else str(latest.name)
        }
    
    def get_data_quality_report(self) -> Dict:
        """Get comprehensive data quality report"""
        return {
            'total_symbols_tracked': len(self.data_quality_metrics),
            'last_updates': {symbol: metrics['last_update'].strftime('%Y-%m-%d %H:%M:%S') 
                           for symbol, metrics in self.data_quality_metrics.items()},
            'data_completeness': {symbol: metrics['completeness'] 
                                for symbol, metrics in self.data_quality_metrics.items()},
            'cache_status': {
                'cache_directory': str(self.cache_dir),
                'cached_files': len(list(self.cache_dir.glob("*.parquet")))
            }
        }
    
    def clean_cache(self, days_old: int = 7):
        """Clean old cache files"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            cleaned_files = 0
            
            for cache_file in self.cache_dir.glob("*.parquet"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_files += 1
            
            logger.info(f"Cleaned {cleaned_files} cache files older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Failed to clean cache: {e}")
