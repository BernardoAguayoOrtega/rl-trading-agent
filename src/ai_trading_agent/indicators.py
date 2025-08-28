"""
Technical Indicators Integration for AI Trading Agent
====================================================

This module integrates your existing technical indicators from notebooks/indicators.ipynb
into a modular system that the AI agent can use dynamically.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

# Import required classes
from .config import DataConfig

logger = logging.getLogger(__name__)


class TechnicalIndicatorFactory:
    """
    Modular system to integrate existing indicator library
    Enables AI agent to dynamically use and combine indicators
    """
    
    def __init__(self):
        self.indicator_registry = self._build_indicator_registry()
        self.cache = {}  # Cache calculated indicators for performance
        
    def _build_indicator_registry(self) -> Dict[str, Dict[str, Any]]:
        """Registry of all available indicators with metadata"""
        return {
            # Trend indicators
            'sma': {
                'function': self.ocpSma,
                'category': 'trend',
                'description': 'Simple Moving Average - trend following',
                'parameters': {'periodo': 20, 'col': 'Close'},
                'min_data_points': 20,
                'output_columns': ['s{periodo}']
            },
            'ema': {
                'function': self.ocpExp,
                'category': 'trend', 
                'description': 'Exponential Moving Average - responsive trend',
                'parameters': {'periodo': 20, 'col': 'Close'},
                'min_data_points': 20,
                'output_columns': ['e{periodo}']
            },
            
            # Momentum indicators
            'rsi': {
                'function': self.ocpRsi,
                'category': 'momentum',
                'description': 'Relative Strength Index - momentum oscillator',
                'parameters': {'periodo': 14, 'col': 'Close'},
                'min_data_points': 30,
                'output_columns': ['rsi{periodo}']
            },
            'macd': {
                'function': self.ocpMacd,
                'category': 'momentum',
                'description': 'MACD - trend and momentum',
                'parameters': {'fast': 12, 'slow': 26, 'suavizado': 9, 'col': 'Close'},
                'min_data_points': 35,
                'output_columns': ['macdL', 'macdS', 'macdH']
            },
            
            # Volatility indicators
            'bb': {
                'function': self.ocpBollingerBands,
                'category': 'volatility',
                'description': 'Bollinger Bands - volatility and mean reversion',
                'parameters': {'periodo': 20, 'std_dev': 2, 'col': 'Close'},
                'min_data_points': 25,
                'output_columns': ['bb_upper{periodo}', 'bb_middle{periodo}', 'bb_lower{periodo}']
            },
            'atr': {
                'function': self.ocpAtr,
                'category': 'volatility',
                'description': 'Average True Range - volatility measure',
                'parameters': {'periodo': 14},
                'min_data_points': 14,
                'output_columns': ['atr{periodo}', 'atr%{periodo}']
            },
            
            # Volume indicators
            'volume_sma': {
                'function': self.ocpVolumeSma,
                'category': 'volume',
                'description': 'Volume Simple Moving Average',
                'parameters': {'periodo': 20},
                'min_data_points': 20,
                'output_columns': ['volume_sma{periodo}']
            }
        }
    
    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str, **kwargs) -> pd.DataFrame:
        """Calculate a specific indicator with custom parameters"""
        if indicator_name not in self.indicator_registry:
            raise ValueError(f"Indicator {indicator_name} not found in registry")
            
        indicator_info = self.indicator_registry[indicator_name]
        
        # Check minimum data requirements
        if len(data) < indicator_info['min_data_points']:
            raise ValueError(f"Insufficient data for {indicator_name}: need {indicator_info['min_data_points']}, got {len(data)}")
        
        # Merge default parameters with custom ones
        params = {**indicator_info['parameters'], **kwargs}
        
        # Create cache key
        cache_key = f"{indicator_name}_{hash(str(sorted(params.items())))}"
        
        # Check cache
        if cache_key in self.cache:
            cached_result, cached_data_hash = self.cache[cache_key]
            current_data_hash = hash(str(data.tail(100).values.tobytes()))  # Hash last 100 rows
            if cached_data_hash == current_data_hash:
                return cached_result
        
        # Calculate indicator
        try:
            result = indicator_info['function'](data, **params)
            
            # Cache result
            current_data_hash = hash(str(data.tail(100).values.tobytes()))
            self.cache[cache_key] = (result, current_data_hash)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate {indicator_name}: {e}")
            raise
    
    def calculate_multiple_indicators(self, data: pd.DataFrame, indicators: List[str], **global_kwargs) -> pd.DataFrame:
        """Calculate multiple indicators efficiently"""
        result_data = data.copy()
        
        for indicator_name in indicators:
            try:
                indicator_result = self.calculate_indicator(data, indicator_name, **global_kwargs)
                
                # Merge results (align by index)
                for col in indicator_result.columns:
                    if col not in result_data.columns:
                        result_data[col] = indicator_result[col]
                        
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator_name}: {e}")
                continue
        
        return result_data
    
    def get_ai_ready_indicators(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate key indicators and format for AI consumption"""
        try:
            # Calculate essential indicators
            essential_indicators = ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']
            indicator_data = self.calculate_multiple_indicators(data, essential_indicators)
            
            # Get latest values
            latest = indicator_data.iloc[-1]
            
            # Format for AI
            formatted_indicators = {
                'sma_20': float(latest.get('s20', 0)),
                'ema_20': float(latest.get('e20', 0)),
                'rsi_14': float(latest.get('rsi14', 50)),
                'macd_line': float(latest.get('macdL', 0)),
                'macd_signal': float(latest.get('macdS', 0)),
                'macd_histogram': float(latest.get('macdH', 0)),
                'bb_upper': float(latest.get('bb_upper20', 0)),
                'bb_middle': float(latest.get('bb_middle20', 0)),
                'bb_lower': float(latest.get('bb_lower20', 0)),
                'atr': float(latest.get('atr14', 0)),
                'atr_percent': float(latest.get('atr%14', 0))
            }
            
            # Add trend analysis
            current_price = float(latest['Close'])
            sma_20 = formatted_indicators['sma_20']
            ema_20 = formatted_indicators['ema_20']
            
            formatted_indicators.update({
                'price_vs_sma20': 'above' if current_price > sma_20 else 'below',
                'price_vs_ema20': 'above' if current_price > ema_20 else 'below',
                'rsi_signal': 'overbought' if formatted_indicators['rsi_14'] > 70 else 'oversold' if formatted_indicators['rsi_14'] < 30 else 'neutral',
                'macd_signal_status': 'bullish' if formatted_indicators['macd_line'] > formatted_indicators['macd_signal'] else 'bearish',
                'bb_position': 'upper' if current_price > formatted_indicators['bb_upper'] else 'lower' if current_price < formatted_indicators['bb_lower'] else 'middle'
            })
            
            return formatted_indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate AI-ready indicators for {symbol}: {e}")
            return {}
    
    # ============= CORE INDICATOR IMPLEMENTATIONS =============
    
    def ocpSma(self, data: pd.DataFrame, periodo: int = 20, col: str = 'Close') -> pd.DataFrame:
        """Simple Moving Average"""
        result = data.copy()
        sma_col = f's{periodo}'
        result[sma_col] = data[col].rolling(window=periodo).mean()
        return result
    
    def ocpExp(self, data: pd.DataFrame, periodo: int = 20, col: str = 'Close') -> pd.DataFrame:
        """Exponential Moving Average"""
        result = data.copy()
        ema_col = f'e{periodo}'
        result[ema_col] = data[col].ewm(span=periodo).mean()
        return result
    
    def ocpRsi(self, data: pd.DataFrame, periodo: int = 14, col: str = 'Close') -> pd.DataFrame:
        """Relative Strength Index"""
        result = data.copy()
        
        # Calculate price changes
        delta = data[col].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate rolling averages
        avg_gains = gains.rolling(window=periodo).mean()
        avg_losses = losses.rolling(window=periodo).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        result[f'rsi{periodo}'] = rsi
        return result
    
    def ocpMacd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, suavizado: int = 9, col: str = 'Close') -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        result = data.copy()
        
        # Calculate EMAs
        ema_fast = data[col].ewm(span=fast).mean()
        ema_slow = data[col].ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=suavizado).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        result['macdL'] = macd_line
        result['macdS'] = signal_line
        result['macdH'] = histogram
        
        return result
    
    def ocpBollingerBands(self, data: pd.DataFrame, periodo: int = 20, std_dev: float = 2, col: str = 'Close') -> pd.DataFrame:
        """Bollinger Bands"""
        result = data.copy()
        
        # Middle line (SMA)
        sma = data[col].rolling(window=periodo).mean()
        
        # Standard deviation
        std = data[col].rolling(window=periodo).std()
        
        # Upper and lower bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        result[f'bb_upper{periodo}'] = upper_band
        result[f'bb_middle{periodo}'] = sma
        result[f'bb_lower{periodo}'] = lower_band
        
        return result
    
    def ocpAtr(self, data: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
        """Average True Range"""
        result = data.copy()
        
        # Calculate True Range
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift())
        low_close_prev = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # ATR is the EMA of True Range
        atr = true_range.ewm(span=periodo).mean()
        
        # ATR as percentage of price
        atr_percent = (atr / data['Close']) * 100
        
        result[f'atr{periodo}'] = atr
        result[f'atr%{periodo}'] = atr_percent
        
        return result
    
    def ocpVolumeSma(self, data: pd.DataFrame, periodo: int = 20) -> pd.DataFrame:
        """Volume Simple Moving Average"""
        result = data.copy()
        volume_sma_col = f'volume_sma{periodo}'
        result[volume_sma_col] = data['Volume'].rolling(window=periodo).mean()
        return result


class MarketDataProcessor:
    """
    High-level processor combining data management and technical indicators
    for comprehensive market analysis
    """
    
    def __init__(self, data_manager=None, data_config: Optional[DataConfig] = None):
        """Initialize the market data processor"""
        if data_manager is not None:
            self.data_manager = data_manager
        else:
            # Import here to avoid circular imports
            from .data_manager import IntelligentDataManager
            self.data_manager = IntelligentDataManager(data_config)
        
        self.indicator_factory = TechnicalIndicatorFactory()
        
        logger.info("MarketDataProcessor initialized")
    
    def get_complete_market_analysis(self, symbol: str, period: str = None, interval: str = None) -> Dict:
        """
        Get complete market analysis including data and indicators
        
        Args:
            symbol: Stock symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Dict: Complete market analysis for AI consumption
        """
        try:
            # Get market data
            market_data = self.data_manager.get_market_data(
                symbol=symbol,
                period=period,
                interval=interval,
                validate=True
            )
            
            # Prepare basic data for AI
            ai_data = self.data_manager.prepare_data_for_ai(market_data, symbol)
            
            # Calculate technical indicators
            technical_indicators = self.indicator_factory.get_ai_ready_indicators(market_data, symbol)
            
            # Combine everything
            complete_analysis = {
                **ai_data,
                'technical_indicators': technical_indicators,
                'data_points': len(market_data),
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(f"Complete market analysis generated for {symbol}")
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Failed to generate complete market analysis for {symbol}: {e}")
            raise
    
    def get_multi_symbol_analysis(self, symbols: List[str], period: str = None, interval: str = None) -> Dict[str, Dict]:
        """Get complete analysis for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                analysis = self.get_complete_market_analysis(symbol, period, interval)
                results[symbol] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def get_real_time_analysis(self, symbol: str, max_age_minutes: int = 5) -> Dict:
        """Get real-time market analysis"""
        try:
            # Get real-time data
            real_time_data = self.data_manager.get_real_time_data(symbol, max_age_minutes)
            
            if real_time_data.empty:
                raise ValueError(f"No real-time data available for {symbol}")
            
            # Prepare for AI
            ai_data = self.data_manager.prepare_data_for_ai(real_time_data, symbol)
            
            # Calculate indicators on available data
            technical_indicators = self.indicator_factory.get_ai_ready_indicators(real_time_data, symbol)
            
            # Combine with real-time flag
            analysis = {
                **ai_data,
                'technical_indicators': technical_indicators,
                'is_real_time': True,
                'data_age_minutes': max_age_minutes,
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to get real-time analysis for {symbol}: {e}")
            raise
        
        # Merge default parameters with provided kwargs
        params = indicator_info['parameters'].copy()
        params.update(kwargs)
        
        # Check minimum data requirements
        min_points = indicator_info['min_data_points']
        if len(data) < min_points:
            raise ValueError(f"Insufficient data: need {min_points}, got {len(data)}")
        
        # Calculate indicator
        result_data = indicator_info['function'](data.copy(), **params)
        return result_data
    
    def calculate_indicator_suite(self, data: pd.DataFrame, indicator_list: Optional[List[Tuple[str, Dict]]] = None) -> Tuple[pd.DataFrame, List[Tuple[str, Dict]]]:
        """Calculate multiple indicators efficiently"""
        if indicator_list is None:
            # Default comprehensive suite
            indicator_list = [
                ('sma', {'periodo': 20}),
                ('sma', {'periodo': 50}),
                ('sma', {'periodo': 200}),
                ('ema', {'periodo': 12}),
                ('rsi', {'periodo': 14}),
                ('macd', {}),
                ('bb', {}),
                ('atr', {})
            ]
        
        result_data = data.copy()
        calculated_indicators = []
        
        for indicator_name, params in indicator_list:
            try:
                result_data = self.calculate_indicator(result_data, indicator_name, **params)
                calculated_indicators.append((indicator_name, params))
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator_name}: {e}")
        
        logger.info(f"Calculated {len(calculated_indicators)} indicators")
        return result_data, calculated_indicators
    
    def get_indicator_categories(self) -> Dict[str, List[Dict[str, str]]]:
        """Get indicators grouped by category for AI understanding"""
        categories = {}
        for name, info in self.indicator_registry.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'name': name,
                'description': info['description']
            })
        return categories
    
    def prepare_indicators_for_ai(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Prepare calculated indicators for AI consumption"""
        indicators = {}
        
        try:
            # Calculate standard indicators
            data_with_indicators, _ = self.calculate_indicator_suite(data)
            
            # Extract latest values for AI
            if len(data_with_indicators) > 0:
                latest = data_with_indicators.iloc[-1]
                
                # RSI
                if 'rsi14' in latest:
                    indicators['RSI'] = float(latest['rsi14'])
                
                # MACD
                if 'macdL' in latest and 'macdS' in latest:
                    indicators['MACD'] = float(latest['macdL'] - latest['macdS'])
                    indicators['MACD_Signal'] = float(latest['macdS'])
                    indicators['MACD_Histogram'] = float(latest['macdH'])
                
                # Moving Averages
                if 's20' in latest:
                    indicators['SMA_20'] = float(latest['s20'])
                if 's50' in latest:
                    indicators['SMA_50'] = float(latest['s50'])
                if 's200' in latest:
                    indicators['SMA_200'] = float(latest['s200'])
                if 'e12' in latest:
                    indicators['EMA_12'] = float(latest['e12'])
                
                # Bollinger Bands
                if all(col in latest for col in ['bb_upper20', 'bb_middle20', 'bb_lower20']):
                    bb_position = (latest['Close'] - latest['bb_lower20']) / (latest['bb_upper20'] - latest['bb_lower20'])
                    indicators['BB_Position'] = float(bb_position)
                    indicators['BB_Upper'] = float(latest['bb_upper20'])
                    indicators['BB_Lower'] = float(latest['bb_lower20'])
                
                # ATR
                if 'atr14' in latest:
                    indicators['ATR'] = float(latest['atr14'])
                if 'atr%14' in latest:
                    indicators['ATR_Percent'] = float(latest['atr%14'])
                
                # Volume indicators
                if 'volume_sma20' in latest and 'Volume' in latest:
                    volume_ratio = latest['Volume'] / latest['volume_sma20'] if latest['volume_sma20'] > 0 else 1.0
                    indicators['Volume_Ratio'] = float(volume_ratio)
        
        except Exception as e:
            logger.error(f"Failed to prepare indicators for AI: {e}")
        
        return indicators
    
    # =============================================================================
    # INDICATOR IMPLEMENTATIONS (from your existing framework)
    # =============================================================================
    
    def ocpSma(self, df: pd.DataFrame, periodo: int = 20, borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """Simple Moving Average - from your indicators library"""
        df[f's{periodo}'] = df[col].rolling(periodo).mean()
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpExp(self, df: pd.DataFrame, periodo: int = 20, borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """Exponential Moving Average - from your indicators library"""
        df[f'e{periodo}'] = df[col].ewm(span=periodo, adjust=False).mean()
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpRsi(self, df: pd.DataFrame, periodo: int = 14, borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """RSI implementation - from your indicators library"""
        df['dif'] = df[col].diff()
        df['win'] = np.where(df['dif'] > 0, df['dif'], 0)
        df['loss'] = np.where(df['dif'] < 0, abs(df['dif']), 0)
        
        df['emaWin'] = df['win'].ewm(span=periodo, adjust=False).mean()
        df['emaLoss'] = df['loss'].ewm(span=periodo, adjust=False).mean()
        
        df['rs'] = df.emaWin / df.emaLoss
        df[f'rsi{periodo}'] = 100 - (100 / (1 + df.rs))
        
        df.drop(['dif', 'win', 'loss', 'emaWin', 'emaLoss', 'rs'], axis=1, inplace=True)
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpMacd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, suavizado: int = 9, 
                borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """MACD implementation - from your indicators library"""
        df['macdL'] = df[col].ewm(span=fast, adjust=False).mean() - df[col].ewm(span=slow, adjust=False).mean()
        df['macdS'] = df.macdL.ewm(span=suavizado, adjust=False).mean()
        df['macdH'] = df.macdL - df.macdS
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpBollingerBands(self, df: pd.DataFrame, periodo: int = 20, std_dev: float = 2, 
                         borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """Bollinger Bands implementation"""
        df[f'bb_middle{periodo}'] = df[col].rolling(periodo).mean()
        bb_std = df[col].rolling(periodo).std()
        
        df[f'bb_upper{periodo}'] = df[f'bb_middle{periodo}'] + (bb_std * std_dev)
        df[f'bb_lower{periodo}'] = df[f'bb_middle{periodo}'] - (bb_std * std_dev)
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpAtr(self, df: pd.DataFrame, periodo: int = 14, borraNan: bool = False) -> pd.DataFrame:
        """ATR implementation - from your indicators library"""
        df['atrHL'] = df.High - df.Low
        df['atrHC'] = abs(df.High - df.Close.shift())
        df['atrLC'] = abs(df.Low - df.Close.shift())
        
        df['atrMax'] = df[['atrHL', 'atrHC', 'atrLC']].max(axis=1)
        df['atrMaxP'] = (df.atrMax / df.Close) * 100
        
        df[f'atr{periodo}'] = df.atrMax.ewm(span=periodo, adjust=False).mean()
        df[f'atr%{periodo}'] = df.atrMaxP.ewm(span=periodo, adjust=False).mean()
        
        df.drop(['atrHL', 'atrHC', 'atrLC', 'atrMax', 'atrMaxP'], axis=1, inplace=True)
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpVolumeSma(self, df: pd.DataFrame, periodo: int = 20, borraNan: bool = False) -> pd.DataFrame:
        """Volume SMA implementation"""
        df[f'volume_sma{periodo}'] = df['Volume'].rolling(periodo).mean()
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def ocpStochastic(self, df: pd.DataFrame, N: int = 14, M: int = 3, borraNan: bool = False) -> pd.DataFrame:
        """Stochastic oscillator implementation"""
        df['lowN'] = df['Low'].rolling(N).min()
        df['highN'] = df['High'].rolling(N).max()
        
        df['K'] = 100 * (df['Close'] - df['lowN']) / (df['highN'] - df['lowN'])
        df['D'] = df['K'].rolling(M).mean()
        
        df.drop(['lowN', 'highN'], axis=1, inplace=True)
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df



