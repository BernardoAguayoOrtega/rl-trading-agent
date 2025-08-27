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


class MarketDataProcessor:
    """
    Advanced data processing for AI consumption
    Combines market data with technical indicators
    """
    
    def __init__(self):
        self.indicator_factory = TechnicalIndicatorFactory()
    
    def process_for_ai_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Process market data and indicators for AI analysis
        
        Args:
            data: Market data DataFrame
            symbol: Stock symbol
            
        Returns:
            Dict: Comprehensive data package for AI
        """
        if data.empty:
            raise ValueError(f"No data provided for {symbol}")
        
        try:
            # Calculate technical indicators
            data_with_indicators = self.indicator_factory.calculate_indicator_suite(data)[0]
            
            # Prepare basic market data
            latest = data_with_indicators.iloc[-1]
            previous = data_with_indicators.iloc[-2] if len(data_with_indicators) > 1 else latest
            
            # Price action analysis
            price_change = latest['Close'] - previous['Close']
            price_change_pct = (price_change / previous['Close']) * 100 if previous['Close'] != 0 else 0
            
            # Volume analysis
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility measures
            volatility_20 = data['Close'].pct_change().tail(20).std() * 100
            
            # Technical indicators for AI
            technical_indicators = self.indicator_factory.prepare_indicators_for_ai(data_with_indicators, symbol)
            
            # Market structure analysis
            market_structure = self._analyze_market_structure(data_with_indicators)
            
            return {
                'symbol': symbol,
                'timestamp': latest.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest.name, 'strftime') else str(latest.name),
                'price_data': {
                    'current_price': float(latest['Close']),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct)
                },
                'volume_data': {
                    'current_volume': int(latest['Volume']),
                    'avg_volume_20': float(avg_volume),
                    'volume_ratio': float(volume_ratio)
                },
                'volatility_data': {
                    'volatility_20': float(volatility_20) if not np.isnan(volatility_20) else 0.0,
                    'atr_percent': technical_indicators.get('ATR_Percent', 0.0)
                },
                'technical_indicators': technical_indicators,
                'market_structure': market_structure,
                'data_quality': {
                    'total_records': len(data),
                    'indicator_coverage': len(technical_indicators),
                    'data_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process data for AI analysis: {e}")
            raise
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure and trends"""
        try:
            # Trend analysis using multiple SMAs
            current_price = data['Close'].iloc[-1]
            
            trend_signals = {}
            if 's20' in data.columns:
                trend_signals['sma20'] = 'bullish' if current_price > data['s20'].iloc[-1] else 'bearish'
            if 's50' in data.columns:
                trend_signals['sma50'] = 'bullish' if current_price > data['s50'].iloc[-1] else 'bearish'
            if 's200' in data.columns:
                trend_signals['sma200'] = 'bullish' if current_price > data['s200'].iloc[-1] else 'bearish'
            
            # Overall trend strength
            bullish_signals = sum(1 for signal in trend_signals.values() if signal == 'bullish')
            trend_strength = bullish_signals / len(trend_signals) if trend_signals else 0.5
            
            # Momentum analysis
            momentum = 'neutral'
            if 'rsi14' in data.columns:
                rsi = data['rsi14'].iloc[-1]
                if rsi > 70:
                    momentum = 'overbought'
                elif rsi < 30:
                    momentum = 'oversold'
                elif rsi > 50:
                    momentum = 'bullish'
                else:
                    momentum = 'bearish'
            
            # Support/Resistance levels (simplified)
            recent_highs = data['High'].tail(20).max()
            recent_lows = data['Low'].tail(20).min()
            
            return {
                'trend_signals': trend_signals,
                'trend_strength': float(trend_strength),
                'momentum': momentum,
                'support_level': float(recent_lows),
                'resistance_level': float(recent_highs),
                'price_position': float((current_price - recent_lows) / (recent_highs - recent_lows)) if recent_highs != recent_lows else 0.5
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze market structure: {e}")
            return {
                'trend_signals': {},
                'trend_strength': 0.5,
                'momentum': 'neutral',
                'support_level': 0.0,
                'resistance_level': 0.0,
                'price_position': 0.5
            }
