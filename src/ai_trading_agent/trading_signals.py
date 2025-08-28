"""
Trading Signals Module for Phase 3: Framework Integration
========================================================

This module converts AI trading decisions into signals compatible with
your existing backtesting framework. It bridges the gap between AI
recommendations and the traditional signal generation system.

Key features:
- Converts AI decisions to TRADE signals (In, p, Out)
- Integrates with existing backtesting functions
- Maintains compatibility with current framework structure
- Adds AI reasoning and confidence to signal metadata
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field

from .trading_engine import TradingDecision, AITradingEngine
from .config import Config, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class AISignal:
    """Enhanced trading signal with AI metadata"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'P' (position), 'cP' (close position), '' (hold)
    ai_action: str    # Original AI decision: BUY, SELL, HOLD
    confidence: float
    reasoning: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    market_outlook: str


class AITradingSignalGenerator:
    """
    Converts AI trading decisions into backtesting framework signals
    
    This class serves as the bridge between modern AI decision making
    and the traditional backtesting framework signal system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the signal generator"""
        self.config = config or Config()
        self.signal_history: List[AISignal] = []
        self.current_positions: Dict[str, Dict] = {}  # Track active positions per symbol
        
        logger.info("AI Trading Signal Generator initialized")
    
    def convert_ai_decision_to_signal(self, 
                                    symbol: str, 
                                    decision: TradingDecision,
                                    current_price: float) -> str:
        """
        Convert AI trading decision to backtesting framework signal
        
        Args:
            symbol: Stock symbol
            decision: AI trading decision
            current_price: Current market price
            
        Returns:
            str: Signal compatible with backtesting framework ('P', 'cP', '')
        """
        try:
            # Check if we have an active position
            has_position = symbol in self.current_positions
            
            # Convert AI decision to framework signal
            if decision.action == "BUY" and not has_position:
                # AI wants to buy and we don't have a position
                signal = 'P'  # Open position
                
                # Track the new position
                self.current_positions[symbol] = {
                    'entry_price': decision.entry_price,
                    'entry_time': datetime.now(),
                    'stop_loss': decision.stop_loss,
                    'take_profit': decision.take_profit,
                    'position_size': decision.position_size
                }
                
            elif decision.action == "SELL" and has_position:
                # AI wants to sell and we have a position
                signal = 'cP'  # Close position
                
                # Remove the position from tracking
                if symbol in self.current_positions:
                    del self.current_positions[symbol]
                    
            elif decision.action == "HOLD":
                # AI wants to hold - no signal
                signal = ''
                
            else:
                # No action (e.g., BUY when already have position, SELL when no position)
                signal = ''
            
            # Create enhanced signal record
            ai_signal = AISignal(
                symbol=symbol,
                timestamp=datetime.now(),
                signal_type=signal,
                ai_action=decision.action,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                position_size=decision.position_size,
                market_outlook=decision.market_outlook
            )
            
            # Add to signal history
            self.signal_history.append(ai_signal)
            
            logger.info(f"Converted AI decision for {symbol}: {decision.action} -> {signal} (confidence: {decision.confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to convert AI decision for {symbol}: {e}")
            return ''  # Return empty signal on error
    
    def apply_ai_signals_to_dataframe(self, 
                                     data: pd.DataFrame, 
                                     ai_decisions: Dict[str, TradingDecision],
                                     symbol: str) -> pd.DataFrame:
        """
        Apply AI-generated signals to a dataframe in backtesting format
        
        Args:
            data: Market data dataframe with OHLCV columns
            ai_decisions: Dictionary of AI decisions indexed by date/timestamp
            symbol: Stock symbol
            
        Returns:
            pd.DataFrame: Enhanced dataframe with AI-generated signals
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Initialize signal column
            df['signal'] = ''
            df['ai_action'] = ''
            df['ai_confidence'] = 0.0
            df['ai_reasoning'] = ''
            
            # Apply AI decisions to corresponding dates
            for date_str, decision in ai_decisions.items():
                try:
                    # Convert date string to datetime if needed
                    if isinstance(date_str, str):
                        decision_date = pd.to_datetime(date_str)
                    else:
                        decision_date = date_str
                    
                    # Find closest date in dataframe
                    if decision_date in df.index:
                        current_price = float(df.loc[decision_date, 'Close'])
                        signal = self.convert_ai_decision_to_signal(symbol, decision, current_price)
                        
                        # Apply the signal
                        df.loc[decision_date, 'signal'] = signal
                        df.loc[decision_date, 'ai_action'] = decision.action
                        df.loc[decision_date, 'ai_confidence'] = decision.confidence
                        df.loc[decision_date, 'ai_reasoning'] = decision.reasoning[:100] + "..." if len(decision.reasoning) > 100 else decision.reasoning
                        
                except Exception as e:
                    logger.warning(f"Failed to apply AI decision for {symbol} on {date_str}: {e}")
                    continue
            
            # Shift signals to avoid look-ahead bias (consistent with existing framework)
            df['position'] = df['signal'].shift(1)
            
            logger.info(f"Applied {len(ai_decisions)} AI decisions to {symbol} dataframe")
            return df
            
        except Exception as e:
            logger.error(f"Failed to apply AI signals to dataframe for {symbol}: {e}")
            return data  # Return original data on error
    
    def generate_backtest_signals(self, 
                                 trading_engine: AITradingEngine,
                                 symbols: List[str],
                                 start_date: str,
                                 end_date: str,
                                 period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive backtesting signals for multiple symbols
        
        Args:
            trading_engine: AI trading engine
            symbols: List of symbols to generate signals for
            start_date: Start date for backtesting
            end_date: End date for backtesting
            period: Data period for analysis
            
        Returns:
            Dict[str, pd.DataFrame]: Backtesting dataframes with AI signals for each symbol
        """
        try:
            logger.info(f"Generating backtest signals for {len(symbols)} symbols from {start_date} to {end_date}")
            
            backtest_data = {}
            
            for symbol in symbols:
                try:
                    logger.info(f"Processing {symbol}...")
                    
                    # Get market data for the symbol
                    market_data = trading_engine.data_manager.get_market_data(
                        symbol, 
                        period=period, 
                        start_date=start_date, 
                        end_date=end_date
                    )
                    
                    # Generate AI decisions for each trading day
                    ai_decisions = {}
                    
                    # Sample AI decisions for key dates (in practice, this would run daily)
                    sample_dates = pd.date_range(
                        start=pd.to_datetime(start_date), 
                        end=pd.to_datetime(end_date), 
                        freq='5D'  # Every 5 days for demonstration
                    )
                    
                    for date in sample_dates:
                        if date in market_data.index:
                            try:
                                # Simulate AI analysis for this date
                                historical_data = market_data.loc[:date]
                                
                                if len(historical_data) > 50:  # Ensure enough data for analysis
                                    market_analysis = trading_engine.analyze_market(symbol, period="3mo")
                                    decision = trading_engine.make_trading_decision(market_analysis)
                                    ai_decisions[date] = decision
                                    
                            except Exception as e:
                                logger.warning(f"Failed to generate AI decision for {symbol} on {date}: {e}")
                                continue
                    
                    # Apply AI signals to market data
                    enhanced_data = self.apply_ai_signals_to_dataframe(
                        market_data, 
                        ai_decisions, 
                        symbol
                    )
                    
                    backtest_data[symbol] = enhanced_data
                    logger.info(f"Generated {len(ai_decisions)} AI signals for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    continue
            
            logger.info(f"Backtest signal generation completed for {len(backtest_data)} symbols")
            return backtest_data
            
        except Exception as e:
            logger.error(f"Backtest signal generation failed: {e}")
            return {}
    
    def create_ai_sistema_function(self, 
                                  ai_decisions_df: pd.DataFrame) -> callable:
        """
        Create a sistema function that uses AI decisions instead of traditional indicators
        
        This creates a function compatible with your existing backtesting framework
        that uses AI decisions instead of traditional RSI/SMA signals.
        
        Args:
            ai_decisions_df: DataFrame with AI decisions and confidence levels
            
        Returns:
            callable: Function compatible with dameSistema from your framework
        """
        def ai_sistema(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
            """
            AI-powered sistema function for backtesting framework
            
            This replaces traditional indicator-based signals with AI decisions
            """
            try:
                # Initialize signal column
                df['signal'] = ''
                
                # Apply AI decisions where available
                for date in df.index:
                    if date in ai_decisions_df.index:
                        ai_row = ai_decisions_df.loc[date]
                        
                        # Use AI confidence as signal strength filter
                        min_confidence = kwargs.get('min_confidence', 0.7)
                        
                        if ai_row['ai_confidence'] >= min_confidence:
                            if ai_row['ai_action'] == 'BUY':
                                df.loc[date, 'signal'] = 'P'
                            elif ai_row['ai_action'] == 'SELL':
                                df.loc[date, 'signal'] = 'cP'
                
                # Apply position logic (avoid look-ahead bias)
                df['position'] = df['signal'].shift(1)
                
                return df
                
            except Exception as e:
                logger.error(f"AI sistema function failed: {e}")
                return df
        
        return ai_sistema
    
    def get_signal_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get statistics about generated signals
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dict with signal statistics
        """
        try:
            # Filter signals by symbol if specified
            if symbol:
                signals = [s for s in self.signal_history if s.symbol == symbol]
            else:
                signals = self.signal_history
            
            if not signals:
                return {}
            
            # Calculate statistics
            total_signals = len(signals)
            buy_signals = len([s for s in signals if s.signal_type == 'P'])
            sell_signals = len([s for s in signals if s.signal_type == 'cP'])
            hold_signals = len([s for s in signals if s.signal_type == ''])
            
            avg_confidence = np.mean([s.confidence for s in signals])
            
            confidence_distribution = {
                'high_confidence (>0.8)': len([s for s in signals if s.confidence > 0.8]),
                'medium_confidence (0.5-0.8)': len([s for s in signals if 0.5 <= s.confidence <= 0.8]),
                'low_confidence (<0.5)': len([s for s in signals if s.confidence < 0.5])
            }
            
            market_outlook_distribution = {}
            for signal in signals:
                outlook = signal.market_outlook
                market_outlook_distribution[outlook] = market_outlook_distribution.get(outlook, 0) + 1
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'avg_confidence': round(avg_confidence, 3),
                'confidence_distribution': confidence_distribution,
                'market_outlook_distribution': market_outlook_distribution,
                'symbol_filter': symbol,
                'active_positions': len(self.current_positions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get signal statistics: {e}")
            return {}
    
    def export_signals_for_backtesting(self, 
                                     symbol: str, 
                                     format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
        """
        Export signals in format suitable for backtesting framework
        
        Args:
            symbol: Stock symbol
            format: Export format ('dataframe' or 'dict')
            
        Returns:
            Signals in requested format
        """
        try:
            # Filter signals for the symbol
            symbol_signals = [s for s in self.signal_history if s.symbol == symbol]
            
            if not symbol_signals:
                logger.warning(f"No signals found for {symbol}")
                return pd.DataFrame() if format == 'dataframe' else {}
            
            if format == 'dataframe':
                # Create DataFrame compatible with backtesting framework
                data = []
                for signal in symbol_signals:
                    data.append({
                        'Date': signal.timestamp,
                        'signal': signal.signal_type,
                        'ai_action': signal.ai_action,
                        'ai_confidence': signal.confidence,
                        'ai_reasoning': signal.reasoning,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'position_size': signal.position_size,
                        'market_outlook': signal.market_outlook
                    })
                
                df = pd.DataFrame(data)
                df.set_index('Date', inplace=True)
                return df
                
            else:  # dict format
                return {
                    signal.timestamp.strftime('%Y-%m-%d'): {
                        'signal': signal.signal_type,
                        'ai_action': signal.ai_action,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'position_size': signal.position_size,
                        'market_outlook': signal.market_outlook
                    }
                    for signal in symbol_signals
                }
                
        except Exception as e:
            logger.error(f"Failed to export signals for {symbol}: {e}")
            return pd.DataFrame() if format == 'dataframe' else {}


class TradingSignalsError(Exception):
    """Custom exception for trading signals errors"""
    pass
