"""
AI Trading Engine for Phase 3: Framework Integration
===================================================

This module serves as the core trading engine that combines:
- AI market analysis and decision making
- Position management and risk control
- Integration with existing backtesting framework
- Real-time trading capabilities

The engine orchestrates all components to create a complete AI-driven trading system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, field

from .config import Config, TradingConfig, default_config
from .ai_client import AITradingClient
from .data_manager import IntelligentDataManager
from .indicators import TechnicalIndicatorFactory, MarketDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Structured trading decision from AI"""
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    position_size: float  # Fraction of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    market_outlook: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Active trading position"""
    symbol: str
    action: str  # long, short
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_date: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class AITradingEngine:
    """
    Core AI Trading Engine for Phase 3
    
    Integrates all components into a complete trading system:
    - Market data acquisition and analysis
    - AI-powered decision making
    - Position management and risk control
    - Performance tracking and optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the AI trading engine"""
        self.config = config or Config()
        
        # Initialize core components
        self.ai_client = AITradingClient(self.config.openai)
        self.data_manager = IntelligentDataManager(self.config.data)
        self.indicator_factory = TechnicalIndicatorFactory()
        self.market_processor = MarketDataProcessor()
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.portfolio_value = self.config.trading.initial_capital
        self.cash_available = self.config.trading.initial_capital
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info("AI Trading Engine initialized successfully")
    
    def analyze_market(self, symbol: str, period: str = "3mo", interval: str = "1d") -> Dict[str, Any]:
        """
        Perform comprehensive market analysis for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            period: Data period for analysis
            interval: Data interval
            
        Returns:
            Dict containing complete market analysis
        """
        try:
            logger.info(f"Starting market analysis for {symbol}")
            
            # Get market data
            market_data = self.data_manager.get_market_data(symbol, period, interval)
            
            # Calculate technical indicators
            ai_indicators = self.indicator_factory.get_ai_ready_indicators(market_data, symbol)
            
            # Prepare data for AI analysis
            ai_data = self.data_manager.prepare_data_for_ai(market_data, symbol)
            ai_data['technical_indicators'] = ai_indicators
            
            # Get AI market analysis
            ai_analysis = self.ai_client.analyze_market_conditions(ai_data)
            
            # Combine all analysis
            complete_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'raw_data': market_data,
                'current_price': ai_data['current_price'],
                'price_change': ai_data.get('price_change', 0),
                'volume': ai_data.get('volume', 0),
                'technical_indicators': ai_indicators,
                'ai_analysis': ai_analysis,
                'market_data': ai_data
            }
            
            logger.info(f"Market analysis completed for {symbol}")
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            raise
    
    def make_trading_decision(self, market_analysis: Dict[str, Any]) -> TradingDecision:
        """
        Make AI-powered trading decision based on market analysis
        
        Args:
            market_analysis: Complete market analysis from analyze_market
            
        Returns:
            TradingDecision object with AI recommendation
        """
        try:
            symbol = market_analysis['symbol']
            logger.info(f"Making trading decision for {symbol}")
            
            # Assess current position risk
            current_position = self.positions.get(symbol)
            position_data = self._get_position_data(symbol, current_position)
            
            # Get risk assessment from AI
            risk_assessment = self.ai_client.assess_risk(
                position_data, 
                market_analysis['ai_analysis']
            )
            
            # Develop strategy based on market conditions
            strategy_params = self.ai_client.develop_strategy(
                market_analysis['ai_analysis'],
                self._get_recent_performance()
            )
            
            # Make final trading decision
            ai_decision = self.ai_client.make_trading_decision(
                market_analysis['market_data'],
                strategy_params,
                risk_assessment
            )
            
            # Create structured trading decision
            decision = TradingDecision(
                action=ai_decision['action'],
                confidence=ai_decision['confidence'],
                position_size=min(
                    ai_decision['position_size'],
                    self.config.trading.max_position_size
                ),
                entry_price=ai_decision['entry_price'],
                stop_loss=ai_decision['stop_loss'],
                take_profit=ai_decision['take_profit'],
                reasoning=ai_decision['reasoning'],
                market_outlook=ai_decision['market_outlook']
            )
            
            logger.info(f"Trading decision made for {symbol}: {decision.action} (confidence: {decision.confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"Trading decision failed for {symbol}: {e}")
            # Return safe default decision
            return TradingDecision(
                action="HOLD",
                confidence=0.0,
                position_size=0.0,
                entry_price=market_analysis['current_price'],
                stop_loss=0.0,
                take_profit=0.0,
                reasoning=f"Error in decision making: {e}",
                market_outlook="neutral"
            )
    
    def execute_decision(self, symbol: str, decision: TradingDecision) -> bool:
        """
        Execute trading decision with position management
        
        Args:
            symbol: Stock symbol
            decision: AI trading decision
            
        Returns:
            bool: True if execution successful
        """
        try:
            # Check if decision meets minimum confidence threshold
            if decision.confidence < self.config.trading.confidence_threshold:
                logger.info(f"Decision confidence {decision.confidence:.2f} below threshold {self.config.trading.confidence_threshold}")
                return False
            
            current_position = self.positions.get(symbol)
            
            if decision.action == "BUY" and current_position is None:
                return self._open_long_position(symbol, decision)
                
            elif decision.action == "SELL" and current_position is not None:
                return self._close_position(symbol, decision)
                
            elif decision.action == "HOLD":
                logger.info(f"Holding position for {symbol}")
                return True
            
            else:
                logger.info(f"No action taken for {symbol} - decision: {decision.action}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute decision for {symbol}: {e}")
            return False
    
    def _open_long_position(self, symbol: str, decision: TradingDecision) -> bool:
        """Open a long position"""
        try:
            # Calculate position size in dollars
            position_value = self.portfolio_value * decision.position_size
            
            # Check if we have enough cash
            if position_value > self.cash_available:
                logger.warning(f"Insufficient cash for {symbol} position: need ${position_value:.2f}, have ${self.cash_available:.2f}")
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                action="long",
                entry_price=decision.entry_price,
                position_size=decision.position_size,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                entry_date=datetime.now(),
                current_price=decision.entry_price
            )
            
            # Update portfolio
            self.positions[symbol] = position
            self.cash_available -= position_value
            
            # Record trade
            self._record_trade({
                'symbol': symbol,
                'action': 'open_long',
                'price': decision.entry_price,
                'size': decision.position_size,
                'value': position_value,
                'timestamp': datetime.now(),
                'reasoning': decision.reasoning,
                'confidence': decision.confidence
            })
            
            logger.info(f"Opened long position for {symbol}: ${position_value:.2f} at ${decision.entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open long position for {symbol}: {e}")
            return False
    
    def _close_position(self, symbol: str, decision: TradingDecision) -> bool:
        """Close an existing position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
            
            # Calculate P&L
            pnl_pct = (decision.entry_price - position.entry_price) / position.entry_price
            position_value = self.portfolio_value * position.position_size
            pnl_value = position_value * pnl_pct
            
            # Update portfolio
            self.cash_available += position_value + pnl_value
            del self.positions[symbol]
            
            # Record trade
            self._record_trade({
                'symbol': symbol,
                'action': 'close_long',
                'price': decision.entry_price,
                'size': position.position_size,
                'value': position_value,
                'pnl_pct': pnl_pct,
                'pnl_value': pnl_value,
                'timestamp': datetime.now(),
                'reasoning': decision.reasoning,
                'confidence': decision.confidence,
                'holding_period': (datetime.now() - position.entry_date).days
            })
            
            # Update performance metrics
            self._update_performance_metrics(pnl_value > 0, pnl_pct)
            
            logger.info(f"Closed position for {symbol}: P&L ${pnl_value:.2f} ({pnl_pct:.2%})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]) -> None:
        """Update all positions with current market prices"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.current_price = current_price
                
                # Calculate unrealized P&L
                pnl_pct = (current_price - position.entry_price) / position.entry_price
                position.unrealized_pnl_pct = pnl_pct
                
                position_value = self.portfolio_value * position.position_size
                position.unrealized_pnl = position_value * pnl_pct
                
                # Check stop loss and take profit
                self._check_exit_conditions(symbol, position)
    
    def _check_exit_conditions(self, symbol: str, position: Position) -> None:
        """Check if position should be closed due to stop loss or take profit"""
        current_price = position.current_price
        entry_price = position.entry_price
        
        # Check stop loss
        if position.stop_loss > 0:
            stop_price = entry_price * (1 - position.stop_loss)
            if current_price <= stop_price:
                logger.info(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
                self._force_close_position(symbol, current_price, "stop_loss")
                return
        
        # Check take profit
        if position.take_profit > 0:
            take_profit_price = entry_price * (1 + position.take_profit)
            if current_price >= take_profit_price:
                logger.info(f"Take profit triggered for {symbol} at ${current_price:.2f}")
                self._force_close_position(symbol, current_price, "take_profit")
                return
    
    def _force_close_position(self, symbol: str, price: float, reason: str) -> None:
        """Force close a position due to stop loss or take profit"""
        # Create a decision to close the position
        decision = TradingDecision(
            action="SELL",
            confidence=1.0,
            position_size=0.0,
            entry_price=price,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=f"Automatic exit: {reason}",
            market_outlook="neutral"
        )
        self._close_position(symbol, decision)
    
    def _get_position_data(self, symbol: str, position: Optional[Position]) -> Dict[str, Any]:
        """Get position data for risk assessment"""
        if position:
            return {
                'symbol': symbol,
                'has_position': True,
                'position_size': position.position_size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'days_held': (datetime.now() - position.entry_date).days,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
        else:
            return {
                'symbol': symbol,
                'has_position': False,
                'position_size': 0.0,
                'cash_available': self.cash_available / self.portfolio_value
            }
    
    def _get_recent_performance(self) -> Optional[Dict[str, Any]]:
        """Get recent performance data for strategy optimization"""
        if len(self.trade_history) < 10:
            return None
        
        recent_trades = self.trade_history[-20:]  # Last 20 trades
        
        returns = [t.get('pnl_pct', 0) for t in recent_trades if 'pnl_pct' in t]
        if not returns:
            return None
        
        return {
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) * 100,
            'avg_return': np.mean(returns) * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': min(returns) * 100 if returns else 0
        }
    
    def _record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a trade in history"""
        self.trade_history.append(trade)
        self.performance_metrics['total_trades'] += 1
        
        # Log significant trades
        if trade.get('pnl_value'):
            pnl = trade['pnl_value']
            if pnl > 0:
                logger.info(f"Profitable trade: {trade['symbol']} +${pnl:.2f}")
            else:
                logger.info(f"Loss trade: {trade['symbol']} ${pnl:.2f}")
    
    def _update_performance_metrics(self, is_winner: bool, return_pct: float) -> None:
        """Update performance tracking metrics"""
        if is_winner:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / total_trades
            )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_position_value = sum(
            self.portfolio_value * pos.position_size for pos in self.positions.values()
        )
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        current_portfolio_value = self.cash_available + total_position_value + total_unrealized_pnl
        total_return = (current_portfolio_value - self.config.trading.initial_capital) / self.config.trading.initial_capital
        
        return {
            'initial_capital': self.config.trading.initial_capital,
            'current_value': current_portfolio_value,
            'cash_available': self.cash_available,
            'invested_value': total_position_value,
            'unrealized_pnl': total_unrealized_pnl,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'active_positions': len(self.positions),
            'position_details': {
                symbol: {
                    'size': pos.position_size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for symbol, pos in self.positions.items()
            },
            'performance_metrics': self.performance_metrics
        }
    
    def run_trading_session(self, symbols: List[str], period: str = "3mo") -> Dict[str, Any]:
        """
        Run a complete trading session for multiple symbols
        
        Args:
            symbols: List of symbols to trade
            period: Data period for analysis
            
        Returns:
            Session results and performance summary
        """
        session_start = datetime.now()
        logger.info(f"Starting trading session for {len(symbols)} symbols")
        
        session_results = {
            'start_time': session_start,
            'symbols': symbols,
            'decisions': {},
            'executions': {},
            'errors': []
        }
        
        # Get current prices for position updates
        current_prices = {}
        for symbol in list(self.positions.keys()):
            try:
                market_data = self.data_manager.get_market_data(symbol, "1d", "1d")
                current_prices[symbol] = float(market_data['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Failed to get current price for {symbol}: {e}")
        
        # Update existing positions
        if current_prices:
            self.update_positions(current_prices)
        
        # Analyze and make decisions for each symbol
        for symbol in symbols:
            try:
                # Market analysis
                market_analysis = self.analyze_market(symbol, period)
                
                # AI decision making
                decision = self.make_trading_decision(market_analysis)
                session_results['decisions'][symbol] = {
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning[:200] + "..." if len(decision.reasoning) > 200 else decision.reasoning
                }
                
                # Execute decision
                executed = self.execute_decision(symbol, decision)
                session_results['executions'][symbol] = executed
                
                logger.info(f"Processed {symbol}: {decision.action} (confidence: {decision.confidence:.2f}, executed: {executed})")
                
            except Exception as e:
                error_msg = f"Failed to process {symbol}: {e}"
                logger.error(error_msg)
                session_results['errors'].append(error_msg)
        
        # Session summary
        session_end = datetime.now()
        session_duration = session_end - session_start
        
        session_results.update({
            'end_time': session_end,
            'duration_seconds': session_duration.total_seconds(),
            'portfolio_summary': self.get_portfolio_summary(),
            'decisions_made': len(session_results['decisions']),
            'successful_executions': sum(session_results['executions'].values()),
            'errors_count': len(session_results['errors'])
        })
        
        logger.info(f"Trading session completed in {session_duration.total_seconds():.1f}s")
        logger.info(f"Decisions: {session_results['decisions_made']}, Executions: {session_results['successful_executions']}, Errors: {session_results['errors_count']}")
        
        return session_results


class TradingEngineError(Exception):
    """Custom exception for trading engine errors"""
    pass
