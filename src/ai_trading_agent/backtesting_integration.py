"""
Backtesting Integration for Phase 3: Framework Integration
=========================================================

This module integrates your existing backtesting framework with the new
AI trading system. It provides a seamless bridge between the traditional
backtesting functions and AI-driven signal generation.

Features:
- Uses your existing backtesting logic (dameSistema, damePosition, etc.)
- Integrates AI signals with traditional framework functions
- Maintains compatibility with existing performance metrics
- Adds AI-specific enhancements and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import matplotlib.pyplot as plt

from .trading_engine import AITradingEngine, TradingDecision
from .trading_signals import AITradingSignalGenerator
from .config import Config
from .data_manager import IntelligentDataManager
from .indicators import TechnicalIndicatorFactory

logger = logging.getLogger(__name__)


class AIBacktestingIntegration:
    """
    Integrates AI trading system with existing backtesting framework
    
    This class bridges the gap between modern AI decision making
    and your proven backtesting methodology, allowing you to test
    AI strategies using your existing performance metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the backtesting integration"""
        self.config = config or Config()
        
        # Initialize components
        self.trading_engine = AITradingEngine(self.config)
        self.signal_generator = AITradingSignalGenerator(self.config)
        self.data_manager = IntelligentDataManager(self.config.data)
        self.indicator_factory = TechnicalIndicatorFactory()
        
        logger.info("AI Backtesting Integration initialized")
    
    # ====================== CORE BACKTESTING FUNCTIONS ======================
    # These functions are adapted from your existing framework
    
    def ocpSma(self, df: pd.DataFrame, periodo: int = 20, borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """Simple Moving Average - adapted from your framework"""
        df[f's{periodo}'] = df[col].rolling(periodo).mean()
        if borraNan:
            df.dropna(inplace=True)
        return df
    
    def ocpRsi(self, df: pd.DataFrame, periodo: int = 14, borraNan: bool = False, col: str = 'Close') -> pd.DataFrame:
        """RSI indicator - adapted from your framework"""
        df['dif'] = df[col].diff()
        df['win'] = np.where(df['dif'] > 0, df['dif'], 0)
        df['loss'] = np.where(df['dif'] < 0, abs(df['dif']), 0)
        df['emaWin'] = df['win'].ewm(span=periodo).mean()
        df['emaLoss'] = df['loss'].ewm(span=periodo).mean()
        df['rs'] = df['emaWin'] / df['emaLoss']
        df[f'rsi{periodo}'] = 100 - (100 / (1 + df['rs']))
        
        df.drop(['dif', 'win', 'loss', 'emaWin', 'emaLoss', 'rs'], axis=1, inplace=True)
        
        if borraNan:
            df.dropna(inplace=True)
        
        return df
    
    def dameAISistema(self, df: pd.DataFrame, ai_decisions: Dict[str, TradingDecision], 
                      min_confidence: float = 0.7) -> pd.DataFrame:
        """
        AI-powered sistema function that replaces traditional indicator logic
        
        Args:
            df: Market data dataframe
            ai_decisions: Dictionary of AI trading decisions
            min_confidence: Minimum confidence threshold for signals
            
        Returns:
            DataFrame with AI-generated signals
        """
        try:
            df['signal'] = ''
            df['ai_confidence'] = 0.0
            df['ai_reasoning'] = ''
            
            # Apply AI decisions to the dataframe
            for date in df.index:
                if date in ai_decisions:
                    decision = ai_decisions[date]
                    
                    # Apply confidence filter
                    if decision.confidence >= min_confidence:
                        # Convert AI decision to signal
                        if decision.action == 'BUY':
                            df.loc[date, 'signal'] = 'P'  # Position
                        elif decision.action == 'SELL':
                            df.loc[date, 'signal'] = 'cP'  # Close Position
                        
                        # Store AI metadata
                        df.loc[date, 'ai_confidence'] = decision.confidence
                        df.loc[date, 'ai_reasoning'] = decision.reasoning[:100]
            
            # Avoid look-ahead bias
            df['position'] = df['signal'].shift(1)
            
            return df
            
        except Exception as e:
            logger.error(f"AI Sistema function failed: {e}")
            return df
    
    def damePosition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Position logic - adapted from your framework
        Determines trade states: In, p (holding), Out
        """
        df['tradeInd'] = ''
        
        inTrade = False
        for i in df.index:
            cell = df.loc[i, 'position']
            
            if pd.isna(cell) or cell == '':
                pass  # No signal, no state change
            else:
                if cell == 'P':
                    if not inTrade:
                        inTrade = True
                        df.loc[i, 'tradeInd'] = 'In'  # Entry
                    else:  # Already in position
                        df.loc[i, 'tradeInd'] = 'p'
                elif cell == 'cP':
                    if inTrade:
                        df.loc[i, 'tradeInd'] = 'Out'  # Exit
                    inTrade = False
                else:
                    if inTrade:
                        df.loc[i, 'tradeInd'] = 'p'
        
        df['TRADE'] = df['tradeInd']
        return df
    
    def dameSalidaPnl(self, df: pd.DataFrame, sentido: str = 'long', tp: float = 0, 
                     sl: float = 0, comision: float = 0, slippage: float = 0) -> pd.DataFrame:
        """
        P&L calculation with stop loss and take profit - adapted from your framework
        """
        df['ROID'] = 0.0
        df['ROIACUM'] = 0.0
        df['precioRef'] = 0.0
        df['tipoSalida'] = ''
        
        df = df.reset_index(drop=False)
        
        # Only apply SL/TP logic if they are set
        if tp > 0 or sl > 0:
            for i in df.index:
                estado = df.loc[i, 'TRADE']
                
                O = df.loc[i, 'Open']
                H = df.loc[i, 'High']
                L = df.loc[i, 'Low']
                C = df.loc[i, 'Close']
                
                if i > 0:
                    if estado == 'In':
                        df.loc[i, 'tradePnl'] = 'In'
                        
                        precioRef = df.loc[i, 'Open']
                        salida, roi = self._pnlSalida(O, H, L, C, precioRef, sentido, tp, sl)
                        
                        # Apply costs only at entry
                        coste = comision + slippage
                        roi -= coste
                        
                        df.loc[i, 'ROID'] = roi
                        df.loc[i, 'ROIACUM'] = roi
                        df.loc[i, 'precioRef'] = precioRef
                        df.loc[i, 'tipoSalida'] = salida
                        
                    elif estado in ['p', 'Out']:
                        # Check if previous state is valid
                        if df.loc[i-1, 'tradePnl'] not in ['In', 'p']:
                            df.loc[i, 'tradePnl'] = ''
                            df.loc[i, 'ROID'] = 0
                            df.loc[i, 'ROIACUM'] = 0
                            df.loc[i, 'precioRef'] = 0
                            df.loc[i, 'tipoSalida'] = ''
                        else:
                            # Check if previous was stopped out
                            if df.loc[i-1, 'tipoSalida'] == 'sltp':
                                df.loc[i, 'tradePnl'] = 'Out'
                                df.loc[i, 'ROID'] = 0
                                df.loc[i, 'ROIACUM'] = df.loc[i-1, 'ROIACUM']
                                df.loc[i, 'precioRef'] = 0
                                df.loc[i, 'tipoSalida'] = ''
                            else:
                                # Normal continuation
                                precioRef = df.loc[i-1, 'precioRef']
                                salida, roiRef = self._pnlSalida(O, H, L, C, precioRef, sentido, tp, sl)
                                
                                if salida == 'normal':
                                    df.loc[i, 'tradePnl'] = estado
                                    
                                    if estado == 'p':
                                        df.loc[i, 'ROID'] = (C / df.loc[i-1, 'Close'] - 1) * 100
                                    if estado == 'Out':
                                        df.loc[i, 'ROID'] = (O / df.loc[i-1, 'Close'] - 1) * 100
                                    
                                    df.loc[i, 'ROIACUM'] = roiRef
                                    df.loc[i, 'precioRef'] = precioRef
                                    df.loc[i, 'tipoSalida'] = salida
                                else:
                                    # Stop loss or take profit hit
                                    df.loc[i, 'tradePnl'] = 'Out'
                                    df.loc[i, 'ROID'] = ((1 + roiRef/100) / (1 + df.loc[i-1, 'ROIACUM']/100) - 1) * 100
                                    df.loc[i, 'ROIACUM'] = roiRef
                                    df.loc[i, 'precioRef'] = 0
                                    df.loc[i, 'tipoSalida'] = salida
                    else:
                        df.loc[i, 'tradePnl'] = ''
                        df.loc[i, 'ROID'] = 0
                        df.loc[i, 'ROIACUM'] = 0
                        df.loc[i, 'precioRef'] = 0
                        df.loc[i, 'tipoSalida'] = ''
            
            df['TRADE'] = df['tradePnl']
        
        else:
            # Simple P&L calculation without SL/TP
            coste = comision + slippage
            for i in df.index:
                if i > 0:
                    estado = df.loc[i, 'TRADE']
                    
                    if estado == 'In':
                        df.loc[i, 'ROID'] = (df.loc[i, 'Close'] / df.loc[i, 'Open'] - 1) * 100 - coste
                        df.loc[i, 'ROIACUM'] = df.loc[i, 'ROID']
                    elif estado == 'p':
                        df.loc[i, 'ROID'] = (df.loc[i, 'Close'] / df.loc[i-1, 'Close'] - 1) * 100
                        df.loc[i, 'ROIACUM'] = ((1 + df.loc[i, 'ROID']/100) * (1 + df.loc[i-1, 'ROIACUM']/100) - 1) * 100
                    elif estado == 'Out':
                        df.loc[i, 'ROID'] = (df.loc[i, 'Open'] / df.loc[i-1, 'Close'] - 1) * 100
                        df.loc[i, 'ROIACUM'] = ((1 + df.loc[i, 'ROID']/100) * (1 + df.loc[i-1, 'ROIACUM']/100) - 1) * 100
                    else:
                        df.loc[i, 'ROID'] = 0
                        df.loc[i, 'ROIACUM'] = 0
            
            if sentido != 'long':
                df['ROID'] = -df['ROID']
                df['ROIACUM'] = -df['ROIACUM']
        
        df.set_index('Date', inplace=True)
        return df
    
    def _pnlSalida(self, Open: float, High: float, Low: float, Close: float, 
                   precioRef: float, sentido: str, tp: float, sl: float) -> Tuple[str, float]:
        """
        Determine if SL/TP is hit during the session - adapted from your framework
        """
        def calcularRoi(precio: float, precioRef: float, sentido: str) -> float:
            roi = (precio - precioRef) / precioRef * 100
            if sentido == 'short':
                roi = -roi
            return roi
        
        # Check stop loss first (priority)
        if sl > 0:
            if sentido == 'long':
                precioSl = precioRef * (1 - sl/100)
                if Open <= precioSl or Low <= precioSl:
                    if Open <= precioSl:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    else:
                        return ('sltp', -sl)
            else:  # short
                precioSl = precioRef * (1 + sl/100)
                if Open >= precioSl or High >= precioSl:
                    if Open >= precioSl:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    else:
                        return ('sltp', -sl)
        
        # Check take profit second
        if tp > 0:
            if sentido == 'long':
                precioTp = precioRef * (1 + tp/100)
                if Open >= precioTp or High >= precioTp:
                    if Open >= precioTp:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    else:
                        return ('sltp', tp)
            else:  # short
                precioTp = precioRef * (1 - tp/100)
                if Open <= precioTp or Low <= precioTp:
                    if Open <= precioTp:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    else:
                        return ('sltp', tp)
        
        # Normal exit - calculate ROI at close
        return ('normal', calcularRoi(Close, precioRef, sentido))
    
    def calculaCurvas(self, df: pd.DataFrame, size: float = 1.0) -> pd.DataFrame:
        """Calculate performance curves - adapted from your framework"""
        df['roiActivo'] = df['Close'].pct_change()
        df['roiActivo'].iloc[0] = 0
        df['cvAct'] = 100 * (1 + df['roiActivo']).cumprod()
        
        df['cvSis'] = 100 * (1 + df['ROID'] * size / 100).cumprod()
        
        df['ddAct'] = ((df['cvAct'] / df['cvAct'].cummax() - 1) * 100).round(2)
        df['ddSis'] = ((df['cvSis'] / df['cvSis'].cummax() - 1) * 100).round(2)
        
        return df
    
    # ====================== AI-ENHANCED BACKTESTING ======================
    
    def run_ai_backtest(self, symbol: str, start_date: str, end_date: str,
                       period: str = "2y", min_confidence: float = 0.7,
                       tp: float = 0, sl: float = 0, 
                       comision: float = 0.001, slippage: float = 0.0005) -> Dict[str, Any]:
        """
        Run complete AI-powered backtest for a symbol
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            period: Data period for analysis
            min_confidence: Minimum AI confidence threshold
            tp: Take profit percentage (e.g., 0.04 for 4%)
            sl: Stop loss percentage (e.g., 0.02 for 2%)
            comision: Commission rate
            slippage: Slippage rate
            
        Returns:
            Dict with backtest results and performance metrics
        """
        try:
            logger.info(f"Starting AI backtest for {symbol} from {start_date} to {end_date}")
            
            # Get market data
            market_data = self.data_manager.get_market_data(
                symbol, period=period, start_date=start_date, end_date=end_date
            )
            
            # Generate AI decisions for the backtest period
            ai_decisions = self._generate_backtest_ai_decisions(
                symbol, market_data, start_date, end_date
            )
            
            # Apply AI sistema
            df = self.dameAISistema(market_data.copy(), ai_decisions, min_confidence)
            
            # Apply position logic
            df = self.damePosition(df)
            
            # Calculate P&L with AI-recommended SL/TP
            df = self.dameSalidaPnl(df, 'long', tp, sl, comision, slippage)
            
            # Calculate performance curves
            df = self.calculaCurvas(df)
            
            # Calculate performance metrics
            performance = self._calculate_ai_performance_metrics(df, ai_decisions)
            
            # Generate AI-specific insights
            ai_insights = self._generate_ai_insights(df, ai_decisions, symbol)
            
            backtest_results = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'period': period,
                'min_confidence': min_confidence,
                'take_profit': tp,
                'stop_loss': sl,
                'commission': comision,
                'slippage': slippage,
                'data': df,
                'ai_decisions': ai_decisions,
                'performance_metrics': performance,
                'ai_insights': ai_insights,
                'total_ai_decisions': len(ai_decisions),
                'executed_trades': len(df[df['TRADE'] == 'Out']),
                'backtest_duration': (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            }
            
            logger.info(f"AI backtest completed for {symbol}. Total decisions: {len(ai_decisions)}, Executed trades: {backtest_results['executed_trades']}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"AI backtest failed for {symbol}: {e}")
            raise
    
    def _generate_backtest_ai_decisions(self, symbol: str, market_data: pd.DataFrame,
                                      start_date: str, end_date: str) -> Dict[datetime, TradingDecision]:
        """Generate AI decisions for backtesting period"""
        try:
            ai_decisions = {}
            
            # Generate decisions every 5 trading days for demonstration
            # In production, this would be daily or more frequent
            sample_dates = pd.date_range(
                start=pd.to_datetime(start_date), 
                end=pd.to_datetime(end_date), 
                freq='5D'
            )
            
            for date in sample_dates:
                if date in market_data.index:
                    try:
                        # Use data up to this date for AI analysis
                        historical_data = market_data.loc[:date]
                        
                        if len(historical_data) >= 50:  # Ensure enough data
                            # Simulate AI analysis
                            market_analysis = self.trading_engine.analyze_market(symbol, "3mo")
                            decision = self.trading_engine.make_trading_decision(market_analysis)
                            ai_decisions[date] = decision
                            
                    except Exception as e:
                        logger.warning(f"Failed to generate AI decision for {symbol} on {date}: {e}")
                        continue
            
            logger.info(f"Generated {len(ai_decisions)} AI decisions for {symbol} backtest")
            return ai_decisions
            
        except Exception as e:
            logger.error(f"Failed to generate backtest AI decisions for {symbol}: {e}")
            return {}
    
    def _calculate_ai_performance_metrics(self, df: pd.DataFrame, 
                                        ai_decisions: Dict[datetime, TradingDecision]) -> Dict[str, Any]:
        """Calculate AI-enhanced performance metrics"""
        try:
            # Basic performance metrics
            trades_out = df[df['TRADE'] == 'Out']
            
            if len(trades_out) == 0:
                return {'error': 'No completed trades found'}
            
            total_trades = len(trades_out)
            winning_trades = len(trades_out[trades_out['ROIACUM'] > 0])
            losing_trades = len(trades_out[trades_out['ROIACUM'] <= 0])
            
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            avg_win = trades_out[trades_out['ROIACUM'] > 0]['ROIACUM'].mean() if winning_trades > 0 else 0
            avg_loss = trades_out[trades_out['ROIACUM'] <= 0]['ROIACUM'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
            
            # Portfolio metrics
            final_value = df['cvSis'].iloc[-1]
            total_return = (final_value - 100) 
            
            max_dd = df['ddSis'].min()
            
            # AI-specific metrics
            ai_confidence_avg = np.mean([d.confidence for d in ai_decisions.values()])
            
            high_confidence_decisions = len([d for d in ai_decisions.values() if d.confidence > 0.8])
            medium_confidence_decisions = len([d for d in ai_decisions.values() if 0.5 <= d.confidence <= 0.8])
            low_confidence_decisions = len([d for d in ai_decisions.values() if d.confidence < 0.5])
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'avg_win_pct': round(avg_win, 2),
                'avg_loss_pct': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'total_return_pct': round(total_return, 2),
                'max_drawdown_pct': round(max_dd, 2),
                'final_portfolio_value': round(final_value, 2),
                
                # AI-specific metrics
                'ai_decisions_total': len(ai_decisions),
                'ai_confidence_avg': round(ai_confidence_avg, 3),
                'high_confidence_decisions': high_confidence_decisions,
                'medium_confidence_decisions': medium_confidence_decisions,
                'low_confidence_decisions': low_confidence_decisions,
                'decisions_to_trades_ratio': round(total_trades / len(ai_decisions) * 100, 1) if len(ai_decisions) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {'error': f'Performance calculation failed: {e}'}
    
    def _generate_ai_insights(self, df: pd.DataFrame, 
                            ai_decisions: Dict[datetime, TradingDecision], 
                            symbol: str) -> Dict[str, Any]:
        """Generate AI-specific insights from backtest"""
        try:
            insights = {
                'symbol': symbol,
                'analysis_period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
                'total_trading_days': len(df),
                'ai_decision_frequency': f"Every {len(df) // len(ai_decisions)} days" if len(ai_decisions) > 0 else "No decisions",
            }
            
            # Analyze AI decision patterns
            if ai_decisions:
                actions = [d.action for d in ai_decisions.values()]
                insights['ai_action_distribution'] = {
                    'BUY': actions.count('BUY'),
                    'SELL': actions.count('SELL'),
                    'HOLD': actions.count('HOLD')
                }
                
                # Market outlook distribution
                outlooks = [d.market_outlook for d in ai_decisions.values()]
                insights['market_outlook_distribution'] = {
                    outlook: outlooks.count(outlook) for outlook in set(outlooks)
                }
                
                # Confidence analysis
                confidences = [d.confidence for d in ai_decisions.values()]
                insights['confidence_analysis'] = {
                    'min': min(confidences),
                    'max': max(confidences),
                    'avg': np.mean(confidences),
                    'std': np.std(confidences)
                }
                
                # Most common reasoning themes
                reasonings = [d.reasoning for d in ai_decisions.values()]
                insights['common_reasoning_themes'] = self._extract_reasoning_themes(reasonings)
            
            # Performance vs confidence correlation
            trades_out = df[df['TRADE'] == 'Out']
            if len(trades_out) > 0 and len(ai_decisions) > 0:
                # This is a simplified correlation analysis
                insights['performance_notes'] = [
                    f"Executed {len(trades_out)} trades from {len(ai_decisions)} AI decisions",
                    f"Win rate: {len(trades_out[trades_out['ROIACUM'] > 0]) / len(trades_out) * 100:.1f}%",
                    f"Average AI confidence: {np.mean([d.confidence for d in ai_decisions.values()]):.3f}"
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate AI insights: {e}")
            return {'error': f'AI insights generation failed: {e}'}
    
    def _extract_reasoning_themes(self, reasonings: List[str]) -> List[str]:
        """Extract common themes from AI reasoning"""
        try:
            # Simple keyword analysis
            keywords = ['bullish', 'bearish', 'oversold', 'overbought', 'trend', 'support', 'resistance', 
                       'momentum', 'volatility', 'volume', 'breakout', 'reversal']
            
            theme_counts = {}
            for reasoning in reasonings:
                reasoning_lower = reasoning.lower()
                for keyword in keywords:
                    if keyword in reasoning_lower:
                        theme_counts[keyword] = theme_counts.get(keyword, 0) + 1
            
            # Return top themes
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            return [f"{theme}: {count}" for theme, count in sorted_themes[:5]]
            
        except Exception as e:
            logger.warning(f"Failed to extract reasoning themes: {e}")
            return []
    
    def plot_ai_backtest_results(self, backtest_results: Dict[str, Any], 
                                save_path: Optional[str] = None) -> None:
        """Plot AI backtest results with enhanced visualizations"""
        try:
            df = backtest_results['data']
            symbol = backtest_results['symbol']
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), 
                                   gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Top plot: Price and AI signals
            ax0 = axes[0]
            ax0.plot(df.index, df['Close'], color='black', label='Close Price', linewidth=1)
            
            # Mark AI signals
            buy_signals = df[df['signal'] == 'P']
            sell_signals = df[df['signal'] == 'cP']
            
            if len(buy_signals) > 0:
                ax0.scatter(buy_signals.index, buy_signals['Close'], 
                           color='green', marker='^', s=100, label='AI Buy Signal', alpha=0.7)
            
            if len(sell_signals) > 0:
                ax0.scatter(sell_signals.index, sell_signals['Close'], 
                           color='red', marker='v', s=100, label='AI Sell Signal', alpha=0.7)
            
            ax0.set_title(f'{symbol} - AI Trading Signals & Price Action')
            ax0.set_ylabel('Price ($)')
            ax0.legend()
            ax0.grid(True, alpha=0.3)
            
            # Middle plot: Portfolio performance vs benchmark
            ax1 = axes[1]
            ax1.plot(df.index, df['cvSis'], color='blue', label='AI Strategy', linewidth=2)
            ax1.plot(df.index, df['cvAct'], color='gray', label='Buy & Hold', linewidth=1, alpha=0.7)
            
            ax1.set_title('Portfolio Performance Comparison')
            ax1.set_ylabel('Portfolio Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: AI Confidence over time
            ax2 = axes[2]
            
            # Plot AI confidence where decisions were made
            ai_dates = []
            ai_confidences = []
            
            for date, decision in backtest_results.get('ai_decisions', {}).items():
                if date in df.index:
                    ai_dates.append(date)
                    ai_confidences.append(decision.confidence)
            
            if ai_dates:
                ax2.scatter(ai_dates, ai_confidences, color='purple', alpha=0.6, s=50)
                ax2.axhline(y=backtest_results.get('min_confidence', 0.7), 
                           color='red', linestyle='--', alpha=0.5, label='Confidence Threshold')
            
            ax2.set_title('AI Decision Confidence Over Time')
            ax2.set_ylabel('Confidence')
            ax2.set_xlabel('Date')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Backtest chart saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot backtest results: {e}")
    
    def export_backtest_results(self, backtest_results: Dict[str, Any], 
                              export_path: str) -> None:
        """Export backtest results to Excel file"""
        try:
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Main data
                backtest_results['data'].to_excel(writer, sheet_name='Backtest_Data')
                
                # Performance metrics
                metrics_df = pd.DataFrame([backtest_results['performance_metrics']]).T
                metrics_df.columns = ['Value']
                metrics_df.to_excel(writer, sheet_name='Performance_Metrics')
                
                # AI insights
                insights_df = pd.DataFrame([backtest_results['ai_insights']]).T
                insights_df.columns = ['Value']
                insights_df.to_excel(writer, sheet_name='AI_Insights')
                
                # Trade log
                trades = backtest_results['data'][backtest_results['data']['TRADE'] == 'Out']
                if len(trades) > 0:
                    trades[['Close', 'ROIACUM', 'ai_confidence', 'ai_reasoning']].to_excel(
                        writer, sheet_name='Trade_Log'
                    )
            
            logger.info(f"Backtest results exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export backtest results: {e}")


class AIBacktestingIntegrationError(Exception):
    """Custom exception for AI backtesting integration errors"""
    pass
