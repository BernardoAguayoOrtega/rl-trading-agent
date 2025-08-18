"""
Dynamic Trading Framework Module

This module provides a flexible trading system that works with pandas-ta indicators
and supports dynamic signal generation for RL agents. It removes the hardcoded
SMA/RSI logic from the original trading_framework.py while maintaining all
P&L calculation and performance analysis functions.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


class DynamicSignalGenerator:
    """
    Generates trading signals based on flexible indicator combinations
    using pandas-ta indicators instead of hardcoded logic.
    """
    
    def __init__(self):
        self.supported_indicators = {
            'sma': self._add_sma,
            'ema': self._add_ema, 
            'rsi': self._add_rsi,
            'macd': self._add_macd,
            'bb': self._add_bollinger_bands,
            'stoch': self._add_stochastic,
            'atr': self._add_atr,
            'adx': self._add_adx,
        }
    
    def add_indicators(self, df, indicator_config):
        """
        Add pandas-ta indicators to dataframe based on configuration
        
        Args:
            df: DataFrame with OHLCV data
            indicator_config: Dictionary with indicator parameters
            
        Returns:
            DataFrame with indicators added
        """
        df_work = df.copy()
        
        # Add indicators based on configuration
        for indicator_name, params in indicator_config.items():
            if indicator_name in self.supported_indicators:
                df_work = self.supported_indicators[indicator_name](df_work, params)
        
        return df_work
    
    def _add_sma(self, df, params):
        """Add Simple Moving Average indicators"""
        periods = params if isinstance(params, list) else [params]
        for period in periods:
            df[f'SMA_{period}'] = ta.sma(df['Close'], length=period)
        return df
    
    def _add_ema(self, df, params):
        """Add Exponential Moving Average indicators"""
        periods = params if isinstance(params, list) else [params]
        for period in periods:
            df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
        return df
    
    def _add_rsi(self, df, params):
        """Add RSI indicators"""
        periods = params if isinstance(params, list) else [params]
        for period in periods:
            df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period)
        return df
    
    def _add_macd(self, df, params):
        """Add MACD indicators"""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26) 
        signal = params.get('signal', 9)
        
        macd_result = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
        if macd_result is not None:
            df = df.join(macd_result)
        return df
    
    def _add_bollinger_bands(self, df, params):
        """Add Bollinger Bands"""
        period = params.get('period', 20)
        std = params.get('std', 2.0)
        
        bb_result = ta.bbands(df['Close'], length=period, std=std)
        if bb_result is not None:
            df = df.join(bb_result)
        return df
    
    def _add_stochastic(self, df, params):
        """Add Stochastic Oscillator"""
        k = params.get('k', 14)
        d = params.get('d', 3)
        
        stoch_result = ta.stoch(df['High'], df['Low'], df['Close'], k=k, d=d)
        if stoch_result is not None:
            df = df.join(stoch_result)
        return df
    
    def _add_atr(self, df, params):
        """Add Average True Range"""
        period = params.get('period', 14)
        df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
        return df
    
    def _add_adx(self, df, params):
        """Add Average Directional Index"""
        period = params.get('period', 14)
        adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        if adx_result is not None:
            df = df.join(adx_result)
        return df
    
    def generate_signals(self, df, strategy_rules):
        """
        Generate trading signals based on flexible strategy rules
        
        Args:
            df: DataFrame with indicators
            strategy_rules: Dictionary with entry/exit conditions
            
        Returns:
            DataFrame with signal and position columns
        """
        df['signal'] = ''
        
        # Apply entry rules
        entry_conditions = strategy_rules.get('entry_conditions', [])
        if entry_conditions:
            entry_mask = self._evaluate_conditions(df, entry_conditions)
            df.loc[entry_mask, 'signal'] = 'P'
        
        # Apply exit rules  
        exit_conditions = strategy_rules.get('exit_conditions', [])
        if exit_conditions:
            exit_mask = self._evaluate_conditions(df, exit_conditions)
            df.loc[exit_mask, 'signal'] = 'cP'
        
        # Avoid lookahead bias - shift signals by 1 period
        df['position'] = df.signal.shift()
        
        return df
    
    def _evaluate_conditions(self, df, conditions):
        """
        Evaluate a list of conditions and return combined mask
        
        Args:
            df: DataFrame with indicators
            conditions: List of condition dictionaries
            
        Returns:
            Boolean mask for rows meeting conditions
        """
        if not conditions:
            return pd.Series(False, index=df.index)
        
        masks = []
        for condition in conditions:
            mask = self._evaluate_single_condition(df, condition)
            masks.append(mask)
        
        # Combine conditions with AND logic by default
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask
            
        return combined_mask
    
    def _evaluate_single_condition(self, df, condition):
        """
        Evaluate a single condition
        
        Args:
            df: DataFrame with indicators
            condition: Dictionary with condition parameters
            
        Returns:
            Boolean mask for the condition
        """
        indicator1 = condition.get('indicator1')
        operator = condition.get('operator')
        indicator2 = condition.get('indicator2')
        value = condition.get('value')
        
        if indicator1 not in df.columns:
            return pd.Series(False, index=df.index)
        
        series1 = df[indicator1]
        
        if operator == '>':
            if indicator2 and indicator2 in df.columns:
                return series1 > df[indicator2]
            elif value is not None:
                return series1 > value
        elif operator == '<':
            if indicator2 and indicator2 in df.columns:
                return series1 < df[indicator2]
            elif value is not None:
                return series1 < value
        elif operator == '>=':
            if indicator2 and indicator2 in df.columns:
                return series1 >= df[indicator2]
            elif value is not None:
                return series1 >= value
        elif operator == '<=':
            if indicator2 and indicator2 in df.columns:
                return series1 <= df[indicator2]
            elif value is not None:
                return series1 <= value
        elif operator == '==':
            if indicator2 and indicator2 in df.columns:
                return series1 == df[indicator2]
            elif value is not None:
                return series1 == value
        elif operator == 'crossover':
            if indicator2 and indicator2 in df.columns:
                # Crossover: series1 crosses above series2
                prev_below = (series1.shift(1) <= df[indicator2].shift(1))
                curr_above = (series1 > df[indicator2])
                return prev_below & curr_above
        elif operator == 'crossunder':
            if indicator2 and indicator2 in df.columns:
                # Crossunder: series1 crosses below series2
                prev_above = (series1.shift(1) >= df[indicator2].shift(1))
                curr_below = (series1 < df[indicator2])
                return prev_above & curr_below
        
        return pd.Series(False, index=df.index)


class DynamicTradingSystem:
    """
    Complete trading system using dynamic signal generation
    and maintaining all original P&L calculation functions
    """
    
    def __init__(self):
        self.signal_generator = DynamicSignalGenerator()
    
    def backtest_strategy(self, df, strategy_config):
        """
        Run complete backtest with dynamic strategy
        
        Args:
            df: DataFrame with OHLCV data
            strategy_config: Dictionary with complete strategy configuration
            
        Returns:
            Tuple of (processed_dataframe, performance_metrics_list)
        """
        # Step 1: Add indicators
        indicator_config = strategy_config.get('indicators', {})
        df_processed = self.signal_generator.add_indicators(df.copy(), indicator_config)
        
        # Step 2: Generate signals
        strategy_rules = strategy_config.get('rules', {})
        df_processed = self.signal_generator.generate_signals(df_processed, strategy_rules)
        
        # Step 3: Convert to trade positions (using original function)
        df_processed = damePosition(df_processed)
        
        # Step 4: Apply candle exit rules (using original function)
        candle_limit = strategy_config.get('candle_limit', 0)
        df_processed = dameSalidaVelas(df_processed, candle_limit)
        
        # Step 5: Calculate P&L (using original function)
        direction = strategy_config.get('direction', 'long')
        take_profit = strategy_config.get('take_profit', 0)
        stop_loss = strategy_config.get('stop_loss', 0)
        commission = strategy_config.get('commission', 0)
        slippage = strategy_config.get('slippage', 0)
        
        df_processed = dameSalidaPnl(df_processed, direction, take_profit, 
                                   stop_loss, commission, slippage)
        
        # Step 6: Calculate performance curves (using original function)
        position_size = strategy_config.get('position_size', 1)
        df_processed = calculaCurvas(df_processed, position_size)
        
        # Step 7: Generate performance metrics (using original function)
        performance_metrics = backSistemaList(df_processed)
        
        return df_processed, performance_metrics


# Import original functions from trading_framework.py
# These remain unchanged as they don't contain hardcoded logic

def damePosition(df):
    """
    Convert position signals to trade indicators
    
    Converts 'P' (position) and 'cP' (close position) signals into:
    - 'In': Entry into trade
    - 'p': Continuation of trade
    - 'Out': Exit from trade
    
    Args:
        df: DataFrame with position column
        
    Returns:
        DataFrame with TRADE column containing trade states
    """
    df['tradeInd'] = ''
    
    inTrade = False
    for i in df.index:
        
        cell = df.loc[i, 'position']
    
        if pd.isna(cell):
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
    
    df['TRADE'] = df.tradeInd

    return df


def dameSalidaVelas(df, num):
    """
    Handle candle-based exit rules
    
    If num > 0, forces exit after maximum number of candles in position
    If num = 0, no candle limit is applied
    
    Args:
        df: DataFrame with TRADE column
        num: Maximum number of candles to stay in position (0 = no limit)
        
    Returns:
        DataFrame with updated TRADE column
    """
    df['velas'] = 0
    df = df.reset_index(drop=False)
    
    for i in df.index:
        if i > 1:
            cell = df.loc[i, 'TRADE']
            
            if cell == 'In':
                df.loc[i, 'velas'] = 1
            elif cell == 'p':
                df.loc[i, 'velas'] = df.loc[i-1, 'velas'] + 1
                
    numMaxVelas = num
    if numMaxVelas:
        df['tradeVela'] = ''
        
        for i in df.index:
            cell = df.loc[i, 'TRADE']
            vela = df.loc[i, 'velas']
            
            if vela <= numMaxVelas:
                df.loc[i, 'tradeVela'] = cell if cell in ['In', 'p'] else ''
        
        df['tradeVela'] = np.where(
            (df['tradeVela'] == '') & (df['tradeVela'].shift().isin(['In', 'p'])),
            'Out', df['tradeVela']
        )
        
        df['TRADE'] = df.tradeVela
    else:
        df['tradeVela'] = df.TRADE

    df.set_index('Date', inplace=True)
    
    return df


def dameSalidaPnl(df, sentido, tp, sl, comision, slippage):
    """
    Calculate P&L with stop-loss and take-profit management
    
    Args:
        df: DataFrame with TRADE column
        sentido: Direction ('long' or 'short')
        tp: Take profit % (e.g., 3.5 for +3.5%)
        sl: Stop loss % (e.g., -1.5 for -1.5%)
        comision: Commission % (e.g., 0.01 for 0.01%)
        slippage: Slippage % (e.g., 0.02 for 0.02%)
        
    Returns:
        DataFrame with ROID and ROIACUM columns
    """
    def pnlSalida(Open, High, Low, Close, precioRef, sentido, tp, sl):
        """
        Determine if SL/TP is hit or normal session during a candle
        
        PARAMETERS:
        - Open, High, Low, Close: OHLC prices of the candle
        - precioRef: Reference price (position entry)
        - sentido: 'long' or 'short'
        - sl: Stop Loss in % (e.g., -2.0 for -2%), 0 if not considered
        - tp: Take Profit in % (e.g., 0.8 for +0.8%), 0 if not considered
       
        LOGIC:
        - Priority: SL first, then TP. In a session, SL is checked first, then TP
        - If exceeded at open, real ROI from open
        - If exceeded during candle, theoretical ROI from SL/TP
    
        RETURNS:
        - Tuple (exitType, roiPercentage)
        - exitType: sltp, or normal
        - roiPercentage: REAL ROI in % relative to precioRef
        """
        # Helper function to calculate real ROI
        def calcularRoi(precio, precioRef, sentido):
            roi = (precio - precioRef) / precioRef * 100
            if sentido == 'short':
                roi = -roi
            return roi
    
        # CHECK SL (PRIORITY 1)
        if sl:
            if sentido == 'long':
                precioSl = precioRef * (1 + sl/100)
                
                # Check if SL is touched
                if Open <= precioSl or Low <= precioSl:
                    # If touched at open, use open ROI
                    if Open <= precioSl:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    # Touched during candle, use theoretical SL    
                    else:
                        return ('sltp', sl)
            # short           
            else:  
                precioSl = precioRef * (1 - sl/100)
                
                # Check if SL is touched
                if Open >= precioSl or High >= precioSl:
                    # If touched at open, use open ROI
                    if Open >= precioSl:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    # Touched during candle, use theoretical SL   
                    else:
                        return ('sltp', sl)
    
        # CHECK TP (PRIORITY 2)
        if tp:
            if sentido == 'long':
                precioTp = precioRef * (1 + tp/100)
                
                # Check if TP is touched
                if Open >= precioTp or High >= precioTp:
                    # If touched at open, use open ROI
                    if Open >= precioTp:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    # Touched during candle, use theoretical TP
                    else:
                        return ('sltp', tp)
            # short           
            else:  
                precioTp = precioRef * (1 - tp/100)
                
                # Check if TP is touched
                if Open <= precioTp or Low <= precioTp:
                    # If touched at open, use open ROI
                    if Open <= precioTp:
                        return ('sltp', calcularRoi(Open, precioRef, sentido))
                    # Touched during candle, use theoretical TP
                    else:
                        return ('sltp', tp)
    
        # NORMAL EXIT - calculate ROI at close
        return ('normal', calcularRoi(Close, precioRef, sentido))

    df['ROID'] = 0
    df['ROIACUM'] = 0  
    
    df['precioRef'] = 0
    df['tipoSalida'] = ''
    
    df = df.reset_index(drop=False)
    
    if tp or sl:
        for i in df.index:
            estado = df.loc[i, 'TRADE']
    
            O = df.loc[i, 'Open']
            H = df.loc[i, 'High']
            L = df.loc[i, 'Low']
            C = df.loc[i, 'Close']
           
            if i > 0:
                if estado == 'In':   
                    df.loc[i,'tradePnl'] = 'In'
                    
                    precioRef = df.loc[i, 'Open']
                    salida, roi = pnlSalida(O, H, L, C, precioRef, sentido, tp, sl)

                    # Apply costs only at position opening
                    coste = (comision + slippage)
                    roi -= coste
    
                    # Both normal and sl/tp exits
                    df.loc[i, 'ROID'] = roi
                    df.loc[i, 'ROIACUM'] = roi
                    df.loc[i, 'precioRef'] = precioRef
                    df.loc[i, 'tipoSalida'] = salida
    
                elif estado in ['p','Out']:
                    # p and Out can only have In or p before, else orphan and remove
                    if df.loc[i-1, 'tradePnl'] not in ['In','p']: 
                        df.loc[i,'tradePnl'] = ''
                        df.loc[i, 'ROID'] = 0
                        df.loc[i, 'ROIACUM'] = 0
                        df.loc[i, 'precioRef'] = 0
                        df.loc[i, 'tipoSalida'] = ''
    
                    # if before is In or p
                    else:
                        # sltp exit in previous
                        if df.loc[i-1,'tipoSalida'] == 'sltp':
                            df.loc[i,'tradePnl'] = 'Out'
                            df.loc[i, 'ROID'] = 0
                            df.loc[i, 'ROIACUM'] = df.loc[i-1, 'ROIACUM']
                            df.loc[i, 'precioRef'] = 0
                            df.loc[i, 'tipoSalida'] = ''
                        
                        # previous is normal
                        else:
                            precioRef = df.loc[i-1, 'precioRef']
                            salida, roiRef = pnlSalida(O, H, L, C, precioRef, sentido, tp, sl)
    
                            # normal exit, no sl or tp touched
                            if salida == 'normal':
                                df.loc[i,'tradePnl'] = estado
                                
                                if estado == 'p':
                                    df.loc[i, 'ROID'] = (df.loc[i, 'Close'] / df.loc[i-1, 'Close'] - 1) * 100
                                if estado == 'Out':
                                    df.loc[i, 'ROID'] = (df.loc[i, 'Open'] / df.loc[i-1, 'Close'] - 1) * 100
                                    
                                df.loc[i, 'ROIACUM'] = roiRef
                                df.loc[i, 'precioRef'] = precioRef
                                df.loc[i, 'tipoSalida'] = salida
    
                            # sl or tp touched
                            else:
                                df.loc[i,'tradePnl'] = 'Out'
                                df.loc[i, 'ROID'] = ((1+roiRef/100) / (1+ df.loc[i-1, 'ROIACUM']/100) - 1) * 100
                                df.loc[i, 'ROIACUM'] = roiRef
                                df.loc[i, 'precioRef'] = 0
                                df.loc[i, 'tipoSalida'] = salida
    
                # When empty
                else:
                    df.loc[i,'tradePnl'] = ''
                    df.loc[i, 'ROID'] = 0
                    df.loc[i, 'ROIACUM'] = 0
                    df.loc[i, 'precioRef'] = 0
                    df.loc[i, 'tipoSalida'] = ''
                
        df['TRADE'] = df.tradePnl
        
    # Both sl and tp are 0, only calculate ROID and ROIACUM
    # df.TRADE already has correct values from df.tradeVela
    else:
        # Apply costs only at position opening
        coste = (comision + slippage)
        for i in df.index:
            if i > 0:
                estado = df.loc[i, 'TRADE']
            
                if estado == 'In':
                    df.loc[i, 'ROID'] = (df.loc[i, 'Close'] / df.loc[i, 'Open'] -1) * 100 - coste
                    df.loc[i, 'ROIACUM'] = df.loc[i, 'ROID']
                    
                elif estado == 'p':
                    df.loc[i, 'ROID'] = (df.loc[i, 'Close'] / df.loc[i-1, 'Close'] - 1) * 100
                    df.loc[i, 'ROIACUM'] = ((1+df.loc[i, 'ROID']/100) * (1+df.loc[i-1, 'ROIACUM']/100) -1)*100
                    
                elif estado == 'Out':
                    df.loc[i, 'ROID'] = (df.loc[i, 'Open'] / df.loc[i-1, 'Close'] - 1) * 100
                    df.loc[i, 'ROIACUM'] = ((1+df.loc[i, 'ROID']/100) * (1+df.loc[i-1, 'ROIACUM']/100) -1)*100
                    
                else:
                    df.loc[i, 'ROID'] = 0
                    df.loc[i, 'ROIACUM'] = 0

        if sentido != 'long':
            df['ROID'] = -df['ROID']
            df['ROIACUM'] = -df['ROIACUM']
            

    df.set_index('Date', inplace=True)

    # For verification: roiComprobar is the roiD calculated from tradeVela
    df['roiComprobar'] = 0

    df = df.reset_index(drop=False)
    for i in df.index:
        estado = df.loc[i, 'tradeVela']
      
        if i > 0:
            if estado == 'In':
                df.loc[i, 'roiComprobar'] = (df.loc[i, 'Close'] / df.loc[i, 'Open'] -1) * 100
            elif estado == 'p':
                df.loc[i, 'roiComprobar'] = (df.loc[i, 'Close'] / df.loc[i-1, 'Close'] - 1) * 100
            elif estado == 'Out':
                df.loc[i, 'roiComprobar'] = (df.loc[i, 'Open'] / df.loc[i-1, 'Close'] - 1) * 100
            else:
                df.loc[i, 'roiComprobar'] = 0
    
    df.set_index('Date', inplace=True)
    
    return df


def calculaCurvas(df, size=1):
    """
    Calculate performance curves and metrics
    
    Args:
        df: DataFrame with ROID and ROIACUM columns
        size: Position size multiplier
        
    Returns:
        DataFrame with additional performance columns
    """
    # Apply position sizing
    df['ROID_sized'] = df['ROID'] * size
    df['ROIACUM_sized'] = df['ROIACUM'] * size
    
    # Calculate equity curve (cumulative returns)
    df['equity'] = (1 + df['ROID_sized']/100).cumprod()
    df['equity_curve'] = df['equity'] * 100  # Convert to percentage
    
    # Calculate drawdown
    df['peak'] = df['equity'].cummax()
    df['drawdown'] = (df['equity'] / df['peak'] - 1) * 100
    
    return df


def backSistemaList(df):
    """
    Calculate comprehensive backtest metrics and return as list
    
    Args:
        df: DataFrame with complete trading results
        
    Returns:
        List with comprehensive trading metrics in specific order
    """
    # Filter only completed trades
    trades = df[df['TRADE'].isin(['In', 'p', 'Out'])].copy()
    
    if trades.empty:
        # Return zeros if no trades
        return [0] * 25
    
    # Basic trade statistics
    total_trades = len(trades[trades['TRADE'] == 'Out'])
    
    if total_trades == 0:
        total_trades = len(trades[trades['TRADE'] == 'In'])  # Open positions
    
    # P&L calculations
    total_return = trades['ROIACUM'].iloc[-1] if not trades.empty else 0
    
    # Winning/Losing trades
    completed_trades = trades[trades['TRADE'] == 'Out']['ROIACUM']
    if completed_trades.empty:
        # Use all trades if no completed ones
        completed_trades = trades['ROIACUM']
    
    winning_trades = completed_trades[completed_trades > 0]
    losing_trades = completed_trades[completed_trades < 0]
    
    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    
    win_rate = (num_wins / (num_wins + num_losses)) * 100 if (num_wins + num_losses) > 0 else 0
    
    avg_win = winning_trades.mean() if not winning_trades.empty else 0
    avg_loss = losing_trades.mean() if not losing_trades.empty else 0
    
    # Profit factor
    gross_profit = winning_trades.sum() if not winning_trades.empty else 0
    gross_loss = abs(losing_trades.sum()) if not losing_trades.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    
    # Risk metrics
    returns = trades['ROID'].dropna()
    
    # Annualized return (CAGR approximation)
    days_total = len(df)
    years = days_total / 252  # Assuming daily data, 252 trading days per year
    cagr = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else total_return
    
    # Maximum drawdown
    max_drawdown = df['drawdown'].min() if 'drawdown' in df.columns else 0
    
    # Sharpe ratio approximation
    if not returns.empty:
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualize
    else:
        sharpe_ratio = 0
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    
    # Return comprehensive metrics list (matching expected format)
    return [
        total_return,           # 0: Total return %
        days_total,            # 1: Total days/periods
        total_trades,          # 2: Number of operations
        num_wins,              # 3: Winning trades
        num_losses,            # 4: Losing trades
        win_rate,              # 5: Win rate %
        avg_win,               # 6: Average winning trade
        avg_loss,              # 7: Average losing trade
        gross_profit,          # 8: Gross profit
        gross_loss,            # 9: Gross loss
        expectancy,            # 10: Expectancy
        sharpe_ratio,          # 11: Sharpe ratio
        cagr,                  # 12: CAGR %
        0,                     # 13: Placeholder
        0,                     # 14: Placeholder
        0,                     # 15: Placeholder
        0,                     # 16: Placeholder
        0,                     # 17: Placeholder
        profit_factor,         # 18: Profit factor
        0,                     # 19: Placeholder
        0,                     # 20: Placeholder
        max_drawdown,          # 21: Maximum drawdown
        0,                     # 22: Placeholder
        0,                     # 23: Placeholder
        0                      # 24: Placeholder
    ]


# Utility functions for RL integration

def create_agent_strategy_config(agent_action):
    """
    Convert RL agent action into strategy configuration
    
    Args:
        agent_action: Dictionary with agent's chosen parameters
        
    Returns:
        Strategy configuration dictionary
    """
    # Extract parameters from agent action
    fast_sma = agent_action.get('fast_sma', 10)
    slow_sma = agent_action.get('slow_sma', 20)
    rsi_period = agent_action.get('rsi_period', 14)
    rsi_entry = agent_action.get('rsi_entry', 30)
    rsi_exit = agent_action.get('rsi_exit', 70)
    
    # Create dynamic strategy configuration
    strategy_config = {
        'indicators': {
            'sma': [fast_sma, slow_sma],
            'rsi': [rsi_period]
        },
        'rules': {
            'entry_conditions': [
                {
                    'indicator1': f'RSI_{rsi_period}',
                    'operator': '<',
                    'value': rsi_entry
                },
                {
                    'indicator1': 'Close',
                    'operator': '>',
                    'indicator2': f'SMA_{slow_sma}'
                }
            ],
            'exit_conditions': [
                {
                    'indicator1': f'RSI_{rsi_period}',
                    'operator': '>',
                    'value': rsi_exit
                },
                {
                    'indicator1': 'Close',
                    'operator': '<',
                    'indicator2': f'SMA_{slow_sma}'
                }
            ]
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


def backtest_agent_strategy(df, agent_action):
    """
    Convenience function to backtest an RL agent's strategy
    
    Args:
        df: DataFrame with OHLCV data
        agent_action: Dictionary with agent's chosen parameters
        
    Returns:
        Tuple of (processed_dataframe, performance_metrics_list)
    """
    # Create strategy configuration from agent action
    strategy_config = create_agent_strategy_config(agent_action)
    
    # Run backtest using dynamic trading system
    trading_system = DynamicTradingSystem()
    return trading_system.backtest_strategy(df, strategy_config)


def create_complex_strategy_config(agent_action):
    """
    Create more complex strategy configurations with multiple indicators
    
    Args:
        agent_action: Dictionary with agent's chosen parameters
        
    Returns:
        Advanced strategy configuration dictionary
    """
    # This can be expanded to support more complex strategies
    # based on the agent's learning progress
    
    fast_ma = agent_action.get('fast_ma', 10)
    slow_ma = agent_action.get('slow_ma', 20)
    rsi_period = agent_action.get('rsi_period', 14)
    macd_fast = agent_action.get('macd_fast', 12)
    macd_slow = agent_action.get('macd_slow', 26)
    macd_signal = agent_action.get('macd_signal', 9)
    
    strategy_config = {
        'indicators': {
            'sma': [fast_ma, slow_ma],
            'rsi': [rsi_period],
            'macd': {
                'fast': macd_fast,
                'slow': macd_slow,
                'signal': macd_signal
            }
        },
        'rules': {
            'entry_conditions': [
                {
                    'indicator1': f'SMA_{fast_ma}',
                    'operator': 'crossover',
                    'indicator2': f'SMA_{slow_ma}'
                },
                {
                    'indicator1': f'RSI_{rsi_period}',
                    'operator': '<',
                    'value': 50
                },
                {
                    'indicator1': f'MACD_{macd_fast}_{macd_slow}_{macd_signal}',
                    'operator': '>',
                    'indicator2': f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
                }
            ],
            'exit_conditions': [
                {
                    'indicator1': f'SMA_{fast_ma}',
                    'operator': 'crossunder',
                    'indicator2': f'SMA_{slow_ma}'
                },
                {
                    'indicator1': f'RSI_{rsi_period}',
                    'operator': '>',
                    'value': 70
                }
            ]
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
