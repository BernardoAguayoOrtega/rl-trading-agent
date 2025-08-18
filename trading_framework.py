"""
Trading Framework Module

This module contains the core backtesting and trading system functions
extracted from the Jupyter notebook. These functions handle signal generation,
position tracking, P&L calculation, and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ocpSma(df, periodo=20, borraNan=False, col='Close'):
    """
    Calculate Simple Moving Average
    
    Args:
        df: DataFrame with OHLCV data
        periodo: Period for SMA calculation
        borraNan: Whether to drop NaN values
        col: Column to calculate SMA on
        
    Returns:
        DataFrame with SMA column added
    """
    df[f's{periodo}'] = df[col].rolling(periodo).mean()
    
    if borraNan:
        df.dropna(inplace=True)
        
    return df


def ocpRsi(df, periodo=14, borraNan=False, col='Close'):
    """
    Calculate RSI (Relative Strength Index)
    
    Args:
        df: DataFrame with OHLCV data
        periodo: Period for RSI calculation
        borraNan: Whether to drop NaN values
        col: Column to calculate RSI on
        
    Returns:
        DataFrame with RSI column added
    """
    df['dif'] = df[col].diff()

    df['win'] = np.where(df['dif'] > 0, df['dif'], 0)
    df['loss'] = np.where(df['dif'] < 0, abs(df['dif']), 0)
    df['emaWin'] = df.win.ewm(span=periodo).mean()
    df['emaLoss'] = df.loss.ewm(span=periodo).mean()
    df['rs'] = df.emaWin / df.emaLoss

    df[f'rsi{periodo}'] = 100 - (100 / (1 + df.rs))

    df.drop(['dif', 'win', 'loss', 'emaWin', 'emaLoss', 'rs'], axis=1, inplace=True)

    if borraNan:
        df.dropna(inplace=True)

    return df


def dameSistema(df, perSma, perRsi, rsiIn, rsiOut):
    """
    Generate trading signals based on SMA and RSI
    
    Larry Connors RSI2 system implementation:
    - Long Entry: RSI < rsiIn AND Close > SMA
    - Long Exit: RSI > rsiOut OR Close < SMA
    
    Args:
        df: DataFrame with OHLCV data
        perSma: SMA period
        perRsi: RSI period
        rsiIn: RSI entry threshold
        rsiOut: RSI exit threshold
        
    Returns:
        DataFrame with signal and position columns
    """
    df['signal'] = ''
    
    # Long entry: RSI < rsiIn AND Close > SMA
    df['signal'] = np.where(
        (df[f'rsi{perRsi}'] < rsiIn) & (df.Close > df[f's{perSma}']), 
        'P', df.signal
    )
    
    # Long exit: RSI > rsiOut OR Close < SMA
    df['signal'] = np.where(
        (df[f'rsi{perRsi}'] > rsiOut) | (df.Close < df[f's{perSma}']), 
        'cP', df.signal
    )
    
    # Avoid lookahead bias - shift signals by 1 period
    df['position'] = ''
    df['position'] = df.signal.shift()

    return df


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


def dameGraficoSistema(df, numVelas, perSma, perRsi, rsiIn, rsiOut):
    """
    Generate system visualization chart
    
    Args:
        df: DataFrame with trading results
        numVelas: Number of recent candles to plot
        perSma: SMA period for labeling
        perRsi: RSI period for labeling
        rsiIn: RSI entry threshold
        rsiOut: RSI exit threshold
    """
    dg = df.tail(numVelas).copy()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # First subplot: Price
    ax0 = axes[0]
    
    ax0.plot(dg.index, dg['Close'], color='black', label='Close')
    ax0.plot(dg.index, dg[f's{perSma}'], color='blue', linestyle='--', label='Sma200', alpha=0.5)
    
    # Add green and red points
    dateInTrade = dg.index[dg['TRADE'] == 'In']
    dateOutTrade = dg.index[dg['TRADE'] == 'Out']
    ax0.scatter(dateInTrade, dg.loc[dateInTrade, 'Close'], color='green', label='In', marker='o', alpha=0.5)
    ax0.scatter(dateOutTrade, dg.loc[dateOutTrade, 'Close'], color='red', label='Out', marker='o', alpha=0.5)
    
    ax0.set_title('PRICE CHART') 
    ax0.set_ylabel('Close')
    ax0.legend(loc='upper left')
    ax0.grid(True)
    
    # Second subplot: Indicators
    ax1 = axes[1]
    
    ax1.plot(dg.index, dg[f'rsi{perRsi}'], color='gray', label='rsi2')
    
    ax1.axhline(90, color='red', linestyle='--', linewidth=0.8)
    ax1.axhline(rsiOut, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axhline(rsiIn, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax1.set_title('INDICATORS') 
    ax1.set_ylabel('rsi2')
    ax1.set_xlabel('Date') 
    ax1.legend(loc='upper left')
    
    # Adjust layout to avoid overlaps
    plt.tight_layout()

    plt.show()

    return
