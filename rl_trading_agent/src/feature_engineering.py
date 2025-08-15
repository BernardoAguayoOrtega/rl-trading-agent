import pandas as pd
import pandas_ta as ta

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a comprehensive set of technical indicators to the OHLCV DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.

    Returns:
        pd.DataFrame: The DataFrame with all indicator columns added.
    """
    print("Starting feature engineering...")
    
    # Ensure columns are in the correct format
    df.columns = [col.capitalize() for col in df.columns]
    
    # --- 1. Trend Indicators ---
    # Moving Averages (short, medium, long)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    
    # MACD (Moving Average Convergence Divergence)
    # This adds MACD_12_26_9, MACDh_12_26_9 (histogram), and MACDs_12_26_9 (signal)
    df.ta.macd(append=True)
    
    # ADX (Average Directional Index)
    # This adds ADX_14, DMN_14 (minus), and DMN_14 (plus)
    df.ta.adx(append=True)
    
    print("  - Added Trend Indicators (SMAs, MACD, ADX)")

    # --- 2. Momentum Indicators ---
    # RSI (Relative Strength Index) - using the standard 14 period
    df.ta.rsi(length=14, append=True)
    
    # Stochastic Oscillator (%K and %D)
    # This adds STOCHk_14_3_3 and STOCHd_14_3_3
    df.ta.stoch(append=True)

    # ROC (Rate of Change)
    df.ta.roc(length=10, append=True)
    
    print("  - Added Momentum Indicators (RSI, Stochastic, ROC)")

    # --- 3. Volatility Indicators ---
    # ATR (Average True Range)
    df.ta.atr(length=14, append=True)

    # Bollinger Bands
    # This adds BBL_20_2.0 (lower), BBM_20_2.0 (middle), BBU_20_2.0 (upper),
    # BBB_20_2.0 (bandwidth), BBP_20_2.0 (percent)
    df.ta.bbands(length=20, append=True)
    
    print("  - Added Volatility Indicators (ATR, Bollinger Bands)")

    # --- 4. Volume Indicators ---
    # OBV (On-Balance Volume)
    df.ta.obv(append=True)

    # Volume Moving Average
    df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    print("  - Added Volume Indicators (OBV, Volume SMA)")

    # --- Clean up ---
    # Drop rows with NaN values created by the indicators
    df.dropna(inplace=True)
    df = df.round(2)
    
    print(f"Feature engineering complete. Shape of final DataFrame: {df.shape}")
    
    return df

if __name__ == '__main__':
    # This is an example of how to use the function.
    # Make sure 'dfAll.xlsx' is in your '../data/' directory relative to this script.
    try:
        # Load the data
        data_path = '../data/dfAll.xlsx'
        original_df = pd.read_excel(data_path, index_col=0)
        
        # Apply the feature engineering
        featured_df = add_all_features(original_df.copy())
        
        print("\n--- DataFrame Head with New Features ---")
        print(featured_df.head())
        
        print("\n--- DataFrame Tail with New Features ---")
        print(featured_df.tail())
        
        print("\n--- All Columns ---")
        print(featured_df.columns.tolist())

    except FileNotFoundError:
        print(f"\nError: Make sure the data file is located at '{data_path}'")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
