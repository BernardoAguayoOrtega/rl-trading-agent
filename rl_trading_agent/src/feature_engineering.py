import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import joblib # For saving the scaler

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
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    print("  - Added Trend Indicators (SMAs, MACD, ADX)")

    # --- 2. Momentum Indicators ---
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.roc(length=10, append=True)
    print("  - Added Momentum Indicators (RSI, Stochastic, ROC)")

    # --- 3. Volatility Indicators ---
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    print("  - Added Volatility Indicators (ATR, Bollinger Bands)")

    # --- 4. Volume Indicators ---
    df.ta.obv(append=True)
    df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
    print("  - Added Volume Indicators (OBV, Volume SMA)")

    # --- Clean up ---
    df.dropna(inplace=True)
    df = df.round(2)
    
    print(f"Feature engineering complete. Shape of final DataFrame: {df.shape}")
    
    return df

def normalize_data(df: pd.DataFrame, scaler_path: str = '../models/scaler.joblib'):
    """
    Normalizes the feature data using MinMaxScaler and saves the scaler.
    
    Args:
        df (pd.DataFrame): The input DataFrame with features.
        scaler_path (str): Path to save the fitted scaler object.

    Returns:
        pd.DataFrame: The DataFrame with normalized values.
    """
    print("\nNormalizing data...")
    
    # We only scale the features, not the date index
    feature_columns = df.columns
    
    # Initialize the scaler to scale data between 0 and 1
    scaler = MinMaxScaler()
    
    # Fit the scaler to the data and transform it
    df_scaled = scaler.fit_transform(df)
    
    # The output of the scaler is a numpy array, so we convert it back to a DataFrame
    df_normalized = pd.DataFrame(df_scaled, columns=feature_columns, index=df.index)
    
    # Save the scaler so we can use it later for new data (e.g., in production)
    joblib.dump(scaler, scaler_path)
    print(f"  - Scaler saved to '{scaler_path}'")
    
    print("Normalization complete.")
    return df_normalized


if __name__ == '__main__':
    # This is an example of how to use the full data preparation pipeline.
    try:
        # --- 1. Load Data ---
        data_path = '../data/dfAll.xlsx'
        original_df = pd.read_excel(data_path, index_col=0)
        
        # --- 2. Add Features ---
        featured_df = add_all_features(original_df.copy())
        
        # --- 3. Normalize Data ---
        # It's crucial to split data BEFORE normalizing to prevent lookahead bias.
        # The scaler should only be FIT on the training data.
        
        # Finalizing the Dataset Split (Last step of Phase 1)
        train_end_date = '2021-12-31'
        train_df = featured_df.loc[:train_end_date]
        test_df = featured_df.loc[train_end_date:]

        print(f"\nSplitting data at {train_end_date}:")
        print(f"  - Training set size: {len(train_df)}")
        print(f"  - Testing (OS) set size: {len(test_df)}")

        # Fit the scaler ONLY on the training data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df)
        
        # Use the SAME scaler to transform the test data
        test_scaled = scaler.transform(test_df)

        # Recreate DataFrames
        train_normalized = pd.DataFrame(train_scaled, columns=train_df.columns, index=train_df.index)
        test_normalized = pd.DataFrame(test_scaled, columns=test_df.columns, index=test_df.index)
        
        # Save the scaler
        scaler_path = '../models/data_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"  - Scaler fitted on training data and saved to '{scaler_path}'")


        print("\n--- Normalized Training Data Head ---")
        print(train_normalized.head())
        
        print("\n--- Normalized Testing Data Head ---")
        print(test_normalized.head())

    except FileNotFoundError:
        print(f"\nError: Make sure the data file is located at '{data_path}'")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
