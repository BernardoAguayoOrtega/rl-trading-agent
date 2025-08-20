# data/loader.py
"""Data loading utilities for the RL trading agent.

The :func:`load_market_data` function pulls OHLCV data for a given ticker and
period.  It supports two modes:

1. **CSV source** – if ``source`` is provided, the function will read the
   CSV file located at that path.  The CSV is expected to contain at least
   the columns ``open``, ``high``, ``low``, ``close`` and ``volume`` (case
   insensitive).  The index should be a datetime column named ``date`` or
   ``timestamp``; if it is not present, the function will try to parse the
   first column as a date.

2. **Yahoo Finance fallback** – if ``source`` is ``None`` the function will
   download the data using :mod:`yfinance`.  The returned DataFrame will be
   normalised to the required column names and index.

The function returns a :class:`pandas.DataFrame` with the following
properties:

* Columns: ``open``, ``high``, ``low``, ``close``, ``volume``
* Datetime index
* ``float`` dtype for price columns and ``int`` for volume

Example
-------
>>> df = load_market_data("AAPL", "2023-01-01", "2023-02-01")
>>> df.head()
            open   high    low  close  volume
2023-01-02  150.0  155.0  149.0  154.0  1000000
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yfinance as yf

__all__ = ["load_market_data"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and keep only the required OHLCV columns.

    Parameters
    ----------
    df:
        DataFrame returned by yfinance or read from CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``open``, ``high``, ``low``, ``close`` and
        ``volume`` and a datetime index.
    """
    # Standardise column names to lower case
    df = df.rename(columns=str.lower)
    # Map common yfinance column names to the required ones
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)
    # Verify that all required columns are present
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    # Keep only the required columns
    df = df[required]
    # Ensure numeric dtypes
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    # Keep original integer dtype for volume (avoid downcasting to int16)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    return df


def load_market_data(
    ticker: str,
    start: str,
    end: str,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Load market data for a ticker.

    Parameters
    ----------
    ticker:
        The ticker symbol to download.
    start, end:
        Date strings in ``YYYY-MM-DD`` format.  ``end`` is inclusive.
    source:
        Optional path to a CSV file.  If provided, the function will read
        the CSV instead of downloading from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        OHLCV data with a datetime index.
    """
    if source:
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV source {source!r} does not exist")
        # Read CSV with the first column as the index (dates) and parse dates
        df = pd.read_csv(source, parse_dates=True, index_col=0)
        # If the index is not datetime, try to set it
        if not isinstance(df.index, pd.DatetimeIndex):
            # Look for a date column
            date_col = None
            for col in ["date", "timestamp", "datetime"]:
                if col in df.columns:
                    date_col = col
                    break
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                # Assume first column is date
                df = df.set_index(df.columns[0])
                df.index = pd.to_datetime(df.index)
        df = _ensure_columns(df)
        # Ensure the index name is None for consistency with make_df()
        df.index.name = None
    else:
        # Use yfinance to download data
        df = yf.download(ticker, start=start, end=end, progress=False, multi_level_index=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker} between {start} and {end}")
        df = _ensure_columns(df)
    # Drop rows with missing values
    df = df.dropna()
    return df
