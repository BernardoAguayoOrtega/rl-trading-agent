import sys
sys.path.append("/Users/bernardo/Documents/GitHub/rl-trading-agent")

import pytest
import pandas as pd
from datetime import datetime

from data.agent import DataAgent
from data.utils import validate_dataframe, resample_to_daily

# Helper to create a simple OHLCV DataFrame

def make_df():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    data = {
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    return pd.DataFrame(data, index=dates)

# ---------- Tests for data.utils ----------

def test_validate_dataframe_success():
    df = make_df()
    # Should not raise
    validate_dataframe(df)


def test_validate_dataframe_missing_columns():
    df = make_df().drop(columns=["volume"])
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df)
    assert "volume" in str(exc.value)


def test_resample_to_daily():
    df = make_df()
    # Resample hourly to daily (should be same as original)
    resampled = resample_to_daily(df, freq="1D")
    pd.testing.assert_frame_equal(df, resampled)


def test_resample_to_daily_hourly():
    # Create hourly data
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    data = {
        "open": range(48),
        "high": range(48),
        "low": range(48),
        "close": range(48),
        "volume": [100]*48,
    }
    df = pd.DataFrame(data, index=dates)
    daily = resample_to_daily(df, freq="1D")
    assert len(daily) == 2
    # Volume should sum
    assert daily.iloc[0]["volume"] == 100*24

# ---------- Tests for data.agent ----------

def test_agent_load_from_csv(tmp_path):
    # Create a CSV file
    df = make_df()
    csv_path = tmp_path / "data.csv"
    # Save with the index so dates are preserved
    df.to_csv(csv_path)
    agent = DataAgent(source=str(csv_path))
    loaded = agent.load("AAPL", "2023-01-01", "2023-01-05")
    # The loader already returns a DataFrame with a DatetimeIndex and the correct columns
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


def test_agent_load_from_yfinance(monkeypatch):
    # Mock yfinance.download to return a simple DataFrame
    import pandas as pd

    def mock_download(ticker, start, end, progress, multi_level_index):
        dates = pd.date_range(start, end, freq="D")
        data = {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0, 1, 2],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [100, 200, 300],
        }
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr("yfinance.download", mock_download)
    agent = DataAgent()
    df = agent.load("AAPL", "2023-01-01", "2023-01-03")
    # Check columns are normalised
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
    assert df.index.is_monotonic_increasing
    assert df.shape == (3, 5)

# ---------- Additional error‑path tests ----------

def test_missing_csv_raises(tmp_path):
    """When the CSV source does not exist a FileNotFoundError should be raised."""
    missing_path = tmp_path / "nonexistent.csv"
    agent = DataAgent(source=str(missing_path))
    with pytest.raises(FileNotFoundError):
        agent.load("AAPL", "2023-01-01", "2023-01-05")


def test_yfinance_returns_empty(monkeypatch):
    """If yfinance returns an empty DataFrame the loader should raise ValueError."""
    def mock_download(ticker, start, end, progress, multi_level_index):
        return pd.DataFrame()
    monkeypatch.setattr("yfinance.download", mock_download)
    agent = DataAgent()
    with pytest.raises(ValueError) as exc:
        agent.load("AAPL", "2023-01-01", "2023-01-03")
    assert "No data returned" in str(exc.value)


def test_validate_dataframe_multiple_missing():
    """validate_dataframe should report all missing columns, not just the first one."""
    df = make_df().drop(columns=["open", "volume"])
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df)
    msg = str(exc.value)
    assert "open" in msg and "volume" in msg


def test_csv_without_index_column(tmp_path):
    """CSV where the date is a regular column (not the index) should still load correctly."""
    df = make_df().reset_index().rename(columns={"index": "date"})
    csv_path = tmp_path / "data_no_index.csv"
    df.to_csv(csv_path, index=False)
    agent = DataAgent(source=str(csv_path))
    loaded = agent.load("AAPL", "2023-01-01", "2023-01-05")
    # The loaded DataFrame should match the original (ignoring the index name)
    pd.testing.assert_frame_equal(make_df(), loaded, check_freq=False)

