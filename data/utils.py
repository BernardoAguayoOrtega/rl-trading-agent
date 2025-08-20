"""Utility helpers for the data package.

These helpers are intentionally lightweight so they can be reused by the
``data.loader`` module and by any downstream code that needs to validate
or resample OHLCV data.

Functions
---------
* :func:`validate_dataframe` – checks that a DataFrame contains the
  required columns ``open``, ``high``, ``low``, ``close`` and ``volume``.
* :func:`resample_to_daily` – resamples a DataFrame to a given frequency
  (default ``1D``) using the last available bar for each period.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["validate_dataframe", "resample_to_daily"]

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that *df* contains the required OHLCV columns.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {', '.join(sorted(missing))}"
        )


def resample_to_daily(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Resample *df* to the given frequency.

    The function assumes that *df* has a DatetimeIndex.  It resamples
    using the last available bar for each period and forwards the
    ``volume`` column using a sum aggregation.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex.
    freq:
        Pandas offset alias (e.g. ``1D`` for daily, ``1H`` for hourly).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex for resampling")

    # Use last for price columns, sum for volume
    agg = {
        "open": "last",
        "high": "last",
        "low": "last",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(freq).agg(agg)
    # Drop periods that have all NaNs (e.g., weekends for daily resample)
    resampled = resampled.dropna(how="all")
    return resampled
