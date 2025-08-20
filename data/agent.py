"""Data agent that wraps :func:`load_market_data`.

The framework expects a *Data Agent* that can be imported from
``data``.  The simplest implementation is a thin wrapper around the
``load_market_data`` function defined in :mod:`data.loader`.

Example usage
-------------
>>> from data.agent import DataAgent
>>> agent = DataAgent()
>>> df = agent.load("AAPL", "2023-01-01", "2023-02-01")
>>> df.head()
            open   high    low  close  volume
2023-01-02  150.0  155.0  149.0  154.0  1000000
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .loader import load_market_data

__all__ = ["DataAgent"]


class DataAgent:
    """Simple data agent exposing a :meth:`load` method.

    Parameters
    ----------
    source:
        Optional path to a CSV file.  If provided, the agent will read the
        CSV instead of downloading from Yahoo Finance.
    """

    def __init__(self, source: Optional[str] = None) -> None:
        self.source = source

    def load(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Load market data for *ticker* between *start* and *end*.

        The method simply forwards to :func:`data.loader.load_market_data`.
        """
        return load_market_data(ticker, start, end, source=self.source)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DataAgent source={self.source!r}>"
