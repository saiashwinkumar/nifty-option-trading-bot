from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class YahooFetchConfig:
    ticker: str = "^NSEI"  # NIFTY 50 index
    start: date | None = None
    end: date | None = None


def fetch_daily_ohlcv(cfg: YahooFetchConfig) -> pd.DataFrame:
    """Fetch daily OHLCV.

    Returns a DataFrame indexed by timestamp (UTC naive) with columns:
    Open, High, Low, Close, Volume.
    """
    df = yf.download(
        cfg.ticker,
        start=cfg.start.isoformat() if cfg.start else None,
        end=cfg.end.isoformat() if cfg.end else None,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned from Yahoo for {cfg.ticker}")

    # yfinance may return multiindex columns; normalize
    if isinstance(df.columns, pd.MultiIndex):
        # keep first level column names
        df.columns = df.columns.get_level_values(0)

    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing columns from Yahoo response: {missing}. Got: {list(df.columns)}"
        )

    df = df[expected].copy()
    df.index = pd.to_datetime(df.index)

    # drop rows with missing close
    df = df.dropna(subset=["Close"]).sort_index()
    return df


def compute_avg_price(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0


def compute_target_next_day_avg_log_return(df: pd.DataFrame) -> pd.Series:
    """Target: log(A(t+1)) - log(C(t)).

    A(t+1) uses t+1 OHLC average, and C(t) is today's close.
    """
    a_next = compute_avg_price(df).shift(-1)
    c_today = df["Close"]
    y = np.log(a_next) - np.log(c_today)
    return y.rename("target")
