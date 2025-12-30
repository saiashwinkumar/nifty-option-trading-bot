from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .data import YahooFetchConfig, fetch_daily_ohlcv


class SentimentProvider:
    """Provides a daily sentiment series aligned to the OHLCV index.

    Contract:
    - `get_series(index)` returns a pd.Series with same DatetimeIndex.
    - Values should be higher when risk-on / greed, lower when fear.
    """

    def get_series(self, index: pd.DatetimeIndex) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class YahooProxySentimentConfig:
    """Fallback sentiment using Yahoo symbols.

    We can’t reliably scrape proprietary indices without an API key, so we offer:
    - India VIX proxy (if available on Yahoo) ^INDIAVIX
    - US VIX ^VIX as a last resort (not ideal for NIFTY)

    We expose both as features, and a combined z-scored proxy.
    """

    start: date | None = None
    end: date | None = None
    india_vix_ticker: str = "^INDIAVIX"
    fallback_vix_ticker: str = "^VIX"


class YahooProxySentiment(SentimentProvider):
    def __init__(self, cfg: YahooProxySentimentConfig | None = None):
        self.cfg = cfg or YahooProxySentimentConfig()

    def _fetch_close(self, ticker: str) -> pd.Series | None:
        try:
            df = fetch_daily_ohlcv(
                YahooFetchConfig(ticker=ticker, start=self.cfg.start, end=self.cfg.end)
            )
        except Exception:
            return None
        return df["Close"].rename(ticker)

    def get_frame(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        vix_in = self._fetch_close(self.cfg.india_vix_ticker)
        vix_us = self._fetch_close(self.cfg.fallback_vix_ticker)

        parts: list[pd.Series] = []
        if vix_in is not None:
            parts.append(vix_in)
        if vix_us is not None:
            parts.append(vix_us)

        if not parts:
            # all missing: return NaNs, caller can drop or impute
            return pd.DataFrame(index=index)

        df = pd.concat(parts, axis=1).reindex(index)

        # Transform: higher sentiment = lower VIX
        for c in df.columns:
            df[c + "_inv"] = -df[c]

        # Combined proxy: mean of available inverted z-scores
        z_cols = []
        for c in df.columns:
            if c.endswith("_inv"):
                z = (df[c] - df[c].rolling(252).mean()) / df[c].rolling(252).std()
                z_cols.append(z.rename(c + "_z"))

        if z_cols:
            zdf = pd.concat(z_cols, axis=1)
            df["sentiment_proxy"] = zdf.mean(axis=1)

        return df

    def get_series(self, index: pd.DatetimeIndex) -> pd.Series:
        df = self.get_frame(index)
        if "sentiment_proxy" not in df.columns:
            return pd.Series(index=index, dtype=float, name="sentiment_proxy")
        return df["sentiment_proxy"].rename("sentiment_proxy")


@dataclass(frozen=True)
class CsvSentimentConfig:
    path: str
    date_col: str = "date"
    value_col: str = "value"
    name: str = "sentiment"


class CsvSentiment(SentimentProvider):
    def __init__(self, cfg: CsvSentimentConfig):
        self.cfg = cfg

    def get_series(self, index: pd.DatetimeIndex) -> pd.Series:
        df = pd.read_csv(self.cfg.path)
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
        s = df.set_index(self.cfg.date_col)[self.cfg.value_col].sort_index().rename(self.cfg.name)
        return s.reindex(index)
