from __future__ import annotations

import pandas as pd

from .data import compute_target_next_day_avg_log_return
from .indicators import build_feature_frame
from .sentiment import SentimentProvider, YahooProxySentiment


def make_dataset(
    df_ohlcv: pd.DataFrame,
    sentiment: SentimentProvider | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned (X, y) with NaNs removed.

    X uses information available at time t.
    y is target for t: log(A(t+1)) - log(C(t)).

    Any rows with missing features or missing target are dropped.
    """
    x = build_feature_frame(df_ohlcv)
    sentiment = sentiment or YahooProxySentiment()
    try:
        s_frame = getattr(sentiment, "get_frame", None)
        if callable(s_frame):
            sent_df = s_frame(x.index)
        else:
            sent_df = sentiment.get_series(x.index).to_frame()
        if sent_df is not None and not sent_df.empty:
            x = x.join(sent_df, how="left")
    except Exception:
        # sentiment is optional; if it fails we proceed with technical-only features
        pass
    y = compute_target_next_day_avg_log_return(df_ohlcv)

    data = x.join(y, how="inner")

    # Sentiment columns may be missing (proxy not available on Yahoo) — don't
    # let them wipe the whole dataset.
    sent_cols = [c for c in data.columns if "sentiment" in c or "VIX" in c]
    if sent_cols:
        data[sent_cols] = data[sent_cols].ffill()

    # Drop rows missing any technicals/target; keep rows even if sentiment is NaN.
    non_sent_cols = [c for c in data.columns if c not in sent_cols]
    data = data.dropna(subset=non_sent_cols)

    y_out = data.pop("target")
    x_out = data

    return x_out, y_out
