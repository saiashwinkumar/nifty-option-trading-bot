from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.rename(f"rsi_{window}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = (fast_ema - slow_ema).rename("macd")
    signal_line = ema(macd_line, signal).rename("macd_signal")
    hist = (macd_line - signal_line).rename("macd_hist")
    return pd.concat([macd_line, signal_line, hist], axis=1)


def stochastics(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> pd.DataFrame:
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    pct_k = 100 * (close - ll) / (hh - ll)
    pct_d = pct_k.rolling(d).mean()
    return pd.concat([pct_k.rename(f"stoch_k_{k}"), pct_d.rename(f"stoch_d_{d}")], axis=1)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    out = tr.ewm(alpha=1 / window, adjust=False).mean()
    return out.rename(f"atr_{window}")


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_w = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * (
        pd.Series(plus_dm, index=high.index)
        .ewm(alpha=1 / window, adjust=False)
        .mean()
        / atr_w
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=high.index)
        .ewm(alpha=1 / window, adjust=False)
        .mean()
        / atr_w
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / window, adjust=False).mean().rename(f"adx_{window}")

    return pd.concat(
        [
            adx_line,
            plus_di.rename(f"plus_di_{window}"),
            minus_di.rename(f"minus_di_{window}"),
        ],
        axis=1,
    )


def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid
    return pd.concat(
        [
            mid.rename(f"bb_mid_{window}"),
            upper.rename(f"bb_upper_{window}"),
            lower.rename(f"bb_lower_{window}"),
            width.rename(f"bb_width_{window}"),
        ],
        axis=1,
    )


def vwap_like(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    # daily data: treat close as typical price proxy
    pv = close * volume
    out = pv.rolling(window).sum() / volume.rolling(window).sum()
    return out.rename(f"vwma_{window}")


def ichimoku_base(high: pd.Series, low: pd.Series, window: int = 26) -> pd.Series:
    base = (high.rolling(window).max() + low.rolling(window).min()) / 2.0
    return base.rename(f"ichimoku_base_{window}")


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    # Classic floor trader pivots using prior day
    h1 = high.shift(1)
    l1 = low.shift(1)
    c1 = close.shift(1)
    pp = (h1 + l1 + c1) / 3.0

    r1 = 2 * pp - l1
    s1 = 2 * pp - h1
    r2 = pp + (h1 - l1)
    s2 = pp - (h1 - l1)

    return pd.concat(
        [
            pp.rename("pp"),
            r1.rename("r1"),
            s1.rename("s1"),
            r2.rename("r2"),
            s2.rename("s2"),
        ],
        axis=1,
    )


def realized_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    # annualization factor for daily data ~ sqrt(252)
    out = log_returns.rolling(window).std() * np.sqrt(252)
    return out.rename(f"realized_vol_{window}")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    logret = np.log(close).diff().rename("logret")

    feats = [
        logret,
        rsi(close, 14),
        ema(close, 10).rename("ema_10"),
        ema(close, 20).rename("ema_20"),
        sma(close, 20).rename("sma_20"),
        sma(close, 50).rename("sma_50"),
        atr(high, low, close, 14),
        realized_vol(logret, 20),
        vwap_like(close, vol, 20),
        ichimoku_base(high, low, 26),
    ]

    feats.append(macd(close))
    feats.append(stochastics(high, low, close))
    feats.append(adx(high, low, close, 14))
    feats.append(bollinger_bands(close, 20, 2.0))
    feats.append(pivot_points(high, low, close))

    x = pd.concat(feats, axis=1)

    # basic price-relative features
    x["close_over_sma20"] = close / x["sma_20"]
    x["close_over_ema20"] = close / x["ema_20"]
    x["atr_pct"] = x["atr_14"] / close

    return x
