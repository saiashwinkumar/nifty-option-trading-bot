import datetime as dt

from nifty_option_trading_bot.data import YahooFetchConfig, fetch_daily_ohlcv
from nifty_option_trading_bot.evaluation import (
    in_sample_view,
    out_of_sample_view,
    rolling_train_test_mape,
)
from nifty_option_trading_bot.features import make_dataset


def test_in_sample_and_oos_views_are_different() -> None:
    # Small sanity test: in-sample and OOS should not be identical series.
    ohlc = fetch_daily_ohlcv(YahooFetchConfig(ticker="^NSEI", start=dt.date(2018, 1, 1)))
    X, y = make_dataset(ohlc)
    ohlc2 = ohlc.loc[X.index]

    res = rolling_train_test_mape(ohlc2, X, y)

    a_tr, p_tr = in_sample_view(res)
    a_te, p_te = out_of_sample_view(res)

    assert len(a_tr) > 0
    assert len(a_te) > 0

    # If both views were the same, these would likely match exactly.
    assert not p_tr.tail(200).equals(p_te.tail(200))
