from __future__ import annotations

import argparse
import json

from .backtest import BacktestConfig, run_backtest, summarize
from .data import YahooFetchConfig, fetch_daily_ohlcv
from .evaluation import RollingMapeConfig, rolling_train_test_mape
from .features import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NIFTY expiry-day options strategy backtest (research)"
    )
    ap.add_argument("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD)")
    ap.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    ap.add_argument("--show-tail", type=int, default=5, help="Print tail rows")
    ap.add_argument(
        "--show-mape",
        action="store_true",
        help="Also compute rolling train/test MAPE for predicting next-day average price",
    )
    args = ap.parse_args()

    cfg = BacktestConfig(start=_parse_date(args.start), initial_capital=float(args.capital))
    bt = run_backtest(cfg)
    summ = summarize(bt)

    if args.show_mape:
        ohlc = fetch_daily_ohlcv(YahooFetchConfig(ticker="^NSEI", start=_parse_date(args.start)))
        X, y = make_dataset(ohlc)
        # Align everything for evaluation
        ohlc2 = ohlc.loc[X.index]
        res = rolling_train_test_mape(
            ohlc2,
            X,
            y,
            RollingMapeConfig(
                train_window_days=cfg.qgbr.train_window_days,
                refit_every=cfg.qgbr.refit_every,
            ),
            cfg.qgbr,
        )
        summ["train_mape"] = res["train_mape"]
        summ["test_mape"] = res["test_mape"]

    print(json.dumps(summ, indent=2, sort_keys=True))
    if args.show_tail:
        print(bt.tail(args.show_tail)[["equity", "pnl", "trade_kind", "regime"]])


def _parse_date(s: str):
    y, m, d = s.split("-")
    from datetime import date

    return date(int(y), int(m), int(d))


if __name__ == "__main__":
    main()
