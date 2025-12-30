from __future__ import annotations

import argparse
import json

from .backtest import BacktestConfig, run_backtest, summarize


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NIFTY expiry-day options strategy backtest (research)"
    )
    ap.add_argument("--start", default="2024-01-01", help="Backtest start date (YYYY-MM-DD)")
    ap.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    ap.add_argument("--show-tail", type=int, default=5, help="Print tail rows")
    args = ap.parse_args()

    cfg = BacktestConfig(start=_parse_date(args.start), initial_capital=float(args.capital))
    bt = run_backtest(cfg)
    summ = summarize(bt)

    print(json.dumps(summ, indent=2, sort_keys=True))
    if args.show_tail:
        print(bt.tail(args.show_tail)[["equity", "pnl", "trade_kind", "regime"]])


def _parse_date(s: str):
    y, m, d = s.split("-")
    from datetime import date

    return date(int(y), int(m), int(d))


if __name__ == "__main__":
    main()
