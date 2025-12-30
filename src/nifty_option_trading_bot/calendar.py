from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd


@dataclass(frozen=True)
class ExpiryCalendarConfig:
    switch_date: date = date(2025, 9, 1)  # after this: Tuesday expiry


def _to_date(ts: pd.Timestamp) -> date:
    return ts.date()


def weekly_expiry_for_week(d: date, cfg: ExpiryCalendarConfig | None = None) -> date:
    """Return weekly expiry date for the week containing date d.

    Assumptions:
    - Before cfg.switch_date: Thursday expiry.
    - On/after cfg.switch_date: Tuesday expiry.
    - We ignore NSE holiday adjustments (can be added later with a holiday calendar).
    """
    cfg = cfg or ExpiryCalendarConfig()

    # choose weekday target
    target_weekday = 3 if d < cfg.switch_date else 1  # Mon=0 ... Thu=3, Tue=1

    # get the target weekday within the same ISO week (Mon..Sun)
    # compute Monday of week
    monday = d - timedelta(days=d.weekday())
    expiry = monday + timedelta(days=target_weekday)
    return expiry


def is_expiry_day(ts: pd.Timestamp, cfg: ExpiryCalendarConfig | None = None) -> bool:
    d = _to_date(ts)
    return d == weekly_expiry_for_week(d, cfg)


def list_expiry_days(
    index: pd.DatetimeIndex, cfg: ExpiryCalendarConfig | None = None
) -> pd.DatetimeIndex:
    cfg = cfg or ExpiryCalendarConfig()
    dates = []
    for ts in index:
        if is_expiry_day(pd.Timestamp(ts), cfg):
            dates.append(pd.Timestamp(ts))
    return pd.DatetimeIndex(dates)


def time_to_expiry_years(entry_ts: pd.Timestamp, expiry_ts: pd.Timestamp) -> float:
    # use ACT/365
    delta_days = (expiry_ts.date() - entry_ts.date()).days
    return max(delta_days, 0) / 365.0
