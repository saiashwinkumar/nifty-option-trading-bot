from datetime import date

import pandas as pd

from nifty_option_trading_bot.calendar import (
    ExpiryCalendarConfig,
    is_expiry_day,
    weekly_expiry_for_week,
)


def test_weekly_expiry_before_switch_is_thursday():
    cfg = ExpiryCalendarConfig(switch_date=date(2025, 9, 1))
    # Week of 2024-01-01 => Thu 2024-01-04
    assert weekly_expiry_for_week(date(2024, 1, 2), cfg) == date(2024, 1, 4)
    assert is_expiry_day(pd.Timestamp("2024-01-04"), cfg)


def test_weekly_expiry_after_switch_is_tuesday():
    cfg = ExpiryCalendarConfig(switch_date=date(2025, 9, 1))
    # Week of 2025-09-01 => Tue 2025-09-02
    assert weekly_expiry_for_week(date(2025, 9, 1), cfg) == date(2025, 9, 2)
    assert is_expiry_day(pd.Timestamp("2025-09-02"), cfg)
