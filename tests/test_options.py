import math

from nifty_option_trading_bot.options import BlackScholesInputs, bs_call_price, bs_put_price


def test_put_call_parity_approx():
    # C - P = S - K e^{-rT}
    inp = BlackScholesInputs(spot=20000.0, strike=20000.0, t_years=7 / 365.0, r=0.06, sigma=0.2)
    c = bs_call_price(inp)
    p = bs_put_price(inp)
    rhs = inp.spot - inp.strike * math.exp(-inp.r * inp.t_years)
    assert abs((c - p) - rhs) < 1e-6
