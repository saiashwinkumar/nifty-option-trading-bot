from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def norm_cdf(x: float) -> float:
    # fast enough: use erf
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


@dataclass(frozen=True)
class BlackScholesInputs:
    spot: float
    strike: float
    t_years: float
    r: float
    sigma: float


def bs_d1_d2(inp: BlackScholesInputs) -> tuple[float, float]:
    if inp.t_years <= 0 or inp.sigma <= 0 or inp.spot <= 0 or inp.strike <= 0:
        return float("nan"), float("nan")
    d1 = (
        np.log(inp.spot / inp.strike) + (inp.r + 0.5 * inp.sigma**2) * inp.t_years
    ) / (inp.sigma * np.sqrt(inp.t_years))
    d2 = d1 - inp.sigma * np.sqrt(inp.t_years)
    return float(d1), float(d2)


def bs_call_price(inp: BlackScholesInputs) -> float:
    d1, d2 = bs_d1_d2(inp)
    if np.isnan(d1) or np.isnan(d2):
        return max(inp.spot - inp.strike, 0.0)
    return float(inp.spot * norm_cdf(d1) - inp.strike * np.exp(-inp.r * inp.t_years) * norm_cdf(d2))


def bs_put_price(inp: BlackScholesInputs) -> float:
    d1, d2 = bs_d1_d2(inp)
    if np.isnan(d1) or np.isnan(d2):
        return max(inp.strike - inp.spot, 0.0)
    return float(
        inp.strike * np.exp(-inp.r * inp.t_years) * norm_cdf(-d2)
        - inp.spot * norm_cdf(-d1)
    )


def strike_from_delta_call(
    spot: float, t_years: float, r: float, sigma: float, delta: float
) -> float:
    """Approx strike for a target call delta (0..1) under BS.

    Uses inverse CDF approximation via scipy normally, but to keep deps minimal,
    we do a simple binary search on strike.
    """

    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    # bounds
    lo = spot * 0.2
    hi = spot * 2.0

    def call_delta(k: float) -> float:
        inp = BlackScholesInputs(spot=spot, strike=k, t_years=t_years, r=r, sigma=sigma)
        d1, _ = bs_d1_d2(inp)
        if np.isnan(d1):
            return 1.0 if spot > k else 0.0
        return norm_cdf(d1)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        d = call_delta(mid)
        if d > delta:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))
