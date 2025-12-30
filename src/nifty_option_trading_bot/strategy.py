from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalConfig:
    # Rolling threshold definition
    tau_lookback: int = 120
    tau_percentile: float = 0.70

    # Position sizing
    k: float = 400.0  # scales log-return to tanh input
    tilt_cap: float = 1.0  # max direction exposure in [-tilt_cap, tilt_cap]

    # Risk adjustment (optional)
    use_interval_width_risk_adj: bool = True
    min_width: float = 1e-4


def compute_z_signal(
    pred_mid: pd.Series,
    cal_lo: pd.Series | None = None,
    cal_hi: pd.Series | None = None,
    cfg: SignalConfig | None = None,
) -> pd.Series:
    """Return z_t: signed expected return (optionally risk-adjusted).

    Default: z_t = pred_mid.
    If interval provided and enabled: z_t = pred_mid / max(width, min_width).
    """

    cfg = cfg or SignalConfig()
    z = pred_mid.copy().rename("z")

    if cfg.use_interval_width_risk_adj and cal_lo is not None and cal_hi is not None:
        width = (cal_hi - cal_lo).abs().rename("width")
        denom = width.clip(lower=cfg.min_width)
        z = (z / denom).rename("z")

    return z


def compute_direction_score(
    pred_mid: pd.Series,
    cfg: SignalConfig | None = None,
) -> pd.Series:
    """Score used for directional sizing.

    We size off the *raw* predicted return (pred_mid), not the risk-adjusted z,
    to avoid saturating tanh when width is small.
    """
    _ = cfg or SignalConfig()
    return pred_mid.rename("score")


def rolling_tau_threshold(z: pd.Series, cfg: SignalConfig | None = None) -> pd.Series:
    cfg = cfg or SignalConfig()
    absz = z.abs()

    def q70(a: np.ndarray) -> float:
        if len(a) == 0:
            return float("nan")
        return float(np.nanquantile(a, cfg.tau_percentile))

    # shift(1) to ensure threshold uses past only
    tau = absz.rolling(cfg.tau_lookback).apply(lambda a: q70(a), raw=True).shift(1)
    return tau.rename("tau")


def position_size(z: pd.Series, cfg: SignalConfig | None = None) -> pd.Series:
    cfg = cfg or SignalConfig()
    pos = np.tanh(cfg.k * z) * cfg.tilt_cap
    return pd.Series(pos, index=z.index, name="position")


def classify_regime(z: pd.Series, tau: pd.Series) -> pd.Series:
    """Directional when |z| > tau, else neutral."""
    reg = pd.Series(index=z.index, dtype="object")
    reg[(z.abs() > tau) & (~tau.isna())] = "directional"
    reg[(z.abs() <= tau) & (~tau.isna())] = "neutral"
    return reg.rename("regime")
