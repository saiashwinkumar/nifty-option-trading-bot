from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class QGBRConfig:
    train_window_days: int = 720
    refit_every: int = 5

    # Quantiles
    q_lo: float = 0.025
    q_mid: float = 0.5
    q_hi: float = 0.975

    # Model hyperparams (sane defaults; tune later)
    max_depth: int | None = 3
    max_iter: int = 300
    learning_rate: float = 0.05
    min_samples_leaf: int = 30
    l2_regularization: float = 0.0
    random_state: int = 42


def _make_quantile_model(q: float, cfg: QGBRConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_depth=cfg.max_depth,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state,
    )


@dataclass
class QGBRState:
    model_lo: HistGradientBoostingRegressor
    model_mid: HistGradientBoostingRegressor
    model_hi: HistGradientBoostingRegressor
    last_fit_end: int


def rolling_qgbr_predict(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: QGBRConfig | None = None,
) -> pd.DataFrame:
    """Walk-forward rolling quantile predictions.

    - Uses past `train_window_days` rows to train.
    - Re-fits every `refit_every` rows.
    - Predicts for each point t using information available at t.

    Returns dataframe aligned to X.index with columns: pred_lo, pred_mid, pred_hi.
    """

    cfg = cfg or QGBRConfig()

    if not X.index.equals(y.index):
        raise ValueError("X and y must be aligned on the same index")

    n = len(X)
    preds = pd.DataFrame(index=X.index, columns=["pred_lo", "pred_mid", "pred_hi"], dtype=float)

    state: QGBRState | None = None

    for i in range(n):
        train_start = i - cfg.train_window_days
        train_end = i  # exclusive

        if train_start < 0:
            continue

        must_refit = state is None or (i - state.last_fit_end) >= cfg.refit_every
        if must_refit:
            Xtr = X.iloc[train_start:train_end]
            ytr = y.iloc[train_start:train_end]

            m_lo = _make_quantile_model(cfg.q_lo, cfg)
            m_mid = _make_quantile_model(cfg.q_mid, cfg)
            m_hi = _make_quantile_model(cfg.q_hi, cfg)

            m_lo.fit(Xtr, ytr)
            m_mid.fit(Xtr, ytr)
            m_hi.fit(Xtr, ytr)

            state = QGBRState(m_lo, m_mid, m_hi, last_fit_end=i)

        xt = X.iloc[[i]]
        preds.iloc[i, 0] = float(state.model_lo.predict(xt)[0])
        preds.iloc[i, 1] = float(state.model_mid.predict(xt)[0])
        preds.iloc[i, 2] = float(state.model_hi.predict(xt)[0])

    return preds


def interval_coverage(y_true: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    mask = (~y_true.isna()) & (~lo.isna()) & (~hi.isna())
    if mask.sum() == 0:
        return float("nan")
    inside = (y_true[mask] >= lo[mask]) & (y_true[mask] <= hi[mask])
    return float(inside.mean())


@dataclass(frozen=True)
class ConformalConfig:
    """Rolling conformal calibration config.

    We calibrate prediction intervals using the last `cal_window` residual scores.
    For two-sided intervals, we use score = max(lo - y, y - hi).

    On each day t, we compute qhat = quantile(score_{t-cal_window:t-1}, level)
    then output calibrated interval:
      [lo - qhat, hi + qhat]
    """

    cal_window: int = 252
    alpha: float = 0.05


def conformalize_intervals(
    y_true: pd.Series,
    pred_lo: pd.Series,
    pred_hi: pd.Series,
    cfg: ConformalConfig | None = None,
) -> pd.DataFrame:
    """Return calibrated intervals aligned to index.

    Notes:
    - The calibration at time t uses only information strictly before t.
    - If insufficient history, outputs NaNs.
    """

    cfg = cfg or ConformalConfig()
    if not (y_true.index.equals(pred_lo.index) and y_true.index.equals(pred_hi.index)):
        raise ValueError("y_true, pred_lo, pred_hi must share the same index")

    idx = y_true.index
    out = pd.DataFrame(index=idx, columns=["cal_lo", "cal_hi", "qhat"], dtype=float)

    # score_t defined when y and preds exist
    score = pd.Series(index=idx, dtype=float)
    mask = (~y_true.isna()) & (~pred_lo.isna()) & (~pred_hi.isna())
    score.loc[mask] = np.maximum(pred_lo[mask] - y_true[mask], y_true[mask] - pred_hi[mask])
    score = score.clip(lower=0)

    level = 1.0 - cfg.alpha

    for i in range(len(idx)):
        start = i - cfg.cal_window
        end = i  # exclusive
        if start < 0:
            continue
        hist = score.iloc[start:end].dropna()
        if hist.empty:
            continue

        # Conformal quantile (finite sample): use the (ceil((n+1)*level)/n) empirical quantile.
        n = len(hist)
        k = int(np.ceil((n + 1) * level))
        k = min(max(k, 1), n)
        qhat = float(np.sort(hist.to_numpy())[k - 1])

        if pd.isna(pred_lo.iloc[i]) or pd.isna(pred_hi.iloc[i]):
            continue

        out.iloc[i, out.columns.get_loc("qhat")] = qhat
        out.iloc[i, out.columns.get_loc("cal_lo")] = float(pred_lo.iloc[i] - qhat)
        out.iloc[i, out.columns.get_loc("cal_hi")] = float(pred_hi.iloc[i] + qhat)

    return out


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball / quantile loss."""
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))
