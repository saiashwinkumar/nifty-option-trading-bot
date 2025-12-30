from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import compute_avg_price
from .model import QGBRConfig, _make_quantile_model


@dataclass(frozen=True)
class RollingMapeConfig:
    train_window_days: int = 720
    refit_every: int = 5

    # we evaluate prediction of A(t+1) (next-day average price)
    eps: float = 1e-9


def mape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-9) -> float:
    mask = (~y_true.isna()) & (~y_pred.isna())
    if mask.sum() == 0:
        return float("nan")
    denom = y_true[mask].abs().clip(lower=eps)
    return float((y_true[mask] - y_pred[mask]).abs().div(denom).mean())


def rolling_train_test_mape(
    ohlc: pd.DataFrame,
    X: pd.DataFrame,
    y_logret: pd.Series,
    cfg: RollingMapeConfig | None = None,
    qgbr: QGBRConfig | None = None,
) -> dict:
    """Compute rolling train/test MAPE using the median model.

    What we measure:
    - The model predicts target log-return: log(A(t+1)) - log(C(t)).
    - Convert to predicted average price:
        Ahat(t+1) = C(t) * exp(yhat(t))
    - Compare to realized A(t+1) and compute MAPE.

    Train MAPE is computed on the training window *using in-sample predictions*
    from the refit model at each refit step.

    Test MAPE is computed on each next refit block (walk-forward), i.e. true
    out-of-sample for those days.

    Returns overall MAPE plus series for inspection.
    """

    cfg = cfg or RollingMapeConfig()
    qgbr = qgbr or QGBRConfig(train_window_days=cfg.train_window_days, refit_every=cfg.refit_every)

    if not (X.index.equals(y_logret.index) and X.index.equals(ohlc.index)):
        # Expect same index; caller can align beforehand.
        raise ValueError("ohlc, X, y must share the same index")

    n = len(X)
    a = compute_avg_price(ohlc)
    a_next = a.shift(-1).rename("a_next")
    c = ohlc["Close"].rename("close")

    train_mape_series = pd.Series(index=X.index, dtype=float, name="train_mape")
    test_mape_series = pd.Series(index=X.index, dtype=float, name="test_mape")

    # Per-day predicted A(t+1)
    pred_a_train = pd.Series(index=X.index, dtype=float, name="pred_a_next_train")
    pred_a_test = pd.Series(index=X.index, dtype=float, name="pred_a_next_test")

    last_fit_i: int | None = None
    model_mid = None

    for i in range(n):
        train_start = i - cfg.train_window_days
        train_end = i
        if train_start < 0:
            continue

        must_refit = last_fit_i is None or (i - last_fit_i) >= cfg.refit_every
        if must_refit:
            Xtr = X.iloc[train_start:train_end]
            ytr = y_logret.iloc[train_start:train_end]

            model_mid = _make_quantile_model(qgbr.q_mid, qgbr)
            model_mid.fit(Xtr, ytr)
            last_fit_i = i

            # in-sample predictions for the training window
            yhat_tr = pd.Series(model_mid.predict(Xtr), index=Xtr.index)
            a_hat_tr = c.loc[Xtr.index] * np.exp(yhat_tr)
            pred_a_train.loc[Xtr.index] = a_hat_tr
            train_mape_series.iloc[train_start:train_end] = (
                (a_next.loc[Xtr.index] - a_hat_tr).abs()
                / a_next.loc[Xtr.index].abs().clip(lower=cfg.eps)
            )

        # out-of-sample prediction for day i
        if model_mid is None:
            continue
        yhat = float(model_mid.predict(X.iloc[[i]])[0])
        pred_a_test.iloc[i] = float(c.iloc[i] * np.exp(yhat))

        # Mark the same day i as test error (true next-day average exists at i)
        if pd.notna(a_next.iloc[i]):
            test_mape_series.iloc[i] = float(
                abs(a_next.iloc[i] - pred_a_test.iloc[i])
                / max(abs(a_next.iloc[i]), cfg.eps)
            )

    out = {
        "train_mape": float(train_mape_series.mean(skipna=True)),
        "test_mape": float(test_mape_series.mean(skipna=True)),
        "train_mape_series": train_mape_series,
        "test_mape_series": test_mape_series,
        "pred_a_next_train": pred_a_train,
        "pred_a_next_test": pred_a_test,
        "a_next": a_next,
    }
    return out


def in_sample_view(res: dict) -> tuple[pd.Series, pd.Series]:
    """Get (actual, predicted) restricted to in-sample points.

    In-sample points are where we have an in-sample prediction for the training
    window (i.e., `pred_a_next_train` is non-NaN).
    """

    actual = res["a_next"].copy()
    pred = res["pred_a_next_train"].copy()
    mask = pred.notna()
    return actual[mask], pred[mask]


def out_of_sample_view(res: dict) -> tuple[pd.Series, pd.Series]:
    """Get (actual, predicted) restricted to out-of-sample points.

    Out-of-sample points are where we have a 1-step-ahead prediction produced
    by the most recent refit model (i.e., `pred_a_next_test` is non-NaN).
    """

    actual = res["a_next"].copy()
    pred = res["pred_a_next_test"].copy()
    mask = pred.notna()
    return actual[mask], pred[mask]
