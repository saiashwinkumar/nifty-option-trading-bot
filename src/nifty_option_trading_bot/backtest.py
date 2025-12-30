from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .calendar import ExpiryCalendarConfig, is_expiry_day
from .data import YahooFetchConfig, fetch_daily_ohlcv
from .features import make_dataset
from .model import ConformalConfig, QGBRConfig, conformalize_intervals, rolling_qgbr_predict
from .options import BlackScholesInputs, bs_call_price, bs_put_price
from .strategy import (
    SignalConfig,
    classify_regime,
    compute_direction_score,
    compute_z_signal,
    position_size,
    rolling_tau_threshold,
)


@dataclass(frozen=True)
class BacktestConfig:
    start: date = date(2024, 1, 1)
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.06

    # Model
    qgbr: QGBRConfig = QGBRConfig()
    conformal: ConformalConfig = ConformalConfig()
    signal: SignalConfig = SignalConfig()

    # Options / risk
    max_risk_frac_per_expiry: float = 0.01  # 1% of capital
    iv_lookback: int = 20
    min_iv: float = 0.10
    max_iv: float = 0.40

    # Strategy params
    width_sigma_mult: float = 1.0  # wing width in sigmas


def _estimate_iv(realized_vol_ann: pd.Series, cfg: BacktestConfig) -> pd.Series:
    # crude proxy: clip realized vol
    iv = realized_vol_ann.clip(lower=cfg.min_iv, upper=cfg.max_iv)
    return iv.rename("iv")


def _expiry_payoff_defined_risk_spread(
    spot_expiry: float,
    spot_entry: float,
    iv: float,
    r: float,
    t_years: float,
    direction: str,
    max_loss_budget: float,
) -> tuple[float, dict]:
    """Simple directional defined-risk debit spread.

    - Bullish: buy call at ATM, sell call OTM to cap risk.
    - Bearish: buy put at ATM, sell put OTM to cap risk.

    We choose the spread width from sigma*sqrt(T)*S and then scale contracts to fit max loss.
    """

    if t_years <= 0:
        return 0.0, {"reason": "no_time"}

    sigma_move = iv * np.sqrt(t_years) * spot_entry
    width = max(50.0, sigma_move)  # rough INR width floor

    k_atm = spot_entry

    if direction == "bull":
        k_long = k_atm
        k_short = k_atm + width
        long_p = bs_call_price(BlackScholesInputs(spot_entry, k_long, t_years, r, iv))
        short_p = bs_call_price(BlackScholesInputs(spot_entry, k_short, t_years, r, iv))

        debit = max(long_p - short_p, 0.0)
        # payoff at expiry
        payoff = max(spot_expiry - k_long, 0.0) - max(spot_expiry - k_short, 0.0)

    else:
        k_long = k_atm
        k_short = k_atm - width
        long_p = bs_put_price(BlackScholesInputs(spot_entry, k_long, t_years, r, iv))
        short_p = bs_put_price(BlackScholesInputs(spot_entry, k_short, t_years, r, iv))

        debit = max(long_p - short_p, 0.0)
        payoff = max(k_long - spot_expiry, 0.0) - max(k_short - spot_expiry, 0.0)

    # treat 1 contract = 1 multiplier (no lot sizing) for research; scale by budget
    if debit <= 0:
        return 0.0, {"reason": "zero_debit"}

    contracts = max_loss_budget / debit
    pnl = contracts * (payoff - debit)

    return float(pnl), {
        "kind": f"{direction}_debit_spread",
        "k_long": float(k_long),
        "k_short": float(k_short),
        "iv": float(iv),
        "t": float(t_years),
        "debit": float(debit),
        "contracts": float(contracts),
    }


def _expiry_payoff_iron_condor(
    spot_expiry: float,
    spot_entry: float,
    iv: float,
    r: float,
    t_years: float,
    max_loss_budget: float,
) -> tuple[float, dict]:
    """Defined-risk iron condor (credit) sized to max loss budget.

    Construction (symmetric):
    - Sell put at Kp_short < S, buy further put Kp_long
    - Sell call at Kc_short > S, buy further call Kc_long

    Width chosen from sigma move.
    """
    if t_years <= 0:
        return 0.0, {"reason": "no_time"}

    sigma_move = iv * np.sqrt(t_years) * spot_entry
    inner = max(50.0, 0.8 * sigma_move)
    wing = max(50.0, 1.2 * sigma_move)

    kp_s = spot_entry - inner
    kp_l = spot_entry - wing
    kc_s = spot_entry + inner
    kc_l = spot_entry + wing

    # prices at entry
    p_sell = bs_put_price(BlackScholesInputs(spot_entry, kp_s, t_years, r, iv))
    p_buy = bs_put_price(BlackScholesInputs(spot_entry, kp_l, t_years, r, iv))
    c_sell = bs_call_price(BlackScholesInputs(spot_entry, kc_s, t_years, r, iv))
    c_buy = bs_call_price(BlackScholesInputs(spot_entry, kc_l, t_years, r, iv))

    credit = (p_sell - p_buy) + (c_sell - c_buy)
    credit = max(credit, 0.0)

    # max loss per 1 contract is wing width - credit
    max_loss_1 = max((wing - inner) - credit, 1e-6)
    contracts = max_loss_budget / max_loss_1

    # payoff at expiry (credit - intrinsic spreads)
    put_spread_intr = max(kp_s - spot_expiry, 0.0) - max(kp_l - spot_expiry, 0.0)
    call_spread_intr = max(spot_expiry - kc_s, 0.0) - max(spot_expiry - kc_l, 0.0)
    pnl_1 = credit - (put_spread_intr + call_spread_intr)

    return float(contracts * pnl_1), {
        "kind": "iron_condor",
        "kp_s": float(kp_s),
        "kp_l": float(kp_l),
        "kc_s": float(kc_s),
        "kc_l": float(kc_l),
        "credit": float(credit),
        "max_loss_1": float(max_loss_1),
        "contracts": float(contracts),
        "iv": float(iv),
        "t": float(t_years),
    }


def run_backtest(cfg: BacktestConfig | None = None) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()

    # Fetch enough history for the rolling 720-day training window, then
    # start trading from cfg.start.
    ohlc = fetch_daily_ohlcv(YahooFetchConfig(ticker="^NSEI", start=date(2018, 1, 1)))

    X, y = make_dataset(ohlc)

    preds = rolling_qgbr_predict(X, y, cfg.qgbr)
    cal = conformalize_intervals(y, preds["pred_lo"], preds["pred_hi"], cfg.conformal)

    # signal
    z = compute_z_signal(preds["pred_mid"], cal["cal_lo"], cal["cal_hi"], cfg.signal)
    tau = rolling_tau_threshold(z, cfg.signal)
    regime = classify_regime(z, tau)
    score = compute_direction_score(preds["pred_mid"], cfg.signal)
    pos = position_size(score, cfg.signal)

    # vol proxy from features if present
    rv = X.get("realized_vol_20")
    if rv is None:
        # fallback
        rv = pd.Series(np.nan, index=X.index, name="realized_vol_20")
    iv = _estimate_iv(rv.ffill(), cfg)

    # backtest only on expiry days
    cal_cfg = ExpiryCalendarConfig()

    df_bt = pd.DataFrame(index=ohlc.index)
    df_bt["close"] = ohlc["Close"]
    df_bt["next_close"] = ohlc["Close"].shift(-1)

    df_bt = df_bt.join(preds).join(cal).join(z).join(tau).join(regime).join(pos).join(iv)

    df_bt["is_expiry"] = [is_expiry_day(ts, cal_cfg) for ts in df_bt.index]

    capital = cfg.initial_capital
    equity = []
    pnl_list = []
    kind_list = []

    for ts, row in df_bt.iterrows():
        if ts < pd.Timestamp(cfg.start):
            equity.append(capital)
            pnl_list.append(0.0)
            kind_list.append(None)
            continue

        pnl = 0.0
        kind = None

        if bool(row.get("is_expiry", False)) and pd.notna(row.get("next_close")):
            # We assume we enter at close of expiry day and settle at next day's close
            # (research simplification, since we don't have intraday option settlement).
            spot_entry = float(row["close"])
            spot_expiry = float(row["next_close"])
            t_years = 1.0 / 365.0
            max_loss = cfg.max_risk_frac_per_expiry * capital

            r = cfg.risk_free_rate
            ivv = float(row.get("iv", cfg.min_iv)) if pd.notna(row.get("iv")) else cfg.min_iv

            if row.get("regime") == "directional":
                direction = "bull" if float(row.get("position", 0.0)) > 0 else "bear"
                pnl, meta = _expiry_payoff_defined_risk_spread(
                    spot_expiry=spot_expiry,
                    spot_entry=spot_entry,
                    iv=ivv,
                    r=r,
                    t_years=t_years,
                    direction=direction,
                    max_loss_budget=max_loss,
                )
                kind = meta.get("kind")
            elif row.get("regime") == "neutral":
                pnl, meta = _expiry_payoff_iron_condor(
                    spot_expiry=spot_expiry,
                    spot_entry=spot_entry,
                    iv=ivv,
                    r=r,
                    t_years=t_years,
                    max_loss_budget=max_loss,
                )
                kind = meta.get("kind")

            # Hard stop: cap loss to max_loss (1% of capital) to enforce constraint
            pnl = max(pnl, -max_loss)

            capital += pnl

        equity.append(capital)
        pnl_list.append(pnl)
        kind_list.append(kind)

    df_bt["pnl"] = pnl_list
    df_bt["equity"] = equity
    df_bt["trade_kind"] = kind_list

    return df_bt


def summarize(df_bt: pd.DataFrame) -> dict:
    trades = df_bt[df_bt["pnl"] != 0.0].copy()
    if trades.empty:
        return {"trades": 0}

    wins = (trades["pnl"] > 0).mean()
    avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean()
    avg_loss = -trades.loc[trades["pnl"] < 0, "pnl"].mean()
    rr = float(avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) else float("nan")

    eq = df_bt["equity"].ffill()
    dd = (eq / eq.cummax()) - 1.0

    return {
        "trades": int(len(trades)),
        "win_rate": float(wins),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "reward_risk": rr,
        "total_pnl": float(trades["pnl"].sum()),
        "final_equity": float(eq.iloc[-1]),
        "max_drawdown": float(dd.min()),
        "by_kind": trades.groupby("trade_kind")["pnl"].agg(["count", "mean", "sum"]).to_dict(),
    }
