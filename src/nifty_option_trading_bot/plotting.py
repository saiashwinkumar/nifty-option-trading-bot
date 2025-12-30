from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    max_points: int = 800


def plot_actual_vs_predicted(
    actual: pd.Series,
    predicted: pd.Series,
    title: str,
    out_path: str | Path,
    cfg: PlotConfig | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> Path:
    """Plot actual vs predicted series and save to a PNG."""

    cfg = cfg or PlotConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat([actual.rename("actual"), predicted.rename("pred")], axis=1)
    if start is not None or end is not None:
        df = df.loc[start:end]
    df = df.dropna()
    if df.empty:
        raise ValueError("No overlapping non-NaN points to plot")

    if len(df) > cfg.max_points:
        df = df.iloc[-cfg.max_points :]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["actual"], label="Actual", linewidth=1.5)
    ax.plot(df.index, df["pred"], label="Predicted", linewidth=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Next-day average price A(t+1)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
