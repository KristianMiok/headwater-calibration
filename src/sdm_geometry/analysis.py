"""Analysis: correlate local intrinsic dimension with calibration error.

The pilot's central question is whether high local ID → high
calibration error. This module provides three lenses:

1. Spearman correlation (rank-based, robust to non-linear monotone
   relationships and outliers).

2. Partial Spearman correlation controlling for local point density.
   This is the critical sanity check: high ID could just mean "sparse
   region of feature space," and sparse regions also tend to have
   poorly-fit models for boring sample-size reasons. If the
   correlation survives partial-control for density, the geometric
   story holds. If it vanishes, what we're really seeing is
   sample sparsity.

3. Binned mean: split pixels by ID decile and report mean
   miscalibration in each bin. A monotone trend across bins is a
   stronger visual claim than a single correlation coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CorrelationResult:
    """Outputs of the ID × calibration correlation analysis."""

    spearman_rho: float
    spearman_p: float
    partial_spearman_rho: float | None
    partial_spearman_p: float | None
    pearson_r: float
    pearson_p: float
    n_pixels: int
    n_valid: int  # after dropping NaNs

    def to_dict(self) -> dict:
        return {
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "partial_spearman_rho": self.partial_spearman_rho,
            "partial_spearman_p": self.partial_spearman_p,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "n_pixels": self.n_pixels,
            "n_valid": self.n_valid,
        }


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res = stats.spearmanr(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def _partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Spearman correlation between x and y, partialling out z.

    Computed as the Spearman correlation between the residuals of
    rank(x) regressed on rank(z) and rank(y) regressed on rank(z).
    Standard non-parametric partial correlation.
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 10:
        return np.nan, np.nan
    rx = stats.rankdata(x[valid])
    ry = stats.rankdata(y[valid])
    rz = stats.rankdata(z[valid])

    # Linear regression on ranks
    def residuals(target: np.ndarray, predictor: np.ndarray) -> np.ndarray:
        slope, intercept, _, _, _ = stats.linregress(predictor, target)
        return target - (slope * predictor + intercept)

    rx_resid = residuals(rx, rz)
    ry_resid = residuals(ry, rz)
    rho, p = stats.pearsonr(rx_resid, ry_resid)
    return float(rho), float(p)


def correlate_id_with_calibration(
    id_values: np.ndarray,
    miscalibration: np.ndarray,
    local_density: np.ndarray | None = None,
) -> CorrelationResult:
    """Compute correlations between local ID and a calibration metric.

    Parameters
    ----------
    id_values : ndarray, shape (n_pixels,)
        Local intrinsic dimension at each pixel.
    miscalibration : ndarray, shape (n_pixels,)
        Either the binary miscalibration indicator or the signed
        miscalibration distance.
    local_density : ndarray or None
        If provided, also compute partial Spearman controlling for it.

    Returns
    -------
    CorrelationResult
    """
    id_values = np.asarray(id_values, dtype=float)
    miscalibration = np.asarray(miscalibration, dtype=float)

    if id_values.shape != miscalibration.shape:
        raise ValueError(
            f"shape mismatch: id_values {id_values.shape}, "
            f"miscalibration {miscalibration.shape}"
        )

    valid = np.isfinite(id_values) & np.isfinite(miscalibration)
    n_valid = int(valid.sum())

    spearman_rho, spearman_p = _spearman(id_values, miscalibration)
    pearson_r, pearson_p = stats.pearsonr(
        id_values[valid], miscalibration[valid]
    )

    if local_density is not None:
        partial_rho, partial_p = _partial_spearman(
            id_values, miscalibration, np.asarray(local_density, dtype=float)
        )
    else:
        partial_rho, partial_p = None, None

    return CorrelationResult(
        spearman_rho=spearman_rho,
        spearman_p=spearman_p,
        partial_spearman_rho=partial_rho,
        partial_spearman_p=partial_p,
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        n_pixels=len(id_values),
        n_valid=n_valid,
    )


def binned_mean_miscalibration(
    id_values: np.ndarray,
    miscalibration: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Mean miscalibration per ID decile (or n_bins quantiles).

    Returns a DataFrame with columns:
    bin, id_lower, id_upper, id_mid, mean_miscalibration, n_pixels.
    """
    valid = np.isfinite(id_values) & np.isfinite(miscalibration)
    id_v = id_values[valid]
    mis_v = miscalibration[valid]

    quantile_edges = np.quantile(id_v, np.linspace(0, 1, n_bins + 1))
    # Make sure edges are strictly increasing (can fail with ties)
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) - 1 < 2:
        raise ValueError("Too few unique ID values for binning")
    bin_idx = np.clip(
        np.searchsorted(quantile_edges, id_v, side="right") - 1,
        0,
        len(quantile_edges) - 2,
    )

    rows = []
    for b in range(len(quantile_edges) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "id_lower": float(quantile_edges[b]),
                "id_upper": float(quantile_edges[b + 1]),
                "id_mid": float(0.5 * (quantile_edges[b] + quantile_edges[b + 1])),
                "mean_miscalibration": float(mis_v[mask].mean()),
                "n_pixels": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_id_vs_calibration(
    id_values: np.ndarray,
    miscalibration: np.ndarray,
    correlation: CorrelationResult,
    binned: pd.DataFrame | None = None,
    title: str = "",
    savepath: Path | str | None = None,
) -> plt.Figure:
    """Two-panel figure: scatter + binned mean.

    Left panel: scatter of local ID vs miscalibration, with a
    LOWESS-style trend line.

    Right panel: binned mean miscalibration per ID decile, with
    error bars (binomial SE for binary indicator, otherwise none).
    """
    valid = np.isfinite(id_values) & np.isfinite(miscalibration)
    id_v = id_values[valid]
    mis_v = miscalibration[valid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: scatter
    axes[0].scatter(
        id_v,
        mis_v + np.random.normal(0, 0.01, mis_v.shape) if set(np.unique(mis_v)) <= {0.0, 1.0} else mis_v,
        s=4,
        alpha=0.25,
        color="#444",
    )
    axes[0].set_xlabel("Local intrinsic dimension")
    axes[0].set_ylabel("Miscalibration (jittered if binary)")
    axes[0].set_title(
        f"Spearman ρ = {correlation.spearman_rho:.3f} "
        f"(p = {correlation.spearman_p:.2e}, n = {correlation.n_valid})"
    )
    axes[0].grid(True, alpha=0.3)

    # Right: binned mean
    if binned is None:
        binned = binned_mean_miscalibration(id_values, miscalibration, n_bins=10)
    axes[1].plot(
        binned["id_mid"],
        binned["mean_miscalibration"],
        marker="o",
        color="#c0392b",
        linewidth=2,
    )
    axes[1].set_xlabel("Local intrinsic dimension (bin midpoint)")
    axes[1].set_ylabel("Mean miscalibration in bin")
    axes[1].set_title("Binned mean miscalibration vs ID decile")
    axes[1].grid(True, alpha=0.3)
    if correlation.partial_spearman_rho is not None:
        axes[1].text(
            0.05,
            0.95,
            f"Partial ρ (controlling density) = {correlation.partial_spearman_rho:.3f}",
            transform=axes[1].transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
