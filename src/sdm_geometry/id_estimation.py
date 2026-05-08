"""Local intrinsic dimension estimation via DADApy.

This module wraps DADApy's ID estimators in a small interface tuned to
our use case: we want a per-point local ID estimate for every row of a
predictor matrix X (n_points × n_features), so we can later correlate
local ID with per-point calibration error.

The "local" in local intrinsic dimension here means we fit the ID
estimator on a local neighbourhood of each point, rather than fitting
once globally. Two strategies are supported:

1. k-NN neighbourhoods (default): for each point, take its k nearest
   neighbours and estimate ID on that local set using TwoNN.
2. Global ID with per-point density: fit TwoNN globally and use the
   per-point local density as a proxy for "where in the manifold am I".

For the pilot we use strategy 1 with TwoNN, which is the simplest
defensible choice. Strategy 2 and the gride estimator are included as
options for follow-up analyses.

References
----------
- Facco et al. (2017) Estimating the intrinsic dimension of datasets
  by a minimal neighborhood information. Sci Rep 7:12140. (TwoNN)
- Denti et al. (2022) The generalised ratios intrinsic dimension
  estimator. Sci Rep 12:20005. (gride)
- Glielmo et al. (2022) DADApy: Distance-based analysis of
  data-manifolds in Python. Patterns 3, 100589.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class LocalIDResult:
    """Container for local ID estimates and metadata.

    Attributes
    ----------
    id_values : ndarray, shape (n_points,)
        Local intrinsic dimension estimate at each point.
    local_density : ndarray, shape (n_points,) or None
        Per-point local density estimate (k-NN-based), if requested.
    method : str
        Which estimator was used.
    k : int
        Neighbourhood size used for local fits.
    n_points : int
        Number of points the estimates were computed on.
    nan_mask : ndarray, shape (n_points,) or None
        True for rows that had at least one NaN in the input X and were
        median-imputed before nearest-neighbour search. Useful to check
        whether imputed (e.g. headwater) rows behave differently from
        non-imputed rows in downstream analysis.
    """

    id_values: np.ndarray
    local_density: np.ndarray | None
    method: str
    k: int
    n_points: int
    nan_mask: np.ndarray | None = None


def standardise(X: np.ndarray) -> np.ndarray:
    """Z-score standardise features. ID estimates are scale-sensitive.

    Uses nanmean/nanstd so columns with NaN entries still produce sensible
    standardisations. Caller is responsible for imputing NaN before fitting
    a NearestNeighbors index — see impute_median().

    Returns a copy; does not modify X in place.
    """
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def impute_median(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN cells with column medians.

    Returns
    -------
    X_imputed : ndarray, shape (n, d)
        Copy of X with NaN cells replaced by column medians.
    nan_mask : ndarray, shape (n,)
        Boolean: True for rows that originally had at least one NaN.
        Useful as a covariate in downstream analyses to check whether
        imputed rows behave differently.
    """
    X = np.asarray(X, dtype=float)
    nan_mask = np.isnan(X).any(axis=1)
    if not nan_mask.any():
        return X.copy(), nan_mask
    medians = np.nanmedian(X, axis=0)
    # Replace any all-NaN columns (medians is NaN there) with 0 — they carry
    # no information anyway and dropping them would change the column count.
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_out = X.copy()
    nan_cells = np.isnan(X_out)
    # Broadcast medians across rows
    col_idx = np.where(nan_cells)
    X_out[col_idx] = medians[col_idx[1]]
    return X_out, nan_mask


def _twonn_local(distances_2: np.ndarray) -> float:
    """TwoNN ID estimate on a local neighbourhood.

    Parameters
    ----------
    distances_2 : ndarray, shape (n_local, 2)
        For each local point, the distances to its first and second
        nearest neighbours (within the local neighbourhood).

    Returns
    -------
    float
        TwoNN ID estimate.

    Notes
    -----
    TwoNN estimator: id = log(N) / sum(log(mu_i)) where
    mu_i = r2_i / r1_i for each point i, and the sum is over points
    with r1_i > 0 (drops near-duplicates).

    This implementation uses the linear regression form:
    if F(mu) is the empirical CDF of the mu values, then
    -log(1 - F(mu)) = id * log(mu), and id is the slope.
    The MLE form (above) and the regression form give the same
    answer at large N; we use MLE because it's slightly more stable
    on small local neighbourhoods.
    """
    r1 = distances_2[:, 0]
    r2 = distances_2[:, 1]
    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 5:
        return np.nan
    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    # MLE: id_hat = N / sum(log(mu_i))
    return float(valid.sum() / log_mu.sum())


def estimate_local_id(
    X: np.ndarray,
    k: int = 50,
    method: Literal["twonn"] = "twonn",
    standardise_X: bool = True,
    return_density: bool = True,
    n_jobs: int = -1,
) -> LocalIDResult:
    """Estimate local intrinsic dimension at each point of X.

    For each point, builds a neighbourhood of size k from the nearest
    neighbours in feature space and fits TwoNN within that neighbourhood.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_features)
        Feature matrix.
    k : int
        Neighbourhood size for local ID estimation. Larger k gives
        smoother estimates but blurs local structure. k in [30, 100]
        is typical for ecological feature matrices.
    method : {"twonn"}
        ID estimator. Currently only TwoNN; gride support pending.
    standardise_X : bool
        Z-score standardise features before computing distances.
        Strongly recommended unless features are already on
        comparable scales.
    return_density : bool
        Also return a per-point local density estimate
        (1 / mean k-NN distance), useful as a control variable.
    n_jobs : int
        Parallelism for nearest-neighbour search. -1 uses all cores.

    Returns
    -------
    LocalIDResult

    Notes
    -----
    For the pilot we use a hand-rolled TwoNN inside k-NN
    neighbourhoods rather than DADApy's global Data.compute_id_2NN(),
    because we want per-point local estimates and DADApy's API is
    primarily designed for global ID with optional decimation.

    A future refactor could use DADApy's gride estimator
    (Data.return_id_scaling_gride) for scale-aware ID, but for the
    pilot's purpose (correlation with calibration error) the simpler
    local TwoNN is enough.
    """
    if method != "twonn":
        raise NotImplementedError(f"method={method!r} not yet supported")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, d = X.shape
    if k < 10:
        raise ValueError(f"k={k} too small; need at least 10 for stable TwoNN")
    if k >= n:
        raise ValueError(f"k={k} must be < n_points={n}")

    if standardise_X:
        X = standardise(X)

    # Median-impute any NaN cells. Required because NearestNeighbors does
    # not accept NaN. The returned nan_mask is currently unused but kept on
    # the result object so callers can audit which rows were imputed.
    X, nan_mask = impute_median(X)

    # Build a single global k-NN index, then for each point compute
    # local TwoNN on its k neighbours. We need k+1 neighbours because
    # the first is the point itself.
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X)
    knn_dist, knn_idx = nbrs.kneighbors(X)
    # knn_dist[:, 0] is the self-distance (zero); drop it.
    knn_dist = knn_dist[:, 1:]
    knn_idx = knn_idx[:, 1:]

    id_values = np.full(n, np.nan, dtype=float)
    for i in range(n):
        local_idx = knn_idx[i]  # the k nearest neighbours of point i
        # We need each local point's two nearest neighbours within
        # the local set (not in the full dataset). Compute pairwise
        # distances among the local points + the centre point.
        local_set = np.concatenate([[i], local_idx])
        local_X = X[local_set]
        # Pairwise distances within the local set
        d2 = np.sum(
            (local_X[:, None, :] - local_X[None, :, :]) ** 2, axis=-1
        )
        np.fill_diagonal(d2, np.inf)
        # For each local point, find r1 and r2 (smallest two distances)
        d2_sorted = np.partition(d2, 2, axis=1)[:, :2]
        d2_sorted = np.sort(d2_sorted, axis=1)
        distances_2 = np.sqrt(d2_sorted)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            id_values[i] = _twonn_local(distances_2)

    local_density = None
    if return_density:
        # Local density proxy: 1 / mean k-NN distance.
        # Higher = denser region.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            local_density = 1.0 / knn_dist.mean(axis=1)

    return LocalIDResult(
        id_values=id_values,
        local_density=local_density,
        method=method,
        k=k,
        n_points=n,
        nan_mask=nan_mask,
    )


def estimate_global_id_twonn(X: np.ndarray, standardise_X: bool = True) -> float:
    """Global TwoNN ID estimate, for sanity checking.

    Useful to compare against the mean of local ID estimates: they
    should be in the same ballpark if local fits are stable.
    """
    X = np.asarray(X, dtype=float)
    if standardise_X:
        X = standardise(X)
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    return _twonn_local(dist[:, 1:3])
