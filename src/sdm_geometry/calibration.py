"""Per-pixel calibration error from ensemble prediction surfaces.

This module computes per-point calibration metrics that we want to
correlate with local intrinsic dimension. The trustworthy-sdm project
already produces panel-level coverage; here we drop down to the
per-stream-segment level so we have one calibration value per row of
the predictor matrix.

Two calibration metrics are supported:

1. Binary miscalibration indicator: 1 if the benchmark prediction
   falls outside the contaminated ensemble's [q_lo, q_hi] interval,
   0 if inside. Easy to interpret, but loses magnitude information.

2. Signed miscalibration distance: how far outside the interval the
   benchmark falls (0 if inside, positive distance to the nearest
   interval edge if outside). Retains magnitude.

For the pilot we report both. The headline analysis uses the binary
indicator because it's directly tied to the coverage probability that
Paper 5 reports, but the signed version is useful for understanding
whether high-ID regions just barely miss vs miss by a lot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerPixelCalibration:
    """Per-pixel calibration outputs.

    Attributes
    ----------
    miscalibrated : ndarray, shape (n_pixels,)
        1 if benchmark is outside the ensemble interval, else 0.
    miscalibration_distance : ndarray, shape (n_pixels,)
        Signed distance from benchmark to nearest interval edge.
        Positive if outside (above q_hi or below q_lo), 0 if inside.
    interval_width : ndarray, shape (n_pixels,)
        q_hi - q_lo at each pixel.
    benchmark : ndarray, shape (n_pixels,)
        Benchmark prediction (clean reference).
    contaminated_mean : ndarray, shape (n_pixels,)
        Mean of the contaminated ensemble's predictions.
    """

    miscalibrated: np.ndarray
    miscalibration_distance: np.ndarray
    interval_width: np.ndarray
    benchmark: np.ndarray
    contaminated_mean: np.ndarray

    @property
    def empirical_coverage(self) -> float:
        """Fraction of pixels where benchmark is inside interval."""
        return float(1.0 - self.miscalibrated.mean())

    @property
    def n_pixels(self) -> int:
        return len(self.miscalibrated)


def compute_per_pixel_calibration(
    replicate_predictions: np.ndarray,
    benchmark: np.ndarray,
    alpha: float = 0.05,
) -> PerPixelCalibration:
    """Compute per-pixel calibration error from raw replicate predictions.

    Parameters
    ----------
    replicate_predictions : ndarray, shape (n_replicates, n_pixels)
        The R replicate predictions for each of n pixels. Typically
        R = 30 in the trustworthy-sdm panel.
    benchmark : ndarray, shape (n_pixels,)
        The benchmark prediction (deterministic clean reference)
        against which coverage is computed.
    alpha : float
        Miscoverage rate; intervals are [q(alpha/2), q(1 - alpha/2)].
        Default 0.05 → 95% intervals.

    Returns
    -------
    PerPixelCalibration
    """
    replicate_predictions = np.asarray(replicate_predictions, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    if replicate_predictions.ndim != 2:
        raise ValueError(
            f"replicate_predictions must be 2D (n_replicates, n_pixels), "
            f"got shape {replicate_predictions.shape}"
        )
    if benchmark.ndim != 1:
        raise ValueError(f"benchmark must be 1D, got shape {benchmark.shape}")
    if replicate_predictions.shape[1] != benchmark.shape[0]:
        raise ValueError(
            f"replicate_predictions has {replicate_predictions.shape[1]} pixels "
            f"but benchmark has {benchmark.shape[0]}"
        )

    q_lo = np.quantile(replicate_predictions, alpha / 2, axis=0)
    q_hi = np.quantile(replicate_predictions, 1 - alpha / 2, axis=0)
    contaminated_mean = replicate_predictions.mean(axis=0)
    interval_width = q_hi - q_lo

    inside = (benchmark >= q_lo) & (benchmark <= q_hi)
    miscalibrated = (~inside).astype(float)

    # Signed distance: 0 if inside, distance to nearest edge if outside
    above = np.maximum(benchmark - q_hi, 0.0)
    below = np.maximum(q_lo - benchmark, 0.0)
    miscalibration_distance = above + below  # exactly one is positive when outside

    return PerPixelCalibration(
        miscalibrated=miscalibrated,
        miscalibration_distance=miscalibration_distance,
        interval_width=interval_width,
        benchmark=benchmark,
        contaminated_mean=contaminated_mean,
    )


def compute_calibration_from_quantiles(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    benchmark: np.ndarray,
    contaminated_mean: np.ndarray | None = None,
) -> PerPixelCalibration:
    """Same as compute_per_pixel_calibration but from precomputed quantiles.

    Useful when the trustworthy-sdm pipeline has already produced
    q_lo and q_hi surfaces and we don't need to keep all 30 replicates
    around.
    """
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)

    if not (q_lo.shape == q_hi.shape == benchmark.shape):
        raise ValueError(
            f"shape mismatch: q_lo {q_lo.shape}, q_hi {q_hi.shape}, "
            f"benchmark {benchmark.shape}"
        )

    interval_width = q_hi - q_lo
    inside = (benchmark >= q_lo) & (benchmark <= q_hi)
    miscalibrated = (~inside).astype(float)
    above = np.maximum(benchmark - q_hi, 0.0)
    below = np.maximum(q_lo - benchmark, 0.0)
    miscalibration_distance = above + below

    if contaminated_mean is None:
        # Approximate by interval midpoint if mean isn't provided
        contaminated_mean = 0.5 * (q_lo + q_hi)
    else:
        contaminated_mean = np.asarray(contaminated_mean, dtype=float)

    return PerPixelCalibration(
        miscalibrated=miscalibrated,
        miscalibration_distance=miscalibration_distance,
        interval_width=interval_width,
        benchmark=benchmark,
        contaminated_mean=contaminated_mean,
    )
