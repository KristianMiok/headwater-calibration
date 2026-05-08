"""Synthetic SDM-like data with known geometric structure.

Used to validate the pilot pipeline before plugging in real data.
The synthetic dataset has a deliberately controlled structure so we
know what the answer should look like:

- Predictor matrix X ∈ R^{n × d_nominal} where most features are
  generated from a low-dimensional latent manifold of dimension
  d_intrinsic_low, but a fraction of points sit in a higher-
  dimensional region (d_intrinsic_high). Local intrinsic dimension
  should detect this difference.

- A "benchmark" suitability surface is a smooth function of the
  latent coordinates.

- An "ensemble" of contaminated predictions adds heteroscedastic
  noise that is *larger* in the high-ID region. This is the signal
  we hope to detect: high-ID regions have wider true uncertainty,
  which the ensemble's variance-based intervals systematically
  underestimate.

If our pipeline finds a strong positive Spearman correlation between
local ID and miscalibration on this synthetic data, the toolchain is
healthy and ready for real data. If it doesn't, there's a bug to fix
before wasting time on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticCell:
    X: np.ndarray
    feature_names: list[str]
    replicates: np.ndarray
    benchmark: np.ndarray
    true_local_id: np.ndarray  # ground-truth ID label at each pixel
    region: np.ndarray  # 0 = low-ID region, 1 = high-ID region


def make_synthetic_cell(
    n_pixels: int = 5000,
    n_nominal_features: int = 12,
    d_intrinsic_low: int = 3,
    d_intrinsic_high: int = 8,
    high_id_fraction: float = 0.25,
    n_replicates: int = 30,
    within_replicate_noise: float = 0.03,
    bias_low: float = 0.02,
    bias_high: float = 0.20,
    seed: int = 0,
) -> SyntheticCell:
    """Generate a synthetic SDM-like cell that reproduces the realistic
    ensemble-miscalibration failure mode.

    The realistic failure mode is *not* that high-ID regions have wider
    noise — wider noise would actually help ensemble intervals cover.
    The realistic mode is the opposite:

      - Across-replicate noise (what the ensemble *sees*) is small in
        both regions, because all 30 replicates share the same model
        family and similar training data.
      - The *bias* (mean replicate minus benchmark) is large in
        high-ID regions, because the model extrapolates poorly there,
        but all replicates are biased in the same direction.

    Result: the ensemble interval is narrow everywhere (low across-
    replicate disagreement) but the benchmark falls outside the
    interval much more often in high-ID regions. That's exactly the
    Paper 5 finding at the per-pixel level, and it's the signal we
    want our pipeline to detect.

    Parameters
    ----------
    n_pixels : int
        Number of stream-segment-like points.
    n_nominal_features : int
        Predictor matrix width.
    d_intrinsic_low, d_intrinsic_high : int
        True intrinsic dimensions of the two regions.
    high_id_fraction : float
        Fraction of pixels in the high-ID region.
    n_replicates : int
        Number of ensemble members.
    within_replicate_noise : float
        SD of replicate-to-replicate disagreement at each pixel.
        Same in both regions — this is the "shared blind spot"
        modelling assumption.
    bias_low, bias_high : float
        Magnitude of (replicate_mean - benchmark) in low-ID and
        high-ID regions. The ensemble cannot detect this bias from
        its own variance.
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)

    # Assign region
    n_high = int(round(high_id_fraction * n_pixels))
    region = np.zeros(n_pixels, dtype=int)
    region[:n_high] = 1
    rng.shuffle(region)

    # Build features. Low-ID region: 3D manifold in n_nominal_features.
    # High-ID region: 8D manifold. Tiny extrinsic noise so TwoNN sees
    # a clean dimensionality.
    A_low = rng.standard_normal((d_intrinsic_low, n_nominal_features))
    z_low = rng.standard_normal((n_pixels, d_intrinsic_low))
    X_low = z_low @ A_low + 0.02 * rng.standard_normal(
        (n_pixels, n_nominal_features)
    )

    A_high = rng.standard_normal((d_intrinsic_high, n_nominal_features))
    z_high = rng.standard_normal((n_pixels, d_intrinsic_high))
    X_high = z_high @ A_high + 0.02 * rng.standard_normal(
        (n_pixels, n_nominal_features)
    )

    X = np.where(region[:, None] == 0, X_low, X_high)

    # Benchmark suitability: smooth function of the first two latent
    # coords of whichever region the point belongs to.
    z_combined = np.where(region[:, None] == 0, z_low[:, :2], z_high[:, :2])
    raw = 1.5 * z_combined[:, 0] - 0.8 * z_combined[:, 1]
    benchmark = 1.0 / (1.0 + np.exp(-raw))

    # Per-pixel bias magnitude: small in low-ID, large in high-ID.
    # Direction is random per pixel but shared across all 30 replicates.
    bias_magnitude = np.where(region == 0, bias_low, bias_high)
    bias_direction = rng.choice([-1.0, 1.0], size=n_pixels)
    per_pixel_bias = bias_magnitude * bias_direction
    replicate_mean = benchmark + per_pixel_bias

    # Replicates: all 30 share the same biased mean, with small
    # within-replicate noise on top. This is the "shared blind spot".
    replicates = (
        replicate_mean[None, :]
        + within_replicate_noise * rng.standard_normal((n_replicates, n_pixels))
    )
    replicates = np.clip(replicates, 0.0, 1.0)

    feature_names = [f"feature_{i:02d}" for i in range(n_nominal_features)]
    true_local_id = np.where(
        region == 0, float(d_intrinsic_low), float(d_intrinsic_high)
    )

    return SyntheticCell(
        X=X,
        feature_names=feature_names,
        replicates=replicates,
        benchmark=benchmark,
        true_local_id=true_local_id,
        region=region,
    )
