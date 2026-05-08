"""Basic tests for sdm_geometry package."""

from __future__ import annotations

import numpy as np
import pytest

from sdm_geometry import analysis, calibration, id_estimation, synth


class TestIDEstimation:
    def test_global_id_on_isotropic_gaussian(self):
        """Isotropic Gaussian in d dimensions should have ID near d."""
        rng = np.random.default_rng(0)
        for true_d in [3, 5, 10]:
            X = rng.standard_normal((2000, true_d))
            id_hat = id_estimation.estimate_global_id_twonn(X)
            # TwoNN typically underestimates slightly; allow generous tolerance
            assert abs(id_hat - true_d) < 1.5, (
                f"Global ID for d={true_d}: got {id_hat}, expected ~{true_d}"
            )

    def test_local_id_returns_correct_shape(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 8))
        result = id_estimation.estimate_local_id(X, k=30)
        assert result.id_values.shape == (500,)
        assert result.local_density.shape == (500,)
        assert result.k == 30
        assert result.n_points == 500

    def test_local_id_distinguishes_regions(self):
        """Local ID should be higher in a high-d region than low-d."""
        rng = np.random.default_rng(2)
        # Low-d region: 3D manifold in 10D
        z_low = rng.standard_normal((300, 3))
        A_low = rng.standard_normal((3, 10))
        X_low = z_low @ A_low + 0.01 * rng.standard_normal((300, 10))
        # High-d region: 8D manifold in 10D
        z_high = rng.standard_normal((300, 8))
        A_high = rng.standard_normal((8, 10))
        X_high = z_high @ A_high + 0.01 * rng.standard_normal((300, 10))
        X = np.vstack([X_low, X_high])

        result = id_estimation.estimate_local_id(X, k=30)
        mean_low = np.nanmean(result.id_values[:300])
        mean_high = np.nanmean(result.id_values[300:])
        assert mean_high > mean_low, (
            f"Expected high-region ID > low-region ID, got "
            f"{mean_high:.2f} vs {mean_low:.2f}"
        )

    def test_invalid_k_raises(self):
        X = np.random.standard_normal((100, 5))
        with pytest.raises(ValueError):
            id_estimation.estimate_local_id(X, k=5)  # k too small
        with pytest.raises(ValueError):
            id_estimation.estimate_local_id(X, k=200)  # k >= n


class TestCalibration:
    def test_perfect_coverage_when_benchmark_inside_intervals(self):
        rng = np.random.default_rng(3)
        n_pixels = 500
        # Replicates centred on 0.5 with std 0.1
        replicates = 0.5 + 0.1 * rng.standard_normal((30, n_pixels))
        # Benchmark also at 0.5 — should be inside intervals almost always
        benchmark = np.full(n_pixels, 0.5)
        calib = calibration.compute_per_pixel_calibration(replicates, benchmark)
        assert calib.empirical_coverage > 0.95

    def test_zero_coverage_when_benchmark_far_outside(self):
        rng = np.random.default_rng(4)
        n_pixels = 200
        replicates = 0.5 + 0.05 * rng.standard_normal((30, n_pixels))
        benchmark = np.full(n_pixels, 1.5)  # well outside any interval
        calib = calibration.compute_per_pixel_calibration(replicates, benchmark)
        assert calib.empirical_coverage == 0.0
        assert (calib.miscalibration_distance > 0).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            calibration.compute_per_pixel_calibration(
                np.zeros((30, 100)), np.zeros(50)
            )


class TestAnalysis:
    def test_correlation_recovers_strong_positive_signal(self):
        rng = np.random.default_rng(5)
        n = 1000
        x = rng.standard_normal(n)
        # y monotonically related to x
        y = x + 0.3 * rng.standard_normal(n)
        corr = analysis.correlate_id_with_calibration(x, y)
        assert corr.spearman_rho > 0.8
        assert corr.spearman_p < 1e-10

    def test_correlation_recovers_no_signal(self):
        rng = np.random.default_rng(6)
        n = 500
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        corr = analysis.correlate_id_with_calibration(x, y)
        assert abs(corr.spearman_rho) < 0.15

    def test_partial_correlation_kills_confounded_signal(self):
        rng = np.random.default_rng(7)
        n = 1000
        z = rng.standard_normal(n)  # confounder
        x = z + 0.1 * rng.standard_normal(n)
        y = z + 0.1 * rng.standard_normal(n)
        # x and y are correlated only via z
        corr = analysis.correlate_id_with_calibration(x, y, local_density=z)
        assert corr.spearman_rho > 0.8  # marginal correlation strong
        assert abs(corr.partial_spearman_rho) < 0.2  # partial correlation weak

    def test_binned_mean_returns_dataframe(self):
        rng = np.random.default_rng(8)
        x = rng.uniform(0, 10, 1000)
        y = (x > 5).astype(float)
        binned = analysis.binned_mean_miscalibration(x, y, n_bins=5)
        assert "id_mid" in binned.columns
        assert "mean_miscalibration" in binned.columns
        # Higher x bins should have higher mean y
        assert binned["mean_miscalibration"].iloc[-1] > binned["mean_miscalibration"].iloc[0]


class TestSynth:
    def test_synthetic_cell_has_expected_shapes(self):
        cell = synth.make_synthetic_cell(
            n_pixels=500, n_nominal_features=10, n_replicates=20, seed=0
        )
        assert cell.X.shape == (500, 10)
        assert cell.replicates.shape == (20, 500)
        assert cell.benchmark.shape == (500,)

    def test_synthetic_pipeline_recovers_signal(self):
        """End-to-end: synthetic data → ID → calibration → correlation."""
        cell = synth.make_synthetic_cell(
            n_pixels=2000, d_intrinsic_low=2, d_intrinsic_high=8,
            high_id_fraction=0.3, within_replicate_noise=0.02,
            bias_low=0.02, bias_high=0.20, seed=10,
        )
        id_res = id_estimation.estimate_local_id(cell.X, k=40)
        calib = calibration.compute_per_pixel_calibration(
            cell.replicates, cell.benchmark
        )
        corr = analysis.correlate_id_with_calibration(
            id_res.id_values, calib.miscalibrated,
            local_density=id_res.local_density,
        )
        assert corr.spearman_rho > 0.2, (
            f"End-to-end pipeline should recover positive correlation, "
            f"got ρ={corr.spearman_rho:.3f}"
        )
