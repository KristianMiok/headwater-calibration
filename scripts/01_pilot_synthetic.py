"""Synthetic-data smoke test for the pilot pipeline.

Run this first, before plugging in real data. It generates a
synthetic SDM-like cell with a known geometric structure (some
pixels on a low-dimensional manifold, others on a higher-dimensional
one, with heteroscedastic prediction noise that the ensemble
under-estimates), runs the full pipeline, and reports whether the
correlation analysis recovers the expected signal.

Expected outcome on this synthetic data:
- Spearman ρ between local ID and miscalibration: strongly positive
  (typically > 0.4)
- Mean miscalibration in the high-ID region: 2-4× higher than in
  the low-ID region
- Partial correlation controlling for density: still positive
  (the heteroscedasticity is by construction independent of density)

If the script runs end-to-end and the signs come out as expected,
the toolchain is healthy and we can move to real data.

Usage:
    python scripts/01_pilot_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running the script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sdm_geometry import analysis, calibration, id_estimation, synth  # noqa: E402


def main() -> int:
    print("=" * 70)
    print("Synthetic pilot: local ID × calibration error")
    print("=" * 70)

    # 1. Generate synthetic cell
    print("\n[1/4] Generating synthetic cell...")
    cell = synth.make_synthetic_cell(
        n_pixels=4000,
        n_nominal_features=12,
        d_intrinsic_low=3,
        d_intrinsic_high=8,
        high_id_fraction=0.25,
        n_replicates=30,
        within_replicate_noise=0.03,
        bias_low=0.02,
        bias_high=0.20,
        seed=42,
    )
    print(f"  X shape: {cell.X.shape}")
    print(f"  replicates shape: {cell.replicates.shape}")
    print(
        f"  region 0 (low-ID) pixels: {(cell.region == 0).sum()}, "
        f"region 1 (high-ID) pixels: {(cell.region == 1).sum()}"
    )

    # 2. Estimate local intrinsic dimension
    print("\n[2/4] Estimating local intrinsic dimension (k=50)...")
    id_result = id_estimation.estimate_local_id(
        cell.X, k=50, method="twonn", standardise_X=True, return_density=True
    )
    print(f"  ID values — mean: {np.nanmean(id_result.id_values):.2f}, "
          f"median: {np.nanmedian(id_result.id_values):.2f}")
    print(
        f"  Mean ID in region 0 (true ID = 3): "
        f"{np.nanmean(id_result.id_values[cell.region == 0]):.2f}"
    )
    print(
        f"  Mean ID in region 1 (true ID = 8): "
        f"{np.nanmean(id_result.id_values[cell.region == 1]):.2f}"
    )
    global_id = id_estimation.estimate_global_id_twonn(cell.X)
    print(f"  Sanity check — global ID (whole dataset): {global_id:.2f}")

    # 3. Compute per-pixel calibration error
    print("\n[3/4] Computing per-pixel calibration error...")
    calib = calibration.compute_per_pixel_calibration(
        replicate_predictions=cell.replicates,
        benchmark=cell.benchmark,
        alpha=0.05,
    )
    print(f"  Empirical coverage: {calib.empirical_coverage:.3f} (target 0.95)")
    print(
        f"  Mean miscalibration in region 0: "
        f"{calib.miscalibrated[cell.region == 0].mean():.3f}"
    )
    print(
        f"  Mean miscalibration in region 1: "
        f"{calib.miscalibrated[cell.region == 1].mean():.3f}"
    )

    # 4. Correlation analysis
    print("\n[4/4] Correlation analysis...")
    corr_binary = analysis.correlate_id_with_calibration(
        id_values=id_result.id_values,
        miscalibration=calib.miscalibrated,
        local_density=id_result.local_density,
    )
    corr_distance = analysis.correlate_id_with_calibration(
        id_values=id_result.id_values,
        miscalibration=calib.miscalibration_distance,
        local_density=id_result.local_density,
    )
    print(f"\n  Local ID vs binary miscalibration:")
    print(f"    Spearman ρ = {corr_binary.spearman_rho:.3f} "
          f"(p = {corr_binary.spearman_p:.2e})")
    print(f"    Pearson r  = {corr_binary.pearson_r:.3f} "
          f"(p = {corr_binary.pearson_p:.2e})")
    if corr_binary.partial_spearman_rho is not None:
        print(
            f"    Partial Spearman (density-controlled) = "
            f"{corr_binary.partial_spearman_rho:.3f} "
            f"(p = {corr_binary.partial_spearman_p:.2e})"
        )

    print(f"\n  Local ID vs miscalibration distance:")
    print(f"    Spearman ρ = {corr_distance.spearman_rho:.3f} "
          f"(p = {corr_distance.spearman_p:.2e})")
    if corr_distance.partial_spearman_rho is not None:
        print(
            f"    Partial Spearman (density-controlled) = "
            f"{corr_distance.partial_spearman_rho:.3f} "
            f"(p = {corr_distance.partial_spearman_p:.2e})"
        )

    # Save outputs
    fig_dir = ROOT / "results" / "figures"
    tab_dir = ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    binned = analysis.binned_mean_miscalibration(
        id_result.id_values, calib.miscalibrated, n_bins=10
    )
    binned.to_csv(tab_dir / "synthetic_binned.csv", index=False)

    pd.DataFrame(
        [
            {**corr_binary.to_dict(), "metric": "binary_miscalibration"},
            {**corr_distance.to_dict(), "metric": "miscalibration_distance"},
        ]
    ).to_csv(tab_dir / "synthetic_correlations.csv", index=False)

    fig = analysis.plot_id_vs_calibration(
        id_values=id_result.id_values,
        miscalibration=calib.miscalibrated,
        correlation=corr_binary,
        binned=binned,
        title="Synthetic pilot: local ID vs miscalibration",
        savepath=fig_dir / "synthetic_pilot.png",
    )

    print(f"\n  Saved figure: {fig_dir / 'synthetic_pilot.png'}")
    print(f"  Saved tables: {tab_dir / 'synthetic_correlations.csv'}, "
          f"{tab_dir / 'synthetic_binned.csv'}")

    # Verdict
    print("\n" + "=" * 70)
    rho = corr_binary.spearman_rho
    if rho > 0.3:
        print(f"PASS: Spearman ρ = {rho:.3f} > 0.3, toolchain looks healthy.")
        return 0
    elif rho > 0.1:
        print(f"WEAK: Spearman ρ = {rho:.3f}; signal exists but is muted. "
              "Check noise parameters in synth.make_synthetic_cell().")
        return 0
    else:
        print(f"FAIL: Spearman ρ = {rho:.3f}; expected positive correlation. "
              "Investigate before running real data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
