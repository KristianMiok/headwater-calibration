"""Pilot on a single real species from the trustworthy-sdm panel.

Run this after copying that species's outputs into data/raw/<entity>/.
See docs/pilot_design.md for the expected file layout and how to
copy from trustworthy-sdm.

Usage:
    python scripts/02_pilot_single_species.py --entity torrentium \\
        --algorithm rf --track full --level 20

Outputs:
    results/figures/<entity>_<algo>_<track>_L<level>_pilot.png
    results/tables/<entity>_<algo>_<track>_L<level>_correlations.csv
    results/tables/<entity>_<algo>_<track>_L<level>_binned.csv

Interpretation:
- Spearman ρ > 0.4 with surviving partial correlation → strong signal,
  worth scaling to all 8 species.
- Spearman ρ ~ 0.2-0.4 → moderate; scale carefully and characterise
  when the signal works.
- Spearman ρ < 0.2 or partial correlation vanishes → hypothesis as
  stated does not hold for this species; check 1-2 more before
  abandoning, but don't force it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sdm_geometry import analysis, calibration, id_estimation, io  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, help="species/entity name (e.g. torrentium)")
    p.add_argument("--algorithm", default="rf", choices=["rf", "xgb"])
    p.add_argument(
        "--track", default="full",
        help="predictor track (e.g. full, upstream_only)",
    )
    p.add_argument(
        "--level", type=int, default=20, choices=[3, 10, 20],
        help="contamination level (3, 10, or 20)",
    )
    p.add_argument(
        "--k", type=int, default=50,
        help="local neighbourhood size for ID estimation (default 50)",
    )
    p.add_argument(
        "--data-root", default=str(ROOT / "data" / "raw"),
        help="path to data/raw containing per-entity subdirectories",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cell_label = f"{args.entity}_{args.algorithm}_{args.track}_L{args.level}"
    print("=" * 70)
    print(f"Real-data pilot: {cell_label}")
    print("=" * 70)

    # 1. Load
    print(f"\n[1/4] Loading cell from {args.data_root}...")
    try:
        cell = io.load_cell(
            data_root=args.data_root,
            entity=args.entity,
            algorithm=args.algorithm,
            track=args.track,
            contamination_level=args.level,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("\nDid you copy data from trustworthy-sdm? "
              "See docs/pilot_design.md.", file=sys.stderr)
        return 2

    print(f"  Loaded: {cell.n_pixels} pixels, {cell.n_features} features, "
          f"{cell.n_replicates} replicates")

    # 2. Local ID
    print(f"\n[2/4] Estimating local ID (k={args.k})...")
    id_result = id_estimation.estimate_local_id(
        cell.X, k=args.k, method="twonn",
        standardise_X=True, return_density=True,
    )
    n_valid_id = int(np.isfinite(id_result.id_values).sum())
    print(
        f"  ID — mean: {np.nanmean(id_result.id_values):.2f}, "
        f"median: {np.nanmedian(id_result.id_values):.2f}, "
        f"valid: {n_valid_id}/{cell.n_pixels}"
    )

    # 3. Calibration
    print("\n[3/4] Computing per-pixel calibration error...")
    calib = calibration.compute_per_pixel_calibration(
        replicate_predictions=cell.replicates,
        benchmark=cell.benchmark,
        alpha=0.05,
    )
    print(f"  Empirical coverage: {calib.empirical_coverage:.3f}")
    print(f"  Mean miscalibration: {calib.miscalibrated.mean():.3f}")
    print(f"  Mean interval width: {calib.interval_width.mean():.4f}")

    # 4. Correlation
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
          f"(p = {corr_binary.spearman_p:.2e}, n = {corr_binary.n_valid})")
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
    binned.to_csv(tab_dir / f"{cell_label}_binned.csv", index=False)

    pd.DataFrame(
        [
            {**corr_binary.to_dict(), "metric": "binary_miscalibration",
             "cell": cell_label},
            {**corr_distance.to_dict(), "metric": "miscalibration_distance",
             "cell": cell_label},
        ]
    ).to_csv(tab_dir / f"{cell_label}_correlations.csv", index=False)

    analysis.plot_id_vs_calibration(
        id_values=id_result.id_values,
        miscalibration=calib.miscalibrated,
        correlation=corr_binary,
        binned=binned,
        title=f"{cell_label}: local ID vs miscalibration",
        savepath=fig_dir / f"{cell_label}_pilot.png",
    )
    print(f"\n  Saved: {fig_dir / f'{cell_label}_pilot.png'}")

    # Verdict
    print("\n" + "=" * 70)
    rho = corr_binary.spearman_rho
    partial = corr_binary.partial_spearman_rho
    print(f"Spearman ρ = {rho:.3f}", end="")
    if partial is not None:
        print(f", partial ρ = {partial:.3f}")
    else:
        print()
    if rho > 0.4 and (partial is None or partial > 0.2):
        print("STRONG signal — scale to all species.")
    elif rho > 0.2:
        print("MODERATE signal — try 1-2 more species before deciding.")
    else:
        print("WEAK signal — hypothesis doesn't hold cleanly here. "
              "Check another species or reconsider.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
