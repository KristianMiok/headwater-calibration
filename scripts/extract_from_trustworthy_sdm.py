#!/Users/kristianmiok/Desktop/Papers/sdm-robustness/venv/bin/python3.12
"""Extract one (entity, algorithm, track, contamination_level) cell from trustworthy-sdm.

Writes three files to sdm-geometry/data/raw/<entity_dir>/:
  predictors.parquet                       — predictor matrix (subc_id, basin_id, features)
  benchmark_<algo>_<track>.npy             — clean benchmark prediction, shape (n_pixels,)
  replicates_<algo>_<track>_L<level>.npy  — contaminated replicates, shape (30, n_pixels)

Requires: run with the sdm-robustness venv Python, which has sdm_robustness importable.
  /Users/kristianmiok/Desktop/Papers/sdm-robustness/venv/bin/python3.12 <this_script>

Or just run this file directly if it is executable (it has the right shebang).
"""

from __future__ import annotations

import sys
import os
import contextlib
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configurable paths — edit these if your directory layout differs
# ---------------------------------------------------------------------------

TRUSTWORTHY_SDM_ROOT = Path("/Users/kristianmiok/Desktop/Papers/trustworthy-sdm")
SDM_ROBUSTNESS_SRC   = Path("/Users/kristianmiok/Desktop/Papers/sdm-robustness/src")
SDM_ROBUSTNESS_ROOT  = Path("/Users/kristianmiok/Desktop/Papers/sdm-robustness")
MASTER_CSV           = SDM_ROBUSTNESS_ROOT / "data/raw/combined_data_true_master.csv"
SDM_GEOMETRY_ROOT    = Path("/Users/kristianmiok/Desktop/Papers/sdm-geometry")

# ---------------------------------------------------------------------------
# Default cell to extract
# ---------------------------------------------------------------------------

DEFAULT_ENTITY    = "Faxonius limosus (alien)"
DEFAULT_ALGO      = "random_forest"
DEFAULT_TRACK     = "combined"
DEFAULT_AXIS      = "lowacc"
DEFAULT_LEVEL     = 20
DEFAULT_N_REPS    = 30

# ---------------------------------------------------------------------------
# Bootstrap: add sdm_robustness to import path
# ---------------------------------------------------------------------------

sys.path.insert(0, str(SDM_ROBUSTNESS_SRC))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inline copy of the entity-dir map (avoids importing trustworthy_sdm itself,
# which in turn requires sdm_robustness to be pip-installed in that venv).
# ---------------------------------------------------------------------------

ENTITY_NAME_TO_DIR: dict[str, str] = {
    "Astacus astacus": "Astacus_astacus",
    "Austropotamobius fulcisianus (pooled)": "Austropotamobius_fulcisianus_pooled",
    "Austropotamobius torrentium (pooled)": "Austropotamobius_torrentium_pooled",
    "Cambarus latimanus": "Cambarus_latimanus",
    "Cambarus striatus": "Cambarus_striatus",
    "Creaserinus fodiens": "Creaserinus_fodiens",
    "Faxonius limosus (alien)": "Faxonius_limosus_alien",
    "Faxonius limosus (native)": "Faxonius_limosus_native",
    "Lacunicambarus diogenes": "Lacunicambarus_diogenes",
    "Pacifastacus leniusculus (alien)": "Pacifastacus_leniusculus_alien",
    "Pontastacus leptodactylus (pooled)": "Pontastacus_leptodactylus_pooled",
    "Procambarus clarkii (alien)": "Procambarus_clarkii_alien",
    "Procambarus clarkii (native)": "Procambarus_clarkii_native",
}


def cell_short(entity_dir: str, algo: str, track: str, axis: str, level: int) -> str:
    """Replicate the cell.short() naming from trustworthy_sdm.io."""
    return f"{entity_dir}__{algo}__{track}__{axis}__L{level}"


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def load_prepared(entity: str, axis: str) -> dict:
    """Call _prepare_entity_data from the companion paper's pipeline."""
    from sdm_robustness.io import load_master_table
    from sdm_robustness.execution.runner import _prepare_entity_data

    print(f"Loading master table from {MASTER_CSV} …", flush=True)
    result = load_master_table(str(MASTER_CSV))
    master = result[0] if isinstance(result, tuple) else result
    print(f"  master: {master.shape[0]:,} rows × {master.shape[1]} cols", flush=True)

    # _prepare_entity_data reads 'config/final_panel.csv' as a relative path;
    # chdir into the sdm-robustness repo root so the relative read resolves.
    print(f"Calling _prepare_entity_data for '{entity}' …", flush=True)
    with contextlib.chdir(SDM_ROBUSTNESS_ROOT):
        prepared = _prepare_entity_data(master, entity)

    return prepared


def extract_predictors(
    prepared: dict,
    track: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (predictors_df, kept_cols) using the same column selection as fit_cv_cell."""
    from sdm_robustness.pipeline.core import get_track_columns, clean_predictors

    benchmark = prepared["benchmark"]
    accessible_area = prepared["accessible_area"]

    # get_track_columns / clean_predictors are called on benchmark (presence records),
    # exactly as fit_cv_cell does.  The resulting column list is then applied to
    # accessible_area to build the predictor matrix.
    feat_cols = get_track_columns(benchmark, track)
    kept = clean_predictors(benchmark, feat_cols)

    print(
        f"  track={track}: {len(feat_cols)} raw feature cols → "
        f"{len(kept)} after missingness/correlation filter",
        flush=True,
    )

    predictors_df = accessible_area[["subc_id", "basin_id"] + kept].copy()
    predictors_df = predictors_df.sort_values("subc_id").reset_index(drop=True)

    return predictors_df, kept


def load_benchmark_surface(
    entity_dir: str,
    algo: str,
    track: str,
    subc_ids: pd.Series,
) -> np.ndarray:
    """Load the saved benchmark surface and align it to the predictors row order."""
    surf_path = (
        TRUSTWORTHY_SDM_ROOT
        / "data/results/grid_b_full"
        / f"{entity_dir}_rfxgb"
        / "surfaces"
        / f"{entity_dir}_{algo}_{track}_benchmark.parquet"
    )
    if not surf_path.exists():
        raise FileNotFoundError(f"Benchmark surface not found: {surf_path}")

    surf_df = pd.read_parquet(surf_path)
    print(f"  benchmark surface: {surf_df.shape[0]:,} rows from {surf_path.name}", flush=True)

    # Align to the predictor row order via subc_id merge
    ref = pd.DataFrame({"subc_id": subc_ids})
    merged = ref.merge(surf_df[["subc_id", "predicted_probability"]], on="subc_id", how="left")

    n_missing = merged["predicted_probability"].isna().sum()
    if n_missing > 0:
        raise ValueError(
            f"Benchmark surface is missing {n_missing} subc_ids present in accessible_area. "
            "Check that the surfaces file was saved with the same accessible_area."
        )

    return merged["predicted_probability"].to_numpy(dtype=float)


def load_replicate_surfaces(
    entity_dir: str,
    algo: str,
    track: str,
    axis: str,
    level: int,
    subc_ids: pd.Series,
    n_reps: int = 30,
) -> np.ndarray:
    """Load all replicate surfaces and stack into (n_reps, n_pixels)."""
    cell_dir = (
        TRUSTWORTHY_SDM_ROOT
        / "data/replicate_surfaces"
        / cell_short(entity_dir, algo, track, axis, level)
    )
    if not cell_dir.exists():
        raise FileNotFoundError(f"Replicate surface directory not found: {cell_dir}")

    rep_files = sorted(cell_dir.glob("rep_*.parquet"))
    if len(rep_files) == 0:
        raise FileNotFoundError(f"No rep_*.parquet files in {cell_dir}")
    if len(rep_files) != n_reps:
        print(
            f"  WARNING: expected {n_reps} replicates, found {len(rep_files)}. "
            "Proceeding with what is available.",
            flush=True,
        )

    ref = pd.DataFrame({"subc_id": subc_ids})
    arrays: list[np.ndarray] = []

    for f in rep_files:
        rep_df = pd.read_parquet(f)
        merged = ref.merge(rep_df[["subc_id", "predicted_probability"]], on="subc_id", how="left")
        n_missing = merged["predicted_probability"].isna().sum()
        if n_missing > 0:
            raise ValueError(
                f"Replicate {f.name} is missing {n_missing} subc_ids. "
                "Replicate and accessible_area subc_ids do not match."
            )
        arrays.append(merged["predicted_probability"].to_numpy(dtype=float))

    stacked = np.stack(arrays, axis=0)  # (n_reps, n_pixels)
    print(
        f"  replicates stacked: {stacked.shape} from {cell_dir.name}",
        flush=True,
    )
    return stacked


def run(
    entity: str,
    algo: str,
    track: str,
    axis: str,
    level: int,
    n_reps: int,
    dry_run: bool,
) -> None:
    entity_dir = ENTITY_NAME_TO_DIR[entity]
    out_dir = SDM_GEOMETRY_ROOT / "data/raw" / entity_dir
    print(f"\n=== Extracting cell ===")
    print(f"  entity     : {entity} ({entity_dir})")
    print(f"  algorithm  : {algo}")
    print(f"  track      : {track}")
    print(f"  axis       : {axis}")
    print(f"  level      : {level}")
    print(f"  output dir : {out_dir}\n")

    # 1. Load prepared entity data (loads master CSV once)
    prepared = load_prepared(entity, axis)
    acc = prepared["accessible_area"]
    print(f"  accessible_area: {acc.shape[0]:,} rows", flush=True)

    # 2. Predictors
    print("\n[1/3] Extracting predictor matrix …", flush=True)
    predictors_df, kept_cols = extract_predictors(prepared, track)
    print(f"  predictors_df: {predictors_df.shape}  (subc_id + basin_id + {len(kept_cols)} features)")

    subc_ids = predictors_df["subc_id"]
    n_pixels = len(subc_ids)

    # 3. Benchmark surface
    print("\n[2/3] Loading benchmark surface …", flush=True)
    benchmark_arr = load_benchmark_surface(entity_dir, algo, track, subc_ids)
    print(f"  benchmark array: shape={benchmark_arr.shape}, "
          f"min={benchmark_arr.min():.4f}, max={benchmark_arr.max():.4f}")

    # 4. Replicate surfaces
    print("\n[3/3] Loading replicate surfaces …", flush=True)
    replicates_arr = load_replicate_surfaces(
        entity_dir, algo, track, axis, level, subc_ids, n_reps=n_reps
    )
    print(f"  replicates array: shape={replicates_arr.shape}")

    # 5. Sanity checks
    assert benchmark_arr.shape == (n_pixels,), \
        f"benchmark shape mismatch: {benchmark_arr.shape} vs ({n_pixels},)"
    assert replicates_arr.shape[1] == n_pixels, \
        f"replicates shape mismatch: {replicates_arr.shape} vs ({replicates_arr.shape[0]}, {n_pixels})"
    assert not np.any(np.isnan(benchmark_arr)), "NaNs in benchmark array"
    assert not np.any(np.isnan(replicates_arr)), "NaNs in replicates array"

    # 6. Save
    pred_path = out_dir / "predictors.parquet"
    bench_path = out_dir / f"benchmark_{algo}_{track}.npy"
    rep_path   = out_dir / f"replicates_{algo}_{track}_L{level}.npy"

    if dry_run:
        print("\n[dry-run] Would write:")
        print(f"  {pred_path}")
        print(f"  {bench_path}")
        print(f"  {rep_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    predictors_df.to_parquet(pred_path, index=False)
    np.save(bench_path, benchmark_arr)
    np.save(rep_path, replicates_arr)

    print(f"\nSaved:")
    print(f"  {pred_path}  ({pred_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {bench_path}  ({bench_path.stat().st_size / 1e3:.1f} KB)")
    print(f"  {rep_path}  ({rep_path.stat().st_size / 1e6:.1f} MB)")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity",  default=DEFAULT_ENTITY,  help="Canonical entity name")
    p.add_argument("--algo",    default=DEFAULT_ALGO,    help="Algorithm: random_forest | xgboost")
    p.add_argument("--track",   default=DEFAULT_TRACK,   help="Track: combined | local_only | upstream_only")
    p.add_argument("--axis",    default=DEFAULT_AXIS,    help="Contamination axis: lowacc | snapping")
    p.add_argument("--level",   default=DEFAULT_LEVEL,   type=int, help="Contamination level (3 | 10 | 20)")
    p.add_argument("--n-reps",  default=DEFAULT_N_REPS,  type=int, help="Expected number of replicates")
    p.add_argument("--dry-run", action="store_true",     help="Print what would be written, don't write")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        entity=args.entity,
        algo=args.algo,
        track=args.track,
        axis=args.axis,
        level=args.level,
        n_reps=args.n_reps,
        dry_run=args.dry_run,
    )
