"""Loaders for trustworthy-sdm outputs.

This module loads three things from a local copy of the
trustworthy-sdm pipeline outputs:

1. Predictor matrix X for a given (entity, track) — the environmental
   features used at each stream segment.
2. The 30 replicate prediction surfaces for a (entity, algorithm,
   track, contamination_level) cell.
3. The benchmark (clean) prediction surface for the same cell.

The exact file layout in trustworthy-sdm is defined by its
notebooks/02_full_panel.py and notebooks/03_conformal_calibration.py;
this loader expects a small subset to have been copied into
data/raw/ following the layout documented in docs/pilot_design.md.

For the pilot, the layout is:

data/raw/
  <entity>/
    predictors.parquet            # predictor matrix (n_pixels, n_features)
    benchmark_<algo>_<track>.npy  # benchmark surface (n_pixels,)
    replicates_<algo>_<track>_L<contam>.npy  # (30, n_pixels) array
    basin_lookup.parquet          # subc_id -> basin_id (for LOBO, optional)

This is a deliberately narrow interface for the pilot. When we scale
to the full panel on VEGA, the loader can be extended to read directly
from the trustworthy-sdm output structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CellData:
    """Everything we need from one (entity, algo, track, level) cell.

    Attributes
    ----------
    entity : str
    algorithm : str
    track : str
    contamination_level : int
    X : ndarray, shape (n_pixels, n_features)
        Predictor matrix.
    feature_names : list[str]
    replicates : ndarray, shape (n_replicates, n_pixels)
        Per-replicate predicted suitabilities.
    benchmark : ndarray, shape (n_pixels,)
        Clean reference prediction.
    subc_ids : ndarray, shape (n_pixels,) or None
        Stream-segment IDs for joining with basin lookup.
    basin_ids : ndarray, shape (n_pixels,) or None
        Basin IDs (for LOBO folds), if available.
    """

    entity: str
    algorithm: str
    track: str
    contamination_level: int
    X: np.ndarray
    feature_names: list[str]
    replicates: np.ndarray
    benchmark: np.ndarray
    subc_ids: np.ndarray | None
    basin_ids: np.ndarray | None

    @property
    def n_pixels(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_replicates(self) -> int:
        return self.replicates.shape[0]


def load_cell(
    data_root: Path | str,
    entity: str,
    algorithm: str,
    track: str,
    contamination_level: int,
) -> CellData:
    """Load one (entity, algo, track, level) cell from data/raw/.

    Raises
    ------
    FileNotFoundError
        If any required file is missing.
    """
    data_root = Path(data_root)
    entity_dir = data_root / entity
    if not entity_dir.is_dir():
        raise FileNotFoundError(f"entity directory not found: {entity_dir}")

    pred_path = entity_dir / "predictors.parquet"
    bench_path = entity_dir / f"benchmark_{algorithm}_{track}.npy"
    repl_path = entity_dir / f"replicates_{algorithm}_{track}_L{contamination_level}.npy"

    for p in (pred_path, bench_path, repl_path):
        if not p.is_file():
            raise FileNotFoundError(f"required file missing: {p}")

    pred_df = pd.read_parquet(pred_path)

    # Identify the predictor columns. By trustworthy-sdm convention,
    # subc_id (and optionally basin_id) are non-feature columns.
    non_feature_cols = {"subc_id", "basin_id"}
    feature_names = [c for c in pred_df.columns if c not in non_feature_cols]
    X = pred_df[feature_names].to_numpy(dtype=float)
    subc_ids = (
        pred_df["subc_id"].to_numpy() if "subc_id" in pred_df.columns else None
    )
    basin_ids = (
        pred_df["basin_id"].to_numpy() if "basin_id" in pred_df.columns else None
    )

    benchmark = np.load(bench_path)
    replicates = np.load(repl_path)

    if benchmark.shape[0] != X.shape[0]:
        raise ValueError(
            f"benchmark has {benchmark.shape[0]} pixels but predictors "
            f"have {X.shape[0]}"
        )
    if replicates.ndim != 2 or replicates.shape[1] != X.shape[0]:
        raise ValueError(
            f"replicates shape {replicates.shape} incompatible with "
            f"X shape {X.shape}"
        )

    # Try to attach basin_ids from the basin_lookup if not in predictors
    if basin_ids is None:
        lookup_path = entity_dir / "basin_lookup.parquet"
        if lookup_path.is_file() and subc_ids is not None:
            lookup = pd.read_parquet(lookup_path)
            if "subc_id" in lookup.columns and "basin_id" in lookup.columns:
                lookup_map = dict(
                    zip(
                        lookup["subc_id"].to_numpy(),
                        lookup["basin_id"].to_numpy(),
                        strict=False,
                    )
                )
                basin_ids = np.array(
                    [lookup_map.get(s, -1) for s in subc_ids], dtype=int
                )

    return CellData(
        entity=entity,
        algorithm=algorithm,
        track=track,
        contamination_level=contamination_level,
        X=X,
        feature_names=feature_names,
        replicates=replicates,
        benchmark=benchmark,
        subc_ids=subc_ids,
        basin_ids=basin_ids,
    )
