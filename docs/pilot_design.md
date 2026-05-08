# Pilot design

## What this pilot tests

**Hypothesis:** Local intrinsic dimension of the environmental feature space at a stream segment correlates positively with the per-pixel calibration error of an ensemble SDM at that segment.

**Why it might be true:** Where the local intrinsic dimension is high, the model is being asked to interpolate or extrapolate in a region of high effective dimensionality with sparse data. In such regions the 30 ensemble replicates tend to all share the same blind spot — their mutual agreement (narrow interval) reflects shared bias, not real certainty. The benchmark prediction is therefore more likely to fall outside the interval.

**Why it might be false:** What we attribute to high intrinsic dimension might be just sample sparsity — fewer training points nearby means worse fit, regardless of geometry. The partial-correlation control for local density tests this.

## Pilot success criteria

We commit to one species first. The pilot is judged on three numbers:

| Metric | Strong | Moderate | Weak |
|---|---|---|---|
| Spearman ρ (ID, miscalibration) | > 0.4 | 0.2–0.4 | < 0.2 |
| Partial ρ controlling for density | > 0.2 | > 0.1 | ≤ 0.1 |
| Mean miscalibration: top-decile ID vs bottom-decile ID | > 2× | 1.5–2× | < 1.5× |

- All three "strong" → scale to all 8 species, draft email to Laio.
- Mixed (e.g. strong correlation but partial vanishes) → pilot a second species before deciding.
- All three "weak" → the geometric story doesn't hold at the per-pixel level. Park the direction.

## Choice of species for the pilot

Pick a species where Paper 5's calibration story was clearest:
- Substantial coverage drop under contamination (so there's miscalibration to correlate with)
- Large enough panel size for stable local ID estimation (n_pixels ≥ 5000)
- Paper 5's panel_summary.csv has `coverage_pre` for every cell; pick the one closest to median calibration failure at L20 — not the worst (likely an outlier) or the best (might be too little signal to detect).

A reasonable default: **torrentium with RF on the `full` track at L20**. Substitute as needed.

## Data layout in `data/raw/`

For each species you pilot, populate:

```
data/raw/<entity>/
├── predictors.parquet                          # required
├── benchmark_<algo>_<track>.npy                # required
├── replicates_<algo>_<track>_L<level>.npy      # required
└── basin_lookup.parquet                        # optional, for LOBO later
```

### `predictors.parquet`

A pandas DataFrame with columns:
- `subc_id` (int): stream-segment ID
- (optional) `basin_id` (int): basin ID for LOBO folds
- All other columns: predictor features (climate, terrain, network position, etc.). The columns must match the features the model was trained on, in the same order.

Number of rows = number of stream segments in the species' study area.

### `benchmark_<algo>_<track>.npy`

A 1-D numpy array of length `n_pixels` containing the benchmark (clean) suitability prediction for each stream segment, in the same row order as `predictors.parquet`.

### `replicates_<algo>_<track>_L<level>.npy`

A 2-D numpy array of shape `(30, n_pixels)` containing the 30 replicate predictions from the contaminated ensemble.

## How to copy from trustworthy-sdm

The trustworthy-sdm repo stores per-replicate surfaces in `data/replicate_surfaces/` after running `trustworthy-sdm-regenerate`. The aggregated panel results are in `figures/panel_summary.csv` and `figures/panel_conformal.csv`.

For each species you want to pilot, you need to extract:

1. **Predictors.** From the `sdm-robustness` companion pipeline that trustworthy-sdm imports, the predictor matrix for an entity/track is what `fit_cv_cell` is called with. Save it as parquet:
   ```python
   import pandas as pd
   # however trustworthy-sdm exposes the predictor matrix for the cell
   X_df.to_parquet("data/raw/<entity>/predictors.parquet")
   ```

2. **Benchmark surface.** The deterministic benchmark for `(entity, algo, track)` — saved during the trustworthy-sdm regen step or available from the companion paper's outputs:
   ```python
   import numpy as np
   np.save("data/raw/<entity>/benchmark_<algo>_<track>.npy", benchmark_array)
   ```

3. **Replicate surfaces.** The 30 per-replicate prediction arrays for the chosen contamination level, stacked into one (30, n_pixels) array:
   ```python
   import numpy as np
   replicates = np.stack(per_replicate_arrays_for_L20, axis=0)
   np.save("data/raw/<entity>/replicates_<algo>_<track>_L<level>.npy", replicates)
   ```

The exact code paths depend on how trustworthy-sdm exposes these in its `io` module. A small extraction script can be added later under `scripts/` once we look at the actual file layout.

## After the pilot

Whatever the result, commit the outputs in `results/tables/` and `results/figures/` with a one-paragraph note in `docs/pilot_results.md` explaining what we found and what we decided to do. Honest pilots that didn't pan out are still valuable — they prevent forced research later.
