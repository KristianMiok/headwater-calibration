# headwater-calibration

Where ensemble species distribution models lose calibration under occurrence-data
contamination — and how group-conditional conformal prediction restores it.

This repository accompanies Miok & Pârvulescu (in prep.). It shows that when ensemble
SDMs are trained on data contaminated with low-accuracy occurrence records, the
resulting prediction intervals fail to cover the truth **specifically at stream-network
headwaters** — the topological tops of the network, where upstream-aggregated
environmental predictors are structurally undefined. The failure is driven by upward
prediction bias that ensemble variance does not detect, it scales with contamination
severity, and it replicates across four freshwater crayfish species and two ensemble
algorithms (random forest, XGBoost).

Standard leave-one-basin-out (LOBO) split conformal calibration — the panel-wide remedy
from the companion `trustworthy-sdm` project (Pârvulescu et al., in prep.) — restores
marginal coverage but leaves headwaters systematically undercovered. A group-conditional
(**Mondrian**) variant, calibrating headwater and non-headwater segments separately,
restores reliable coverage in both simultaneously at no additional computational cost,
and reallocates interval width toward the segments that need it.

A local-intrinsic-dimension analysis (TwoNN; Facco et al. 2017) — originally the
project's central hypothesis — is retained only as a supplementary within-headwater
robustness check.

## Status

Manuscript complete; in preparation for submission to *Ecography*. The committed scripts
reproduce the figures and the sparsity-control robustness analysis; panel diagnostic
outputs are in `results/tables/`.

## Repository layout

\`\`\`
headwater-calibration/
├── pyproject.toml
├── README.md
├── src/sdm_geometry/            # package name retained from the project's origin
│   ├── io.py                    # load extracted per-cell surfaces
│   ├── id_estimation.py         # local intrinsic dimension + density (TwoNN)
│   ├── calibration.py           # per-pixel calibration error, coverage, width
│   ├── analysis.py              # correlation / partial correlation / plotting
│   └── synth.py                 # synthetic generator for the smoke test
├── scripts/
│   ├── extract_from_trustworthy_sdm.py   # pull one cell's surfaces from the companion pipeline
│   ├── make_paper_figures.py             # Figure 1 (coverage), Figure 3 (width inflation)
│   ├── make_paper_figures_2_4.py         # Figure 2 (bias), Figure S1 (intrinsic dimension)
│   ├── robustness_density_control.py     # feature-space sparsity control for the headwater effect
│   ├── 01_pilot_synthetic.py             # end-to-end smoke test (no real data needed)
│   ├── 02_pilot_single_species.py        # single-cell pilot
│   └── run_pilot_vega.sbatch             # SLURM script for HPC runs
├── manuscript/                  # manuscript draft, tables, and figure legends
├── results/
│   ├── figures/                 # generated figures (gitignored)
│   └── tables/                  # diagnostic and panel tables
├── tests/
├── docs/
└── data/raw/                    # gitignored: surfaces extracted from trustworthy-sdm
\`\`\`

## Installation

Tested on Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

\`\`\`bash
cd headwater-calibration
uv venv --python 3.12
uv sync
\`\`\`

Verify:

\`\`\`bash
uv run python -c "from sdm_geometry import calibration; print('package OK')"
\`\`\`

## Quick start — synthetic smoke test

Before touching real data, verify the pipeline end-to-end on a constructed dataset with
known structure:

\`\`\`bash
uv run python scripts/01_pilot_synthetic.py
\`\`\`

## Reproducing the analysis

The analysis runs on per-cell prediction surfaces (30 ensemble replicates plus a clean
deterministic benchmark) extracted from the companion `trustworthy-sdm` pipeline. After
extracting the four headwater-bearing entities into `data/raw/`
(see `scripts/extract_from_trustworthy_sdm.py`):

\`\`\`bash
# main figures
uv run python scripts/make_paper_figures.py
uv run python scripts/make_paper_figures_2_4.py

# sparsity-control robustness for the headwater effect
uv run python scripts/robustness_density_control.py
\`\`\`

Panel diagnostic tables (contamination response, bias/width decomposition, standard vs
Mondrian conformal coverage and width) are written to `results/tables/`, e.g.
`panel_contamination_x_headwater.csv`, `panel_mondrian.csv`, `panel_mondrian_xgboost.csv`,
and `robustness_density_control.csv`.

## Data availability

The `data/` directory is gitignored. Prediction surfaces are extracted locally from the
companion `trustworthy-sdm` outputs; the underlying crayfish occurrences are from the
curated World of Crayfish® database, and environmental predictors from Hydrography90m and
GeoFRESH. See the manuscript's data-availability statement for full details.

## License

MIT.
