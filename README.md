# sdm-geometry

Geometric perspectives on calibration risk in species distribution models.

This project investigates whether the local intrinsic dimension of the environmental feature space predicts where calibration of ensemble SDM prediction intervals will fail. The hypothesis: regions of high local intrinsic dimension correspond to regions where ensemble variance underestimates true uncertainty, even after standard contamination-robust calibration.

The work builds on the panel of calibration results from the companion `trustworthy-sdm` project (Miok et al., in prep) and uses intrinsic-dimension estimators from DADApy (Glielmo et al. 2022).

## Status

**Pilot phase.** Currently testing whether the core hypothesis — local intrinsic dimension correlates with per-pixel calibration error — holds on a single species before scaling to the full 8-entity panel.

## Repository layout

```
sdm-geometry/
├── pyproject.toml              # package config and dependencies
├── README.md
├── .gitignore
├── src/sdm_geometry/
│   ├── __init__.py
│   ├── io.py                   # data loading from trustworthy-sdm outputs
│   ├── id_estimation.py        # DADApy wrappers for local ID estimation
│   ├── calibration.py          # per-pixel calibration error computation
│   ├── analysis.py             # correlation, partial correlation, plotting
│   └── synth.py                # synthetic SDM-like data generator (for testing)
├── scripts/
│   ├── 01_pilot_synthetic.py   # smoke test with synthetic data (no real data needed)
│   ├── 02_pilot_single_species.py  # the actual pilot on one species
│   └── run_on_vega.sbatch      # SLURM script for full-panel scaling
├── notebooks/
│   └── pilot_exploration.ipynb # interactive exploration
├── tests/
│   └── test_id_estimation.py
├── data/
│   ├── raw/                    # gitignored: copied from trustworthy-sdm
│   └── processed/              # gitignored: ID estimates, correlations
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    └── pilot_design.md         # what the pilot tests and why
```

## Installation

Tested on Python 3.12.

```bash
cd sdm-geometry
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import dadapy; print('DADApy', dadapy.__version__)"
python -c "from sdm_geometry import id_estimation; print('package OK')"
```

## Quick start — synthetic pilot

Before plugging in real data, verify the toolchain works:

```bash
python scripts/01_pilot_synthetic.py
```

This generates a synthetic dataset with a known geometric structure, runs the full pilot pipeline (local ID estimation, calibration error simulation, correlation analysis), and produces a figure in `results/figures/synthetic_pilot.png`. If this runs end-to-end, the toolchain is healthy and you can move to real data.

## Real-data pilot

After copying single-species Paper 5 outputs into `data/raw/` (see `docs/pilot_design.md`):

```bash
python scripts/02_pilot_single_species.py --species torrentium --algorithm rf --track full
```

Output: correlation table in `results/tables/`, scatter plot in `results/figures/`.

## Data

The `data/` directory is gitignored. Real data is copied locally from `trustworthy-sdm` outputs for development; full-panel runs happen on VEGA.

## License

MIT.
