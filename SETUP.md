# Setup guide

Step-by-step instructions for getting `sdm-geometry` running locally on your Mac, pushing to GitHub, and pulling on VEGA when you're ready to scale.

## 1. Local setup on your MacBook

Open Terminal. These commands assume you keep your code in `~/Code/`; adjust if you use a different path.

```bash
# Move to where you keep code
cd ~/Code

# (You'll get the project folder by unzipping sdm-geometry.zip here)
cd sdm-geometry

# Confirm Python version
python3 --version
# Should print: Python 3.12.x

# Create the virtual environment
python3 -m venv .venv

# Activate it (do this every time you open a new terminal)
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package and dev dependencies
pip install -e ".[dev]"

# This installs: numpy, pandas, scipy, scikit-learn, matplotlib,
# seaborn, dadapy, tqdm, pyarrow, plus pytest/ruff/jupyter for dev.
# Takes 1-3 minutes.
```

## 2. Verify the toolchain

```bash
# Quick import check
python -c "import dadapy; print('dadapy', dadapy.__version__)"
python -c "from sdm_geometry import id_estimation; print('package OK')"

# Run the test suite
pytest tests/

# Should report: 13 passed in ~2 seconds.
```

## 3. Run the synthetic pilot

This validates the full pipeline before plugging in real data:

```bash
python scripts/01_pilot_synthetic.py
```

Expected output ends with:

```
PASS: Spearman ρ = 0.641 > 0.3, toolchain looks healthy.
```

A figure is saved to `results/figures/synthetic_pilot.png`. Open it — you should see a clear separation in the binned mean miscalibration between low-ID and high-ID regions.

If the toolchain passes, you're ready for real data.

## 4. Open in PyCharm

```bash
# From inside sdm-geometry/
open -a "PyCharm.app" .
```

Or: PyCharm → File → Open → select the `sdm-geometry` folder.

In PyCharm:
- **Settings → Project → Python Interpreter** → select the `.venv/bin/python` you just created (PyCharm usually auto-detects it).
- **Settings → Tools → Python Integrated Tools** → set Default test runner to pytest.
- Mark `src` as Sources Root (right-click → Mark Directory as → Sources Root) so imports resolve cleanly in the editor.

You should now be able to run any script or test directly from PyCharm.

## 5. Initialise git and push to GitHub

First create the repo on GitHub:
1. Go to https://github.com/new
2. Repository name: `sdm-geometry`
3. Description: "Geometric perspectives on calibration risk in species distribution models"
4. Public or private — your choice; I'd suggest private until you have a result
5. **Do NOT** initialise with README, .gitignore, or license (we have those already)
6. Click "Create repository"

Then locally:

```bash
cd ~/Code/sdm-geometry

# Initialise git
git init -b main

# Stage everything (except what's in .gitignore)
git add .

# Sanity check what will be committed
git status

# Commit
git commit -m "Initial project skeleton with synthetic-pilot verification"

# Connect to GitHub (replace USERNAME if not KristianMiok)
git remote add origin git@github.com:KristianMiok/sdm-geometry.git

# Push
git push -u origin main
```

If you get an SSH auth error, you can use HTTPS instead:
```bash
git remote set-url origin https://github.com/KristianMiok/sdm-geometry.git
git push -u origin main
```

## 6. Plug in real data (when ready)

Read `docs/pilot_design.md` for the expected data layout. Then copy the relevant Paper 5 outputs for one species into `data/raw/<entity>/`. The data directory is gitignored, so this stays local.

Once data is in place:

```bash
python scripts/02_pilot_single_species.py \
    --entity torrentium \
    --algorithm rf \
    --track full \
    --level 20
```

Output goes to `results/figures/` and `results/tables/`.

## 7. Scale on VEGA (when the local pilot is promising)

```bash
# On your Mac: push your latest code
git add . && git commit -m "Latest pilot results" && git push

# Connect to VEGA
ssh loginvega

# Clone (first time only)
cd ~
git clone git@github.com:KristianMiok/sdm-geometry.git
cd sdm-geometry

# Or if already cloned: git pull

# Set up venv on VEGA (first time only)
module load Python/3.12  # adjust if VEGA uses a different module name
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Stage data on VEGA — typically by extracting from the
# trustworthy-sdm output directory directly. See docs/pilot_design.md.

# Submit the SLURM job
sbatch scripts/run_pilot_vega.sbatch torrentium rf full 20

# Monitor
squeue -u $USER
tail -f logs/pilot_*.out
```

When the job is done, `scp` the small `results/` files back to your Mac for inspection, or commit-and-push them on VEGA so they show up in the GitHub repo.

## Troubleshooting

**`pip install dadapy` is slow or fails.** DADApy depends on Cython-compiled extensions. If it fails on macOS, make sure Xcode command-line tools are installed (`xcode-select --install`) and try again.

**`ModuleNotFoundError: No module named 'sdm_geometry'` when running scripts.** The `scripts/01_pilot_synthetic.py` and `scripts/02_pilot_single_species.py` files insert `src/` onto sys.path, so they should work even without `pip install -e .`. If you hit this from elsewhere, run `pip install -e .` from the project root.

**Tests fail on the global ID estimate.** TwoNN is stochastic in finite samples and the test tolerances are deliberately generous. If a single test fails by a small margin, run again — it's almost certainly a rare RNG draw.

**Figure looks wrong after running on real data.** Check that the per-pixel ordering in `predictors.parquet` matches the row order of the benchmark and replicate `.npy` arrays. The loader does basic shape checks but cannot detect row-order mismatches.
