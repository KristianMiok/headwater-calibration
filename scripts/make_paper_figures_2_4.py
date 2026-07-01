#!/usr/bin/env python3
"""Paper figures 2 (per-pixel bias by population) and 4 (within-population
local-ID vs miscalibration correlation), at 20% contamination.

Headwater = topological headwater = any upstream ('u_') predictor is NaN.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sdm_geometry import analysis, calibration, id_estimation, io

DATA_ROOT = ROOT / "data" / "raw"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

ENTITIES = [
    ("Astacus_astacus", "Astacus astacus"),
    ("Austropotamobius_torrentium_pooled", "Austropotamobius torrentium"),
    ("Faxonius_limosus_alien", "Faxonius limosus alien"),
    ("Pacifastacus_leniusculus_alien", "Pacifastacus leniusculus alien"),
]
ALGOS = [("random_forest", "Random forest"), ("xgboost", "XGBoost")]
TRACK = "combined"
LEVEL = 20
K = 50
POP_COLORS = {"headwater": "#d1495b", "non-headwater": "#30638e"}
POPS = ["headwater", "non-headwater"]
elabels = [e[1] for e in ENTITIES]


def headwater_mask(cell) -> np.ndarray:
    u_idx = [i for i, n in enumerate(cell.feature_names) if n.startswith("u_")]
    if not u_idx:
        raise RuntimeError("no upstream ('u_') features found")
    return np.isnan(cell.X[:, u_idx]).any(axis=1)


bias_arrays = {}   # (alabel, elabel, pop) -> ndarray
bias_summary = []
corr_records = []

for ekey, elabel in ENTITIES:
    print(f"\n=== {elabel} ===")
    try:
        rf_cell = io.load_cell(DATA_ROOT, ekey, "random_forest", TRACK, LEVEL)
    except Exception as e:
        print(f"  SKIP {elabel}: {e}")
        continue
    hw = headwater_mask(rf_cell)
    print(f"  headwater: {int(hw.sum())}/{hw.size} ({100*hw.mean():.2f}%)")

    # Figure 2 — bias per population, both algorithms
    for akey, alabel in ALGOS:
        try:
            cell = rf_cell if akey == "random_forest" else io.load_cell(
                DATA_ROOT, ekey, akey, TRACK, LEVEL)
        except Exception as e:
            print(f"  skip bias {elabel}/{alabel}: {e}")
            continue
        bias = cell.replicates.mean(axis=0) - cell.benchmark
        for pop, m in [("headwater", hw), ("non-headwater", ~hw)]:
            b = bias[m]
            b = b[np.isfinite(b)]
            if b.size == 0:
                continue
            bias_arrays[(alabel, elabel, pop)] = b
            bias_summary.append({
                "entity": elabel, "algorithm": alabel, "population": pop,
                "median_bias": float(np.median(b)),
                "q25": float(np.percentile(b, 25)),
                "q75": float(np.percentile(b, 75)), "n": int(b.size)})

    # Figure 4 — within-population ID vs miscalibration (RF)
    idr = id_estimation.estimate_local_id(
        rf_cell.X, k=K, method="twonn", standardise_X=True, return_density=True)
    calib = calibration.compute_per_pixel_calibration(
        replicate_predictions=rf_cell.replicates, benchmark=rf_cell.benchmark,
        alpha=0.05)
    for pop, m in [("headwater", hw), ("non-headwater", ~hw)]:
        dens = idr.local_density[m] if idr.local_density is not None else None
        corr = analysis.correlate_id_with_calibration(
            id_values=idr.id_values[m], miscalibration=calib.miscalibrated[m],
            local_density=dens)
        corr_records.append({"entity": elabel, "population": pop,
                             "spearman_rho": corr.spearman_rho, "n": corr.n_valid})
        print(f"  {pop}: rho={corr.spearman_rho:.3f} (n={corr.n_valid})")

pd.DataFrame(bias_summary).to_csv(TAB / "figure2_bias_summary.csv", index=False)
corr_df = pd.DataFrame(corr_records)
corr_df.to_csv(TAB / "figure4_within_population_correlations.csv", index=False)

# ===== Figure 2: bias boxplots =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for c, (akey, alabel) in enumerate(ALGOS):
    ax = axes[c]
    for ei, el in enumerate(elabels):
        for j, pop in enumerate(POPS):
            if (alabel, el, pop) not in bias_arrays:
                continue
            vals = bias_arrays[(alabel, el, pop)]
            pos = ei + (j - 0.5) * 0.32
            bp = ax.boxplot(vals, positions=[pos], widths=0.28,
                            patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor=POP_COLORS[pop], alpha=0.75,
                        edgecolor="black", linewidth=0.6)
            for med in bp["medians"]:
                med.set(color="black", linewidth=1.0)
            for wk in bp["whiskers"] + bp["caps"]:
                wk.set(color="black", linewidth=0.6)
    ax.axhline(0, ls="--", lw=1, color="grey")
    ax.set_xticks(range(len(elabels)))
    ax.set_xticklabels(elabels, rotation=25, ha="right", fontsize=8)
    ax.set_title(alabel)
    if c == 0:
        ax.set_ylabel("Per-pixel bias (ensemble mean − benchmark)")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=POP_COLORS[p], alpha=0.75,
                         edgecolor="black") for p in POPS]
fig.legend(handles, POPS, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Per-pixel bias by population at 20% contamination", y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(FIG / "figure2_bias_boxplots.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "figure2_bias_boxplots.png")

# ===== Figure 4: within-population correlations (RF) =====
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ENTITIES))
w = 0.38
for j, pop in enumerate(POPS):
    vals = []
    for el in elabels:
        sub = corr_df[(corr_df.entity == el) & (corr_df.population == pop)]
        vals.append(sub["spearman_rho"].iloc[0] if len(sub) else np.nan)
    ax.bar(x + (j - 0.5) * w, vals, width=w, color=POP_COLORS[pop],
           alpha=0.85, edgecolor="black", linewidth=0.6, label=pop)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(elabels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Spearman ρ (local ID vs miscalibration)")
ax.set_title("Within-population local-ID vs miscalibration (RF, 20% contamination)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "figure4_id_correlations.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "figure4_id_correlations.png")
