"""Robustness: does the headwater miscalibration gap survive controlling for
local feature-space density? Run: uv run python scripts/robustness_density_control.py"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sdm_geometry import io, id_estimation, calibration

ENTITIES = ["Faxonius_limosus_alien", "Austropotamobius_torrentium_pooled",
            "Astacus_astacus", "Pacifastacus_leniusculus_alien"]
ALGO, LEVEL, N_BINS = "random_forest", 20, 5
rows, logit_rows = [], []

for entity in ENTITIES:
    cell = io.load_cell("data/raw", entity, ALGO, "combined", LEVEL)
    print(f"\n=== {entity} (n={cell.n_pixels}) ===", flush=True)
    id_res = id_estimation.estimate_local_id(cell.X, k=50)
    density, is_hw = id_res.local_density, id_res.nan_mask
    calib = calibration.compute_per_pixel_calibration(cell.replicates, cell.benchmark)
    miscal = calib.miscalibrated.astype(float)
    gap_overall = miscal[is_hw].mean() - miscal[~is_hw].mean()
    print(f"  overall HW gap: {gap_overall:+.3f}")
    finite = np.isfinite(density)
    q = np.quantile(density[finite], np.linspace(0, 1, N_BINS + 1)); q[0], q[-1] = -np.inf, np.inf
    bin_idx = np.digitize(density, q) - 1
    print(f"  {'bin':>3} {'n_hw':>7} {'n_nhw':>7} {'m_hw':>8} {'m_nhw':>8} {'gap':>8}")
    per_bin = []
    for b in range(N_BINS):
        m = bin_idx == b; hw_m, nh_m = m & is_hw, m & (~is_hw)
        if hw_m.sum() < 30 or nh_m.sum() < 30: continue
        g = miscal[hw_m].mean() - miscal[nh_m].mean(); per_bin.append(g)
        print(f"  {b:>3} {hw_m.sum():>7} {nh_m.sum():>7} {miscal[hw_m].mean():>8.3f} {miscal[nh_m].mean():>8.3f} {g:>+8.3f}")
        rows.append({"entity": entity, "bin": b, "gap": g})
    if per_bin: print(f"  mean density-matched gap: {np.mean(per_bin):+.3f}  (overall {gap_overall:+.3f})")
    try:
        from sklearn.linear_model import LogisticRegression
        z = np.nan_to_num((density - np.nanmean(density)) / np.nanstd(density), nan=0.0)
        Xd = np.column_stack([is_hw.astype(float), z]); ok = np.isfinite(Xd).all(axis=1)
        lr = LogisticRegression(max_iter=1000).fit(Xd[ok], miscal[ok])
        c_hw, c_d = lr.coef_[0]
        print(f"  logistic  is_hw={c_hw:+.3f}  density(z)={c_d:+.3f}")
        logit_rows.append({"entity": entity, "coef_is_hw": c_hw, "coef_density_z": c_d})
    except Exception as e:
        print(f"  [logistic skipped: {e}]")

print("\n================ SUMMARY ================")
df = pd.DataFrame(rows)
if len(df):
    print(df.groupby("entity")["gap"].mean().round(3).to_string())
    print(f"\nGrand mean density-matched gap: {df['gap'].mean():+.3f}")
if logit_rows:
    ld = pd.DataFrame(logit_rows)
    print("\nLogistic is_hw coef (density-controlled):"); print(ld.set_index("entity").round(3).to_string())
    print(f"\nMean is_hw coefficient: {ld['coef_is_hw'].mean():+.3f}")
df.to_csv("results/tables/robustness_density_control.csv", index=False)
print("\nSaved results/tables/robustness_density_control.csv")
