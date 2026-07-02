"""Protocol B (algorithm-consensus ensemble) headwater analysis.
Tests whether the headwater calibration failure + Mondrian fix hold under the
four-algorithm consensus ensemble (GLM, GAM, RF, XGBoost), not only under the
30-replicate single-algorithm (Protocol A) ensemble.
Run from project root: python scripts/protocol_b_headwater_analysis.py"""
import sys
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
import pandas as pd
from sdm_geometry.conformal import CellID, load_ensemble, nonconformity_scores, conformal_quantile, ALPHA

SURFACES_ROOT = Path("data/replicate_surfaces_protocol_b")
TRACK, LEVELS = "combined", [3, 10, 20]
SPECIES = [
    ("Astacus astacus", "Astacus_astacus"),
    ("Austropotamobius torrentium (pooled)", "Austropotamobius_torrentium_pooled"),
    ("Faxonius limosus (alien)", "Faxonius_limosus_alien"),
    ("Pacifastacus leniusculus (alien)", "Pacifastacus_leniusculus_alien"),
]

def consensus_benchmark(entity):
    return load_ensemble(CellID(entity, "consensus", TRACK, axis="benchmark", level=0), SURFACES_ROOT).mean(axis=1)

def headwater_and_basin(entity_dir):
    df = pd.read_parquet(f"data/raw/{entity_dir}/predictors.parquet")
    feats = [c for c in df.columns if c not in {"subc_id", "basin_id"}]
    is_hw = np.isnan(df[feats].to_numpy(dtype=float)).sum(axis=1) >= 20
    return pd.DataFrame({"is_hw": is_hw, "basin": df["basin_id"].to_numpy()},
                        index=df["subc_id"].to_numpy())

def lobo_mondrian(bench, q_lo, q_hi, basin, is_hw):
    idx = bench.index
    q_lo_l = pd.Series(np.nan, index=idx); q_hi_l = pd.Series(np.nan, index=idx)
    q_lo_m = pd.Series(np.nan, index=idx); q_hi_m = pd.Series(np.nan, index=idx)
    for b in pd.unique(basin):
        test = basin == b; calib = ~test
        if int(calib.sum()) < 20 or int(test.sum()) == 0: continue
        qh = conformal_quantile(nonconformity_scores(bench[calib], q_lo[calib], q_hi[calib]), alpha=ALPHA)
        q_lo_l[test] = q_lo[test] - qh; q_hi_l[test] = q_hi[test] + qh
        for gv in (True, False):
            cm = calib & (is_hw == gv); tm = test & (is_hw == gv)
            if int(cm.sum()) < 10 or int(tm.sum()) == 0:
                if int(tm.sum()) > 0: q_lo_m[tm] = q_lo[tm] - qh; q_hi_m[tm] = q_hi[tm] + qh
                continue
            qhg = conformal_quantile(nonconformity_scores(bench[cm], q_lo[cm], q_hi[cm]), alpha=ALPHA)
            q_lo_m[tm] = q_lo[tm] - qhg; q_hi_m[tm] = q_hi[tm] + qhg
    return ((bench >= q_lo_l) & (bench <= q_hi_l)), ((bench >= q_lo_m) & (bench <= q_hi_m))

rows = []
for entity, entity_dir in SPECIES:
    hb = headwater_and_basin(entity_dir)
    for level in LEVELS:
        cell = CellID(entity, "consensus", TRACK, axis="lowacc", level=level)
        if not (SURFACES_ROOT / cell.short()).exists():
            print(f"SKIP {entity_dir} L{level}: no dir {cell.short()}"); continue
        ens = load_ensemble(cell, SURFACES_ROOT); bench = consensus_benchmark(entity)
        shared = ens.index.intersection(bench.index).intersection(hb.index)
        ens = ens.loc[shared]; bench = bench.loc[shared]
        is_hw = pd.Series(hb.loc[shared, "is_hw"].to_numpy().astype(bool), index=shared)
        basin = pd.Series(hb.loc[shared, "basin"].to_numpy(), index=shared)
        q_lo = ens.quantile(ALPHA/2, axis=1); q_hi = ens.quantile(1-ALPHA/2, axis=1)
        cov_before = (bench >= q_lo) & (bench <= q_hi)
        cov_l, cov_m = lobo_mondrian(bench, q_lo, q_hi, basin, is_hw)
        hw = is_hw.to_numpy()
        cb, cl, cmv = cov_before.to_numpy(), cov_l.to_numpy(), cov_m.to_numpy()
        for grp, m in (("headwater", hw), ("non_hw", ~hw)):
            rows.append({"entity": entity_dir, "level": level, "group": grp, "n": int(m.sum()),
                         "n_members": int(ens.shape[1]), "cov_before": float(cb[m].mean()),
                         "cov_lobo": float(cl[m].mean()), "cov_mondrian": float(cmv[m].mean())})
        print(f"{entity_dir} L{level} (members={ens.shape[1]}): "
              f"HW {cb[hw].mean():.3f}/{cl[hw].mean():.3f}/{cmv[hw].mean():.3f}  "
              f"nonHW {cb[~hw].mean():.3f}/{cl[~hw].mean():.3f}/{cmv[~hw].mean():.3f}")

res = pd.DataFrame(rows)
if len(res):
    res.to_csv("results/tables/protocol_b_headwater.csv", index=False)
    print("\nSaved results/tables/protocol_b_headwater.csv")
