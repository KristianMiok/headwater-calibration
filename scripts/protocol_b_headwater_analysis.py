"""Protocol B (algorithm-consensus ensemble) headwater analysis.
Tests whether the headwater calibration failure + Mondrian fix hold under the
four-algorithm consensus ensemble (GLM, GAM, RF, XGBoost), not only under the
30-replicate single-algorithm (Protocol A) ensemble.
Run from project root: uv run python scripts/protocol_b_headwater_analysis.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TSDM_SRC = Path("/Users/kristianmiok/Desktop/Papers/trustworthy-sdm/src")
SURFACES_ROOT = Path("/Users/kristianmiok/Desktop/Papers/trustworthy-sdm/data/replicate_surfaces_protocol_b")
sys.path.insert(0, str(TSDM_SRC))

from trustworthy_sdm.io import CellID
from trustworthy_sdm.analysis import load_ensemble, ALPHA
from trustworthy_sdm.conformal import nonconformity_scores, conformal_quantile

HC = Path("/Users/kristianmiok/Desktop/Papers/headwater-calibration")
TRACK, LEVELS = "combined", [3, 10, 20]
SPECIES = [
    ("Astacus astacus", "Astacus_astacus"),
    ("Austropotamobius torrentium (pooled)", "Austropotamobius_torrentium_pooled"),
    ("Faxonius limosus (alien)", "Faxonius_limosus_alien"),
    ("Pacifastacus leniusculus (alien)", "Pacifastacus_leniusculus_alien"),
]

def consensus_benchmark(entity):
    cell = CellID(entity, "consensus", TRACK, axis="benchmark", level=0)
    return load_ensemble(cell, SURFACES_ROOT).mean(axis=1)

def headwater_and_basin(entity_dir):
    df = pd.read_parquet(HC / "data/raw" / entity_dir / "predictors.parquet")
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
        try:
            ens = load_ensemble(cell, SURFACES_ROOT); bench = consensus_benchmark(entity)
        except Exception as e:
            print(f"SKIP {entity_dir} L{level}: {e}"); continue
        shared = ens.index.intersection(bench.index).intersection(hb.index)
        ens = ens.loc[shared]; bench = bench.loc[shared]
        is_hw = pd.Series(hb.loc[shared, "is_hw"].to_numpy().astype(bool), index=shared)
        basin = pd.Series(hb.loc[shared, "basin"].to_numpy(), index=shared)
        q_lo = ens.quantile(ALPHA/2, axis=1); q_hi = ens.quantile(1-ALPHA/2, axis=1)
        cov_before = (bench >= q_lo) & (bench <= q_hi); bias = ens.mean(axis=1) - bench
        cov_l, cov_m = lobo_mondrian(bench, q_lo, q_hi, basin, is_hw)
        hw = is_hw.to_numpy()
        cb, cl, cmv, bi = cov_before.to_numpy(), cov_l.to_numpy(), cov_m.to_numpy(), bias.to_numpy()
        for grp, m in (("headwater", hw), ("non_hw", ~hw)):
            rows.append({"entity": entity_dir, "level": level, "group": grp, "n": int(m.sum()),
                         "n_members": int(ens.shape[1]), "cov_before": float(cb[m].mean()),
                         "cov_lobo": float(cl[m].mean()), "cov_mondrian": float(cmv[m].mean()),
                         "bias": float(bi[m].mean())})
        print(f"{entity_dir} L{level} (members={ens.shape[1]}, n_hw={int(hw.sum())}): "
              f"HW before/LOBO/Mond = {cb[hw].mean():.3f}/{cl[hw].mean():.3f}/{cmv[hw].mean():.3f}   "
              f"nonHW = {cb[~hw].mean():.3f}/{cl[~hw].mean():.3f}/{cmv[~hw].mean():.3f}")

res = pd.DataFrame(rows)
if len(res):
    res.to_csv(HC / "results/tables/protocol_b_headwater.csv", index=False)
    print("\nSaved results/tables/protocol_b_headwater.csv\n=== Protocol B summary ===")
    for level in LEVELS:
        hw = res[(res.level==level)&(res.group=="headwater")]; nh = res[(res.level==level)&(res.group=="non_hw")]
        if len(hw) and len(nh):
            print(f"  L{level}: HW {hw.cov_before.mean():.3f}/{hw.cov_lobo.mean():.3f}/{hw.cov_mondrian.mean():.3f}   "
                  f"nonHW {nh.cov_before.mean():.3f}/{nh.cov_lobo.mean():.3f}/{nh.cov_mondrian.mean():.3f}")
else:
    print("No cells processed -- check Protocol B combined-track consensus dirs exist.")
