"""Export per-segment map layer for Lucian/Antonio's spatial figure.
One CSV per species (Random Forest, Protocol A, 20% contamination), one row per
stream segment, keyed by the RAW Hydrography90m subc_id.
Run from headwater-calibration root: uv run python scripts/export_map_layer.py"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "/Users/kristianmiok/Desktop/Papers/trustworthy-sdm/src")
import numpy as np
import pandas as pd
from trustworthy_sdm.conformal import nonconformity_scores, conformal_quantile

ALPHA, ALGO, LEVEL = 0.05, "random_forest", 20
SPECIES = ["Austropotamobius_torrentium_pooled", "Pacifastacus_leniusculus_alien"]

for entity in SPECIES:
    df = pd.read_parquet(f"data/raw/{entity}/predictors.parquet")
    features = [c for c in df.columns if c not in {"subc_id", "basin_id"}]
    X = df[features].to_numpy(dtype=float)
    is_hw = (np.isnan(X).sum(axis=1) >= 20)
    subc_id, basin_id = df["subc_id"].to_numpy(), df["basin_id"].to_numpy()

    benchmark = np.load(f"data/raw/{entity}/benchmark_{ALGO}_combined.npy")
    reps = np.load(f"data/raw/{entity}/replicates_{ALGO}_combined_L{LEVEL}.npy")
    q_lo = np.quantile(reps, ALPHA/2, axis=0)
    q_hi = np.quantile(reps, 1-ALPHA/2, axis=0)
    ens_mean = reps.mean(axis=0)

    bias = ens_mean - benchmark
    # signed distance from benchmark to nearest interval edge (pre-correction):
    #   >0 benchmark above upper edge, <0 below lower edge, 0 inside
    miscal_distance = np.where(benchmark > q_hi, benchmark - q_hi,
                       np.where(benchmark < q_lo, benchmark - q_lo, 0.0))
    covered_before = ((benchmark >= q_lo) & (benchmark <= q_hi)).astype(int)

    q_lo_l = np.full_like(q_lo, np.nan); q_hi_l = np.full_like(q_hi, np.nan)
    q_lo_m = np.full_like(q_lo, np.nan); q_hi_m = np.full_like(q_hi, np.nan)
    valid = ~np.isnan(basin_id) if basin_id.dtype.kind == "f" else np.ones(len(basin_id), bool)
    for b in np.unique(basin_id[valid]):
        test = basin_id == b; calib = (~test) & valid
        if calib.sum() < 20 or test.sum() == 0: continue
        s = nonconformity_scores(pd.Series(benchmark[calib]), pd.Series(q_lo[calib]), pd.Series(q_hi[calib]))
        qh = conformal_quantile(s, alpha=ALPHA)
        q_lo_l[test] = q_lo[test] - qh; q_hi_l[test] = q_hi[test] + qh
        for gv in (True, False):
            cm = calib & (is_hw == gv); tm = test & (is_hw == gv)
            if cm.sum() < 10 or tm.sum() == 0:
                if tm.sum() > 0: q_lo_m[tm] = q_lo[tm] - qh; q_hi_m[tm] = q_hi[tm] + qh
                continue
            sg = nonconformity_scores(pd.Series(benchmark[cm]), pd.Series(q_lo[cm]), pd.Series(q_hi[cm]))
            qhg = conformal_quantile(sg, alpha=ALPHA)
            q_lo_m[tm] = q_lo[tm] - qhg; q_hi_m[tm] = q_hi[tm] + qhg

    covered_stdLOBO = ((benchmark >= q_lo_l) & (benchmark <= q_hi_l)).astype(int)
    covered_mondrian = ((benchmark >= q_lo_m) & (benchmark <= q_hi_m)).astype(int)

    out = pd.DataFrame({
        "subc_id": subc_id, "headwater": is_hw.astype(int),
        "miscal_distance": miscal_distance, "bias": bias,
        "covered_before": covered_before, "covered_stdLOBO": covered_stdLOBO,
        "covered_mondrian": covered_mondrian, "basin": basin_id,
    })
    n_uncal = int(np.isnan(q_lo_m).sum())
    outpath = f"results/tables/map_layer_{entity}_{ALGO}_L{LEVEL}.csv"
    out.to_csv(outpath, index=False)
    print(f"\n{entity}: wrote {len(out)} rows -> {outpath}")
    print(f"  subc_id range: {subc_id.min()} .. {subc_id.max()}")
    print(f"    (MUST be large scattered Hydrography90m IDs, NOT 0..{len(out)-1} -- if 0..N the join breaks silently)")
    print(f"  headwater: {is_hw.sum()} ({100*is_hw.mean():.1f}%)   basins: {len(np.unique(basin_id[valid]))}")
    print(f"  uncalibrated pixels (tiny basins, covered=0): {n_uncal}")
    print(f"  HW coverage before/LOBO/Mondrian: {covered_before[is_hw].mean():.3f} / {covered_stdLOBO[is_hw].mean():.3f} / {covered_mondrian[is_hw].mean():.3f}")
