"""Self-contained conformal-calibration primitives for the headwater analysis.

Split-conformal scoring and quantiles, the ensemble-surface loader, and the
cell descriptor used to locate per-replicate surfaces. Vendored here so the
analysis in this repository runs stand-alone, without any external pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05  # 95% interval

ENTITY_NAME_TO_DIR: dict[str, str] = {
    "Astacus astacus": "Astacus_astacus",
    "Austropotamobius fulcisianus (pooled)": "Austropotamobius_fulcisianus_pooled",
    "Austropotamobius torrentium (pooled)": "Austropotamobius_torrentium_pooled",
    "Faxonius limosus (alien)": "Faxonius_limosus_alien",
    "Pacifastacus leniusculus (alien)": "Pacifastacus_leniusculus_alien",
    "Pontastacus leptodactylus (pooled)": "Pontastacus_leptodactylus_pooled",
    "Procambarus clarkii (alien)": "Procambarus_clarkii_alien",
    "Procambarus clarkii (native)": "Procambarus_clarkii_native",
}


@dataclass(frozen=True)
class CellID:
    """Identifies one (entity, algorithm, track, axis, level) cell.
    A benchmark cell uses axis="benchmark" and level=0."""
    entity: str
    algorithm: str
    track: str
    axis: str
    level: int

    @property
    def entity_dir(self) -> str:
        return ENTITY_NAME_TO_DIR[self.entity]

    def short(self) -> str:
        return f"{self.entity_dir}__{self.algorithm}__{self.track}__{self.axis}__L{self.level}"


def load_ensemble(cell: CellID, surfaces_root: Path) -> pd.DataFrame:
    """Load all replicate/member surfaces for one cell as a wide DataFrame
    indexed by subc_id with one column per replicate (or algorithm member)."""
    cell_dir = Path(surfaces_root) / cell.short()
    if not cell_dir.exists():
        raise FileNotFoundError(f"missing surface dir: {cell_dir}")
    frames = []
    for path in sorted(cell_dir.glob("rep_*.parquet")):
        s = pd.read_parquet(path).set_index("subc_id")["predicted_probability"]
        frames.append(s.rename(path.stem))
    if not frames:
        raise RuntimeError(f"no surfaces in {cell_dir}")
    return pd.concat(frames, axis=1)


def nonconformity_scores(bench: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    """Per-pixel non-conformity: s_i = max(lo_i - bench_i, bench_i - hi_i, 0).
    Zero inside [lo, hi], positive outside. NaN rows dropped."""
    aligned = pd.concat({"bench": bench, "lo": lo, "hi": hi}, axis=1).dropna()
    below = aligned["lo"] - aligned["bench"]
    above = aligned["bench"] - aligned["hi"]
    return pd.concat([below, above], axis=1).max(axis=1).clip(lower=0.0)


def conformal_quantile(scores: pd.Series, alpha: float = ALPHA) -> float:
    """Finite-sample (1 - alpha) split-conformal quantile:
    the ceil((n + 1) * (1 - alpha)) / n -th order statistic."""
    s = np.asarray(scores)
    n = len(s)
    if n == 0:
        raise ValueError("conformal_quantile requires non-empty scores")
    q_rank = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q_rank, method="higher"))
