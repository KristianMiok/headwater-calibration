#!/usr/bin/env python3
"""Paper figures 1 (coverage) and 3 (width inflation) from the Mondrian panels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

rf = pd.read_csv(TAB / "panel_mondrian.csv")
if "algorithm" not in rf.columns:
    rf.insert(0, "algorithm", "random_forest")
xgb = pd.read_csv(TAB / "panel_mondrian_xgboost.csv")
panel = pd.concat([rf, xgb], ignore_index=True)
panel["group"] = panel["group"].replace({"non_hw": "non-headwater"})

ALGOS = [("random_forest", "Random forest"), ("xgboost", "XGBoost")]
GROUPS = ["headwater", "non-headwater"]
SCHEMES = [("cov_pre", "Before"), ("cov_lobo", "Standard\nLOBO"), ("cov_mond", "Mondrian\nLOBO")]

entities = sorted(panel["entity"].unique())
eidx = {e: i for i, e in enumerate(entities)}
cmap = plt.get_cmap("tab10")
ecol = {e: cmap(i % 10) for i, e in enumerate(entities)}

# ---------- Figure 1: coverage ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
for r, (algo, algo_label) in enumerate(ALGOS):
    for c, grp in enumerate(GROUPS):
        ax = axes[r, c]
        sub = panel[(panel["algorithm"] == algo) & (panel["group"] == grp)]
        for x, (col, _) in enumerate(SCHEMES):
            for _, row in sub.iterrows():
                off = (eidx[row["entity"]] - (len(entities) - 1) / 2) * 0.12
                ax.scatter(x + off, row[col], color=ecol[row["entity"]], s=36,
                           alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
            ax.scatter(x, sub[col].mean(), marker="_", s=900, color="black", zorder=4)
        ax.axhline(0.95, ls="--", lw=1, color="grey", zorder=1)
        ax.set_xticks(range(len(SCHEMES)))
        ax.set_xticklabels([s[1] for s in SCHEMES])
        ax.set_ylim(0.55, 1.02)
        if c == 0:
            ax.set_ylabel(f"{algo_label}\n\nEmpirical coverage")
        if r == 0:
            ax.set_title(grp)
handles = [plt.Line2D([0], [0], marker='o', ls='', color=ecol[e],
                      label=e.replace("_", " ")) for e in entities]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Empirical coverage of 95% prediction intervals", y=0.98)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG / "figure1_coverage.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "figure1_coverage.png")

# ---------- Figure 3: width inflation at L20 ----------
l20 = panel[panel["level"] == 20].copy()
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
xpos = {"headwater": 0, "non-headwater": 1}
for c, (algo, algo_label) in enumerate(ALGOS):
    ax = axes[c]
    sub = l20[l20["algorithm"] == algo]
    for _, row in sub.iterrows():
        x0 = xpos[row["group"]]
        ax.scatter(x0 - 0.12, row["infl_lobo"], color=ecol[row["entity"]], s=40, marker="o", zorder=3)
        ax.scatter(x0 + 0.12, row["infl_mond"], color=ecol[row["entity"]], s=40, marker="^", zorder=3)
        ax.plot([x0 - 0.12, x0 + 0.12], [row["infl_lobo"], row["infl_mond"]],
                color=ecol[row["entity"]], lw=0.8, alpha=0.6, zorder=2)
    ax.axhline(1.0, ls="--", lw=1, color="grey")
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["headwater", "non-headwater"])
    ax.set_title(algo_label)
    if c == 0:
        ax.set_ylabel("Median width inflation factor (log scale)")
shape_handles = [plt.Line2D([0], [0], marker='o', ls='', color='grey', label='Standard LOBO'),
                 plt.Line2D([0], [0], marker='^', ls='', color='grey', label='Mondrian LOBO')]
ent_handles = [plt.Line2D([0], [0], marker='s', ls='', color=ecol[e],
                          label=e.replace("_", " ")) for e in entities]
fig.legend(handles=shape_handles + ent_handles, loc="lower center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Width inflation at 20% contamination: standard vs Mondrian LOBO", y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(FIG / "figure3_width_inflation.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote", FIG / "figure3_width_inflation.png")
