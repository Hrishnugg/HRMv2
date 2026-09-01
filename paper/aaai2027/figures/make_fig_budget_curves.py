"""Supplementary figure: success-vs-budget step curves per dynamic suite.

Reads docs/experiments/analysis/c8_budget_curves.json (exact step curves
derived from the single 4x-binding eval pass by expansion thresholding).
Euclid vs fixed blind U-Net (aware twin dashed, oracle dotted); the binding
budget is marked. Budgets normalized to multiples of the binding budget so
all six suites share an x-axis.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_JSON = sys.argv[1] if len(sys.argv) > 1 else "c8_budget_curves.json"
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "docs", "experiments",
                                    "analysis", _JSON))
BLUE, VERM, MUTED, INK, GRID = "#0072B2", "#C05500", "#5f5f5c", "#1a1a18", "#e4e4e1"

# AAAI-27 kit: figure text >=9 pt (we set 10 pt), strokes >=0.8 pt,
# colors >=4.5:1 contrast on white.
plt.rcParams.update({
    "font.size": 10.0, "axes.titlesize": 10.0, "axes.labelsize": 10.0,
    "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "legend.fontsize": 10.0,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

ORDER = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
         "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]


def step_xy(budgets, success, binding, hi):
    """Right-continuous step curve on the normalized-budget axis."""
    xs, ys = [], []
    prev = 0.0
    for b, s in zip(budgets, success):
        xs.extend([b / binding, b / binding])
        ys.extend([prev, s])
        prev = s
    xs.append(hi / binding)
    ys.append(prev)
    return xs, ys


def main():
    data = json.load(open(SRC))
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.3), sharey=True)
    for ax, suite in zip(axes.flat, ORDER):
        d = data["suites"][suite]
        binding, hi = d["binding"], d["high"]
        curves = d["curves"]
        style = {
            "euclid": dict(color=MUTED, lw=1.1, ls="-", label="Euclid"),
            "field_unet_blind": dict(color=BLUE, lw=1.3, ls="-", label="blind U-Net"),
            "field_unet": dict(color=VERM, lw=0.9, ls="--", label="aware U-Net"),
            "oracle": dict(color=INK, lw=0.7, ls=":", label="oracle"),
        }
        for prov, st in style.items():
            c = curves.get(prov)
            if not c:
                continue
            xs, ys = step_xy(c["budgets"], c["success"], binding, hi)
            ax.plot(xs, ys, **st)
        ax.axvline(1.0, color=INK, lw=0.6, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xlim(min(0.02, 1.0 / binding), hi / binding)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0.1, 1.0, 4.0])
        ax.set_xticklabels(["0.1", "1", "4"])
        ax.set_title(f"{d['label']} (B*={binding})", fontsize=10.0, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[1, 0].set_xlabel("budget / binding budget")
    axes[1, 1].set_xlabel("budget / binding budget")
    axes[1, 2].set_xlabel("budget / binding budget")
    axes[0, 0].set_ylabel("success")
    axes[1, 0].set_ylabel("success")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(h_pad=0.9, w_pad=0.7)
    fig.savefig(os.path.join(HERE, "fig_budget_curves.pdf"))
    print("wrote fig_budget_curves.pdf")


if __name__ == "__main__":
    main()
