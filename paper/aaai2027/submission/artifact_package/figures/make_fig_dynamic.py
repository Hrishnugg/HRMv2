"""Dynamic-transfer headline figure (paper Figure 1).

All numbers from docs/experiments/analysis/c8_fixed_provider.json (map-level
reanalysis of canonical c8_local_heavy raw rows; fixed provider = field U-Net
blind, additive mode, binding budgets; 20 maps/suite; 10k bootstraps).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
_JSON = sys.argv[1] if len(sys.argv) > 1 else "c8r_fresh_cohort.json"
_AN = os.path.normpath(os.path.join(HERE, "..", "..", "..", "docs", "experiments",
                                    "analysis"))
SRC = os.path.join(_AN, _JSON)
# Per-seed analyses for panel (c): primary seed + the two retrained seeds.
SEED_JSONS = [("1234", _JSON), ("2001", "c8r_seed2001.json"), ("2002", "c8r_seed2002.json")]
BLUE, VERM, MUTED, INK, GRID = "#0072B2", "#C05500", "#5f5f5c", "#1a1a18", "#e4e4e1"

plt.rcParams.update({
    "font.size": 10.0, "axes.titlesize": 10.0, "axes.labelsize": 10.0,
    "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "legend.fontsize": 10.0,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

data = json.load(open(SRC))
ORDER = ["C_dyn_crossing", "C_dyn_maze", "C_dyn_maze_dense",
         "C_dyn_rooms", "C_dyn_rooms_large", "C_dyn_spiral"]
suites = [data["suites"][s] for s in ORDER]
labels = [s["label"] for s in suites]
ys = np.arange(len(suites))[::-1]


def style_ax(ax, grid_axis="x"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})

# Panel A: success dumbbells + delta CIs
ax = axes[0]
for y, s in zip(ys, suites):
    ax.plot([s["euclid_success"], s["primary_success"]], [y, y],
            color=MUTED, linewidth=1.0, zorder=2)
    ax.plot(s["euclid_success"], y, "s", markerfacecolor="white",
            markeredgecolor=MUTED, markersize=4.0, markeredgewidth=1.1, zorder=3)
    ax.plot(s["primary_success"], y, "o", color=BLUE, markersize=4.6, zorder=4)
    lo, hi = s.get("dsucc_ci", (s["dsucc"], s["dsucc"]))
    ax.text(1.05, y, f"+{s['dsucc']:.2f} [{lo:+.2f},{hi:+.2f}]",
            va="center", fontsize=10.0, color=BLUE)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=10.0)
ax.set_xlim(0, 1.98)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xticklabels(["0", "0.5", "1"])
ax.set_title("(a) Success (anchor $\\rightarrow$ learned)", fontsize=10.0, loc="left")
style_ax(ax)

# Panel B: matched ratios with CIs
ax = axes[1]
for y, s in zip(ys, suites):
    med, (lo, hi), n = s["ratio_median"], s["ratio_ci"], s["matched_n"]
    if n > 1:
        ax.plot([lo, hi], [y, y], color=BLUE, linewidth=1.0, zorder=2)
        ax.plot(med, y, "o", color=BLUE, markersize=4.4, zorder=3)
    else:
        ax.plot(med, y, "o", markerfacecolor="white", markeredgecolor=BLUE,
                markersize=4.4, markeredgewidth=1.1, zorder=3)
    ax.text(1.32, y, f"n={n}", va="center", fontsize=10.0, color=MUTED)
ax.axvline(1.0, color=INK, linewidth=0.9, linestyle=(0, (3, 2)))
ax.set_yticks(ys)
ax.set_yticklabels([])
ax.set_xscale("log")
ax.set_xlim(0.017, 2.6)
ax.set_xticks([0.05, 0.25, 1.0])
ax.set_xticklabels(["0.05", "0.25", "1"])
ax.set_title("(b) Expansion ratio", fontsize=10.0, loc="left")
style_ax(ax)

# Panel C: aware - blind success deltas, all three training seeds
ax = axes[2]
SEED_STYLE = {"1234": (VERM, "o"), "2001": ("#008561", "^"), "2002": ("#3A7A9E", "D")}
seed_data = []
for tag, fname in SEED_JSONS:
    try:
        seed_data.append((tag, json.load(open(os.path.join(_AN, fname)))))
    except FileNotFoundError:
        pass
n_seeds = max(1, len(seed_data))
for si, (tag, d2) in enumerate(seed_data):
    off = (si - (n_seeds - 1) / 2.0) * 0.24
    col, mk = SEED_STYLE.get(tag, (VERM, "o"))
    for y, sname in zip(ys, ORDER):
        s2 = d2["suites"][sname]
        d, (lo, hi) = s2["aware_minus_blind_succ"], s2["aware_minus_blind_succ_ci"]
        ax.plot([lo, hi], [y + off, y + off], color=col, linewidth=1.1,
                alpha=0.9, zorder=2)
        ax.plot(d, y + off, mk, color=col, markersize=4.2, zorder=3)
ax.axvline(0.0, color=INK, linewidth=0.9)
ax.set_yticks(ys)
ax.set_yticklabels([])
ax.set_xlim(-0.36, 0.42)
ax.set_xticks([-0.25, 0.0, 0.25])
ax.set_title("(c) Aware $-$ blind", fontsize=10.0, loc="left")
style_ax(ax)

handles = [
    Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=MUTED,
           linestyle="none", markersize=4.6, label="Euclid anchor"),
    Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=4.6,
           label="learned (blind)"),
    Line2D([], [], marker="o", color=SEED_STYLE["1234"][0], linestyle="none",
           markersize=4.6, label="seed 1234"),
    Line2D([], [], marker="^", color=SEED_STYLE["2001"][0], linestyle="none",
           markersize=4.6, label="seed 2001"),
    Line2D([], [], marker="D", color=SEED_STYLE["2002"][0], linestyle="none",
           markersize=4.6, label="seed 2002"),
]
fig.tight_layout(w_pad=1.0)
fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, -0.055), columnspacing=1.1, handletextpad=0.3)
fig.savefig(os.path.join(HERE, "fig_dynamic_headline.pdf"))
print("wrote fig_dynamic_headline.pdf")
