"""C14 matched-label factorial figure (replaces the K-indexed Figure 2).

Data: docs/experiments/analysis/c14_analysis.json (180 cells: domain x N x
diversity x method x seed; success delta vs the frozen zero-shot source on the
frozen test cohorts at binding budgets).

Design: two vertically stacked panels (static / dynamic), x = N (log2),
y = success delta vs source. Methods: full FT (vermillion), LoRA (blue),
scratch (gray). Concentrated = solid lines, filled markers; distributed =
dashed lines, open markers. Small markers show individual adaptation seeds;
bold lines join seed means. The w_min row under each panel gives the
concentrated condition's minimal world count at each N (distributed uses 8x).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "docs",
                                    "experiments", "analysis",
                                    "c14_analysis.json"))
OUT = os.path.join(HERE, "fig_c14_factorial.pdf")

BLUE, VERM, MUTED, INK, GRID = "#0072B2", "#C05500", "#5f5f5c", "#1a1a18", "#e4e4e1"

# AAAI-27 kit: figure text >=9 pt after scaling (we set 10 pt), strokes
# >=0.5 pt (we use >=0.8 pt), colors >=4.5:1 contrast on white.
plt.rcParams.update({
    "font.size": 10.0, "axes.titlesize": 10.0, "axes.labelsize": 10.0,
    "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "legend.fontsize": 10.0,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

NS = [256, 1024, 4096, 16384, 65536]
X = np.log2(NS)
METHODS = [("full_ft", VERM, "o"), ("lora", BLUE, "s"), ("scratch", MUTED, "^")]
WMIN = {"static": ["2", "6", "22", "89", "356"],
        "dynamic": ["1", "1", "1", "1", "4"]}

cells = json.load(open(SRC))["cells"]


def get(domain, N, div, method):
    return [c for c in cells
            if c["domain"] == domain and c["N"] == N and c["div"] == div
            and c["method"] == method]


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True,
                         gridspec_kw={"wspace": 0.16, "bottom": 0.30})

for ax, domain, letter in zip(axes, ("static", "dynamic"), "ab"):
    # Shade the N levels re-drawn with independent world sets (C14-R).
    _repl = [256] if domain == "static" else [1024, 16384]
    for _N in _repl:
        ax.axvspan(np.log2(_N) - 0.45, np.log2(_N) + 0.45,
                   color="#000000", alpha=0.06, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color=MUTED, linewidth=0.9, linestyle=":", zorder=1)

    for method, color, marker in METHODS:
        for div, ls, mfc in (("conc", "-", color), ("dist", "--", "white")):
            means = []
            for N, x in zip(NS, X):
                cs = get(domain, N, div, method)
                vals = [c["dsucc"] for c in cs]
                means.append(float(np.mean(vals)))
                # individual seeds, slightly jittered
                jit = np.linspace(-0.10, 0.10, len(vals))
                ax.plot(x + jit, vals, marker, color=color, markersize=2.0,
                        markerfacecolor=color if div == "conc" else "white",
                        markeredgewidth=0.8, linestyle="none", alpha=0.55,
                        zorder=3)
            ax.plot(X, means, ls, color=color, linewidth=1.3,
                    marker=marker, markersize=3.6,
                    markerfacecolor=mfc, markeredgecolor=color,
                    markeredgewidth=0.9, zorder=4)

    if domain == "static":
        ax.set_ylabel("success delta vs. source")
    # Plain-text titles: matplotlib cannot render the LaTeX small-caps
    # stage labels (C9/C9b), so the caption carries the substrate names.
    dom_label = "Static substrate" if domain == "static" else "Dynamic substrate"
    ax.set_title(f"({letter}) {dom_label}", loc="left", fontsize=10.0)
    if domain == "static":
        ax.set_ylim(-0.44, 0.16)
    else:
        ax.set_ylim(-0.41, 0.30)
    wtxt = ", ".join(WMIN[domain])
    if domain == "static":
        ax.text(0.985, 0.97,
                f"min. worlds (conc.): {wtxt}\nshaded bands: redrawn $N$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.0, color=MUTED, linespacing=1.2)
    else:
        ax.text(0.985, 0.97, f"min. worlds (conc.): {wtxt}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10.0, color=MUTED)
        for _N in (1024, 16384):
            ax.text(np.log2(_N), -0.398, "redrawn", ha="center",
                    va="bottom", fontsize=10.0, color=MUTED, zorder=2)

for ax in axes:
    ax.set_xticks(X)
    ax.set_xticklabels(["256", "1024", "4096", "16384", "65536"])
    ax.set_xlabel("supervised states $N$ (log scale)")

handles = [
    Line2D([], [], color=VERM, marker="o", markersize=4.6, linewidth=1.4,
           label="full FT"),
    Line2D([], [], color=BLUE, marker="s", markersize=4.6, linewidth=1.4,
           label="LoRA"),
    Line2D([], [], color=MUTED, marker="^", markersize=4.6, linewidth=1.4,
           label="scratch"),
    Line2D([], [], color=INK, linestyle="-", linewidth=1.2,
           label="concentrated (conc.)"),
    Line2D([], [], color=INK, linestyle="--", linewidth=1.2,
           label="distributed ($8\\times$ worlds)"),
]
fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center",
           bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0,
           handletextpad=0.4)

fig.savefig(OUT)
print("wrote", OUT)


# --- Single-panel dynamic variant for the main paper (full 2-panel goes to
# the appendix). Same data, same styling, one column wide at 9 pt.
fig2, ax = plt.subplots(figsize=(3.4, 2.7), gridspec_kw={"bottom": 0.34})
for _N in (1024, 16384):
    ax.axvspan(np.log2(_N) - 0.45, np.log2(_N) + 0.45,
               color="#000000", alpha=0.06, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.axhline(0.0, color=MUTED, linewidth=0.9, linestyle=":", zorder=1)
for method, color, marker in METHODS:
    for div, ls, mfc in (("conc", "-", color), ("dist", "--", "white")):
        means = []
        for N, x in zip(NS, X):
            cs = get("dynamic", N, div, method)
            vals = [c["dsucc"] for c in cs]
            means.append(float(np.mean(vals)))
            jit = np.linspace(-0.10, 0.10, len(vals))
            ax.plot(x + jit, vals, marker, color=color, markersize=2.2,
                    markerfacecolor=color if div == "conc" else "white",
                    markeredgewidth=0.8, linestyle="none", alpha=0.55, zorder=3)
        ax.plot(X, means, ls, color=color, linewidth=1.4, marker=marker,
                markersize=4.0, markerfacecolor=mfc, markeredgecolor=color,
                markeredgewidth=0.9, zorder=4)
# Short label: the full "vs. source" phrase is taller than this panel's axes
# and the tight bbox clips its last glyph; the caption states the baseline.
ax.set_ylabel("success delta")
ax.set_ylim(-0.41, 0.31)
ax.text(0.985, 0.97, "min. worlds (conc.): 1, 1, 1, 1, 4",
        transform=ax.transAxes, ha="right", va="top", fontsize=10.0,
        color=MUTED)
for _N in (1024, 16384):
    ax.text(np.log2(_N), -0.398, "redrawn", ha="center", va="bottom",
            fontsize=10.0, color=MUTED, zorder=2)
ax.set_xticks(X)
ax.set_xticklabels(["256", "1024", "4096", "16384", "65536"])
ax.set_xlabel("supervised states $N$ (log scale)")
# Short labels so the tight bounding box stays at column width (legend text
# wider than the axes would grow the PDF and shrink effective font size).
handles2 = [
    Line2D([], [], color=VERM, marker="o", markersize=4.6, linewidth=1.4,
           label="full FT"),
    Line2D([], [], color=BLUE, marker="s", markersize=4.6, linewidth=1.4,
           label="LoRA"),
    Line2D([], [], color=MUTED, marker="^", markersize=4.6, linewidth=1.4,
           label="scratch"),
    Line2D([], [], color=INK, linestyle="-", linewidth=1.2, label="conc."),
    Line2D([], [], color=INK, linestyle="--", linewidth=1.2,
           label="dist. ($8{\\times}$)"),
]
fig2.legend(handles=handles2, ncol=3, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, -0.02), handlelength=1.3, columnspacing=0.6,
            handletextpad=0.3, fontsize=10.0)
OUT2 = os.path.join(HERE, "fig_c14_dynamic.pdf")
fig2.savefig(OUT2)
print("wrote", OUT2)
