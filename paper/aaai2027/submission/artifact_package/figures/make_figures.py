"""Generate all paper figures as vector PDFs.

Every number is traced to docs/experiments/MASTER_EXPERIMENT_SYNTHESIS.md
(snapshot 2026-07-20) or the canonical C13 result document. No value is
interpolated or invented; sparse curves plot only recorded points.

Palette: Okabe-Ito subset validated with the dataviz six-check validator
(light surface): blue #0072B2, vermilion #D55E00, green #009E73,
orange #E69F00 (direct-labeled), purple #CC79A7 (direct-labeled).


Provenance of in-source constants: every plotted array below is transcribed
from the committed raw rows / result documents in the artifact package --
fig2 (integration): discrete clean-v3 + focal-pilot rows and the continuous
C7 matched ratios; fig3 (K-indexed adaptation): C9/C9b curve summaries
(record-level, illustrative; map-level inference lives in the analysis
scripts); fig4 (C11): mission evaluation and would-halt tables; fig5 (C13):
the frozen C13-M confirmation summary. The dynamic, budget-curve, and
factorial figures read their JSONs directly and embed nothing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE, VERM, GREEN, ORANGE, PURPLE = "#0072B2", "#C05500", "#008561", "#9C6C00", "#9F5E82"
INK, MUTED, GRID = "#1a1a18", "#5f5f5c", "#e4e4e1"

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


def style_ax(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1
# Harness-layer program map (schematic; verdicts from the synthesis).
def fig1():
    fig, ax = plt.subplots(figsize=(6.9, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    rows = [
        ("Representation\n& training", 21.5, [
            ("C5: scalar HRM collapses\nto residual cap", VERM),
            ("C6: field target rescues HRM\n(succ. .625→.975)", BLUE),
            ("C7: well-trained scalar\nis also viable", MUTED),
        ]),
        ("Planner\nintegration", 12.5, [
            ("Discrete: additive residual\ninert vs Manhattan", VERM),
            ("Same weights, focal rank\nw=1.0: −6–15% exp.", BLUE),
            ("C7–C8: additive beats\nfocal vs loose Euclid", BLUE),
            ("C13: fresh certifier 0/6 →\nshared queue 6/6", BLUE),
        ]),
        ("Formulation,\ninformation,\nsupervision", 3.5, [
            ("C8/C9b: future window\nno benefit (0/9)", MUTED),
            ("C9/C9h: LoRA plateau =\ncapacity; labels set regime", MUTED),
            ("C11/C12: no depth or\nhierarchy dose-response", VERM),
            ("C13-M: bounded obs. +\nlocal Bellman: −15.95%", BLUE),
        ]),
    ]
    X0, PITCH, W = 17.5, 20.7, 19.3
    for label, y, chips in rows:
        ax.text(0.6, y + 3.2, label, fontsize=10.0, fontweight="bold",
                va="center", ha="left", color=INK)
        for j, (text, color) in enumerate(chips):
            x = X0 + j * PITCH
            box = FancyBboxPatch((x, y), W, 6.4,
                                 boxstyle="round,pad=0.35,rounding_size=0.9",
                                 linewidth=0.9, edgecolor=color, facecolor="white")
            ax.add_patch(box)
            ax.text(x + W / 2, y + 3.2, text, fontsize=9.5,
                    va="center", ha="center", color=INK)
    fig.savefig(os.path.join(HERE, "fig1_program_map.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
# Integration principle: expansion ratios by (domain, integration mode).
def fig2():
    rows = [
        # (label, values, kind)
        ("Discrete additive\n(vs Manhattan)", [1.0103, 1.0041, 1.1116, 1.1466], "pts"),
        ("Discrete focal $w{=}1.0$", [0.85, 0.94, 0.93, 0.85, 0.85], "pts"),
        ("Continuous additive\n(field HRM vs Euclid)", [0.521, 0.804, 0.829, 0.850, 0.714, 0.839], "pts"),
        ("Continuous focal\n(best $w{=}1.1$, range)", [0.789, 0.977], "range"),
    ]
    colors = [VERM, BLUE, BLUE, MUTED]
    fig, ax = plt.subplots(figsize=(3.3, 1.95))
    for i, ((label, vals, kind), c) in enumerate(zip(rows, colors)):
        y = len(rows) - 1 - i
        if kind == "range":
            ax.plot(vals, [y, y], color=c, linewidth=1.4, zorder=3)
            ax.plot(vals, [y, y], "o", color="white", markeredgecolor=c,
                    markersize=4.2, markeredgewidth=1.1, zorder=4)
        else:
            ax.plot(vals, [y] * len(vals), "o", color=c, markersize=4.2,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.axvline(1.0, color=INK, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(1.005, 3.42, "no gain", fontsize=9.5, color=MUTED, ha="left")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10.0)
    ax.set_xlabel("matched-solved expansion ratio (lower is better)")
    ax.set_xlim(0.35, 1.22)
    style_ax(ax, grid_axis="x")
    fig.savefig(os.path.join(HERE, "fig2_integration.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
# Transfer: static label-scarce crossover vs dynamic label-dense regime.
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=False)

    ax = axes[0]
    ax.plot([1, 16], [0.650, 0.730], "-o", color=BLUE, markersize=3.4, linewidth=1.2, label="LoRA r8")
    ax.plot([1, 16, 32], [0.744, 0.571, 0.552], "-s", color=VERM, markersize=3.4, linewidth=1.2, label="full FT")
    ax.plot([1, 16], [1.008, 0.808], "-^", color=MUTED, markersize=3.4, linewidth=1.2, label="scratch")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 16, 32])
    ax.set_xticklabels(["1", "16", "32"])
    ax.set_title("Static (maze-dense, HRM)", fontsize=10.0)
    ax.set_xlabel("K labeled maps")
    ax.set_ylabel("expansion ratio")
    ax.legend(frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    style_ax(ax)

    ax = axes[1]
    ax.plot([1, 16], [0.165, 0.059], "-o", color=BLUE, markersize=3.4, linewidth=1.2, label="LoRA r8")
    ax.plot([1, 16], [0.112, 0.101], "-s", color=VERM, markersize=3.4, linewidth=1.2, label="full FT")
    ax.axhline(0.145, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(1.05, 0.150, "zero-shot", fontsize=9.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 16])
    ax.set_xticklabels(["1", "16"])
    ax.set_title("Dynamic (maze-dense, U-Net)", fontsize=10.0)
    ax.set_xlabel("K labeled maps")
    ax.legend(frameon=False, handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    style_ax(ax)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(HERE, "fig3_transfer.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
# C11: oracle headroom grows with K; learned halting moves the wrong way.
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))

    ax = axes[0]
    ks = [2, 4, 8]
    ax.plot(ks, [0.155, 0.121, 0.082], "-o", color=BLUE, markersize=3.4, linewidth=1.2, label="A maze/wp")
    ax.plot(ks, [0.225, 0.208, 0.103], "-s", color=VERM, markersize=3.4, linewidth=1.2, label="B rooms/wp")
    ax.plot(ks, [0.144, 0.128, 0.084], "-^", color=GREEN, markersize=3.4, linewidth=1.2, label="C keys/doors")
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.26)
    ax.set_title("Oracle/leg-sum ratio", fontsize=10.0)
    ax.set_xlabel("mission length K")
    ax.legend(frameon=False, handlelength=1.3, borderpad=0.1, labelspacing=0.25)
    style_ax(ax)

    ax = axes[1]
    bars = ax.bar([0, 1, 2], [6.79, 7.01, 5.30], width=0.62, color=VERM, zorder=3)
    for rect, v in zip(bars, [6.79, 7.01, 5.30]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, f"{v:.2f}",
                ha="center", fontsize=10.0, color=INK)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["K=2", "K=4", "K=8"])
    ax.set_ylim(0, 8.4)
    ax.set_title("Mean would-halt ACT steps (record level)", fontsize=10.0)
    style_ax(ax)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(HERE, "fig4_c11.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
# C13 ladder and C13-M per-suite confirmation.
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    rungs = [
        ("C: fresh certifier (oracle)", +11.50),
        ("D: shared queue (oracle)", -6.83),
        ("E: exact rollout rank", +1.33),
        ("F: monotone recalibration", +1.33),
        ("G: analytic local escape", +6.83),
        ("I: one-suite model, live", +15.36),
        ("J: suite-balanced, static", +16.17),
        ("K: + local Bellman backup", -1.21),
        ("L: + scale $\\alpha{=}1.5$", -13.04),
        ("M: 144-map confirmation", -12.96),
    ]
    ys = np.arange(len(rungs))[::-1]
    vals = [r[1] for r in rungs]
    cols = [BLUE if v < 0 else VERM for v in vals]
    ax.barh(ys, vals, height=0.62, color=cols, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.9)
    for y, (label, v) in zip(ys, rungs):
        ax.text(-0.6 if v >= 0 else 0.6, y, label, va="center",
                ha="right" if v >= 0 else "left", fontsize=9.5, color=INK)
    ax.set_yticks([])
    ax.set_xlim(-24, 26)
    ax.set_xlabel("$\\Delta$ expansions vs. matched comparator (lower is better)")
    ax.set_title("(a) Mechanism ladder", fontsize=10.0, loc="left")
    style_ax(ax, grid_axis="x")

    ax = axes[1]
    suites = ["Maze", "Dense maze", "Rooms", "Spiral", "Bugtrap", "Large rooms", "Pooled"]
    deltas = [-36.750, -9.125, -8.125, -1.083, -9.333, -13.333, -12.958]
    lo = [-45.583, -15.208, -14.333, -5.875, -17.792, -18.917, -16.299]
    hi = [-27.833, -3.125, -1.708, 4.208, -2.917, -7.500, -9.743]
    ys = np.arange(len(suites))[::-1]
    err_lo = [d - l for d, l in zip(deltas, lo)]
    err_hi = [h - d for d, h in zip(deltas, hi)]
    cols = [BLUE] * 6 + [INK]
    ax.errorbar(deltas, ys, xerr=[err_lo, err_hi], fmt="o", color=BLUE,
                ecolor=BLUE, elinewidth=1.0, capsize=2.0, markersize=3.6, zorder=3)
    ax.plot(deltas[-1], ys[-1], "D", color=INK, markersize=4.4, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(suites, fontsize=10.0)
    ax.set_xlabel("paired $\\Delta$ expansions vs. field HRM")
    ax.set_title("(b) Confirmation (144 maps)", fontsize=10.0, loc="left")
    style_ax(ax, grid_axis="x")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(HERE, "fig5_c13.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures written:", [f for f in sorted(os.listdir(HERE)) if f.endswith(".pdf")])
