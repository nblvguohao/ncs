#!/usr/bin/env python3
"""
Figure 1: Paradigm-shift schematic — "Homology Illusion vs Scientific Reality"

A 3-panel double-column figure:
  (a) Schematic: random split allows homologous leakage → inflated AUC
  (b) Leakage gradient: AUC decays smoothly with increasing stringency
  (c) Remediation: BW-site physicochemical features recover genuine signal

Nature Computational Science format (183mm wide, vector PDF + 600dpi PNG).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.plotting.nature_style import (
    set_nature_style, NATURE_COLORS as NC, DOUBLE_COL_WIDTH,
    add_panel_label, save_nature_fig,
)

OUT_DIR = os.path.join(PROJECT_DIR, "figures", "nature")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    set_nature_style()

    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 4.8))

    # Layout: top row = (a) schematic spanning full width
    #         bottom row = (b) leakage gradient | (c) remediation bar
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.07, right=0.97, top=0.95, bottom=0.08)

    # ── Panel (a): Schematic — homology illusion ──────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 3)
    ax_a.axis("off")
    add_panel_label(ax_a, "a", x=-0.03, y=1.05)

    # Left: Random split (leaky)
    box_leak = FancyBboxPatch((0.3, 0.5), 3.8, 2.0, boxstyle="round,pad=0.15",
                               facecolor="#FFE0E0", edgecolor=NC["red"], linewidth=1.2)
    ax_a.add_patch(box_leak)
    ax_a.text(2.2, 2.2, "Random Split", fontsize=8, fontweight="bold",
              ha="center", color=NC["red"])
    ax_a.text(2.2, 1.6, "Homologous sequences\nin train & test", fontsize=6,
              ha="center", color="#666666")
    ax_a.text(2.2, 0.9, "AUC = 0.835", fontsize=9, fontweight="bold",
              ha="center", color=NC["red"])
    ax_a.text(2.2, 0.6, '"Performance Illusion"', fontsize=6, fontstyle="italic",
              ha="center", color=NC["red"])

    # Arrow
    ax_a.annotate("", xy=(5.8, 1.5), xytext=(4.3, 1.5),
                  arrowprops=dict(arrowstyle="-|>", color=NC["black"],
                                  lw=1.5, mutation_scale=15))
    ax_a.text(5.05, 1.85, "Leakage\nAudit", fontsize=6, ha="center",
              fontweight="bold", color=NC["black"])
    ax_a.text(5.05, 1.1, "ΔAUC = 0.24", fontsize=7, ha="center",
              fontweight="bold", color=NC["orange"])

    # Right: No-leak split (real)
    box_real = FancyBboxPatch((5.9, 0.5), 3.8, 2.0, boxstyle="round,pad=0.15",
                               facecolor="#E0F0FF", edgecolor=NC["blue"], linewidth=1.2)
    ax_a.add_patch(box_real)
    ax_a.text(7.8, 2.2, "Subfamily-Grouped CV", fontsize=8, fontweight="bold",
              ha="center", color=NC["blue"])
    ax_a.text(7.8, 1.6, "Zero subfamily overlap\nbetween train & test", fontsize=6,
              ha="center", color="#666666")
    ax_a.text(7.8, 0.9, "AUC = 0.599", fontsize=9, fontweight="bold",
              ha="center", color=NC["blue"])
    ax_a.text(7.8, 0.6, '"Scientific Reality"', fontsize=6, fontstyle="italic",
              ha="center", color=NC["blue"])

    # ── Panel (b): Leakage gradient ──────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    add_panel_label(ax_b, "b", x=-0.18, y=1.08)

    strategies = ["Random\nCV", "Seq 0.6", "Seq 0.5", "Seq 0.4",
                  "Seq 0.3", "Seq 0.2", "Subfamily\nCV"]
    bw_aucs = [0.835, 0.839, 0.840, 0.820, 0.787, 0.766, 0.599]
    esm_aucs = [0.878, 0.851, 0.860, 0.824, 0.809, 0.777, 0.623]

    x = np.arange(len(strategies))
    ax_b.plot(x, bw_aucs, "o-", color=NC["blue"], markersize=4,
              label="BW-site (145d)", linewidth=1.2, zorder=3)
    ax_b.plot(x, esm_aucs, "s--", color=NC["red"], markersize=4,
              label="ESM-2 (9280d)", linewidth=1.0, zorder=3)

    # Fill gradient zone
    ax_b.fill_between(x, bw_aucs, 0.5, alpha=0.08, color=NC["blue"])
    ax_b.axhline(0.5, color=NC["lightgrey"], ls="--", lw=0.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(strategies, fontsize=5, rotation=0)
    ax_b.set_ylabel("AUC-ROC")
    ax_b.set_ylim(0.45, 0.95)
    ax_b.set_title("Leakage gradient", fontsize=7, fontweight="bold")
    ax_b.legend(fontsize=5, loc="lower left")

    # Annotation: gradient arrow
    ax_b.annotate("", xy=(6, 0.60), xytext=(0, 0.84),
                  arrowprops=dict(arrowstyle="->", color=NC["orange"],
                                  lw=1.5, connectionstyle="arc3,rad=0.15"))
    ax_b.text(3, 0.68, "Increasing\nstringency", fontsize=5,
              ha="center", color=NC["orange"], fontstyle="italic")

    # ── Panel (c): Remediation bar chart ─────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    add_panel_label(ax_c, "c", x=-0.18, y=1.08)

    features = ["Handcrafted\n(99d)", "ESM-2\n(320d)", "Combined\n(244d)",
                "BW-site\n(145d)"]
    noleak_aucs = [0.487, 0.623, 0.521, 0.599]
    colors = [NC["grey"], NC["red"], NC["orange"], NC["blue"]]

    bars = ax_c.bar(range(4), noleak_aucs, color=colors,
                    edgecolor="white", linewidth=0.5)
    ax_c.axhline(0.5, color=NC["lightgrey"], ls="--", lw=0.5)

    # Value labels
    for i, v in enumerate(noleak_aucs):
        ax_c.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6,
                  fontweight="bold" if i == 3 else "normal")

    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(features, fontsize=5.5)
    ax_c.set_ylabel("AUC-ROC (no-leak CV)")
    ax_c.set_ylim(0.35, 0.72)
    ax_c.set_title("Structure-aligned remediation", fontsize=7, fontweight="bold")

    # Star on BW-site bar
    ax_c.text(3, noleak_aucs[3] + 0.04, "*", ha="center", fontsize=10,
              fontweight="bold", color=NC["blue"])

    save_nature_fig(fig, os.path.join(OUT_DIR, "fig1_paradigm_shift"))
    print("Done: fig1_paradigm_shift.pdf + .png")


if __name__ == "__main__":
    main()
