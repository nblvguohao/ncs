"""
Nature Computational Science figure style configuration.

Specifications (Nature portfolio):
  - Single column: 89 mm (3.50 in)
  - Double column: 183 mm (7.20 in)
  - Max height: 247 mm (9.72 in)
  - Min font size: 5 pt (axis ticks), 7 pt (labels), 8 pt (panel labels)
  - Resolution: 300 dpi (halftone), 600 dpi (line art), vector preferred
  - Fonts: Helvetica / Arial (sans-serif)
  - File formats: PDF (vector), TIFF/EPS for raster
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Nature-inspired colour palette ────────────────────────────────────────
NATURE_COLORS = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "red":       "#D55E00",
    "purple":    "#CC79A7",
    "cyan":      "#56B4E9",
    "yellow":    "#F0E442",
    "grey":      "#999999",
    "black":     "#000000",
    "lightgrey": "#CCCCCC",
}

NATURE_PALETTE = [
    NATURE_COLORS["blue"],
    NATURE_COLORS["orange"],
    NATURE_COLORS["green"],
    NATURE_COLORS["red"],
    NATURE_COLORS["purple"],
    NATURE_COLORS["cyan"],
    NATURE_COLORS["yellow"],
    NATURE_COLORS["grey"],
]

# ── Column widths in inches ───────────────────────────────────────────────
SINGLE_COL_WIDTH = 3.50    # 89 mm
DOUBLE_COL_WIDTH = 7.20    # 183 mm
MAX_HEIGHT = 9.72          # 247 mm


def set_nature_style():
    """Apply Nature-style matplotlib rcParams globally."""
    style = {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,

        # Lines and markers
        "lines.linewidth": 1.0,
        "lines.markersize": 3,

        # Axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=NATURE_PALETTE),

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid (off by default for Nature)
        "axes.grid": False,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,

        # PDF
        "pdf.fonttype": 42,       # TrueType (editable in Illustrator)
        "ps.fonttype": 42,

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
    }
    mpl.rcParams.update(style)


def nature_single_col(height_ratio=0.75):
    """Create a single-column figure with Nature dimensions."""
    set_nature_style()
    h = SINGLE_COL_WIDTH * height_ratio
    return plt.figure(figsize=(SINGLE_COL_WIDTH, h))


def nature_double_col(height_ratio=0.45):
    """Create a double-column figure with Nature dimensions."""
    set_nature_style()
    h = DOUBLE_COL_WIDTH * height_ratio
    return plt.figure(figsize=(DOUBLE_COL_WIDTH, h))


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=10):
    """Add bold panel label (a, b, c...) in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


def save_nature_fig(fig, path, formats=("pdf", "png")):
    """Save figure in multiple formats suitable for Nature submission."""
    import os
    base, _ = os.path.splitext(path)
    for fmt in formats:
        out = f"{base}.{fmt}"
        dpi = 600 if fmt == "png" else 300
        fig.savefig(out, dpi=dpi, format=fmt, bbox_inches="tight",
                    pad_inches=0.02)
        print(f"  Saved: {out}")
    plt.close(fig)
