"""Publication-quality plotting utilities for Nature Computational Science."""

from .nature_style import set_nature_style, NATURE_COLORS, NATURE_PALETTE
from .nature_style import nature_single_col, nature_double_col

__all__ = [
    "set_nature_style", "NATURE_COLORS", "NATURE_PALETTE",
    "nature_single_col", "nature_double_col",
]
