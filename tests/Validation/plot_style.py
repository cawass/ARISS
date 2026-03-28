from __future__ import annotations

from ariss.utils.ploting import (
    DEFAULT_DPI,
    DEFAULT_FONT_SIZE,
    DEFAULT_PAGE_FIGSIZE,
    DEFAULT_SERIES_COLORS,
    MPL_RC,
    PALETTE,
    add_summary_box,
    apply_plot_style,
    format_summary,
    style_axis,
    style_legend,
    summarize_series,
)


def apply_validation_style() -> None:
    apply_plot_style(font_size=DEFAULT_FONT_SIZE, figsize=DEFAULT_PAGE_FIGSIZE)


__all__ = [
    "PALETTE",
    "MPL_RC",
    "DEFAULT_DPI",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_PAGE_FIGSIZE",
    "DEFAULT_SERIES_COLORS",
    "apply_validation_style",
    "apply_plot_style",
    "style_axis",
    "style_legend",
    "summarize_series",
    "format_summary",
    "add_summary_box",
]

