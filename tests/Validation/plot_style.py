from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "primary_text": "#111111",
    "secondary_text": "#4A4A4A",
    "muted_text": "#7A7A7A",
    "background": "#FFFFFF",
    "panel_bg": "#FFFFFF",
    "light_grid": "#D8D8D8",
    "mid_grid": "#BEBEBE",
    "node_gray": "#B8B8B8",
    "edge_gray": "#9A9A9A",
    "sernn_pink": "#F08FA7",
    "sernn_pink_fill": "#F6B8C5",
    "l1_teal": "#5BC8D0",
    "l1_teal_fill": "#A8E3E4",
    "goal_dark": "#1E7F78",
    "goal_mid": "#6FC6D2",
    "goal_light": "#D9F4F2",
    "choice_dark": "#8A5A12",
    "choice_mid": "#C59A4A",
    "choice_light": "#F1E4C8",
    "sweet_spot_pink": "#F39AAA",
    "zone_orange": "#DDA57D",
    "zone_blue": "#AFC8E2",
    "cat_yellow": "#DCCB4F",
    "cat_green": "#76C56E",
    "cat_purple": "#9A5CB8",
    "cat_red": "#E85C62",
}

MPL_RC = {
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#111111",
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.color": "#111111",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "legend.frameon": False,
    "legend.fontsize": 12,
    "grid.color": "#D8D8D8",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


def apply_validation_style() -> None:
    plt.rcParams.update(MPL_RC)


def style_axis(axis, *, grid: bool = True, boxed: bool = False) -> None:
    axis.set_facecolor(PALETTE["panel_bg"])
    if grid:
        axis.grid(True, color=PALETTE["light_grid"], linewidth=0.6, alpha=0.6)
    else:
        axis.grid(False)

    axis.tick_params(colors=PALETTE["secondary_text"], labelsize=12, width=0.8)
    axis.xaxis.label.set_color(PALETTE["primary_text"])
    axis.yaxis.label.set_color(PALETTE["primary_text"])
    axis.title.set_color(PALETTE["primary_text"])

    if boxed:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#444444")
            spine.set_linewidth(0.8)
    else:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#444444")
        axis.spines["bottom"].set_color("#444444")
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_color(PALETTE["secondary_text"])

