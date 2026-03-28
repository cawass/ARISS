from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter


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

DEFAULT_FONT_SIZE = 12
DEFAULT_PAGE_FIGSIZE = (11.2, 4.8)
DEFAULT_DPI = 150
DEFAULT_SERIES_COLORS = [
    PALETTE["secondary_text"],
    PALETTE["l1_teal"],
    PALETTE["sernn_pink"],
    PALETTE["choice_mid"],
    PALETTE["goal_dark"],
    PALETTE["cat_purple"],
    PALETTE["cat_green"],
    PALETTE["cat_red"],
]

MPL_RC = {
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor": PALETTE["panel_bg"],
    "savefig.facecolor": PALETTE["background"],
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.labelcolor": PALETTE["primary_text"],
    "axes.titlesize": DEFAULT_FONT_SIZE,
    "axes.labelsize": DEFAULT_FONT_SIZE,
    "xtick.color": PALETTE["secondary_text"],
    "ytick.color": PALETTE["secondary_text"],
    "xtick.labelsize": DEFAULT_FONT_SIZE,
    "ytick.labelsize": DEFAULT_FONT_SIZE,
    "text.color": PALETTE["primary_text"],
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "legend.frameon": False,
    "legend.fontsize": DEFAULT_FONT_SIZE,
    "grid.color": PALETTE["light_grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
}


def apply_plot_style(
    *,
    font_size: float = DEFAULT_FONT_SIZE,
    figsize: tuple[float, float] = DEFAULT_PAGE_FIGSIZE,
) -> tuple[float, float]:
    plt.rcParams.update(MPL_RC)
    plt.rcParams["figure.figsize"] = figsize
    for key in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    ):
        plt.rcParams[key] = font_size
    return figsize


def style_axis(axis, *, grid: bool = True, boxed: bool = False) -> None:
    axis.set_facecolor(PALETTE["panel_bg"])
    if grid:
        axis.grid(True, color=PALETTE["light_grid"], linewidth=0.6, alpha=0.6)
    else:
        axis.grid(False)

    axis.tick_params(colors=PALETTE["secondary_text"], labelsize=DEFAULT_FONT_SIZE, width=0.8)
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


def summarize_series(
    series: Mapping[str, Sequence[Any]],
    *,
    converged: Sequence[bool] | None = None,
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for label, values in series.items():
        numeric = np.asarray(
            [float(value) if value is not None else float("nan") for value in values],
            dtype=float,
        )
        finite = numeric[np.isfinite(numeric)]
        stats: dict[str, float | int] = {
            "count": int(numeric.size),
            "finite_count": int(finite.size),
        }
        if finite.size > 0:
            stats["min"] = float(np.min(finite))
            stats["max"] = float(np.max(finite))
            stats["mean"] = float(np.mean(finite))
            stats["start"] = float(finite[0])
            stats["end"] = float(finite[-1])
        summary[label] = stats

    if converged is not None:
        total = len(converged)
        success = int(np.count_nonzero(np.asarray(converged, dtype=bool)))
        summary["_convergence"] = {
            "count": total,
            "finite_count": success,
            "min": float(success),
            "max": float(total),
            "mean": (100.0 * success / total) if total > 0 else 0.0,
        }

    return summary


def format_summary(summary: Mapping[str, Mapping[str, float | int]], *, max_items: int = 6) -> str:
    lines: list[str] = []
    shown = 0
    for label, stats in summary.items():
        if shown >= max_items:
            break
        if label == "_convergence":
            total = int(stats.get("count", 0))
            success = int(stats.get("finite_count", 0))
            percentage = float(stats.get("mean", 0.0))
            lines.append(f"Convergence: {success}/{total} ({percentage:.1f}%)")
            shown += 1
            continue

        finite_count = int(stats.get("finite_count", 0))
        if finite_count <= 0:
            lines.append(f"{label}: no finite values")
            shown += 1
            continue

        min_val = float(stats.get("min", float("nan")))
        max_val = float(stats.get("max", float("nan")))
        mean_val = float(stats.get("mean", float("nan")))
        lines.append(f"{label}: min={min_val:.3g}, max={max_val:.3g}, mean={mean_val:.3g}")
        shown += 1

    return "\n".join(lines)


def add_summary_box(
    axis,
    summary: Mapping[str, Mapping[str, float | int]],
    *,
    max_items: int = 6,
    fontsize: float = 10.5,
    location: tuple[float, float] = (0.98, 0.98),
) -> None:
    text = format_summary(summary, max_items=max_items)
    if not text:
        return
    axis.text(
        location[0],
        location[1],
        text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=PALETTE["secondary_text"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFFFFFCC", "edgecolor": PALETTE["light_grid"]},
    )


PLOT_GEOMETRY_ASPECT_RATIO_BARS = 0
PLOT_SENSITIVITY = 1
PLOT_MULTI_SENSITIVITY = 2
PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE = 3

DEFAULT_ORIGINAL_SENSITIVITY_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_GEOMETRY_ASPECT_RATIOS = [0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_GEOMETRY_CONFIGS = [
    {"label": "Sq intake / Sq body", "overrides": {"geometry.S_in": "s", "geometry.S_body": "s"}},
    {"label": "Sq intake / Circ body", "overrides": {"geometry.S_in": "s", "geometry.S_body": "c"}},
    {"label": "Circ intake / Circ body", "overrides": {"geometry.S_in": "c", "geometry.S_body": "c"}},
    {"label": "Circ intake / Sq body", "overrides": {"geometry.S_in": "c", "geometry.S_body": "s"}},
]

__all__ = [
    "PALETTE",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_PAGE_FIGSIZE",
    "DEFAULT_DPI",
    "DEFAULT_SERIES_COLORS",
    "apply_plot_style",
    "style_axis",
    "style_legend",
    "summarize_series",
    "format_summary",
    "add_summary_box",
    "PLOT_GEOMETRY_ASPECT_RATIO_BARS",
    "PLOT_SENSITIVITY",
    "PLOT_MULTI_SENSITIVITY",
    "PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE",
    "DEFAULT_ORIGINAL_SENSITIVITY_VALUES",
    "DEFAULT_GEOMETRY_ASPECT_RATIOS",
    "DEFAULT_GEOMETRY_CONFIGS",
    "build_original_sensitivity_cases",
    "run_original_sensitivity_cases",
    "build_geometry_sensitivity_cases",
    "run_geometry_sensitivity_cases",
    "plot_geometry_aspect_ratio_bars",
    "plot_sensitivity",
    "plot_multi_sensitivity",
    "plot_multi_sensitivity_side_by_side",
    "plot_by_index",
]


def _format_plain_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:g}"


def _apply_horizontal_grid_and_y_ticks(axis, *, log_y: bool) -> None:
    # Horizontal grid emphasis requested by user.
    axis.grid(False)

    if log_y:
        axis.yaxis.set_major_locator(LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        axis.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: (
                    "" if (not np.isfinite(value) or value <= 0.0) else _format_plain_number(value)
                )
            )
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: _format_plain_number(value))
        )

    axis.grid(which="major", axis="y", color="0.86", linewidth=0.8)
    axis.grid(which="minor", axis="y", color="0.92", linewidth=0.55)


def _compute_boundary_converged_point(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    converged_flags: Sequence[bool],
    *,
    log_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return one real converged point adjacent to the first failed block.

    Behavior:
    - If failures occur after convergence, return the last converged point
      immediately before the first failed block.
    - If failures are leading, return the first converged point immediately
      after that failed block.
    - Never fabricate y-values.
    """
    finite_x = np.isfinite(x_arr)
    finite_y = np.isfinite(y_arr)
    valid_y = finite_y & ((y_arr > 0.0) if log_y else True)

    converged = np.asarray(list(converged_flags), dtype=bool)
    failed = (~converged) & finite_x

    if not np.any(failed):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    failed_indices = np.where(failed)[0]

    # First contiguous failed block
    block_start = int(failed_indices[0])
    block_end = block_start
    for idx in failed_indices[1:]:
        if idx == block_end + 1:
            block_end = int(idx)
        else:
            break

    converged_valid = converged & finite_x & valid_y
    idx_all = np.arange(x_arr.size)

    # Prefer the last converged point before the failed block.
    left_candidates = np.where(converged_valid & (idx_all < block_start))[0]
    if left_candidates.size > 0:
        idx = int(left_candidates[-1])
        return (
            np.asarray([x_arr[idx]], dtype=float),
            np.asarray([y_arr[idx]], dtype=float),
        )

    # For leading failures, fall back to the first converged point after the block.
    right_candidates = np.where(converged_valid & (idx_all > block_end))[0]
    if right_candidates.size > 0:
        idx = int(right_candidates[0])
        return (
            np.asarray([x_arr[idx]], dtype=float),
            np.asarray([y_arr[idx]], dtype=float),
        )

    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def _case_x_values(case_data: Mapping[str, Any]) -> list[Any]:
    if "variable_values" in case_data:
        return list(case_data["variable_values"])
    return list(case_data.get("values", []))


def _case_x_label(case_data: Mapping[str, Any]) -> str:
    if "variable_path" in case_data:
        return str(case_data["variable_path"])
    return str(case_data.get("x_label", "x"))


def _first_output_path(data: Mapping[str, Any], output_path: str | None) -> str:
    if output_path is not None:
        return output_path
    output_paths = list(data.get("output_paths", []))
    if output_paths:
        return str(output_paths[0])
    outputs = data.get("outputs")
    if isinstance(outputs, Mapping) and outputs:
        return str(next(iter(outputs.keys())))
    raise KeyError("Could not infer output path from data.")


def build_original_sensitivity_cases(
    values: Sequence[float] | None = None,
    *,
    epsilon_path: str = "geometry.epsilon_body",
) -> list[dict[str, Any]]:
    sweep_values = [
        float(value)
        for value in (
            DEFAULT_ORIGINAL_SENSITIVITY_VALUES
            if values is None
            else values
        )
    ]
    definitions = [
        ("Thruster efficiency", "thruster.eff"),
        ("Collection efficiency", "refueling.coll_eff"),
        ("Accommodation coefficient", epsilon_path),
        ("Solar-cell efficiency", "solar.eta_solar"),
    ]

    cases: list[dict[str, Any]] = []
    for label, path in definitions:
        cases.append(
            {
                "label": label,
                "values": list(sweep_values),
                "x_label": "Parameter value [-]",
                "assign": (lambda x, p=path: {p: float(x)}),
            }
        )
    return cases


def run_original_sensitivity_cases(
    values: Sequence[float] | None = None,
    *,
    epsilon_path: str = "geometry.epsilon_body",
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ariss.core.sensitivity import sweep

    cases = build_original_sensitivity_cases(values, epsilon_path=epsilon_path)
    altitude_result = sweep(
        cases=cases,
        output_paths=["orbit.altitude"],
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
        mode="direct",
    )
    refuel_result = sweep(
        cases=cases,
        output_paths=["refueling.t_refuel"],
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
        mode="refuel_search",
    )
    return altitude_result, refuel_result


def build_geometry_sensitivity_cases(
    aspect_ratios: Sequence[float] | None = None,
    geometry_configs: Sequence[Mapping[str, Any]] | None = None,
    *,
    active_refueling: bool = False,
) -> tuple[list[dict[str, Any]], list[float]]:
    ars = [
        float(value)
        for value in (
            DEFAULT_GEOMETRY_ASPECT_RATIOS
            if aspect_ratios is None
            else aspect_ratios
        )
    ]
    configs = list(DEFAULT_GEOMETRY_CONFIGS) if geometry_configs is None else list(geometry_configs)

    cases: list[dict[str, Any]] = []
    for config in configs:
        label = str(config.get("label", "Geometry"))
        overrides = dict(config.get("overrides", {}))
        overrides["mission_profile.active_refueling"] = bool(active_refueling)
        cases.append(
            {
                "label": label,
                "values": list(ars),
                "x_label": "Aspect ratio [-]",
                "assign": (
                    lambda ar, ov=overrides: {
                        **ov,
                        "geometry.AR_in": float(ar),
                        "geometry.AR_body": float(ar),
                    }
                ),
            }
        )
    return cases, ars


def run_geometry_sensitivity_cases(
    aspect_ratios: Sequence[float] | None = None,
    geometry_configs: Sequence[Mapping[str, Any]] | None = None,
    *,
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> dict[str, dict[str, Any]]:
    from ariss.core.sensitivity import sweep

    cases_no_refuel, ars = build_geometry_sensitivity_cases(
        aspect_ratios=aspect_ratios,
        geometry_configs=geometry_configs,
        active_refueling=False,
    )
    cases_refuel, _ = build_geometry_sensitivity_cases(
        aspect_ratios=aspect_ratios,
        geometry_configs=geometry_configs,
        active_refueling=True,
    )

    no_refuel = sweep(
        cases=cases_no_refuel,
        output_paths=["orbit.altitude"],
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
        mode="direct",
    )
    with_refuel = sweep(
        cases=cases_refuel,
        output_paths=["refueling.t_refuel"],
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
        mode="refuel_search",
    )

    labels = [str(case["label"]) for case in cases_no_refuel]

    return {
        "without_refueling": {
            "active_refueling": False,
            "geometry_labels": labels,
            "aspect_ratios": ars,
            "values": [
                [None if value is None else float(value) for value in no_refuel["cases"][label]["outputs"]["orbit.altitude"]]
                for label in labels
            ],
            "value_label": "Orbit altitude [km]",
            "converged": [list(no_refuel["cases"][label]["converged"]) for label in labels],
            "errors": [list(no_refuel["cases"][label]["errors"]) for label in labels],
        },
        "with_refueling": {
            "active_refueling": True,
            "geometry_labels": labels,
            "aspect_ratios": ars,
            "values": [
                [None if value is None else float(value) for value in with_refuel["cases"][label]["outputs"]["refueling.t_refuel"]]
                for label in labels
            ],
            "value_label": "Minimum refueling time [months]",
            "converged": [list(with_refuel["cases"][label]["converged"]) for label in labels],
            "errors": [list(with_refuel["cases"][label]["errors"]) for label in labels],
        },
    }


def plot_geometry_aspect_ratio_bars(
    sensitivity_data: Mapping[str, Mapping[str, Any]],
    *,
    show: bool = True,
    title_left: str = "No Refueling",
    title_right: str = "With Refueling",
):
    # Inputs:
    #   sensitivity_data: output of run_geometry_aspect_ratio_sensitivity.
    #   show: if True display figure.
    #   title_left/title_right: subplot titles.
    #
    # Output:
    #   (figure, axes) matplotlib objects.

    required_keys = {"without_refueling", "with_refueling"}
    if not required_keys.issubset(set(sensitivity_data.keys())):
        missing = required_keys - set(sensitivity_data.keys())
        raise KeyError(f"sensitivity_data missing required modes: {sorted(missing)}")

    apply_plot_style()

    left = sensitivity_data["without_refueling"]
    right = sensitivity_data["with_refueling"]

    geometry_labels = list(left["geometry_labels"])
    aspect_ratios = list(left["aspect_ratios"])
    if geometry_labels != list(right["geometry_labels"]):
        raise ValueError("Geometry labels differ between refueling modes.")
    if aspect_ratios != list(right["aspect_ratios"]):
        raise ValueError("Aspect-ratio lists differ between refueling modes.")

    n_groups = len(geometry_labels)
    n_ar = len(aspect_ratios)
    if n_groups == 0 or n_ar == 0:
        raise ValueError("No geometry/aspect-ratio data available for plotting.")

    x_centers = np.arange(n_groups, dtype=float)
    group_width = 0.82
    bar_width = group_width / max(n_ar, 1)

    fig = plt.figure(figsize=(11.2, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.45))
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    bar_colors = [
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_yellow", "#DCCB4F"),
        PALETTE.get("cat_red", "#E85C62"),
    ]

    def draw_mode(axis, mode_data: Mapping[str, Any], subplot_title: str) -> bool:
        value_grid = mode_data["values"]
        converged_grid = mode_data["converged"]

        non_converged_present = False
        finite_values: list[float] = []
        for row in value_grid:
            for value in row:
                if value is not None and np.isfinite(value):
                    finite_values.append(float(value))

        for ar_idx, ar_value in enumerate(aspect_ratios):
            x_offset = (ar_idx - 0.5 * (n_ar - 1)) * bar_width
            x_positions = x_centers + x_offset

            heights: list[float] = []
            for group_idx in range(n_groups):
                value = value_grid[group_idx][ar_idx]
                heights.append(np.nan if value is None else float(value))

            axis.bar(
                x_positions,
                np.asarray(heights, dtype=float),
                width=bar_width * 0.92,
                color=bar_colors[ar_idx % len(bar_colors)],
                edgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidth=0.6,
                zorder=2,
            )

            for group_idx, x_pos in enumerate(x_positions):
                if not bool(converged_grid[group_idx][ar_idx]):
                    non_converged_present = True
                    if finite_values:
                        marker_y = float(min(finite_values)) - 0.04 * max(
                            float(max(finite_values) - min(finite_values)),
                            1.0,
                        )
                    else:
                        marker_y = 0.0
                    axis.scatter(
                        [x_pos],
                        [marker_y],
                        marker="s",
                        s=28,
                        facecolors=PALETTE.get("cat_red", "#E85C62"),
                        edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                        linewidths=0.7,
                        zorder=4,
                    )

        axis.set_xticks(x_centers)
        two_line_labels = [label.replace(" / ", "\n") for label in geometry_labels]
        axis.set_xticklabels(two_line_labels)
        axis.set_xlabel("Geometry configuration")
        axis.set_title(subplot_title)
        style_axis(axis)
        _apply_horizontal_grid_and_y_ticks(axis, log_y=False)

        if finite_values:
            y_min = float(min(finite_values))
            y_max = float(max(finite_values))
            spread = max(y_max - y_min, 1.0)
            low_pad = 0.10 * spread
            high_pad = 0.08 * spread
            axis.set_ylim(y_min - low_pad, y_max + high_pad)

        return non_converged_present

    left_has_non_converged = draw_mode(ax_left, left, title_left)
    right_has_non_converged = draw_mode(ax_right, right, title_right)
    ax_left.set_ylabel(str(left["value_label"]))
    ax_right.set_ylabel(str(right["value_label"]))

    legend_handles = [
        Patch(
            facecolor=bar_colors[idx % len(bar_colors)],
            edgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidth=0.6,
            label=f"AR = {float(ar_value):g}",
        )
        for idx, ar_value in enumerate(aspect_ratios)
    ]

    if left_has_non_converged or right_has_non_converged:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=7,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.8,
                label="Non-converged",
            )
        )

    legend = legend_axis.legend(
        handles=legend_handles,
        title="Aspect Ratio",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=1.8,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.92, wspace=0.20)

    if show:
        plt.show(block=True)

    return fig, (ax_left, ax_right)


def plot_sensitivity(
    result: Mapping[str, Any],
    output_path: str | None = None,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    log_y: bool = True,
    show: bool = True,
):
    # Inputs:
    #   result: output from run_sensitivity/run_sensitivity_analysis.
    #   output_path: output variable path to plot (defaults to first output path).
    #   x_label, y_label, title: optional custom labels/title.
    #   show: if True, display the figure.
    #
    # Output:
    #   (fig, ax) matplotlib objects.

    apply_plot_style()

    selected_output = _first_output_path(result, output_path)
    if selected_output not in result["outputs"]:
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {list(result['outputs'].keys())}"
        )

    x_vals = _case_x_values(result)
    y_vals = list(result["outputs"][selected_output])
    y_numeric = [float(v) if v is not None else float("nan") for v in y_vals]

    fig = plt.figure(figsize=(11.2, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34))
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_numeric, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if log_y:
        valid &= y_arr > 0.0

    ax.plot(
        x_arr[valid],
        y_arr[valid],
        color=PALETTE.get("secondary_text", "#4A4A4A"),
        linewidth=2.2,
        label="ARISS full loop",
    )
    ax.scatter(
        x_arr[valid],
        y_arr[valid],
        color=PALETTE.get("l1_teal", "#5BC8D0"),
        edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
        linewidths=0.8,
        s=30,
        zorder=3,
        label="Sweep points",
    )

    boundary_x, boundary_y = _compute_boundary_converged_point(
        x_arr,
        y_arr,
        result["converged"],
        log_y=log_y,
    )
    if boundary_x.size > 0:
        ax.scatter(
            boundary_x,
            boundary_y,
            marker="s",
            s=42,
            facecolors=PALETTE.get("cat_red", "#E85C62"),
            edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidths=0.8,
            zorder=4,
            label="Last converged",
        )

    ax.set_xlabel(x_label or _case_x_label(result))
    if y_label is not None:
        ax.set_ylabel(y_label)
    elif selected_output == "refueling.t_refuel":
        ax.set_ylabel("refueling.t_refuel [months]")
    else:
        ax.set_ylabel(selected_output)
    if title is not None:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")

    style_axis(ax)
    _apply_horizontal_grid_and_y_ticks(ax, log_y=log_y)
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_axis.legend(
        handles,
        labels,
        title="Curves",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=0.05)

    if show:
        plt.show(block=True)

    return fig, ax


def plot_multi_sensitivity(
    multi_result: Mapping[str, Any],
    output_path: str | None = None,
    *,
    x_label: str | None = "Parameter value [-]",
    y_label: str | None = None,
    title: str | None = None,
    log_y: bool = True,
    show: bool = True,
):
    # Overlay multiple sensitivity sweeps on one figure.
    apply_plot_style()

    selected_output = _first_output_path(multi_result, output_path)
    if selected_output not in multi_result.get("output_paths", [selected_output]):
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {multi_result['output_paths']}"
        )

    fig = plt.figure(figsize=(11.2, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34))
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    color_cycle = [
        PALETTE.get("secondary_text", "#4A4A4A"),
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_red", "#E85C62"),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<"]
    boundary_legend_added = False

    for idx, (label, result) in enumerate(multi_result["cases"].items()):
        if selected_output not in result["outputs"]:
            raise KeyError(
                f"Case '{label}' does not contain output '{selected_output}'. "
                f"Available: {list(result['outputs'].keys())}"
            )

        x_vals = _case_x_values(result)
        y_vals = result["outputs"][selected_output]
        y_numeric = [float(v) if v is not None else float("nan") for v in y_vals]
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_numeric, dtype=float)
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        if log_y:
            valid &= y_arr > 0.0
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]

        ax.plot(
            x_arr[valid],
            y_arr[valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            label=label,
        )

        boundary_x, boundary_y = _compute_boundary_converged_point(
            x_arr,
            y_arr,
            result["converged"],
            log_y=log_y,
        )
        if boundary_x.size > 0:
            boundary_label = "Last converged" if not boundary_legend_added else "_nolegend_"
            boundary_legend_added = True
            ax.scatter(
                boundary_x,
                boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
                label=boundary_label,
            )

    ax.set_xlabel(x_label or "Parameter value [-]")
    if y_label is not None:
        ax.set_ylabel(y_label)
    elif selected_output == "refueling.t_refuel":
        ax.set_ylabel("refueling.t_refuel [months]")
    else:
        ax.set_ylabel(selected_output)
    if title is not None:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")

    style_axis(ax)
    _apply_horizontal_grid_and_y_ticks(ax, log_y=log_y)
    handles, labels = ax.get_legend_handles_labels()
    legend = legend_axis.legend(
        handles,
        labels,
        title="Sensitivity Cases",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=0.05)

    if show:
        plt.show(block=True)

    return fig, ax


def plot_multi_sensitivity_side_by_side(
    multi_result: Mapping[str, Any],
    right_multi_result: Mapping[str, Any] | None = None,
    *,
    left_output: str = "orbit.altitude",
    right_output: str = "refueling.t_refuel",
    left_log_y: bool = False,
    right_log_y: bool = True,
    left_title: str = "Orbit altitude",
    right_title: str = "Refueling time",
    x_label: str = "Parameter value [-]",
    show: bool = True,
):
    # Plot two outputs side by side with one shared legend.
    apply_plot_style()

    left_result = multi_result
    right_result = multi_result if right_multi_result is None else right_multi_result

    left_paths = left_result.get("output_paths", [])
    right_paths = right_result.get("output_paths", [])
    if left_paths and left_output not in left_paths:
        raise KeyError(f"Unknown left_output '{left_output}'. Available: {left_paths}")
    if right_paths and right_output not in right_paths:
        raise KeyError(f"Unknown right_output '{right_output}'. Available: {right_paths}")

    left_labels = list(left_result["cases"].keys())
    right_labels = list(right_result["cases"].keys())
    if set(left_labels) != set(right_labels):
        raise ValueError(
            "Left and right sensitivity results must have identical case labels. "
            f"Left={left_labels}, Right={right_labels}"
        )

    fig = plt.figure(figsize=(12.6, 5.4), dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.46))
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    color_cycle = [
        PALETTE.get("secondary_text", "#4A4A4A"),
        PALETTE.get("l1_teal", "#5BC8D0"),
        PALETTE.get("sernn_pink", "#F08FA7"),
        PALETTE.get("choice_mid", "#C59A4A"),
        PALETTE.get("goal_dark", "#1E7F78"),
        PALETTE.get("cat_purple", "#9A5CB8"),
        PALETTE.get("cat_green", "#76C56E"),
        PALETTE.get("cat_red", "#E85C62"),
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<"]

    boundary_present = False
    case_handles: list[Line2D] = []

    for idx, label in enumerate(left_labels):
        left_case = left_result["cases"][label]
        right_case = right_result["cases"][label]
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]
        case_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.0,
                marker=marker,
                markersize=5.0,
                markerfacecolor=color,
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.6,
                label=label,
            )
        )

        left_x = np.asarray(_case_x_values(left_case), dtype=float)
        left_y_vals = left_case["outputs"][left_output]
        left_y = np.asarray([float(v) if v is not None else float("nan") for v in left_y_vals], dtype=float)
        left_valid = np.isfinite(left_x) & np.isfinite(left_y)
        if left_log_y:
            left_valid &= left_y > 0.0

        ax_left.plot(
            left_x[left_valid],
            left_y[left_valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            zorder=2,
        )

        left_boundary_x, left_boundary_y = _compute_boundary_converged_point(
            left_x,
            left_y,
            left_case["converged"],
            log_y=left_log_y,
        )
        if left_boundary_x.size > 0:
            boundary_present = True
            ax_left.scatter(
                left_boundary_x,
                left_boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
            )

        right_x = np.asarray(_case_x_values(right_case), dtype=float)
        right_y_vals = right_case["outputs"][right_output]
        right_y = np.asarray([float(v) if v is not None else float("nan") for v in right_y_vals], dtype=float)
        right_valid = np.isfinite(right_x) & np.isfinite(right_y)
        if right_log_y:
            right_valid &= right_y > 0.0

        ax_right.plot(
            right_x[right_valid],
            right_y[right_valid],
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=0.6,
            zorder=2,
        )

        right_boundary_x, right_boundary_y = _compute_boundary_converged_point(
            right_x,
            right_y,
            right_case["converged"],
            log_y=right_log_y,
        )
        if right_boundary_x.size > 0:
            boundary_present = True
            ax_right.scatter(
                right_boundary_x,
                right_boundary_y,
                marker="s",
                s=36,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=0.75,
                zorder=5,
            )

    ax_left.set_xlabel(x_label)
    ax_right.set_xlabel(x_label)
    ax_left.set_ylabel("Orbit altitude [km]" if left_output == "orbit.altitude" else left_output)
    ax_right.set_ylabel("Refueling time [months]" if right_output == "refueling.t_refuel" else right_output)
    ax_left.set_title(left_title)
    ax_right.set_title(right_title)

    if left_log_y:
        ax_left.set_yscale("log")
    if right_log_y:
        ax_right.set_yscale("log")

    style_axis(ax_left)
    style_axis(ax_right)
    _apply_horizontal_grid_and_y_ticks(ax_left, log_y=left_log_y)
    _apply_horizontal_grid_and_y_ticks(ax_right, log_y=right_log_y)

    legend_handles: list[Any] = list(case_handles)
    if boundary_present:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=7,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.8,
                label="Last converged",
            )
        )

    legend = legend_axis.legend(
        handles=legend_handles,
        title="Sensitivity Cases",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.92, wspace=0.22)

    if show:
        plt.show(block=True)

    return fig, (ax_left, ax_right)



def plot_by_index(plot_index: int, data: Any, /, **kwargs):
    if plot_index == PLOT_GEOMETRY_ASPECT_RATIO_BARS:
        return plot_geometry_aspect_ratio_bars(data, **kwargs)

    if plot_index == PLOT_SENSITIVITY:
        return plot_sensitivity(data, **kwargs)

    if plot_index == PLOT_MULTI_SENSITIVITY:
        return plot_multi_sensitivity(data, **kwargs)

    if plot_index == PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE:
        if isinstance(data, (tuple, list)):
            if len(data) == 0:
                raise ValueError("For plot index 3, provide at least one multi-sensitivity result dictionary.")
            left = data[0]
            right = data[1] if len(data) > 1 else None
            return plot_multi_sensitivity_side_by_side(left, right_multi_result=right, **kwargs)

        return plot_multi_sensitivity_side_by_side(data, **kwargs)

    raise ValueError(
        f"Unknown plot_index {plot_index}. "
        "Valid indices: 0 (geometry bars), 1 (single sensitivity), "
        "2 (multi sensitivity), 3 (side-by-side multi sensitivity)."
    )

