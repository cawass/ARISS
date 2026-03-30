# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS - Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      
# This module is a centralized collection of plotting utilities, styles, and templates used
#
#  Project:        ARISS
#  Module:         atmosphere.py
#  Author:         Carlos Carrasco Requejo
# ============================================================================


from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, MultipleLocator, NullFormatter


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
DEFAULT_PAGE_FIGSIZE = (13.44, 4.8)
DEFAULT_DPI = 150
GLOBAL_FONT_SCALE = 0.85
GLOBAL_FIGURE_WIDTH_SCALE = 1.15
VALIDATION_FIGURE_WIDTH_SCALE = 1.265
CRANDALL_WSPACE_SCALE = 3.99
BASE_VALIDATION_WSPACE = 0.07
CRANDALL_VALIDATION_WSPACE = BASE_VALIDATION_WSPACE * CRANDALL_WSPACE_SCALE
UNIFORM_MARKER_SIZE = 6.0
UNIFORM_MARKER_EDGE_WIDTH = 1.0

GOCEE_DRAG_FONT_SIZE = DEFAULT_FONT_SIZE * 1.30 * 1.20
CRANDALL_DRAG_FONT_SIZE = DEFAULT_FONT_SIZE * 1.66
CRANDALL_FIG11_FONT_SIZE = DEFAULT_FONT_SIZE * 1.66
CRANDALL_FIG2627_FONT_SIZE = DEFAULT_FONT_SIZE * 1.33
MANSUR_EFFICIENCY_FONT_SIZE = DEFAULT_FONT_SIZE * 1.35
MANSUR_ENVELOPE_FONT_SIZE = DEFAULT_FONT_SIZE * 1.35
MANSUR_THRUSTER_MAP_FONT_SIZE = DEFAULT_FONT_SIZE * 1.25
MANSUR_FONT_SCALE = 1.33
MANSUR_ENVELOPE_LEGEND_RIGHT_SHIFT = 0.25
MANSUR_MARKER_SCALE = 1.75
MANSUR_ENVELOPE_SOURCE_X_SCALE = 1.00
MANSUR_ENVELOPE_SOURCE_Y_SCALE = 1.05
MANSUR_THRUSTER_FAMILIES_ANCHOR_Y = 0.66
MANSUR_THRUSTER_FAMILIES_NCOL = 3
MANSUR_THRUSTER_FAMILIES_COLUMN_SPACING = 1.12
MANSUR_THRUSTER_LEGEND_X_SHIFT = 0.05
MANSUR_THRUSTER_FIGSIZE_SCALE = 1.08
MANSUR_THRUSTER_ETA_LEVELS_ANCHOR_Y = 0.42 * 0.85
MANSUR_THRUSTER_MDOT_ANCHOR_Y = min(0.98, 0.88 * 1.15)
MANSUR_THRUSTER_AIN_ANCHOR_Y = 0.42 * 0.85
SENSITIVITY_CASES_WSPACE_SCALE = 1.50
SENSITIVITY_MULTI_WSPACE = 0.05 * SENSITIVITY_CASES_WSPACE_SCALE
SENSITIVITY_SIDE_BY_SIDE_WSPACE = 0.20 * SENSITIVITY_CASES_WSPACE_SCALE
SENSITIVITY_REFUELING_LABEL_X = -0.12
GEOMETRY_SENSITIVITY_FONT_SIZE = DEFAULT_FONT_SIZE * 1.10
PARAMETER_SENSITIVITY_FONT_SIZE = DEFAULT_FONT_SIZE * 1.50
PARAMETER_SENSITIVITY_PAPER_FONT_SIZE = 9.5 * 1.50
MANSUR_EXTRA_WIDTH_SCALE = 1.15
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
    "lines.markersize": UNIFORM_MARKER_SIZE,
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

    axis.tick_params(colors=PALETTE["secondary_text"], width=0.8)
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


VALIDATION_MAJOR_GRID_COLOR = "0.88"
VALIDATION_MINOR_GRID_COLOR = "0.94"


def style_validation_axis(
    axis,
    *,
    x_minor_divisions: int | None = 2,
    y_minor_divisions: int | None = 2,
    black_axes: bool = False,
) -> None:
    # Shared style for validation figures so all cases look consistent.
    style_axis(axis)
    axis.tick_params(axis="both", which="both")

    if x_minor_divisions is not None and axis.get_xscale() == "linear":
        axis.xaxis.set_minor_locator(AutoMinorLocator(x_minor_divisions))
    if y_minor_divisions is not None and axis.get_yscale() == "linear":
        axis.yaxis.set_minor_locator(AutoMinorLocator(y_minor_divisions))

    axis.grid(which="major", color=VALIDATION_MAJOR_GRID_COLOR, linewidth=0.7)
    axis.grid(which="minor", color=VALIDATION_MINOR_GRID_COLOR, linewidth=0.5)

    if black_axes:
        for spine in axis.spines.values():
            spine.set_color("black")
        axis.tick_params(colors="black")


def adjust_validation_layout(
    figure,
    *,
    left: float = 0.08,
    right: float = 0.98,
    bottom: float = 0.12,
    top: float = 0.95,
    wspace: float = 0.07,
) -> None:
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)


def _scaled_validation_figsize(page_figsize: tuple[float, float]) -> tuple[float, float]:
    return (float(page_figsize[0]) * VALIDATION_FIGURE_WIDTH_SCALE, float(page_figsize[1]))


def _scaled_mansur_figsize(page_figsize: tuple[float, float]) -> tuple[float, float]:
    width, height = _scaled_validation_figsize(page_figsize)
    return (width * MANSUR_EXTRA_WIDTH_SCALE, height)


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

DEFAULT_ORIGINAL_SENSITIVITY_VALUES = [float(v) for v in np.linspace(0.0, 1.0, 21)]
DEFAULT_DRAG_EPSILON_PATHS = [
    "geometry.epsilon_in",
    "geometry.epsilon_body",
    "geometry.epsilon_solar",
    "geometry.epsilon_rad",
    "geometry.epsilon_in_norm",
]
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
    "UNIFORM_MARKER_SIZE",
    "UNIFORM_MARKER_EDGE_WIDTH",
    "DEFAULT_SERIES_COLORS",
    "apply_plot_style",
    "style_axis",
    "style_validation_axis",
    "style_legend",
    "adjust_validation_layout",
    "summarize_series",
    "format_summary",
    "add_summary_box",
    "PLOT_GEOMETRY_ASPECT_RATIO_BARS",
    "PLOT_SENSITIVITY",
    "PLOT_MULTI_SENSITIVITY",
    "PLOT_MULTI_SENSITIVITY_SIDE_BY_SIDE",
    "DEFAULT_ORIGINAL_SENSITIVITY_VALUES",
    "DEFAULT_DRAG_EPSILON_PATHS",
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
    "plot_validation_gocee_fig5",
    "plot_validation_crandall_fig11",
    "plot_validation_crandall_fig26_fig27",
    "plot_validation_crandall_fig6_drag",
    "plot_validation_mansur_efficiency",
    "plot_validation_mansur_envelope",
    "plot_validation_mansur_thruster_map",
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
    epsilon_paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    sweep_values = [
        float(value)
        for value in (
            DEFAULT_ORIGINAL_SENSITIVITY_VALUES
            if values is None
            else values
        )
    ]
    accommodation_paths = (
        list(epsilon_paths)
        if epsilon_paths is not None
        else (DEFAULT_DRAG_EPSILON_PATHS if epsilon_path == "geometry.epsilon_body" else [epsilon_path])
    )
    definitions = [
        ("Thruster efficiency", ["thruster.eff"]),
        ("Collection efficiency", ["refueling.coll_eff"]),
        ("Accommodation coefficient", [str(path) for path in accommodation_paths]),
        ("Solar-cell efficiency", ["solar.eta_solar"]),
    ]

    cases: list[dict[str, Any]] = []
    for label, paths in definitions:
        cases.append(
            {
                "label": label,
                "variable_paths": list(paths),
                "variable_values": list(sweep_values),
                "x_label": "Parameter value [-]",
            }
        )
    return cases


def run_original_sensitivity_cases(
    values: Sequence[float] | None = None,
    *,
    epsilon_path: str = "geometry.epsilon_body",
    epsilon_paths: Sequence[str] | None = None,
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ariss.core.sensitivity import run_sensitivity_analysis
    from ariss.core.simulation import load_spacecraft_from_base_config

    base_sc = load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )
    cases = build_original_sensitivity_cases(values, epsilon_path=epsilon_path, epsilon_paths=epsilon_paths)
    altitude_cases: dict[str, Any] = {}
    refuel_cases: dict[str, Any] = {}

    for case in cases:
        label = str(case["label"])
        variable_paths = [str(path) for path in case.get("variable_paths", [])]
        if not variable_paths:
            variable_paths = [str(case["variable_path"])]
        sweep_values = [float(v) for v in case["variable_values"]]
        if len(variable_paths) == 1:
            analysis_paths: str | list[str] = variable_paths[0]
            analysis_values: list[float] | list[list[float]] = list(sweep_values)
        else:
            analysis_paths = list(variable_paths)
            analysis_values = [list(sweep_values) for _ in variable_paths]

        altitude_case = run_sensitivity_analysis(
            variable_paths=analysis_paths,
            variable_values=analysis_values,
            output_paths=["orbit.altitude"],
            combine="zip",
            base_sc=base_sc,
            max_iterations=max_iterations,
            mode="direct",
        )
        altitude_case["label"] = label
        altitude_case["x_label"] = str(case["x_label"])
        altitude_case["variable_paths"] = list(variable_paths)
        altitude_case["variable_path"] = (
            variable_paths[0] if len(variable_paths) == 1 else ", ".join(variable_paths)
        )
        altitude_case["variable_values"] = list(sweep_values)
        altitude_cases[label] = altitude_case

        refuel_case = run_sensitivity_analysis(
            variable_paths=analysis_paths,
            variable_values=analysis_values,
            output_paths=["refueling.t_refuel"],
            combine="zip",
            base_sc=base_sc,
            max_iterations=max_iterations,
            mode="refuel_search",
        )
        refuel_case["label"] = label
        refuel_case["x_label"] = str(case["x_label"])
        refuel_case["variable_paths"] = list(variable_paths)
        refuel_case["variable_path"] = (
            variable_paths[0] if len(variable_paths) == 1 else ", ".join(variable_paths)
        )
        refuel_case["variable_values"] = list(sweep_values)
        refuel_cases[label] = refuel_case

    altitude_result = {"output_paths": ["orbit.altitude"], "cases": altitude_cases}
    refuel_result = {"output_paths": ["refueling.t_refuel"], "cases": refuel_cases}
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
        cases.append(
            {
                "label": label,
                "aspect_ratios": list(ars),
                "x_label": "Aspect ratio [-]",
                "overrides": overrides,
                "active_refueling": bool(active_refueling),
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
    from ariss.core.sensitivity import run_sensitivity_analysis, set_path
    from ariss.core.simulation import load_spacecraft_from_base_config

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
    base_sc = load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    no_refuel_cases: dict[str, Any] = {}
    with_refuel_cases: dict[str, Any] = {}
    for case in cases_no_refuel:
        label = str(case["label"])
        sc = deepcopy(base_sc)
        for path, value in dict(case["overrides"]).items():
            set_path(sc, str(path), value)
        set_path(sc, "mission_profile.active_refueling", bool(case["active_refueling"]))

        run = run_sensitivity_analysis(
            variable_paths=["geometry.AR_in", "geometry.AR_body"],
            variable_values=[list(ars), list(ars)],
            combine="zip",
            output_paths=["orbit.altitude"],
            base_sc=sc,
            max_iterations=max_iterations,
            mode="direct",
        )
        run["label"] = label
        run["x_label"] = str(case["x_label"])
        run["variable_values"] = list(ars)
        no_refuel_cases[label] = run

    for case in cases_refuel:
        label = str(case["label"])
        sc = deepcopy(base_sc)
        for path, value in dict(case["overrides"]).items():
            set_path(sc, str(path), value)
        set_path(sc, "mission_profile.active_refueling", bool(case["active_refueling"]))

        run = run_sensitivity_analysis(
            variable_paths=["geometry.AR_in", "geometry.AR_body"],
            variable_values=[list(ars), list(ars)],
            combine="zip",
            output_paths=["refueling.t_refuel"],
            base_sc=sc,
            max_iterations=max_iterations,
            mode="refuel_search",
        )
        run["label"] = label
        run["x_label"] = str(case["x_label"])
        run["variable_values"] = list(ars)
        with_refuel_cases[label] = run

    no_refuel = {"output_paths": ["orbit.altitude"], "cases": no_refuel_cases}
    with_refuel = {"output_paths": ["refueling.t_refuel"], "cases": with_refuel_cases}

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

    apply_plot_style(font_size=GEOMETRY_SENSITIVITY_FONT_SIZE)

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

    fig = plt.figure(figsize=(15.2, 5.3), dpi=150)
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
                        s=UNIFORM_MARKER_SIZE**2,
                        facecolors=PALETTE.get("cat_red", "#E85C62"),
                        edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                        linewidths=UNIFORM_MARKER_EDGE_WIDTH,
                        zorder=4,
                    )

        axis.set_xticks(x_centers)
        two_line_labels = [label.replace(" / ", "\n") for label in geometry_labels]
        axis.set_xticklabels(two_line_labels, linespacing=0.95)
        axis.tick_params(axis="x", labelsize=DEFAULT_FONT_SIZE * 0.82, pad=5)
        axis.tick_params(axis="y", labelsize=DEFAULT_FONT_SIZE * 0.90)
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
                markersize=UNIFORM_MARKER_SIZE,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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

    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.26, top=0.92, wspace=0.22)

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

    apply_plot_style(font_size=PARAMETER_SENSITIVITY_FONT_SIZE)

    selected_output = _first_output_path(result, output_path)
    if selected_output not in result["outputs"]:
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {list(result['outputs'].keys())}"
        )

    x_vals = _case_x_values(result)
    y_vals = list(result["outputs"][selected_output])
    y_numeric = [float(v) if v is not None else float("nan") for v in y_vals]

    fig = plt.figure(figsize=(13.44, 4.8), dpi=150)
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
        linewidths=UNIFORM_MARKER_EDGE_WIDTH,
        s=UNIFORM_MARKER_SIZE**2,
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
            s=UNIFORM_MARKER_SIZE**2,
            facecolors=PALETTE.get("cat_red", "#E85C62"),
            edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidths=UNIFORM_MARKER_EDGE_WIDTH,
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
    if y_label is None and selected_output == "refueling.t_refuel":
        ax.yaxis.set_label_coords(SENSITIVITY_REFUELING_LABEL_X, 0.5)
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
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=SENSITIVITY_MULTI_WSPACE)

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
    apply_plot_style(font_size=PARAMETER_SENSITIVITY_FONT_SIZE)

    selected_output = _first_output_path(multi_result, output_path)
    if selected_output not in multi_result.get("output_paths", [selected_output]):
        raise KeyError(
            f"Unknown output_path '{selected_output}'. Available: {multi_result['output_paths']}"
        )

    fig = plt.figure(figsize=(13.44, 4.8), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.34))
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    color_cycle = list(DEFAULT_SERIES_COLORS)
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
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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
                s=UNIFORM_MARKER_SIZE**2,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=UNIFORM_MARKER_EDGE_WIDTH,
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
    if y_label is None and selected_output == "refueling.t_refuel":
        ax.yaxis.set_label_coords(SENSITIVITY_REFUELING_LABEL_X, 0.5)
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
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94, wspace=SENSITIVITY_MULTI_WSPACE)

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
    paper_style: bool = True,
    show: bool = True,
):
    # Plot two outputs side by side with one shared legend.
    if paper_style:
        apply_plot_style(font_size=PARAMETER_SENSITIVITY_PAPER_FONT_SIZE, figsize=(13.2, 3.6))
    else:
        apply_plot_style(font_size=PARAMETER_SENSITIVITY_FONT_SIZE)

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

    fig = plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=150)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.46))
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    legend_axis = fig.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    if paper_style:
        fig.patch.set_facecolor("#E8E8E8")
        ax_left.set_facecolor("#E8E8E8")
        ax_right.set_facecolor("#E8E8E8")
        legend_axis.set_facecolor("#E8E8E8")

    color_cycle = list(DEFAULT_SERIES_COLORS)
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
                markersize=UNIFORM_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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
            linewidth=1.4 if paper_style else 2.0,
            marker=marker,
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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
                s=UNIFORM_MARKER_SIZE**2,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=UNIFORM_MARKER_EDGE_WIDTH,
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
            linewidth=1.4 if paper_style else 2.0,
            marker=marker,
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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
                s=UNIFORM_MARKER_SIZE**2,
                facecolors=PALETTE.get("cat_red", "#E85C62"),
                edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
                linewidths=UNIFORM_MARKER_EDGE_WIDTH,
                zorder=5,
            )

    ax_left.set_xlabel(x_label)
    ax_right.set_xlabel(x_label)
    ax_left.set_ylabel("Orbit altitude [km]" if left_output == "orbit.altitude" else left_output)
    ax_right.set_ylabel("Refueling time [months]" if right_output == "refueling.t_refuel" else right_output)
    if left_output == "refueling.t_refuel":
        ax_left.yaxis.set_label_coords(SENSITIVITY_REFUELING_LABEL_X, 0.5)
    if right_output == "refueling.t_refuel":
        ax_right.yaxis.set_label_coords(SENSITIVITY_REFUELING_LABEL_X, 0.5)
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

    if paper_style:
        ax_left.grid(which="major", axis="y", color="#CFCFCF", linewidth=0.6)
        ax_right.grid(which="major", axis="y", color="#CFCFCF", linewidth=0.6)
        ax_left.grid(which="minor", axis="y", color="#DFDFDF", linewidth=0.45)
        ax_right.grid(which="minor", axis="y", color="#DFDFDF", linewidth=0.45)

    # Match the paper sweep framing when values are normalized 0..1.
    all_x_values: list[float] = []
    for case_data in left_result["cases"].values():
        all_x_values.extend([float(v) for v in _case_x_values(case_data)])
    if all_x_values:
        x_min = float(np.nanmin(all_x_values))
        x_max = float(np.nanmax(all_x_values))
        if x_min >= -1.0e-12 and x_max <= 1.0 + 1.0e-12:
            ax_left.set_xlim(0.0, 1.0)
            ax_right.set_xlim(0.0, 1.0)

    legend_handles: list[Any] = list(case_handles)
    if boundary_present:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=UNIFORM_MARKER_SIZE,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
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

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.88, wspace=SENSITIVITY_SIDE_BY_SIDE_WSPACE)

    if show:
        plt.show(block=True)

    return fig, (ax_left, ax_right)


def plot_validation_gocee_fig5(
    dataset: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    compute_ariss_body_cd_curve,
    output_path: str | Path,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
) -> Path:
    scaled_figsize = _scaled_mansur_figsize(page_figsize)
    apply_plot_style(font_size=GOCEE_DRAG_FONT_SIZE, figsize=scaled_figsize)
    output_path = Path(output_path)

    figure = plt.figure(figsize=scaled_figsize)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=BASE_VALIDATION_WSPACE)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    x_all: list[float] = []
    y_all: list[float] = []

    x_ref = []
    for key in ("Mansur", "Koppenwallner"):
        if key in dataset:
            x_ref.extend(np.asarray(dataset[key][0], dtype=float).tolist())
    if x_ref:
        x_min = float(np.nanmin(x_ref))
        x_max = float(np.nanmax(x_ref))
    else:
        x_min, x_max = 8.0, 13.0

    x_ariss = np.linspace(x_min, x_max, 140)
    x_ariss, y_ariss = compute_ariss_body_cd_curve(x_ariss)
    axis.plot(x_ariss, y_ariss, color=PALETTE["sernn_pink"], lw=2.2, zorder=2)
    x_all.extend(np.asarray(x_ariss, dtype=float).tolist())
    y_all.extend(np.asarray(y_ariss, dtype=float).tolist())

    if "Mansur" in dataset:
        x_m, y_m = dataset["Mansur"]
        axis.plot(x_m, y_m, color=PALETTE["l1_teal"], lw=2.2, ls=(0, (4, 2)), zorder=2)
        x_all.extend(np.asarray(x_m, dtype=float).tolist())
        y_all.extend(np.asarray(y_m, dtype=float).tolist())

    if "Koppenwallner" in dataset:
        x_k, y_k = dataset["Koppenwallner"]
        axis.plot(
            x_k,
            y_k,
            linestyle="None",
            marker="x",
            markersize=UNIFORM_MARKER_SIZE,
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
            color=PALETTE["choice_mid"],
            zorder=3,
        )
        x_all.extend(np.asarray(x_k, dtype=float).tolist())
        y_all.extend(np.asarray(y_k, dtype=float).tolist())

    axis.set_xlabel(r"Speed ratio $S_0$")
    axis.set_ylabel(r"Drag coefficient $C_D$")
    style_validation_axis(axis, x_minor_divisions=2, y_minor_divisions=2)

    if x_all and y_all:
        x_min = float(np.nanmin(x_all))
        x_max = float(np.nanmax(x_all))
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        x_pad = 0.03 * (x_max - x_min) if x_max > x_min else 0.2
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        axis.set_xlim(x_min - x_pad, x_max + x_pad)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)

    source_handles = [
        Line2D([0], [0], color=PALETTE["sernn_pink"], lw=2.2, label="ARISS drag model"),
        Line2D([0], [0], color=PALETTE["l1_teal"], lw=2.2, ls=(0, (4, 2)), label=r"Mansur model $(A_{in}/A_{ref}=28.2)$"),
        Line2D([0], [0], color=PALETTE["choice_mid"], lw=0.0, marker="x", markersize=UNIFORM_MARKER_SIZE, markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label=r"Koppenwallner GOCE $C_D$"),
    ]
    legend = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    adjust_validation_layout(figure)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_validation_crandall_fig11(
    results: Mapping[str, Mapping[str, np.ndarray]],
    dataset: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    solar_cases: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
) -> Path:
    scaled_figsize = _scaled_validation_figsize(page_figsize)
    apply_plot_style(font_size=CRANDALL_FIG11_FONT_SIZE, figsize=scaled_figsize)
    output_path = Path(output_path)

    figure = plt.figure(figsize=scaled_figsize)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=CRANDALL_VALIDATION_WSPACE)
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    x_all: list[float] = []
    y_all: list[float] = []

    for spec in solar_cases:
        label = str(spec["label"])
        color = spec["color"]
        marker = spec["marker"]

        model = results.get(label, {})
        tp = np.asarray(model.get("tp", np.array([])), dtype=float)
        altitude = np.asarray(model.get("altitude", np.array([])), dtype=float)

        valid_model = np.isfinite(tp) & np.isfinite(altitude)
        axis.plot(tp[valid_model], altitude[valid_model], color=color, lw=2.2, zorder=2)

        if label in dataset:
            x_ref, y_ref = dataset[label]
            valid_ref = np.isfinite(x_ref) & np.isfinite(y_ref)
            axis.plot(
                x_ref[valid_ref],
                y_ref[valid_ref],
                color=color,
                lw=1.4,
                ls=(0, (4, 2)),
                marker=marker,
                markersize=UNIFORM_MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
                zorder=3,
            )
            x_all.extend(np.asarray(x_ref[valid_ref], dtype=float).tolist())
            y_all.extend(np.asarray(y_ref[valid_ref], dtype=float).tolist())

        x_all.extend(np.asarray(tp[valid_model], dtype=float).tolist())
        y_all.extend(np.asarray(altitude[valid_model], dtype=float).tolist())

    axis.set_xlabel("T/P (mN/kW)")
    axis.set_ylabel("Minimum operating altitude (km)")
    style_validation_axis(axis, x_minor_divisions=2, y_minor_divisions=2)

    if x_all and y_all:
        x_min = float(np.nanmin(x_all))
        x_max = float(np.nanmax(x_all))
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        axis.set_xlim(0.95 * x_min, 1.03 * x_max)
        axis.set_ylim(y_min - 1.0, y_max + 1.0)

    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.2, label="ARISS full loop"),
        Line2D(
            [0],
            [0],
            color=PALETTE["secondary_text"],
            lw=1.4,
            ls=(0, (4, 2)),
            marker="o",
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=PALETTE["secondary_text"],
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
            label="Crandall-Wirz Data",
        ),
    ]
    case_handles = [Line2D([0], [0], color=spec["color"], lw=2.2, label=str(spec["label"])) for spec in solar_cases]

    legend_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend_source)
    if legend_source is not None and legend_source.get_title() is not None:
        legend_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(legend_source)

    legend_cases = legend_axis.legend(
        handles=case_handles,
        title="Solar Activity",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.58),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(legend_cases)
    if legend_cases is not None and legend_cases.get_title() is not None:
        legend_cases.get_title().set_fontweight("bold")

    adjust_validation_layout(figure, wspace=CRANDALL_VALIDATION_WSPACE)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_validation_crandall_fig26_fig27(
    fig26_model: Mapping[str, tuple[np.ndarray, np.ndarray]],
    fig27_model: Mapping[str, tuple[np.ndarray, np.ndarray]],
    fig26_dataset: Mapping[str, tuple[np.ndarray, np.ndarray]],
    fig27_dataset: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    solar_cases: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
) -> Path:
    scaled_figsize = _scaled_validation_figsize(page_figsize)
    apply_plot_style(font_size=CRANDALL_FIG2627_FONT_SIZE, figsize=scaled_figsize)
    output_path = Path(output_path)

    figure = plt.figure(figsize=scaled_figsize)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=CRANDALL_VALIDATION_WSPACE)
    ax_left = figure.add_subplot(grid[0, 0])
    ax_right = figure.add_subplot(grid[0, 1], sharey=ax_left)
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    axes = [ax_left, ax_right]

    for axis in axes:
        style_validation_axis(axis, x_minor_divisions=2, y_minor_divisions=2)

    axes[0].set_xlabel("Solar-cell efficiency (-)")
    axes[1].set_xlabel("Accommodation coefficient (-)")
    figure.supylabel("Minimum operating altitude (km)", x=0.04)

    y_all: list[float] = []
    for case in solar_cases:
        label = str(case["label"])
        color = case["color"]
        marker = case["marker"]

        if label in fig26_model:
            x_mod, y_mod = fig26_model[label]
            axes[0].plot(x_mod, y_mod, color=color, lw=2.2, zorder=2)
            y_all.extend(np.asarray(y_mod, dtype=float).tolist())
        if label in fig26_dataset:
            x_ref, y_ref = fig26_dataset[label]
            axes[0].plot(x_ref, y_ref, color=color, lw=1.5, ls=(0, (4, 2)), marker=marker, markersize=UNIFORM_MARKER_SIZE, markerfacecolor="white", markeredgecolor=color, markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, zorder=3)
            y_all.extend(np.asarray(y_ref, dtype=float).tolist())

        if label in fig27_model:
            x_mod, y_mod = fig27_model[label]
            axes[1].plot(x_mod, y_mod, color=color, lw=2.2, zorder=2)
            y_all.extend(np.asarray(y_mod, dtype=float).tolist())
        if label in fig27_dataset:
            x_ref, y_ref = fig27_dataset[label]
            axes[1].plot(x_ref, y_ref, color=color, lw=1.5, ls=(0, (4, 2)), marker=marker, markersize=UNIFORM_MARKER_SIZE, markerfacecolor="white", markeredgecolor=color, markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, zorder=3)
            y_all.extend(np.asarray(y_ref, dtype=float).tolist())

    axes[0].set_xlim(0.24, 0.505)
    axes[1].set_xlim(0.0, 1.0)
    if y_all:
        y_arr = np.asarray(y_all, dtype=float)
        finite = np.isfinite(y_arr)
        if np.any(finite):
            y_min = float(np.min(y_arr[finite]))
            y_max = float(np.max(y_arr[finite]))
            axes[0].set_ylim(y_min - 1.5, y_max + 1.5)

    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.2, label="ARISS full loop"),
        Line2D(
            [0],
            [0],
            color=PALETTE["secondary_text"],
            lw=1.5,
            ls=(0, (4, 2)),
            marker="o",
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=PALETTE["secondary_text"],
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
            label="Crandall-Wirz Data",
        ),
    ]
    case_handles = [Line2D([0], [0], color=case["color"], lw=2.2, label=str(case["label"])) for case in solar_cases]

    source_legend = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(source_legend)
    if source_legend is not None and source_legend.get_title() is not None:
        source_legend.get_title().set_fontweight("bold")
    legend_axis.add_artist(source_legend)

    case_legend = legend_axis.legend(
        handles=case_handles,
        title="Solar Activity",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.58),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(case_legend)
    if case_legend is not None and case_legend.get_title() is not None:
        case_legend.get_title().set_fontweight("bold")

    adjust_validation_layout(
        figure,
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.12,
        wspace=CRANDALL_VALIDATION_WSPACE,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_validation_crandall_fig6_drag(
    models: Sequence[Mapping[str, np.ndarray]],
    datasets: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    *,
    case_specs: Sequence[Mapping[str, Any]],
    component_specs: Sequence[tuple[str, str, str, str]],
    altitude_min: float,
    altitude_max: float,
    output_path: str | Path,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
) -> Path:
    def plot_case(axis, title: str, model: Mapping[str, np.ndarray], dataset: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        x_lower = np.inf
        x_upper = 0.0
        for label, key, color, marker in component_specs:
            x_model = np.asarray(model[key], dtype=float)
            y_model = np.asarray(model["altitude"], dtype=float)
            valid_model = np.isfinite(x_model) & np.isfinite(y_model) & (x_model > 0.0)
            axis.plot(x_model[valid_model], y_model[valid_model], color=color, lw=2.0, zorder=2)
            if np.any(valid_model):
                x_lower = min(x_lower, float(np.min(x_model[valid_model])))
                x_upper = max(x_upper, float(np.max(x_model[valid_model])))

            if label in dataset:
                x_ref, y_ref = dataset[label]
                valid_ref = np.isfinite(x_ref) & np.isfinite(y_ref) & (x_ref > 0.0)
                axis.plot(
                    x_ref[valid_ref],
                    y_ref[valid_ref],
                    linestyle=(0, (4, 2)),
                    lw=1.2,
                    color=color,
                    marker=marker,
                    markersize=UNIFORM_MARKER_SIZE,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
                    zorder=3,
                )
                if np.any(valid_ref):
                    x_lower = min(x_lower, float(np.min(x_ref[valid_ref])))
                    x_upper = max(x_upper, float(np.max(x_ref[valid_ref])))

        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_xlabel("Drag [mN]")
        axis.yaxis.set_major_locator(MultipleLocator(10))
        axis.yaxis.set_minor_locator(MultipleLocator(5))
        style_validation_axis(axis, x_minor_divisions=None, y_minor_divisions=None)
        axis.xaxis.label.set_size(CRANDALL_DRAG_FONT_SIZE)
        axis.yaxis.label.set_size(CRANDALL_DRAG_FONT_SIZE)
        axis.title.set_size(CRANDALL_DRAG_FONT_SIZE)
        if np.isfinite(x_lower) and np.isfinite(x_upper) and x_upper > x_lower > 0.0:
            axis.set_xlim(0.9 * x_lower, 1.15 * x_upper)

    scaled_figsize = _scaled_validation_figsize(page_figsize)
    apply_plot_style(font_size=CRANDALL_DRAG_FONT_SIZE, figsize=scaled_figsize)
    output_path = Path(output_path)

    figure = plt.figure(figsize=scaled_figsize)
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=CRANDALL_VALIDATION_WSPACE)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    axes[1].sharey(axes[0])
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")

    for axis, spec, model, dataset in zip(axes, case_specs, models, datasets):
        plot_case(axis, str(spec["title"]), model, dataset)

    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_ylim(float(altitude_min) - 1.0, float(altitude_max) + 1.0)

    component_handles = [Line2D([0], [0], color=color, lw=2.0, label=label) for label, _, color, _ in component_specs]
    source_handles = [
        Line2D([0], [0], color=PALETTE["secondary_text"], lw=2.0, label="ARISS drag-only model"),
        Line2D([0], [0], marker="o", color=PALETTE["secondary_text"], markerfacecolor="white", markersize=UNIFORM_MARKER_SIZE, lw=1.2, ls=(0, (4, 2)), label="Crandall-Wirz Data"),
    ]

    legend_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=0.6,
    )
    style_legend(legend_source)
    if legend_source is not None and legend_source.get_title() is not None:
        legend_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(legend_source)

    legend_components = legend_axis.legend(
        handles=component_handles,
        title="Components",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.62),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=0.6,
    )
    style_legend(legend_components)
    if legend_components is not None and legend_components.get_title() is not None:
        legend_components.get_title().set_fontweight("bold")

    adjust_validation_layout(figure, wspace=CRANDALL_VALIDATION_WSPACE)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return output_path


def plot_validation_mansur_efficiency(
    results: Mapping[float, tuple[np.ndarray, np.ndarray]],
    paper: Mapping[float, tuple[np.ndarray, np.ndarray]] | None,
    *,
    collection_efficiencies: Sequence[float],
    plot_colors: Sequence[str],
    output_path: str | Path,
    vector_output_path: str | Path | None = None,
    pdf_output_path: str | Path | None = None,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
    relative_and_corr_stats_fn=None,
) -> Path:
    def sanitize_color(color: str) -> str:
        if str(color).lower() in {"grey", "gray", "#808080", "#7f7f7f"}:
            return "black"
        return color

    def soft_curve_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.size < 3:
            return x, y
        from scipy.interpolate import PchipInterpolator

        seg = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate(([0.0], np.cumsum(seg)))
        if s[-1] == 0:
            return x, y
        t = s / s[-1]
        t, idx = np.unique(t, return_index=True)
        if t.size < 3:
            return x[idx], y[idx]
        fx = PchipInterpolator(t, x[idx])
        fy = PchipInterpolator(t, y[idx])
        tt = np.linspace(0, 1, max(140, 24 * (t.size - 1)))
        return fx(tt), fy(tt)

    def plot_curve(ax, x, y, color, marker, linestyle) -> None:
        color = sanitize_color(color)
        sx, sy = soft_curve_points(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        ax.plot(sx, sy, color=color, linestyle=linestyle, linewidth=1.1)
        ax.plot(
            x,
            y,
            linestyle="None",
            marker=marker,
            markersize=UNIFORM_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH,
        )

    def plot_sweep(ax, x, y, color) -> None:
        if len(x) == 0:
            return
        idx = int(np.argmin(x))
        plot_curve(ax, x[: idx + 1], y[: idx + 1], color, "o", "-")
        if idx < len(x) - 1:
            plot_curve(ax, x[idx:], y[idx:], color, "o", "-")

    def plot_reference(ax, x, y, color) -> None:
        if len(x) == 0:
            return
        order = np.argsort(y)[::-1]
        plot_curve(ax, np.asarray(x)[order], np.asarray(y)[order], color, "s", "--")

    scaled_figsize = _scaled_validation_figsize(page_figsize)
    mansur_efficiency_font_size = MANSUR_EFFICIENCY_FONT_SIZE * MANSUR_FONT_SCALE
    apply_plot_style(font_size=mansur_efficiency_font_size, figsize=scaled_figsize)
    output_path = Path(output_path)
    vector_output = Path(vector_output_path) if vector_output_path is not None else None
    pdf_output = Path(pdf_output_path) if pdf_output_path is not None else None

    fig = plt.figure(figsize=scaled_figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.82], wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    handles: list[Line2D] = []
    labels: list[str] = []

    print("Datapoint relative-error and correlation against Mansur efficiency curves:")
    pearson_values: list[float] = []

    for color, eff in zip(plot_colors, collection_efficiencies):
        x, y = results.get(float(eff), (np.array([]), np.array([])))
        px, py = (paper or {}).get(float(eff), (np.array([]), np.array([])))
        if len(x) == 0:
            continue

        clean_color = sanitize_color(color)
        plot_sweep(ax, np.asarray(x, dtype=float), np.asarray(y, dtype=float), clean_color)
        plot_reference(ax, np.asarray(px, dtype=float), np.asarray(py, dtype=float), clean_color)

        if relative_and_corr_stats_fn is not None and len(px) > 0 and len(py) > 0:
            stats = relative_and_corr_stats_fn(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(px, dtype=float), np.asarray(py, dtype=float))
            if stats is None:
                print(f"  eta_c={float(eff):.2f} n/a")
            else:
                max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  eta_c={float(eff):.2f} "
                    f"max_relative_error={max_rel:10.6f} ({100.0 * max_rel:7.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel:10.6f} ({100.0 * mean_rel:7.3f}%), "
                    f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values.append(pearson_r)

        h_lim = float(np.min(np.asarray(x, dtype=float)))
        handles.append(Line2D([0], [0], color=clean_color, marker="o", linestyle="-"))
        labels.append(f"ARISS ηc={float(eff):.2f}, h_lim={h_lim:.1f} km")
        if len(px) > 0:
            h_lim_ref = float(np.min(np.asarray(px, dtype=float)))
            handles.append(Line2D([0], [0], color=clean_color, marker="s", linestyle="--"))
            labels.append(f"Mansur ηc={float(eff):.2f}, h_lim={h_lim_ref:.1f} km")

    if pearson_values:
        print(f"  Minimum Pearson correlation coefficient: {min(pearson_values):.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    ax.set_xlabel("Converged altitude (km)")
    ax.set_ylabel("Isp (s)")
    ax.set_xlim(150, 230)
    ax.set_ylim(2000, 6000)
    style_validation_axis(ax, x_minor_divisions=2, y_minor_divisions=2, black_axes=True)
    x_ticks = ax.get_xticks()
    if len(x_ticks) > 0:
        first_x_tick = float(x_ticks[0])
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position, first_tick=first_x_tick: (
                    "" if np.isclose(value, first_tick) else f"{value:g}"
                )
            )
        )

    if handles:
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

    adjust_validation_layout(fig, left=0.055, right=0.985, bottom=0.13, top=0.95, wspace=0.03)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if vector_output is not None:
        fig.savefig(vector_output, bbox_inches="tight")
    if pdf_output is not None:
        fig.savefig(pdf_output, bbox_inches="tight")

    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_validation_mansur_envelope(
    ariss: Mapping[float, Sequence[tuple[np.ndarray, np.ndarray]]],
    paper: Mapping[float, tuple[np.ndarray, np.ndarray]],
    *,
    alt_levels: Sequence[float],
    output_path: str | Path | None = None,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
    save: bool = True,
    return_figure: bool = False,
) -> Path | tuple[Any, tuple[Any, Any]]:
    from scipy.interpolate import PchipInterpolator

    def clean_xy(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def smooth_xy(x, y, n=300):
        x, y = clean_xy(x, y)
        if len(x) < 3:
            return x, y
        x, idx = np.unique(x, return_index=True)
        y = y[idx]
        f = PchipInterpolator(x, y)
        xs = np.linspace(x.min(), x.max(), n)
        return xs, f(xs)

    def smooth_by_y(x, y, n=300):
        x, y = clean_xy(x, y)
        if len(x) < 3:
            return x, y
        order = np.argsort(y)
        y, x = y[order], x[order]
        y, idx = np.unique(y, return_index=True)
        x = x[idx]
        f = PchipInterpolator(y, x)
        ys = np.linspace(y.min(), y.max(), n)
        return f(ys), ys

    def spaced_marker_indices(x, y, n_markers=2, pad_fraction=0.14):
        x, y = clean_xy(x, y)
        if len(x) < 2:
            return []
        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        total = s[-1]
        if not np.isfinite(total) or total <= 0:
            return [len(x) // 2]
        if n_markers <= 1:
            targets = np.array([0.5 * total])
        else:
            lo = pad_fraction * total
            hi = (1.0 - pad_fraction) * total
            targets = np.array([0.5 * total]) if hi <= lo else np.linspace(lo, hi, n_markers)
        idx = []
        for t in targets:
            i = int(np.argmin(np.abs(s - t)))
            if i not in idx:
                idx.append(i)
        return idx

    def plot_curve_with_markers(ax, x, y, *, color, marker, lw, ls, alpha, zorder, filled, halo=False, n_markers=2):
        x, y = clean_xy(x, y)
        if len(x) < 2:
            return
        mark_idx = spaced_marker_indices(x, y, n_markers=n_markers, pad_fraction=0.14)
        line, = ax.plot(
            x,
            y,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
            dash_capstyle="round",
            marker=marker,
            markevery=mark_idx if mark_idx else None,
            ms=envelope_marker_size,
            mec="white" if filled else color,
            mew=UNIFORM_MARKER_EDGE_WIDTH,
            mfc=color if filled else "white",
        )
        if halo:
            line.set_path_effects([pe.Stroke(linewidth=lw + 1.6, foreground="white"), pe.Normal()])

    scaled_figsize = _scaled_mansur_figsize(page_figsize)
    envelope_marker_size = UNIFORM_MARKER_SIZE * MANSUR_MARKER_SCALE
    envelope_font_size = MANSUR_ENVELOPE_FONT_SIZE * MANSUR_FONT_SCALE
    legend_anchor_x = -0.07 + MANSUR_ENVELOPE_LEGEND_RIGHT_SHIFT
    source_anchor_x = legend_anchor_x * MANSUR_ENVELOPE_SOURCE_X_SCALE
    source_anchor_y = 0.91 * MANSUR_ENVELOPE_SOURCE_Y_SCALE
    apply_plot_style(font_size=envelope_font_size, figsize=scaled_figsize)
    resolved_output_path: Path | None = Path(output_path) if output_path is not None else None
    if save and resolved_output_path is None:
        raise ValueError("output_path is required when save=True for Mansur envelope plot.")

    cmap = plt.get_cmap("viridis")
    if len(alt_levels) <= 1:
        color_positions = [0.62]
    else:
        color_positions = np.linspace(0.10, 0.92, len(alt_levels))
    colors = {
        float(h): cmap(float(color_positions[idx]))
        for idx, h in enumerate(alt_levels)
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    markers = {float(h): m for h, m in zip(alt_levels, marker_cycle)}

    fig = plt.figure(figsize=scaled_figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.82], wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    for h in alt_levels:
        h = float(h)
        if h not in paper:
            continue
        c = colors[h]
        mk = markers[h]
        x, y = smooth_xy(*paper[h], n=320)
        plot_curve_with_markers(ax, x, y, color=c, marker=mk, lw=1.2, ls=(0, (4, 2)), alpha=0.38, zorder=1, filled=False, halo=False, n_markers=2)

    for h in alt_levels:
        h = float(h)
        if h not in ariss:
            continue
        c = colors[h]
        mk = markers[h]
        branches = sorted(ariss[h], key=lambda b: len(b[0]), reverse=True)
        for x, y in branches:
            xs, ys = smooth_by_y(x, y, n=320)
            plot_curve_with_markers(ax, xs, ys, color=c, marker=mk, lw=2.2, ls="-", alpha=0.98, zorder=3, filled=True, halo=True, n_markers=2)

    ax.set_xlabel("T/P (mN/kW)")
    ax.set_ylabel("Isp (s)")
    ax.set_xlim(5, 60)
    ax.set_ylim(2500, 6000)
    style_validation_axis(ax, x_minor_divisions=2, y_minor_divisions=2, black_axes=True)

    source_handles = [Line2D([0], [0], color="black", lw=2.2, ls="-", label="ARISS"), Line2D([0], [0], color="black", lw=1.2, ls=(0, (4, 2)), label="Mansur")]
    leg_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(source_anchor_x, source_anchor_y),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_source)
    if leg_source is not None and leg_source.get_title() is not None:
        leg_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_source)

    alt_handles = [
        Line2D([0], [0], linestyle="None", marker=markers[float(h)], markersize=envelope_marker_size, markerfacecolor=colors[float(h)], markeredgecolor="black", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label=f"{float(h):g} km")
        for h in alt_levels
    ]
    leg_alt = legend_axis.legend(
        handles=alt_handles,
        title="Altitude",
        loc="upper left",
        bbox_to_anchor=(legend_anchor_x, 0.60),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        labelspacing=0.7,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_alt)
    if leg_alt is not None and leg_alt.get_title() is not None:
        leg_alt.get_title().set_fontweight("bold")

    adjust_validation_layout(fig, left=0.055, right=0.985, bottom=0.13, top=0.95, wspace=0.03)
    if save and resolved_output_path is not None:
        fig.savefig(resolved_output_path, dpi=220, bbox_inches="tight")

    interactive_show = show and "agg" not in plt.get_backend().lower()
    if interactive_show:
        plt.show()

    if return_figure:
        return fig, (ax, legend_axis)

    if not interactive_show:
        plt.close(fig)

    if resolved_output_path is not None:
        return resolved_output_path
    return fig, (ax, legend_axis)


def plot_validation_mansur_thruster_map(
    efficiency_lines: Mapping[float, Sequence[tuple[np.ndarray, np.ndarray]]],
    mass_flow_lines: Mapping[float, Sequence[tuple[np.ndarray, np.ndarray]]],
    intake_area_lines: Mapping[float, Sequence[tuple[np.ndarray, np.ndarray]]],
    mansur_eta: Mapping[float, tuple[np.ndarray, np.ndarray]],
    mansur_mdot: Mapping[float, tuple[np.ndarray, np.ndarray]],
    mansur_ain: Mapping[float, tuple[np.ndarray, np.ndarray]],
    *,
    eff_levels: Sequence[float],
    mdot_levels: Sequence[float],
    ain_levels: Sequence[float],
    eff_color: str,
    mdot_color: str,
    ain_color: str,
    marker_cycle: Sequence[str],
    marker_size: float,
    eff_marker_spec: Mapping[str, Any],
    mdot_marker_spec: Mapping[str, Any],
    ain_marker_spec: Mapping[str, Any],
    output_path: str | Path | None = None,
    page_figsize: tuple[float, float] = (15.84, 5.4),
    show: bool = True,
    save: bool = True,
    return_figure: bool = False,
) -> Path | tuple[Any, tuple[Any, Any, Any]]:
    marker_size = float(UNIFORM_MARKER_SIZE) * MANSUR_MARKER_SCALE

    def clean_xy(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def spaced_marker_indices(x, y, n_markers=2, pad_fraction=0.14):
        x, y = clean_xy(x, y)
        if len(x) < 2:
            return []
        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        total = s[-1]
        if not np.isfinite(total) or total <= 0:
            return [len(x) // 2]
        if n_markers <= 1:
            targets = np.array([0.5 * total])
        else:
            lo = pad_fraction * total
            hi = (1.0 - pad_fraction) * total
            targets = np.array([0.5 * total]) if hi <= lo else np.linspace(lo, hi, n_markers)
        idx = []
        for t in targets:
            i = int(np.argmin(np.abs(s - t)))
            if i not in idx:
                idx.append(i)
        return idx

    def target_marker_indices(x, y, *, x_targets=None, y_targets=None, fallback_count=2):
        x, y = clean_xy(x, y)
        if len(x) < 2:
            return []
        idx = []
        if x_targets is not None:
            xmin = float(np.min(x))
            xmax = float(np.max(x))
            for target in x_targets:
                if xmin <= float(target) <= xmax:
                    i = int(np.argmin(np.abs(x - float(target))))
                    if i not in idx:
                        idx.append(i)
        elif y_targets is not None:
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            for target in y_targets:
                if ymin <= float(target) <= ymax:
                    i = int(np.argmin(np.abs(y - float(target))))
                    if i not in idx:
                        idx.append(i)
        if not idx:
            return spaced_marker_indices(x, y, n_markers=fallback_count)
        return idx

    def marker_indices_from_spec(x, y, marker_spec):
        mode = marker_spec.get("mode", "arc")
        if mode == "x_targets":
            return target_marker_indices(x, y, x_targets=marker_spec.get("targets", []), fallback_count=marker_spec.get("fallback_count", 2))
        if mode == "y_targets":
            return target_marker_indices(x, y, y_targets=marker_spec.get("targets", []), fallback_count=marker_spec.get("fallback_count", 2))
        return spaced_marker_indices(x, y, n_markers=marker_spec.get("count", 2))

    def plot_curve_with_markers(axis, x, y, *, color, marker, lw, ls, alpha, zorder, filled, halo=False, marker_spec=None):
        x, y = clean_xy(x, y)
        if len(x) < 2:
            return
        mark_idx = spaced_marker_indices(x, y, n_markers=2) if marker_spec is None else marker_indices_from_spec(x, y, marker_spec)
        line, = axis.plot(
            x,
            y,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
            dash_capstyle="round",
            marker=marker,
            markevery=mark_idx if mark_idx else None,
            ms=marker_size,
            mec="white" if filled else color,
            mew=UNIFORM_MARKER_EDGE_WIDTH,
            mfc=color if filled else "white",
        )
        if halo:
            line.set_path_effects([pe.Stroke(linewidth=lw + 1.4, foreground="white"), pe.Normal()])

    def build_marker_map(levels):
        if len(levels) > len(marker_cycle):
            raise ValueError("Not enough markers defined for the requested levels.")
        return {float(level): marker_cycle[i] for i, level in enumerate(levels)}

    def plot_ariss_family(axis, lines, levels, color, linewidth, markers, marker_spec, zorder):
        for level in levels:
            key = float(level)
            if key not in lines:
                continue
            branches = sorted(lines[key], key=lambda b: len(b[0]), reverse=True)
            for x_vals, y_vals in branches:
                plot_curve_with_markers(axis, x_vals, y_vals, color=color, marker=markers[key], lw=linewidth, ls="-", alpha=0.98, zorder=zorder, filled=True, halo=True, marker_spec=marker_spec)

    def plot_mansur_family(axis, dataset, levels, color, linewidth, markers, marker_spec, zorder):
        for level in levels:
            key = float(level)
            if key not in dataset:
                continue
            x_vals, y_vals = dataset[key]
            plot_curve_with_markers(axis, x_vals, y_vals, color=color, marker=markers[key], lw=linewidth, ls=(0, (4, 2)), alpha=0.42, zorder=zorder, filled=False, halo=False, marker_spec=marker_spec)

    scaled_figsize = _scaled_mansur_figsize(page_figsize)
    scaled_figsize = (
        float(scaled_figsize[0]) * MANSUR_THRUSTER_FIGSIZE_SCALE,
        float(scaled_figsize[1]) * MANSUR_THRUSTER_FIGSIZE_SCALE,
    )
    thruster_font_size = MANSUR_THRUSTER_MAP_FONT_SIZE * MANSUR_FONT_SCALE
    apply_plot_style(font_size=thruster_font_size, figsize=scaled_figsize)
    resolved_output_path: Path | None = Path(output_path) if output_path is not None else None
    if save and resolved_output_path is None:
        raise ValueError("output_path is required when save=True for Mansur thruster-map plot.")
    eta_markers = build_marker_map(eff_levels)
    mdot_markers = build_marker_map(mdot_levels)
    ain_markers = build_marker_map(ain_levels)

    figure = plt.figure(figsize=scaled_figsize)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.03)
    axis = figure.add_subplot(grid[0, 0])
    axis.xaxis.label.set_size(thruster_font_size)
    axis.yaxis.label.set_size(thruster_font_size)
    axis.tick_params(axis="both", which="both", labelsize=thruster_font_size)
    legend_grid = grid[0, 1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    legend_axis_left = figure.add_subplot(legend_grid[0, 0])
    legend_axis_right = figure.add_subplot(legend_grid[0, 1])
    legend_axis_left.axis("off")
    legend_axis_right.axis("off")
    adjust_validation_layout(figure, left=0.055, right=0.985, bottom=0.13, top=0.95, wspace=0.03)

    plot_mansur_family(axis, mansur_eta, eff_levels, color=eff_color, linewidth=1.0, markers=eta_markers, marker_spec=eff_marker_spec, zorder=1)
    plot_mansur_family(axis, mansur_mdot, mdot_levels, color=mdot_color, linewidth=0.95, markers=mdot_markers, marker_spec=mdot_marker_spec, zorder=1)
    plot_mansur_family(axis, mansur_ain, ain_levels, color=ain_color, linewidth=0.9, markers=ain_markers, marker_spec=ain_marker_spec, zorder=1)
    plot_ariss_family(axis, efficiency_lines, eff_levels, color=eff_color, linewidth=1.35, markers=eta_markers, marker_spec=eff_marker_spec, zorder=4)
    plot_ariss_family(axis, mass_flow_lines, mdot_levels, color=mdot_color, linewidth=1.20, markers=mdot_markers, marker_spec=mdot_marker_spec, zorder=3)
    plot_ariss_family(axis, intake_area_lines, ain_levels, color=ain_color, linewidth=1.05, markers=ain_markers, marker_spec=ain_marker_spec, zorder=2)

    axis.set_xlabel("T/P (mN/kW)")
    axis.set_ylabel("Isp (s)")
    axis.set_xlim(5, 60)
    axis.set_ylim(2500, 6000)
    style_validation_axis(axis, x_minor_divisions=2, y_minor_divisions=2, black_axes=True)
    axis.tick_params(axis="both", which="major", width=0.9, length=5, labelsize=thruster_font_size)
    axis.tick_params(axis="both", which="minor", width=0.7, length=3, labelsize=thruster_font_size)

    source_handles = [
        Line2D([0], [0], color="black", lw=1.25, ls="-", marker="o", markersize=marker_size, markerfacecolor="black", markeredgecolor="white", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label="ARISS"),
        Line2D([0], [0], color="black", lw=1.0, ls=(0, (4, 2)), marker="o", markersize=marker_size, markerfacecolor="white", markeredgecolor="black", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label="Mansur"),
    ]
    leg_source = legend_axis_left.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(MANSUR_THRUSTER_LEGEND_X_SHIFT, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_source)
    if leg_source is not None and leg_source.get_title() is not None:
        leg_source.get_title().set_fontweight("bold")
    legend_axis_left.add_artist(leg_source)

    family_handles = [
        Line2D([0], [0], color=eff_color, lw=1.35, label=r"$\eta_T$"),
        Line2D([0], [0], color=mdot_color, lw=1.20, label=r"$\dot{m}$ (mg/s)"),
        Line2D([0], [0], color=ain_color, lw=1.05, label=r"$A_i$ (m$^2$)"),
    ]
    leg_family = legend_axis_left.legend(
        handles=family_handles,
        title="Families",
        loc="upper left",
        bbox_to_anchor=(MANSUR_THRUSTER_LEGEND_X_SHIFT, MANSUR_THRUSTER_FAMILIES_ANCHOR_Y),
        frameon=False,
        borderaxespad=0.0,
        ncol=MANSUR_THRUSTER_FAMILIES_NCOL,
        columnspacing=MANSUR_THRUSTER_FAMILIES_COLUMN_SPACING,
        labelspacing=0.8,
        handlelength=2.6,
        handletextpad=0.6,
    )
    style_legend(leg_family)
    if leg_family is not None and leg_family.get_title() is not None:
        leg_family.get_title().set_fontweight("bold")
    legend_axis_left.add_artist(leg_family)

    eta_handles = [
        Line2D([0], [0], linestyle="None", marker=eta_markers[float(level)], markersize=marker_size, markerfacecolor=eff_color, markeredgecolor="black", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label=f"{float(level):g}")
        for level in eff_levels
    ]
    leg_eta = legend_axis_left.legend(
        handles=eta_handles,
        title=r"$\eta_T$ levels",
        loc="upper left",
        bbox_to_anchor=(MANSUR_THRUSTER_LEGEND_X_SHIFT, MANSUR_THRUSTER_ETA_LEVELS_ANCHOR_Y),
        frameon=False,
        borderaxespad=0.0,
        ncol=3,
        columnspacing=0.9,
        labelspacing=0.75,
        handletextpad=0.45,
    )
    style_legend(leg_eta)
    if leg_eta is not None and leg_eta.get_title() is not None:
        leg_eta.get_title().set_fontweight("bold")
    legend_axis_left.add_artist(leg_eta)

    mdot_handles = [
        Line2D([0], [0], linestyle="None", marker=mdot_markers[float(level)], markersize=marker_size, markerfacecolor=mdot_color, markeredgecolor="black", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label=f"{float(level):g}")
        for level in mdot_levels
    ]
    leg_mdot = legend_axis_right.legend(
        handles=mdot_handles,
        title=r"$\dot{m}$ (mg/s)",
        loc="upper left",
        bbox_to_anchor=(MANSUR_THRUSTER_LEGEND_X_SHIFT, MANSUR_THRUSTER_MDOT_ANCHOR_Y),
        frameon=False,
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.9,
        labelspacing=0.75,
        handletextpad=0.45,
    )
    style_legend(leg_mdot)
    if leg_mdot is not None and leg_mdot.get_title() is not None:
        leg_mdot.get_title().set_fontweight("bold")
    legend_axis_right.add_artist(leg_mdot)

    ain_handles = [
        Line2D([0], [0], linestyle="None", marker=ain_markers[float(level)], markersize=marker_size, markerfacecolor=ain_color, markeredgecolor="black", markeredgewidth=UNIFORM_MARKER_EDGE_WIDTH, label=f"{float(level):g}")
        for level in ain_levels
    ]
    leg_ain = legend_axis_right.legend(
        handles=ain_handles,
        title=r"$A_i$ (m$^2$)",
        loc="upper left",
        bbox_to_anchor=(MANSUR_THRUSTER_LEGEND_X_SHIFT, MANSUR_THRUSTER_AIN_ANCHOR_Y),
        frameon=False,
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.9,
        labelspacing=0.75,
        handletextpad=0.45,
    )
    style_legend(leg_ain)
    if leg_ain is not None and leg_ain.get_title() is not None:
        leg_ain.get_title().set_fontweight("bold")

    if save and resolved_output_path is not None:
        figure.savefig(resolved_output_path, dpi=220, bbox_inches="tight")

    interactive_show = show and "agg" not in plt.get_backend().lower()
    if interactive_show:
        plt.show()

    if return_figure:
        return figure, (axis, legend_axis_left, legend_axis_right)

    if not interactive_show:
        plt.close(figure)

    if resolved_output_path is not None:
        return resolved_output_path
    return figure, (axis, legend_axis_left, legend_axis_right)


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
