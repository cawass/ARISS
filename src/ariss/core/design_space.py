from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import io
import sys
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import (
    load_spacecraft_from_base_config,
    logger as simulation_logger,
    run_sizing_loop,
)


@dataclass
class DesignPointRequirements:
    # Two-sided requirements for design-point extraction.
    eta_min: float | None = None
    eta_max: float | None = None
    mdot_min_mg_per_s: float | None = None
    mdot_max_mg_per_s: float | None = None
    ain_min_m2: float | None = None
    ain_max_m2: float | None = None
    altitude_min_km: float | None = None
    altitude_max_km: float | None = None


@dataclass
class DesignPointCandidate:
    isp_s: float
    tp_mN_per_kW: float
    eta: float
    mdot_mg_per_s: float
    ain_m2: float
    altitude_km: float
    score: float


@dataclass
class DesignMapResult:
    isp_values: np.ndarray
    efficiency_values: np.ndarray
    isp_grid: np.ndarray
    tp_mN_per_kW: np.ndarray
    eta: np.ndarray
    mdot_mg_per_s: np.ndarray
    ain_m2: np.ndarray
    altitude_km: np.ndarray
    converged: np.ndarray
    requirements: DesignPointRequirements
    eta_mask: np.ndarray
    mdot_mask: np.ndarray
    ain_mask: np.ndarray
    altitude_mask: np.ndarray
    feasible_mask: np.ndarray
    boundaries: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    design_point: DesignPointCandidate | None


def _load_plot_style():
    repo_root = Path(__file__).resolve().parents[3]
    validation_dir = repo_root / "tests" / "Validation"
    if validation_dir.exists():
        style_path = str(validation_dir)
        if style_path not in sys.path:
            sys.path.insert(0, style_path)

    try:
        from plot_style import PALETTE, apply_validation_style, style_axis, style_legend
    except Exception:
        PALETTE = {
            "secondary_text": "#4A4A4A",
            "l1_teal": "#5BC8D0",
            "sernn_pink": "#F08FA7",
            "choice_mid": "#C59A4A",
            "goal_dark": "#1E7F78",
            "cat_green": "#76C56E",
            "cat_red": "#E85C62",
        }

        def apply_validation_style():
            pass

        def style_axis(axis):
            axis.grid(True, linewidth=0.6, alpha=0.6)

        def style_legend(legend):
            return legend

    return PALETTE, apply_validation_style, style_axis, style_legend


def _extract_level_contours(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    finite = np.isfinite(field)
    if not np.any(finite):
        return []

    values = field[finite]
    fmin = float(np.min(values))
    fmax = float(np.max(values))
    if not (fmin <= float(level) <= fmax):
        return []

    fig = plt.figure(figsize=(4.0, 3.0), dpi=100)
    axis = fig.add_subplot(111)
    contour = axis.contour(
        x_grid,
        y_grid,
        np.ma.masked_invalid(field),
        levels=[float(level)],
    )

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in contour.allsegs[0]:
        if segment.shape[0] >= 2:
            lines.append((segment[:, 0].copy(), segment[:, 1].copy()))

    plt.close(fig)
    return lines


def _within_bounds(field: np.ndarray, lower: float | None, upper: float | None) -> np.ndarray:
    mask = np.isfinite(field)
    if lower is not None:
        mask &= field >= float(lower)
    if upper is not None:
        mask &= field <= float(upper)
    return mask


def _center_distance_score(
    values: np.ndarray,
    *,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    # Lower score is better. Prefers the center when both bounds are present.
    eps = 1.0e-12
    if lower is not None and upper is not None:
        lower_f = float(lower)
        upper_f = float(upper)
        center = 0.5 * (lower_f + upper_f)
        half_span = max(0.5 * (upper_f - lower_f), eps)
        return ((values - center) / half_span) ** 2
    if lower is not None:
        scale = max(abs(float(lower)), 1.0)
        return ((values - float(lower)) / scale) ** 2
    if upper is not None:
        scale = max(abs(float(upper)), 1.0)
        return ((float(upper) - values) / scale) ** 2
    return np.zeros_like(values, dtype=float)


def _format_range_label(
    symbol: str,
    lower: float | None,
    upper: float | None,
    *,
    unit: str = "",
) -> str:
    unit_txt = f" {unit}" if unit else ""
    if lower is not None and upper is not None:
        return f"{symbol} in [{float(lower):g}, {float(upper):g}]{unit_txt}"
    if lower is not None:
        return f"{symbol} >= {float(lower):g}{unit_txt}"
    if upper is not None:
        return f"{symbol} <= {float(upper):g}{unit_txt}"
    return f"{symbol} unconstrained"


def _select_design_point(
    requirements: DesignPointRequirements,
    feasible_mask: np.ndarray,
    isp_grid: np.ndarray,
    tp_grid: np.ndarray,
    eta: np.ndarray,
    mdot: np.ndarray,
    ain: np.ndarray,
    altitude: np.ndarray,
) -> DesignPointCandidate | None:
    if not np.any(feasible_mask):
        return None

    eta_score = _center_distance_score(
        eta,
        lower=requirements.eta_min,
        upper=requirements.eta_max,
    )
    mdot_score = _center_distance_score(
        mdot,
        lower=requirements.mdot_min_mg_per_s,
        upper=requirements.mdot_max_mg_per_s,
    )
    ain_score = _center_distance_score(
        ain,
        lower=requirements.ain_min_m2,
        upper=requirements.ain_max_m2,
    )
    alt_score = _center_distance_score(
        altitude,
        lower=requirements.altitude_min_km,
        upper=requirements.altitude_max_km,
    )

    score = np.where(
        feasible_mask,
        eta_score + mdot_score + ain_score + alt_score,
        np.nan,
    )

    idx = np.nanargmin(score)
    iy, ix = np.unravel_index(idx, score.shape)
    return DesignPointCandidate(
        isp_s=float(isp_grid[iy, ix]),
        tp_mN_per_kW=float(tp_grid[iy, ix]),
        eta=float(eta[iy, ix]),
        mdot_mg_per_s=float(mdot[iy, ix]),
        ain_m2=float(ain[iy, ix]),
        altitude_km=float(altitude[iy, ix]),
        score=float(score[iy, ix]),
    )


def run_design_map_thruster_efficiency_sweep(
    requirements: DesignPointRequirements,
    *,
    isp_values: Sequence[float],
    efficiency_values: Sequence[float],
    case_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
    max_iterations: int = 200,
) -> DesignMapResult:
    # Sweep only thruster efficiency (as requested), with Isp as the second axis.
    base_sc = load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    isp_arr = np.asarray(list(isp_values), dtype=float)
    eff_arr = np.asarray(list(efficiency_values), dtype=float)
    if isp_arr.size == 0 or eff_arr.size == 0:
        raise ValueError("isp_values and efficiency_values cannot be empty.")

    isp_grid = np.full((isp_arr.size, eff_arr.size), np.nan, dtype=float)
    tp_grid = np.full_like(isp_grid, np.nan, dtype=float)
    eta_grid = np.full_like(isp_grid, np.nan, dtype=float)
    mdot_grid = np.full_like(isp_grid, np.nan, dtype=float)
    ain_grid = np.full_like(isp_grid, np.nan, dtype=float)
    altitude_grid = np.full_like(isp_grid, np.nan, dtype=float)
    converged = np.zeros_like(isp_grid, dtype=bool)

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)

    try:
        for i, isp_s in enumerate(isp_arr):
            for j, eta in enumerate(eff_arr):
                sc = deepcopy(base_sc)
                sc.thruster.specific_impulse = float(isp_s)
                sc.thruster.eff = float(eta)

                with redirect_stdout(io.StringIO()):
                    final_sc, conv, _ = run_sizing_loop(sc, max_iterations=max_iterations)

                if not conv:
                    continue

                thrust = float(final_sc.thruster.thrust)
                power = float(final_sc.thruster.power)
                if not np.isfinite(power) or abs(power) < 1.0e-12:
                    continue

                isp_grid[i, j] = float(final_sc.thruster.specific_impulse)
                tp_grid[i, j] = 1.0e6 * thrust / power  # [mN/kW] (same numeric value as uN/W)
                eta_grid[i, j] = float(final_sc.thruster.eff)
                mdot_grid[i, j] = 1.0e6 * float(final_sc.thruster.m_flow)  # [mg/s]
                ain_grid[i, j] = float(final_sc.geometry.A_in)  # [m^2]
                altitude_grid[i, j] = float(final_sc.orbit.altitude)  # [km]
                converged[i, j] = True
    finally:
        simulation_logger.setLevel(old_level)

    eta_mask = _within_bounds(
        eta_grid,
        lower=requirements.eta_min,
        upper=requirements.eta_max,
    )
    mdot_mask = _within_bounds(
        mdot_grid,
        lower=requirements.mdot_min_mg_per_s,
        upper=requirements.mdot_max_mg_per_s,
    )
    ain_mask = _within_bounds(
        ain_grid,
        lower=requirements.ain_min_m2,
        upper=requirements.ain_max_m2,
    )
    altitude_mask = _within_bounds(
        altitude_grid,
        lower=requirements.altitude_min_km,
        upper=requirements.altitude_max_km,
    )

    feasible_mask = converged & eta_mask & mdot_mask & ain_mask & altitude_mask & np.isfinite(tp_grid)

    boundaries: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    if requirements.eta_min is not None:
        boundaries["eta_min"] = _extract_level_contours(tp_grid, isp_grid, eta_grid, float(requirements.eta_min))
    if requirements.eta_max is not None:
        boundaries["eta_max"] = _extract_level_contours(tp_grid, isp_grid, eta_grid, float(requirements.eta_max))
    if requirements.mdot_min_mg_per_s is not None:
        boundaries["mdot_min"] = _extract_level_contours(
            tp_grid, isp_grid, mdot_grid, float(requirements.mdot_min_mg_per_s)
        )
    if requirements.mdot_max_mg_per_s is not None:
        boundaries["mdot_max"] = _extract_level_contours(
            tp_grid, isp_grid, mdot_grid, float(requirements.mdot_max_mg_per_s)
        )
    if requirements.ain_min_m2 is not None:
        boundaries["ain_min"] = _extract_level_contours(
            tp_grid, isp_grid, ain_grid, float(requirements.ain_min_m2)
        )
    if requirements.ain_max_m2 is not None:
        boundaries["ain_max"] = _extract_level_contours(
            tp_grid, isp_grid, ain_grid, float(requirements.ain_max_m2)
        )
    if requirements.altitude_min_km is not None:
        boundaries["altitude_min"] = _extract_level_contours(
            tp_grid, isp_grid, altitude_grid, float(requirements.altitude_min_km)
        )
    if requirements.altitude_max_km is not None:
        boundaries["altitude_max"] = _extract_level_contours(
            tp_grid, isp_grid, altitude_grid, float(requirements.altitude_max_km)
        )

    design_point = _select_design_point(
        requirements=requirements,
        feasible_mask=feasible_mask,
        isp_grid=isp_grid,
        tp_grid=tp_grid,
        eta=eta_grid,
        mdot=mdot_grid,
        ain=ain_grid,
        altitude=altitude_grid,
    )

    return DesignMapResult(
        isp_values=isp_arr,
        efficiency_values=eff_arr,
        isp_grid=isp_grid,
        tp_mN_per_kW=tp_grid,
        eta=eta_grid,
        mdot_mg_per_s=mdot_grid,
        ain_m2=ain_grid,
        altitude_km=altitude_grid,
        converged=converged,
        requirements=requirements,
        eta_mask=eta_mask,
        mdot_mask=mdot_mask,
        ain_mask=ain_mask,
        altitude_mask=altitude_mask,
        feasible_mask=feasible_mask,
        boundaries=boundaries,
        design_point=design_point,
    )


def plot_design_map_boundaries(
    result: DesignMapResult,
    *,
    title: str | None = None,
    show: bool = True,
):
    # Plot only requirement boundaries and fill the common intersection area.
    PALETTE, apply_validation_style, style_axis, style_legend = _load_plot_style()
    apply_validation_style()

    fig = plt.figure(figsize=(11.8, 5.2), dpi=150)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.40), wspace=0.05)
    axis = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    def _marker_indices_by_arclength(
        x_curve: np.ndarray,
        y_curve: np.ndarray,
        *,
        n_markers: int = 3,
        endpoint_margin: float = 0.12,
    ) -> np.ndarray:
        """
        Place markers based on geometric distance along the contour, not raw
        point index. That keeps spacing visually uniform even when the contour
        sampling density is irregular.
        """
        n = int(len(x_curve))
        if n == 0:
            return np.asarray([], dtype=int)
        if n <= 2:
            return np.arange(n, dtype=int)

        seg_len = np.hypot(np.diff(x_curve), np.diff(y_curve))
        s = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_len = float(s[-1])

        if not np.isfinite(total_len) or total_len <= 1.0e-12:
            # Fallback for degenerate segments.
            interior = np.arange(1, max(n - 1, 1), dtype=int)
            if interior.size == 0:
                return np.asarray([0], dtype=int)
            count = min(max(n_markers, 1), interior.size)
            return np.unique(
                np.linspace(interior[0], interior[-1], count, dtype=int)
            )

        max_interior = max(n - 2, 1)
        count = min(max(n_markers, 1), max_interior)

        start_s = endpoint_margin * total_len
        end_s = (1.0 - endpoint_margin) * total_len
        if end_s <= start_s:
            start_s, end_s = 0.0, total_len

        targets = np.linspace(start_s, end_s, count)
        idx = np.searchsorted(s, targets, side="left")
        idx = np.clip(idx, 1, n - 2)

        prev_idx = np.clip(idx - 1, 0, n - 1)
        use_prev = np.abs(s[prev_idx] - targets) < np.abs(s[idx] - targets)
        idx = np.where(use_prev, prev_idx, idx)

        return np.unique(idx.astype(int))

    def _plot_boundary_with_symbol(
        x_curve: np.ndarray,
        y_curve: np.ndarray,
        *,
        color: str,
        marker_text: str,
        zorder: float,
    ) -> None:
        # Base contour line.
        axis.plot(
            x_curve,
            y_curve,
            color=color,
            lw=2.1,
            ls="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )

        idx = _marker_indices_by_arclength(
            x_curve,
            y_curve,
            n_markers=3,
            endpoint_margin=0.12,
        )
        if idx.size == 0:
            return

        x_mark = np.asarray(x_curve, dtype=float)[idx]
        y_mark = np.asarray(y_curve, dtype=float)[idx]

        # White knockout so the marker stays readable over the fill/grid/line.
        axis.scatter(
            x_mark,
            y_mark,
            s=92,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
            zorder=zorder + 0.25,
            clip_on=True,
        )

        # Clean symbol on top.
        axis.scatter(
            x_mark,
            y_mark,
            s=70,
            marker=marker_text,   # '$+$' or '$-$'
            c=color,
            zorder=zorder + 0.35,
            clip_on=True,
        )

    def _make_boundary_legend_handle(
        *,
        color: str,
        marker_text: str,
        label: str,
    ) -> tuple[tuple[Line2D, Line2D, Line2D], str]:
        return (
            (
                Line2D([0], [0], color=color, lw=2.1),
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    marker="o",
                    markersize=8.2,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    marker=marker_text,
                    markersize=7.8,
                    color=color,
                ),
            ),
            label,
        )

    common_region_plotted = False
    if np.any(result.feasible_mask):
        masked_common = np.ma.masked_where(
            ~result.feasible_mask,
            result.feasible_mask.astype(float),
        )
        axis.contourf(
            result.tp_mN_per_kW,
            result.isp_grid,
            masked_common,
            levels=[0.5, 1.5],
            colors=[PALETTE.get("cat_green", "#76C56E")],
            alpha=0.30,
            zorder=0.8,
        )
        common_region_plotted = True

    curve_specs = [
        {
            "base_key": "eta",
            "min_key": "eta_min",
            "max_key": "eta_max",
            "color": PALETTE.get("l1_teal", "#5BC8D0"),
            "symbol": "eta",
            "unit": "",
            "min_value": result.requirements.eta_min,
            "max_value": result.requirements.eta_max,
        },
        {
            "base_key": "mdot",
            "min_key": "mdot_min",
            "max_key": "mdot_max",
            "color": PALETTE.get("sernn_pink", "#F08FA7"),
            "symbol": "m_dot",
            "unit": "mg/s",
            "min_value": result.requirements.mdot_min_mg_per_s,
            "max_value": result.requirements.mdot_max_mg_per_s,
        },
        {
            "base_key": "ain",
            "min_key": "ain_min",
            "max_key": "ain_max",
            "color": PALETTE.get("choice_mid", "#C59A4A"),
            "symbol": "A_in",
            "unit": "m^2",
            "min_value": result.requirements.ain_min_m2,
            "max_value": result.requirements.ain_max_m2,
        },
        {
            "base_key": "altitude",
            "min_key": "altitude_min",
            "max_key": "altitude_max",
            "color": PALETTE.get("goal_dark", "#1E7F78"),
            "symbol": "h",
            "unit": "km",
            "min_value": result.requirements.altitude_min_km,
            "max_value": result.requirements.altitude_max_km,
        },
    ]

    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for spec in curve_specs:
        color = spec["color"]
        symbol = spec["symbol"]
        unit = spec["unit"]

        min_value = spec["min_value"]
        min_lines = result.boundaries.get(spec["min_key"], [])
        if min_value is not None and min_lines:
            handle, label = _make_boundary_legend_handle(
                color=color,
                marker_text="$-$",
                label=_format_range_label(symbol, min_value, None, unit=unit),
            )
            legend_handles.append(handle)
            legend_labels.append(label)

            for x_curve, y_curve in min_lines:
                _plot_boundary_with_symbol(
                    x_curve,
                    y_curve,
                    color=color,
                    marker_text="$-$",
                    zorder=3.0,
                )

        max_value = spec["max_value"]
        max_lines = result.boundaries.get(spec["max_key"], [])
        if max_value is not None and max_lines:
            handle, label = _make_boundary_legend_handle(
                color=color,
                marker_text="$+$",
                label=_format_range_label(symbol, None, max_value, unit=unit),
            )
            legend_handles.append(handle)
            legend_labels.append(label)

            for x_curve, y_curve in max_lines:
                _plot_boundary_with_symbol(
                    x_curve,
                    y_curve,
                    color=color,
                    marker_text="$+$",
                    zorder=3.0,
                )

    if result.design_point is not None:
        axis.scatter(
            [result.design_point.tp_mN_per_kW],
            [result.design_point.isp_s],
            marker="o",
            s=44,
            facecolors=PALETTE.get("cat_red", "#E85C62"),
            edgecolors=PALETTE.get("secondary_text", "#4A4A4A"),
            linewidths=0.8,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=PALETTE.get("cat_red", "#E85C62"),
                markeredgecolor=PALETTE.get("secondary_text", "#4A4A4A"),
                markeredgewidth=0.8,
            )
        )
        legend_labels.append("Design point")

    if common_region_plotted:
        legend_handles.append(
            Patch(
                facecolor=PALETTE.get("cat_green", "#76C56E"),
                edgecolor="none",
                alpha=0.30,
            )
        )
        legend_labels.append("Feasible design space")

    axis.set_xlabel("T/P [mN/kW]")
    axis.set_ylabel("Isp [s]")
    if title is not None:
        axis.set_title(title)

    finite_tp = result.tp_mN_per_kW[np.isfinite(result.tp_mN_per_kW)]
    finite_isp = result.isp_grid[np.isfinite(result.isp_grid)]
    if finite_tp.size > 0:
        x_pad = 0.05 * max(float(np.max(finite_tp) - np.min(finite_tp)), 1.0)
        axis.set_xlim(float(np.min(finite_tp) - x_pad), float(np.max(finite_tp) + x_pad))
    if finite_isp.size > 0:
        y_pad = 0.05 * max(float(np.max(finite_isp) - np.min(finite_isp)), 1.0)
        axis.set_ylim(float(np.min(finite_isp) - y_pad), float(np.max(finite_isp) + y_pad))

    style_axis(axis)

    legend = legend_axis.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Boundaries",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.4,
        handletextpad=0.6,
        handler_map={tuple: HandlerTuple(ndivide=1)},
    )
    style_legend(legend)
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.93)
    if show:
        plt.show(block=True)

    return fig, axis


if __name__ == "__main__":
    requirements = DesignPointRequirements(
        eta_min=0.45,
        eta_max=0.85,
        mdot_min_mg_per_s=0.8,
        mdot_max_mg_per_s=2.0,
        ain_min_m2=0.2,
        ain_max_m2=0.6,
        altitude_min_km=150.0,
        altitude_max_km=260.0,
    )

    result = run_design_map_thruster_efficiency_sweep(
        requirements=requirements,
        isp_values=np.linspace(1500.0, 6000.0, 80),
        efficiency_values=np.geomspace(0.05, 1.0, 80),
        max_iterations=200,
    )

    plot_design_map_boundaries(
        result,
        title="Design Point Map (eta sweep only)",
        show=True,
    )
