# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
# ============================================================================== #
#  ARISS - Atmospheric Refueling Iterative System Solver
# -----------------------------------------------------------------------------
#  Description:
#      Atmospheric refueling power model based on intake mass flow and tank
#
#  Project:        ARISS
#  Module:         history_ui.py
#  Author:         Carlos Carrasco Requejo
#
#  Notes:
#      Refactor performed to:
#        - enforce true equal-scale 3D axes in Matplotlib,
#        - reduce repeated panel-drawing logic,
#        - remove unused helper functions,
#        - add comments around the less obvious geometry/plotting paths.
# ============================================================================

from __future__ import annotations

import io
import sys
import traceback
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass, fields, is_dataclass
from math import ceil, pi, sqrt
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:  # pragma: no cover - UI optional in test environments
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None

from ariss.core.simulation import (
    compute_drag_diagnostics,
    load_spacecraft_from_base_config,
    run_sizing_loop,
)
from ariss.core.spacecraft import GeometryState, SpacecraftState
from ariss.modules.Thermal import ThermalDiagnostics, thermal_model
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmosphere_properties_from_height, atmos

NASA_BG = "#ffffff"
NASA_PANEL = "#f4efe2"
NASA_GRID = "#b9b1a2"
NASA_TEXT = "#1c2833"
HISTORY_PLOT_COLORS = ["#0f4c81", "#d95d39", "#2a9d8f", "#d4a017", "#6d597a", "#457b9d", "#bc4749", "#5f6f52"]
NASA_LINE = ["#1a7bc0", "#c44e52", "#dd8452", "#4c72b0", "#55a868", "#8172b2", "#937860"]
GEOM_BODY = "#f5efe4"
GEOM_INTAKE = "#ece6da"
GEOM_SOLAR = "#2f6db3"
GEOM_RAD = "#f2f5f7"
GEOM_PROP = "#c44e52"
GEOM_EDGE = "#8f877b"
WAKE_CMAP = cm.get_cmap("RdYlGn_r")
WAKE_NORM = colors.Normalize(vmin=0.0, vmax=1.0)

PlotSpec = tuple[str | Sequence[str], str, bool]
SpacecraftInput = SpacecraftState | str | PathLike[str]
DEFAULT_HISTORY_SPECS: list[PlotSpec] = [
    ("orbit.altitude", "ORBITAL HEIGHT", False),
    (["power.Power_total", "power.Power_in", "power.Power_body", "power.Power_solar", "power.Power_rad", "power.Power_prop", "power.Power_ADCS", "power.Power_payload", "power.Power_refprop"], "POWER BUDGETS", False),
    (["mass.Mass_total", "mass.Mass_in", "mass.Mass_body", "mass.Mass_solar", "mass.Mass_rad", "mass.Mass_prop", "mass.Mass_ADCS", "mass.Mass_payload", "mass.Mass_refprop"], "MASS BUDGETS", False),
    (["geometry.A_body", "geometry.A_in", "geometry.A_in_drag", "geometry.A_prop", "geometry.A_solar", "geometry.L_body", "geometry.L_in"], "KEY GEOMETRY", False),
]

ATMOSPHERE_SPECS: list[PlotSpec] = [("orbit.altitude", "ALTITUDE", False), ("orbit.density", "DENSITY", True), ("orbit.temperature", "TEMPERATURE", False), ("orbit.molar_mass", "MOLAR MASS", True), ("orbit.velocity", "ORBITAL VELOCITY", False), ("drag.drag_total", "TOTAL DRAG", True)]

BUDGET_SPECS: list[PlotSpec] = [("mass.Mass_total", "TOTAL MASS", False), ("mass.Mass_in", "INLET MASS", False), ("mass.Mass_solar", "SOLAR ARRAY MASS", False), ("power.Power_total", "TOTAL POWER", False), ("power.Power_prop", "PROPULSION POWER", True), ("power.Power_solar", "SOLAR POWER", True)]

DIMENSION_SPECS: list[PlotSpec] = [("geometry.A_in", "INTAKE AREA", False), ("geometry.A_in_drag", "DRAG INTAKE AREA", False), ("geometry.A_prop", "PROPULSIVE AREA", False), ("geometry.A_solar", "SOLAR AREA", False), ("geometry.L_in", "INTAKE LENGTH", False), ("geometry.L_body", "BODY LENGTH", False), ("geometry.AR_in", "INTAKE ASPECT RATIO", False)]

DRAG_SPECS: list[PlotSpec] = [("drag.drag_total", "TOTAL DRAG", True), ("drag.drag_body_side", "BODY SIDE DRAG", True), ("drag.drag_inlet_side", "INLET SIDE DRAG", True), ("drag.drag_inlet_front", "INLET FRONT DRAG", True), ("drag.drag_solar", "SOLAR DRAG", True), ("geometry.A_in_drag", "DRAG INTAKE AREA", False), ("orbit.density", "DENSITY", True)]

POWER_SPECS: list[PlotSpec] = [("power.Power_total", "TOTAL POWER", False), ("power.Power_prop", "PROPULSION POWER", True), ("power.Power_solar", "SOLAR POWER", True), ("geometry.A_solar", "SOLAR ARRAY AREA", False), ("solar.eta_power", "POWER EFFICIENCY", False), ("solar.av_aligment", "ARRAY ALIGNMENT", False)]

PROPULSION_SPECS: list[PlotSpec] = [("geometry.A_prop", "REQUIRED PROPULSIVE AREA", False), ("geometry.A_in", "INTAKE AREA", False), ("geometry.A_in_drag", "DRAG INTAKE AREA", False), ("thruster.power", "POWER REQUIRED", True), ("thruster.thrust", "THRUST", True), ("thruster.m_flow", "PROPELLANT MASS FLOW", True), ("orbit.density", "INFERRED DENSITY", True)]

SIM_BUDGET_SPECS: list[PlotSpec] = [("mass.Mass_total", "TOTAL MASS", False), ("power.Power_total", "TOTAL POWER", False), ("drag.drag_total", "TOTAL DRAG", True), ("drag.drag_inlet_front", "FRONT INLET DRAG", True), ("geometry.A_prop", "PROPULSIVE AREA", False), ("orbit.altitude", "ALTITUDE", False), ("orbit.density", "DENSITY", True)]


@dataclass(frozen=True)
class _SectionShape:
    width: float
    height: float
    semi_y: float
    semi_z: float
    is_square: bool


# ---------------------------------------------------------------------------
# History and data flattening helpers
# ---------------------------------------------------------------------------
def run_sizing_with_history(
    sc: SpacecraftState,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
) -> tuple[SpacecraftState, bool, list[SpacecraftState]]:
    """Run the same sizing loop as ``simulation.py`` and retain its history."""
    return run_sizing_loop(
        sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
    )


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Flatten dataclass / dict trees into dotted numeric channels."""
    if is_dataclass(value):
        field_names: set[str] = set()
        for field_info in fields(value):
            field_names.add(field_info.name)
            child = getattr(value, field_info.name)
            child_prefix = f"{prefix}.{field_info.name}" if prefix else field_info.name
            _flatten_numeric(child_prefix, child, out)
        for name, child in vars(value).items():
            if name in field_names:
                continue
            child_prefix = f"{prefix}.{name}" if prefix else name
            _flatten_numeric(child_prefix, child, out)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(child_prefix, child, out)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def _history_series(history: list[SpacecraftState]) -> dict[str, list[float]]:
    """Build time-series arrays for every numeric state channel seen in history."""
    series: dict[str, list[float]] = {}
    for index, state in enumerate(history):
        current: dict[str, float] = {}
        _flatten_numeric("", state, current)

        for key, values in series.items():
            if key not in current:
                values.append(float("nan"))

        for key, value in current.items():
            if key not in series:
                series[key] = [float("nan")] * index
            series[key].append(value)

    return {key: series[key] for key in sorted(series)}


def _resolve_spacecraft_input(sc: SpacecraftInput | None) -> SpacecraftState:
    if sc is None:
        return load_spacecraft_from_base_config()
    if isinstance(sc, SpacecraftState):
        return sc
    return load_spacecraft_from_base_config(Path(sc))


def _spacecraft_case_name(sc: SpacecraftState) -> str:
    return str(getattr(sc, "name", "")).strip()


def _normalize_paths(path: str | Sequence[str] | None, available_paths: list[str]) -> list[str]:
    if path is None:
        default_path = "mass.Mass_total" if "mass.Mass_total" in available_paths else available_paths[0]
        return [default_path]

    candidates = [path] if isinstance(path, str) else list(path)
    selected = [candidate for candidate in candidates if candidate in available_paths]
    if selected:
        return selected

    default_path = "mass.Mass_total" if "mass.Mass_total" in available_paths else available_paths[0]
    return [default_path]


def _default_title(selected_paths: list[str]) -> str:
    if len(selected_paths) == 1:
        return selected_paths[0]
    if len(selected_paths) == 2:
        return f"{selected_paths[0]} / {selected_paths[1]}"
    return f"{selected_paths[0]} + {len(selected_paths) - 1} more"


def _history_plot_color(index: int) -> str:
    return HISTORY_PLOT_COLORS[index % len(HISTORY_PLOT_COLORS)]


def _selected_listbox_values(listbox) -> list[str]:
    return [listbox.get(i) for i in listbox.curselection()]


def _style_legend_text(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_color(NASA_TEXT)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(np.nan_to_num(value, nan=default, posinf=default, neginf=default))


def _positive(value: float, floor: float = 1.0e-6) -> float:
    return max(abs(float(value)), floor)


# ---------------------------------------------------------------------------
# Geometry sizing helpers
# ---------------------------------------------------------------------------
def _rect_dims(area: float, aspect_ratio: float) -> tuple[float, float]:
    area = _positive(area)
    aspect_ratio = _positive(aspect_ratio)
    width = sqrt(area * aspect_ratio)
    height = area / width
    return width, height


def _panel_planform_dims(area: float, aspect_ratio: float) -> tuple[float, float]:
    area = _positive(area)
    aspect_ratio = _positive(aspect_ratio)
    length = sqrt(area / aspect_ratio)
    width = area / length
    return width, length


def _mounted_axial_bounds(center_from_aft: float, length: float, mount_length: float) -> tuple[float, float]:
    length = _positive(length)
    half_length = 0.5 * length
    center = float(np.clip(center_from_aft, 0.0, _positive(mount_length)))
    return center - half_length, center + half_length


def _ellipse_radii(area: float, aspect_ratio: float) -> tuple[float, float]:
    area = _positive(area)
    aspect_ratio = _positive(aspect_ratio)
    semi_y = sqrt(area * aspect_ratio / pi)
    semi_z = area / (pi * semi_y)
    return semi_y, semi_z


def _is_square(shape: str) -> bool:
    normalized = (shape or "").strip().lower()
    return normalized.startswith("s") or normalized.startswith("r")


def _shape_power(shape_code: str, square_power: float = 24.0) -> float:
    return square_power if _is_square(shape_code) else 2.0


def _section_dims(area: float, aspect_ratio: float, shape_code: str) -> tuple[float, float]:
    if area <= 0.0 or aspect_ratio <= 0.0:
        return 0.0, 0.0
    if _is_square(shape_code):
        height = sqrt(area / aspect_ratio)
        width = aspect_ratio * height
    else:
        height = sqrt(4.0 * area / (pi * aspect_ratio))
        width = aspect_ratio * height
    return float(width), float(height)


def _local_section(
    x: float,
    body_length: float,
    intake_length: float,
    body_width: float,
    body_height: float,
    body_power: float,
    intake_width: float,
    intake_height: float,
    intake_power: float,
) -> tuple[float, float, float]:
    """Return local super-ellipse dimensions at an axial location."""
    if x <= body_length or intake_length <= 0.0:
        return body_width, body_height, body_power
    progress = float(np.clip((x - body_length) / intake_length, 0.0, 1.0))
    width = body_width + progress * (intake_width - body_width)
    height = body_height + progress * (intake_height - body_height)
    power = body_power + progress * (intake_power - body_power)
    return float(width), float(height), float(power)


def _superellipse_half_span(
    coord: np.ndarray | float,
    semi_primary: float,
    semi_other: float,
    power: float,
) -> np.ndarray:
    coord_array = np.asarray(coord, dtype=float)
    out = np.zeros_like(coord_array)
    if semi_primary <= 0.0 or semi_other <= 0.0 or power <= 0.0:
        return out
    ratio = np.abs(coord_array) / semi_primary
    inside = ratio <= 1.0
    out[inside] = semi_other * np.power(1.0 - np.power(ratio[inside], power), 1.0 / power)
    return out


def _shape_from_geometry(area: float, aspect_ratio: float, shape: str) -> _SectionShape:
    if _is_square(shape):
        width, height = _rect_dims(area, aspect_ratio)
        return _SectionShape(width, height, 0.5 * width, 0.5 * height, True)
    semi_y, semi_z = _ellipse_radii(area, aspect_ratio)
    return _SectionShape(2.0 * semi_y, 2.0 * semi_z, semi_y, semi_z, False)


def _interpolate_shape(start: _SectionShape, end: _SectionShape, progress: float) -> _SectionShape:
    t = float(np.clip(progress, 0.0, 1.0))
    width = start.width + (end.width - start.width) * t
    height = start.height + (end.height - start.height) * t
    semi_y = start.semi_y + (end.semi_y - start.semi_y) * t
    semi_z = start.semi_z + (end.semi_z - start.semi_z) * t
    return _SectionShape(width, height, semi_y, semi_z, start.is_square and end.is_square)


def _shape_at_x(
    x_geom: float,
    body_length: float,
    intake_length: float,
    body_shape: _SectionShape,
    intake_shape: _SectionShape,
) -> _SectionShape:
    if intake_length <= 0.0 or x_geom <= body_length:
        return body_shape
    progress = (x_geom - body_length) / intake_length
    return _interpolate_shape(body_shape, intake_shape, progress)


# ---------------------------------------------------------------------------
# Generic plotting helpers
# ---------------------------------------------------------------------------
def _style_2d_axis(axis) -> None:
    """Apply the shared 2D styling used across history, drag, and thermal views."""
    axis.set_facecolor(NASA_BG)
    axis.grid(True, color=NASA_GRID, alpha=0.65, linestyle="--", linewidth=0.7)
    axis.tick_params(colors=NASA_TEXT, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(NASA_TEXT)


def _box_faces(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> list[list[tuple[float, float, float]]]:
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _add_poly3d(
    axis,
    faces: Sequence[Sequence[Sequence[float]]],
    color: Any,
    alpha: float = 0.75,
    edgecolor: str = NASA_TEXT,
    linewidth: float = 0.9,
    zorder: float | None = None,
) -> None:
    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    if zorder is not None:
        collection.set_zorder(zorder)
        collection.set_zsort("max")
        collection.set_sort_zpos(zorder)
    axis.add_collection3d(collection)


def _add_box(
    axis,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    color: Any,
    alpha: float = 0.75,
    edgecolor: str = NASA_TEXT,
    linewidth: float = 0.9,
    zorder: float | None = None,
) -> None:
    _add_poly3d(axis, _box_faces(x0, x1, y0, y1, z0, z1), color, alpha, edgecolor, linewidth, zorder)


def _add_box_edges(
    axis,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    color: str,
    linewidth: float = 1.3,
    zorder: float | None = None,
) -> None:
    vertices = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for start, end in edges:
        axis.plot(
            [vertices[start][0], vertices[end][0]],
            [vertices[start][1], vertices[end][1]],
            [vertices[start][2], vertices[end][2]],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )


def _add_tapered_box(
    axis,
    x0: float,
    x1: float,
    front_width: float,
    front_height: float,
    rear_width: float,
    rear_height: float,
    color: Any,
    alpha: float = 0.75,
    edgecolor: str = NASA_TEXT,
    linewidth: float = 0.9,
    zorder: float | None = None,
) -> None:
    f0 = (x0, -0.5 * front_width, -0.5 * front_height)
    f1 = (x0, 0.5 * front_width, -0.5 * front_height)
    f2 = (x0, 0.5 * front_width, 0.5 * front_height)
    f3 = (x0, -0.5 * front_width, 0.5 * front_height)
    r0 = (x1, -0.5 * rear_width, -0.5 * rear_height)
    r1 = (x1, 0.5 * rear_width, -0.5 * rear_height)
    r2 = (x1, 0.5 * rear_width, 0.5 * rear_height)
    r3 = (x1, -0.5 * rear_width, 0.5 * rear_height)
    faces = [[f0, f1, f2, f3], [r0, r1, r2, r3], [f0, f1, r1, r0], [f1, f2, r2, r1], [f2, f3, r3, r2], [f3, f0, r0, r3]]
    _add_poly3d(axis, faces, color, alpha, edgecolor, linewidth, zorder)


def _add_tapered_elliptic_tube(
    axis,
    x0: float,
    length: float,
    front_semi_y: float,
    front_semi_z: float,
    rear_semi_y: float,
    rear_semi_z: float,
    color: Any,
    alpha: float = 0.75,
    edgecolor: str = NASA_TEXT,
    linewidth: float = 0.3,
    zorder: float | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    x = np.linspace(x0, x0 + length, 20)
    progress = np.linspace(0.0, 1.0, len(x))
    semi_y = front_semi_y + (rear_semi_y - front_semi_y) * progress
    semi_z = front_semi_z + (rear_semi_z - front_semi_z) * progress
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="ij")
    y_grid = np.tile(semi_y, (len(theta), 1)) * np.cos(theta_grid)
    z_grid = np.tile(semi_z, (len(theta), 1)) * np.sin(theta_grid)
    surface = axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        shade=False,
    )
    if zorder is not None:
        surface.set_zorder(zorder)
        surface.set_zsort("max")
        surface.set_sort_zpos(zorder)


def _add_tapered_top_face(
    axis,
    x0: float,
    x1: float,
    front_width: float,
    rear_width: float,
    front_height: float,
    rear_height: float,
    color: Any,
    alpha: float = 0.92,
    edgecolor: str = "none",
    linewidth: float = 0.4,
    zorder: float | None = None,
) -> None:
    epsilon = 1.0e-3 * max(front_height, rear_height, 1.0)
    front_z = 0.5 * front_height + epsilon
    rear_z = 0.5 * rear_height + epsilon
    face = [(x0, -0.5 * front_width, front_z), (x0, 0.5 * front_width, front_z), (x1, 0.5 * rear_width, rear_z), (x1, -0.5 * rear_width, rear_z)]
    _add_poly3d(axis, [face], color, alpha, edgecolor, linewidth, zorder)


def _add_tapered_elliptic_shell(
    axis,
    x0: float,
    length: float,
    front_semi_y: float,
    front_semi_z: float,
    rear_semi_y: float,
    rear_semi_z: float,
    theta_min: float,
    theta_max: float,
    color: Any,
    alpha: float = 0.9,
    edgecolor: str = "none",
    linewidth: float = 0.0,
    zorder: float | None = None,
) -> None:
    theta = np.linspace(theta_min, theta_max, 32)
    x = np.linspace(x0, x0 + length, 20)
    progress = np.linspace(0.0, 1.0, len(x))
    semi_y = front_semi_y + (rear_semi_y - front_semi_y) * progress
    semi_z = front_semi_z + (rear_semi_z - front_semi_z) * progress
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="ij")
    y_grid = np.tile(semi_y, (len(theta), 1)) * np.cos(theta_grid)
    z_grid = np.tile(semi_z, (len(theta), 1)) * np.sin(theta_grid)
    surface = axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        shade=False,
    )
    if zorder is not None:
        surface.set_zorder(zorder)
        surface.set_zsort("max")
        surface.set_sort_zpos(zorder)


def _add_rectangle_outline(
    axis,
    x: float,
    width: float,
    height: float,
    color: Any,
    zorder: float | None = None,
) -> None:
    y0, y1 = -0.5 * width, 0.5 * width
    z0, z1 = -0.5 * height, 0.5 * height
    axis.plot([x, x, x, x, x], [y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color=color, linewidth=2.0, zorder=zorder)


def _extend_bounds(
    bounds: dict[str, list[float]],
    x_values: Sequence[float],
    y_values: Sequence[float],
    z_values: Sequence[float],
) -> None:
    bounds["x"].extend(x_values)
    bounds["y"].extend(y_values)
    bounds["z"].extend(z_values)


def _set_equal_limits(axis, bounds: dict[str, list[float]], padding: float = 0.05) -> None:
    """
    Force equal scaling in all three axes.

    ``mplot3d`` often looks stretched even when limits are symmetric. The extra
    ``set_box_aspect((1, 1, 1))`` call is what actually fixes the rendered cube.
    """
    x_min, x_max = min(bounds["x"]), max(bounds["x"])
    y_min, y_max = min(bounds["y"]), max(bounds["y"])
    z_min, z_max = min(bounds["z"]), max(bounds["z"])

    x_span = max(x_max - x_min, 1.0e-9)
    y_span = max(y_max - y_min, 1.0e-9)
    z_span = max(z_max - z_min, 1.0e-9)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    radius = 0.5 * max(x_span, y_span, z_span, 1.0) * (1.0 + padding)
    axis.set_xlim(x_mid - radius, x_mid + radius)
    axis.set_ylim(y_mid - radius, y_mid + radius)
    axis.set_zlim(z_mid - radius, z_mid + radius)

    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect((1.0, 1.0, 1.0))


def _style_3d_axis(axis, title: str) -> None:
    axis.set_facecolor(NASA_BG)
    if hasattr(axis, "computed_zorder"):
        axis.computed_zorder = False

    axis.set_title(title, color=NASA_TEXT, fontsize=11, fontfamily="Courier New")
    axis.set_xlabel("X [m]", color=NASA_TEXT, fontsize=9)
    axis.set_ylabel("Y [m]", color=NASA_TEXT, fontsize=9)
    axis.set_zlabel("Z [m]", color=NASA_TEXT, fontsize=9)
    axis.tick_params(colors=NASA_TEXT, labelsize=8)
    axis.view_init(elev=24, azim=-58)

    grid_color = (185 / 255, 177 / 255, 162 / 255, 0.4)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor(NASA_GRID)

    for ax in (axis.xaxis, axis.yaxis, axis.zaxis):
        try:
            ax._axinfo["grid"]["color"] = grid_color
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Wake / drag helpers
# ---------------------------------------------------------------------------
def _wake_profile_arrays(
    total_length: float,
    geometry: GeometryState,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return piecewise wake-factor arrays along the spacecraft length."""
    if x_values is None or np.asarray(x_values, dtype=float).size == 0:
        if total_length > 0.0:
            x_drag = np.array([0.0, total_length], dtype=float)
        else:
            x_drag = np.array([0.0], dtype=float)
    else:
        x_drag = np.clip(np.asarray(x_values, dtype=float), 0.0, total_length)
        x_drag = np.unique(np.sort(x_drag))

    body_end = float(np.clip(_safe_float(geometry.L_body), 0.0, total_length))
    wake_body = float(getattr(geometry, "wake_body", 1.0))
    wake_in = float(getattr(geometry, "wake_in", 1.0))
    profile = np.where(x_drag < body_end, wake_body, wake_in).astype(float)
    return x_drag, profile.copy(), profile.copy()


def _wake_fraction_at_x(
    x_geom: float,
    total_length: float,
    x_drag: np.ndarray,
    fy_profile: np.ndarray,
    fz_profile: np.ndarray,
) -> tuple[float, float, float]:
    if total_length <= 0.0 or x_drag.size == 0:
        return 1.0, 1.0, 1.0
    x_drag_value = float(np.clip(x_geom, 0.0, total_length))
    fy = float(np.interp(x_drag_value, x_drag, fy_profile))
    fz = float(np.interp(x_drag_value, x_drag, fz_profile))
    return fy, fz, 0.5 * (fy + fz)


def _wake_color(value: float) -> tuple[float, float, float, float]:
    clipped = float(np.clip(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
    return WAKE_CMAP(WAKE_NORM(clipped))


def _drag_geometry_samples(
    body_length: float,
    intake_length: float,
    max_segments: int = 80,
) -> np.ndarray:
    total_length = body_length + intake_length
    anchors = np.array([0.0, body_length, total_length], dtype=float)
    if total_length <= 0.0:
        return np.unique(anchors)

    samples = np.linspace(0.0, total_length, max_segments + 1, dtype=float)
    samples = np.unique(np.concatenate([anchors, samples]))
    samples = samples[(samples >= 0.0) & (samples <= total_length)]
    samples.sort()
    return samples


def _plot_drag_slice_lines(
    axis,
    geometry: GeometryState,
    n_x: int = 140,
    n_cuts: int = 7,
) -> None:
    total_length = max(_safe_float(geometry.L_body + geometry.L_in), 0.0)
    if total_length <= 0.0 or n_x < 2 or n_cuts < 1:
        return

    w_in, h_in = _section_dims(geometry.A_in, geometry.AR_in, geometry.S_in)
    w_body, h_body = _section_dims(geometry.A_body, geometry.AR_body, geometry.S_body)
    p_body = _shape_power(geometry.S_body)
    p_in = _shape_power(geometry.S_in)

    max_half_width = 0.5 * max(w_body, w_in)
    max_half_height = 0.5 * max(h_body, h_in)
    if max_half_width <= 0.0 or max_half_height <= 0.0:
        return

    x_values = np.linspace(0.0, total_length, n_x, dtype=float)
    y_cuts = np.linspace(-0.9 * max_half_width, 0.9 * max_half_width, n_cuts, dtype=float)
    z_cuts = np.linspace(-0.9 * max_half_height, 0.9 * max_half_height, n_cuts, dtype=float)

    for y_cut in y_cuts:
        z_plus = np.full_like(x_values, np.nan)
        z_minus = np.full_like(x_values, np.nan)
        for i, x_value in enumerate(x_values):
            w_loc, h_loc, p_loc = _local_section(
                x_value,
                geometry.L_body,
                geometry.L_in,
                w_body,
                h_body,
                p_body,
                w_in,
                h_in,
                p_in,
            )
            half_width = 0.5 * w_loc
            half_height = 0.5 * h_loc
            if half_width <= 0.0 or half_height <= 0.0 or abs(y_cut) > half_width:
                continue
            z_value = float(_superellipse_half_span(y_cut, half_width, half_height, p_loc))
            z_plus[i] = z_value
            z_minus[i] = -z_value

        y_values = np.full_like(x_values, y_cut)
        for z_values in (z_plus, z_minus):
            valid = np.isfinite(z_values)
            if np.any(valid):
                axis.plot(
                    x_values[valid],
                    y_values[valid],
                    z_values[valid],
                    color="#c44e52",
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=42.0,
                )

    for z_cut in z_cuts:
        y_plus = np.full_like(x_values, np.nan)
        y_minus = np.full_like(x_values, np.nan)
        for i, x_value in enumerate(x_values):
            w_loc, h_loc, p_loc = _local_section(
                x_value,
                geometry.L_body,
                geometry.L_in,
                w_body,
                h_body,
                p_body,
                w_in,
                h_in,
                p_in,
            )
            half_width = 0.5 * w_loc
            half_height = 0.5 * h_loc
            if half_width <= 0.0 or half_height <= 0.0 or abs(z_cut) > half_height:
                continue
            y_value = float(_superellipse_half_span(z_cut, half_height, half_width, p_loc))
            y_plus[i] = y_value
            y_minus[i] = -y_value

        z_values = np.full_like(x_values, z_cut)
        for y_values in (y_plus, y_minus):
            valid = np.isfinite(y_values)
            if np.any(valid):
                axis.plot(
                    x_values[valid],
                    y_values[valid],
                    z_values[valid],
                    color="#2a9d8f",
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=42.0,
                )


# ---------------------------------------------------------------------------
# 3D spacecraft rendering
# ---------------------------------------------------------------------------
def _add_symmetric_panel_pair(
    axis,
    *,
    x0: float,
    x1: float,
    lateral_extent: float,
    normal_thickness: float,
    offset: float,
    along_axis: str,
    color: Any,
    edgecolor: str,
    alpha: float,
    face_zorder: float,
    edge_zorder: float,
    bounds: dict[str, list[float]],
) -> None:
    """
    Draw a mirrored pair of rectangular appendages.

    ``along_axis`` controls whether the pair extends along Y (solar-style wings)
    or Z (radiator-style top/bottom plates).
    """
    if along_axis == "y":
        pos0 = offset
        pos1 = offset + lateral_extent
        neg1 = -offset
        neg0 = -(offset + lateral_extent)

        _add_box(axis, x0, x1, pos0, pos1, -0.5 * normal_thickness, 0.5 * normal_thickness, color, alpha=alpha, zorder=face_zorder)
        _add_box(axis, x0, x1, neg0, neg1, -0.5 * normal_thickness, 0.5 * normal_thickness, color, alpha=alpha, zorder=face_zorder)
        _add_box_edges(axis, x0, x1, pos0, pos1, -0.5 * normal_thickness, 0.5 * normal_thickness, color=edgecolor, linewidth=1.2, zorder=edge_zorder)
        _add_box_edges(axis, x0, x1, neg0, neg1, -0.5 * normal_thickness, 0.5 * normal_thickness, color=edgecolor, linewidth=1.2, zorder=edge_zorder)
        _extend_bounds(bounds, [x0, x1], [neg0, pos1], [-0.5 * normal_thickness, 0.5 * normal_thickness])
        return

    if along_axis == "z":
        pos0 = offset
        pos1 = offset + lateral_extent
        neg1 = -offset
        neg0 = -(offset + lateral_extent)

        _add_box(axis, x0, x1, -0.5 * normal_thickness, 0.5 * normal_thickness, pos0, pos1, color, alpha=alpha, zorder=face_zorder)
        _add_box(axis, x0, x1, -0.5 * normal_thickness, 0.5 * normal_thickness, neg0, neg1, color, alpha=alpha, zorder=face_zorder)
        _add_box_edges(axis, x0, x1, -0.5 * normal_thickness, 0.5 * normal_thickness, pos0, pos1, color=edgecolor, linewidth=1.0, zorder=edge_zorder)
        _add_box_edges(axis, x0, x1, -0.5 * normal_thickness, 0.5 * normal_thickness, neg0, neg1, color=edgecolor, linewidth=1.0, zorder=edge_zorder)
        _extend_bounds(bounds, [x0, x1], [-0.5 * normal_thickness, 0.5 * normal_thickness], [neg0, pos1])
        return

    raise ValueError(f"Unsupported panel axis: {along_axis}")


def _render_spacecraft(
    axis,
    geometry: GeometryState,
    color_fn: Callable[[str, float, float], Any],
    x_samples: np.ndarray | None = None,
    core_edgecolor: str | None = None,
) -> dict[str, list[float]]:
    """
    Draw the body/intake shell plus all mounted appendages.

    The function returns axis bounds so the caller can apply equal-scale limits.
    """
    bounds = {"x": [], "y": [], "z": []}
    body_length = _positive(geometry.L_body, 0.1)
    intake_length = _positive(geometry.L_in, 0.1)
    total_length = body_length + intake_length

    body_shape = _shape_from_geometry(geometry.A_body, geometry.AR_body, geometry.S_body)
    intake_shape = _shape_from_geometry(geometry.A_in, geometry.AR_in, geometry.S_in)

    if x_samples is None:
        x_samples = np.array([0.0, body_length, total_length], dtype=float)
    else:
        x_samples = np.asarray(x_samples, dtype=float)
        x_samples = np.unique(np.clip(x_samples, 0.0, total_length))
        if x_samples.size < 2:
            x_samples = np.array([0.0, body_length, total_length], dtype=float)

    # Draw the centerbody in axial segments so wake coloring can vary along x.
    for x0, x1 in zip(x_samples[:-1], x_samples[1:]):
        if x1 <= x0:
            continue

        x_mid = 0.5 * (x0 + x1)
        shape_0 = _shape_at_x(x0, body_length, intake_length, body_shape, intake_shape)
        shape_1 = _shape_at_x(x1, body_length, intake_length, body_shape, intake_shape)
        segment_color = color_fn("core", x_mid, total_length)
        zorder = 6.0 if x_mid >= body_length else 5.0

        in_body = x_mid <= body_length
        segment_is_square = body_shape.is_square if in_body else intake_shape.is_square

        if segment_is_square:
            _add_tapered_box(
                axis,
                x0,
                x1,
                shape_0.width,
                shape_0.height,
                shape_1.width,
                shape_1.height,
                segment_color,
                alpha=0.95,
                edgecolor=core_edgecolor or "none",
                linewidth=0.8 if core_edgecolor else 0.0,
                zorder=zorder,
            )
        else:
            _add_tapered_elliptic_tube(
                axis,
                x0,
                x1 - x0,
                shape_0.semi_y,
                shape_0.semi_z,
                shape_1.semi_y,
                shape_1.semi_z,
                segment_color,
                alpha=0.95,
                edgecolor=core_edgecolor or "none",
                linewidth=0.35 if core_edgecolor else 0.0,
                zorder=zorder,
            )

        # When solar area exists, paint the top shell to make exposure obvious.
        if geometry.A_solar > 0.0:
            solar_cover_color = color_fn("solar", x_mid, total_length)
            solar_zorder = zorder + 3.0
            if segment_is_square:
                _add_tapered_top_face(
                    axis,
                    x0,
                    x1,
                    shape_0.width,
                    shape_1.width,
                    shape_0.height,
                    shape_1.height,
                    solar_cover_color,
                    alpha=0.95,
                    edgecolor=GEOM_EDGE,
                    linewidth=0.35,
                    zorder=solar_zorder,
                )
            else:
                _add_tapered_elliptic_shell(
                    axis,
                    x0,
                    x1 - x0,
                    1.002 * shape_0.semi_y,
                    1.002 * shape_0.semi_z,
                    1.002 * shape_1.semi_y,
                    1.002 * shape_1.semi_z,
                    0.0,
                    np.pi,
                    solar_cover_color,
                    alpha=0.9,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=solar_zorder,
                )

    max_width = max(body_shape.width, intake_shape.width)
    max_height = max(body_shape.height, intake_shape.height)
    _extend_bounds(bounds, [0.0, total_length], [-0.5 * max_width, 0.5 * max_width], [-0.5 * max_height, 0.5 * max_height])

    if geometry.A_prop > 0.0:
        prop_width, prop_height = _rect_dims(geometry.A_prop, geometry.AR_in)
        prop_x = body_length + 0.95 * intake_length
        _add_rectangle_outline(axis, prop_x, prop_width, prop_height, color_fn("prop", prop_x, total_length), zorder=40.0)
        _extend_bounds(bounds, [prop_x], [-0.5 * prop_width, 0.5 * prop_width], [-0.5 * prop_height, 0.5 * prop_height])

    # Solar wings: mirrored plates extending along Y.
    if geometry.A_solar > 0.0:
        solar_area_each = 0.5 * geometry.A_solar
        solar_span, solar_chord = _panel_planform_dims(solar_area_each, geometry.AR_solar)
        solar_thickness = max(0.06, 0.06 * min(body_shape.width, body_shape.height))
        solar_clearance = max(0.04, 1.25 * solar_thickness)
        solar_x0, solar_x1 = _mounted_axial_bounds(geometry.X_solar, solar_chord, body_length)
        solar_color = color_fn("solar", 0.5 * (solar_x0 + solar_x1), total_length)
        _add_symmetric_panel_pair(
            axis,
            x0=solar_x0,
            x1=solar_x1,
            lateral_extent=solar_span,
            normal_thickness=solar_thickness,
            offset=0.5 * body_shape.width + solar_clearance,
            along_axis="y",
            color=solar_color,
            edgecolor=NASA_TEXT,
            alpha=0.92,
            face_zorder=30.0,
            edge_zorder=31.0,
            bounds=bounds,
        )

    # Radiators: mirrored plates extending along Z.
    if geometry.A_rad > 0.0:
        rad_area_each = 0.5 * geometry.A_rad
        rad_span, rad_chord = _panel_planform_dims(rad_area_each, geometry.AR_rad)
        rad_thickness = max(0.02, 0.03 * min(body_shape.width, body_shape.height))
        rad_clearance = max(0.03, 1.1 * rad_thickness)
        rad_x0, rad_x1 = _mounted_axial_bounds(geometry.X_rad, rad_chord, body_length)
        rad_color = color_fn("rad", 0.5 * (rad_x0 + rad_x1), total_length)
        _add_symmetric_panel_pair(
            axis,
            x0=rad_x0,
            x1=rad_x1,
            lateral_extent=rad_span,
            normal_thickness=rad_thickness,
            offset=0.5 * body_shape.height + rad_clearance,
            along_axis="z",
            color=rad_color,
            edgecolor=GEOM_EDGE,
            alpha=0.90,
            face_zorder=20.0,
            edge_zorder=21.0,
            bounds=bounds,
        )

    return bounds


def draw_spacecraft_geometry(axis, geometry: GeometryState, iteration: int | None = None) -> None:
    """Render a simplified spacecraft geometry derived from ``GeometryState``."""
    axis.clear()
    body_length = _positive(geometry.L_body, 0.1)

    def _color(component: str, x_mid: float, _total_length: float) -> Any:
        if component == "core":
            return GEOM_BODY if x_mid <= body_length else GEOM_INTAKE
        if component == "solar":
            return GEOM_SOLAR
        if component == "rad":
            return GEOM_RAD
        if component == "prop":
            return GEOM_PROP
        return GEOM_BODY

    bounds = _render_spacecraft(axis, geometry, _color, core_edgecolor=GEOM_EDGE)
    if not bounds["x"]:
        _extend_bounds(bounds, [0.0, 1.0], [-0.5, 0.5], [-0.5, 0.5])
    _style_3d_axis(axis, f"SC Geometry | Iteration {iteration if iteration is not None else 0}")
    _set_equal_limits(axis, bounds)


def draw_spacecraft_drag_geometry(
    figure,
    state: SpacecraftState,
    iteration: int | None = None,
) -> None:
    """Render spacecraft geometry colored by wake-factor profiles."""
    figure.clear()
    axis = figure.add_subplot(111, projection="3d")
    body_length = max(_safe_float(state.geometry.L_body), 0.0)
    intake_length = max(_safe_float(state.geometry.L_in), 0.0)
    total_length = body_length + intake_length

    x_samples = _drag_geometry_samples(body_length, intake_length)
    x_drag, fy_profile, fz_profile = _wake_profile_arrays(total_length, state.geometry, x_samples)

    def _color(component: str, x_mid: float, _total: float) -> Any:
        _, _, wake_mean = _wake_fraction_at_x(x_mid, total_length, x_drag, fy_profile, fz_profile)
        return _wake_color(wake_mean)

    bounds = _render_spacecraft(axis, state.geometry, _color, x_samples=x_samples)
    _plot_drag_slice_lines(axis, state.geometry)
    if not bounds["x"]:
        _extend_bounds(bounds, [0.0, 1.0], [-0.5, 0.5], [-0.5, 0.5])
    _style_3d_axis(axis, f"3D Drag Exposure | Iteration {iteration if iteration is not None else 0}")
    _set_equal_limits(axis, bounds)

    axis.text2D(
        0.02,
        0.02,
        "Surface color = local wake factor | red = y=c cuts | green = z=c cuts",
        transform=axis.transAxes,
        color=NASA_TEXT,
        fontsize=9,
        fontfamily="Courier New",
    )

    scalar_map = cm.ScalarMappable(norm=WAKE_NORM, cmap=WAKE_CMAP)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=axis, fraction=0.045, pad=0.08)
    colorbar.set_label("Wake factor [-]", color=NASA_TEXT, fontsize=9)
    colorbar.ax.tick_params(colors=NASA_TEXT, labelsize=8)
    colorbar.outline.set_edgecolor(NASA_TEXT)
    figure.patch.set_facecolor(NASA_BG)
    figure.tight_layout()


# ---------------------------------------------------------------------------
# Drag / thermal / propulsion diagnostics
# ---------------------------------------------------------------------------
def _momentum_exchange_terms(state: SpacecraftState) -> dict[str, float]:
    rho = _safe_float(state.orbit.density)
    velocity = _safe_float(state.orbit.velocity)
    area_ref = _safe_float(state.geometry.A_ref)
    area_prop = _safe_float(state.geometry.A_prop)
    refueling_exchange = rho * area_ref * velocity * velocity
    propulsive_exchange = rho * area_prop * velocity * velocity
    total_drag = _safe_float(state.drag.drag_total)
    total_exchange = refueling_exchange + propulsive_exchange
    return {
        "refueling_exchange": refueling_exchange,
        "propulsive_exchange": propulsive_exchange,
        "total_drag": total_drag,
        "total_exchange": total_exchange,
        "total_load": total_drag + total_exchange,
    }


def _drag_component_label(attr_name: str) -> str:
    labels = {
        "drag_body_side": "Body Side",
        "drag_inlet_side": "Inlet Side",
        "drag_inlet_front": "Inlet Front",
        "drag_body": "Body",
        "drag_inlet": "Inlet",
        "drag_inlet_normal": "Inlet Normal",
        "drag_solar": "Solar",
        "drag_rad": "Radiator",
    }
    return labels.get(attr_name, attr_name.removeprefix("drag_").replace("_", " ").title())


def _drag_component_map(state: SpacecraftState) -> dict[str, float]:
    drag_values: dict[str, Any] = {}
    drag_state = state.drag

    if is_dataclass(drag_state):
        for field_info in fields(drag_state):
            drag_values[field_info.name] = getattr(drag_state, field_info.name)
    for name, value in vars(drag_state).items():
        drag_values[name] = value

    preferred_order = ["drag_body_side", "drag_inlet_side", "drag_inlet_front", "drag_solar", "drag_rad", "drag_body", "drag_inlet", "drag_inlet_normal"]
    component_names = [name for name in preferred_order if name in drag_values]
    component_names.extend(
        sorted(
            name
            for name in drag_values
            if name.startswith("drag_") and name not in component_names and name != "drag_total"
        )
    )
    return {name: _safe_float(drag_values[name]) for name in component_names}


def _thermal_component_label(attr_name: str) -> str:
    labels = {
        "Q_drag": "Drag",
        "Q_sun": "Sun",
        "Q_albedo": "Albedo",
        "Q_ir": "Earth IR",
        "Q_internal": "Internal",
        "Q_radiated": "Radiated",
    }
    return labels.get(attr_name, attr_name.removeprefix("Q_").replace("_", " ").title())


def _thermal_component_map(diagnostics: ThermalDiagnostics) -> dict[str, float]:
    preferred_order = ["Q_drag", "Q_sun", "Q_albedo", "Q_ir", "Q_internal", "Q_radiated"]
    return {name: _safe_float(getattr(diagnostics, name, 0.0)) for name in preferred_order}


def draw_drag_distribution(
    figure,
    state: SpacecraftState,
    iteration: int | None = None,
) -> None:
    """Render drag diagnostics using active wake-based model quantities."""
    figure.clear()
    axes = figure.subplots(2, 2, squeeze=False)

    drag_coefficients = {
        "Body Side": _safe_float(state.drag.cd_body_side),
        "Inlet Side": _safe_float(state.drag.cd_inlet_side),
        "Inlet Front": _safe_float(state.drag.cd_inlet_front),
        "Solar": _safe_float(state.drag.cd_solar),
        "Radiator": _safe_float(state.drag.cd_rad),
    }
    drag_components = _drag_component_map(state)
    wake_factors = {
        "Body": _safe_float(getattr(state.geometry, "wake_body", 1.0)),
        "Inlet": _safe_float(getattr(state.geometry, "wake_in", 1.0)),
        "Solar": _safe_float(getattr(state.geometry, "wake_solar", 1.0)),
        "Radiator": _safe_float(getattr(state.geometry, "wake_radiator", 1.0)),
    }
    exchange_terms = _momentum_exchange_terms(state)
    orbit_density = _safe_float(state.orbit.density)
    orbit_velocity = _safe_float(state.orbit.velocity)
    dynamic_pressure = 0.5 * orbit_density * orbit_velocity * orbit_velocity

    cd_axis = axes[0][0]
    _style_2d_axis(cd_axis)
    cd_labels = list(drag_coefficients.keys())
    cd_values = list(drag_coefficients.values())
    cd_axis.bar(
        cd_labels,
        cd_values,
        color=[NASA_LINE[i % len(NASA_LINE)] for i in range(len(cd_labels))],
        edgecolor=NASA_TEXT,
        linewidth=0.8,
    )
    cd_axis.set_title("EFFECTIVE DRAG COEFFICIENTS", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    cd_axis.set_ylabel("Cd [-]", color=NASA_TEXT, fontsize=8)

    force_axis = axes[0][1]
    _style_2d_axis(force_axis)
    force_labels = [_drag_component_label(name) for name in drag_components]
    force_values = list(drag_components.values())
    force_axis.bar(
        force_labels,
        force_values,
        color=[NASA_LINE[i % len(NASA_LINE)] for i in range(len(force_labels))],
        edgecolor=NASA_TEXT,
        linewidth=0.8,
    )
    shown_drag_sum = float(np.sum(force_values))
    total_drag = exchange_terms["total_drag"]
    force_axis.axhline(total_drag, color="#6b7280", linestyle="--", linewidth=1.1, label=f"State total = {total_drag:.3e} N")
    if not np.isclose(shown_drag_sum, total_drag, rtol=1.0e-6, atol=1.0e-12):
        force_axis.axhline(shown_drag_sum, color="#2f6db3", linestyle="-.", linewidth=1.1, label=f"Shown sum = {shown_drag_sum:.3e} N")
    force_axis.set_title("DRAG FORCE BY SURFACE", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    force_axis.set_ylabel("Force [N]", color=NASA_TEXT, fontsize=8)
    force_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

    wake_axis = axes[1][0]
    _style_2d_axis(wake_axis)
    wake_labels = list(wake_factors.keys())
    wake_values = list(wake_factors.values())
    wake_axis.bar(
        wake_labels,
        wake_values,
        color=["#c44e52", "#4c72b0", "#2f6db3", "#6d597a"],
        edgecolor=NASA_TEXT,
        linewidth=0.8,
    )
    wake_axis.set_ylim(-0.05, 1.05)
    wake_axis.set_title("WAKE FACTORS", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    wake_axis.set_ylabel("Wake factor [-]", color=NASA_TEXT, fontsize=8)

    terms_axis = axes[1][1]
    _style_2d_axis(terms_axis)
    thrust_available = _safe_float(getattr(state.thruster, "thrust", 0.0))
    balance_residual = thrust_available - exchange_terms["total_load"]
    term_labels = ["Aero Drag", "Refuel Ram", "Prop Ram", "Required Load", "Thrust"]
    term_values = [exchange_terms["total_drag"], exchange_terms["refueling_exchange"], exchange_terms["propulsive_exchange"], exchange_terms["total_load"], thrust_available]
    terms_axis.bar(
        term_labels,
        term_values,
        color=["#6b7280", "#dd8452", "#55a868", "#c44e52", "#2f6db3"],
        edgecolor=NASA_TEXT,
        linewidth=0.8,
    )
    terms_axis.set_title("FORCE BALANCE (AERO + RAM)", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    terms_axis.set_ylabel("Force [N]", color=NASA_TEXT, fontsize=8)
    terms_axis.text(
        0.02,
        0.98,
        f"rho = {orbit_density:.3e} kg/m^3\nV = {orbit_velocity:.2f} m/s\nq = {dynamic_pressure:.3e} Pa\n"
        f"Residual (T-Load) = {balance_residual:.3e} N",
        transform=terms_axis.transAxes,
        ha="left",
        va="top",
        color=NASA_TEXT,
        fontsize=8,
        fontfamily="Courier New",
    )

    figure.patch.set_facecolor(NASA_BG)
    figure.suptitle(
        f"DRAG DIAGNOSTICS | Iteration {iteration if iteration is not None else 0}",
        color=NASA_TEXT,
        fontsize=14,
        fontfamily="Courier New",
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])


def draw_thermal_distribution(
    figure,
    state: SpacecraftState,
    diagnostics: ThermalDiagnostics,
    iteration: int | None = None,
) -> None:
    """Render thermal contribution diagnostics for one spacecraft state."""
    figure.clear()
    axes = figure.subplots(2, 2, squeeze=False)
    components = _thermal_component_map(diagnostics)
    input_names = [name for name in components if name != "Q_radiated"]
    input_labels = [_thermal_component_label(name) for name in input_names]
    input_values = np.asarray([components[name] for name in input_names], dtype=float)
    total_input = float(np.sum(input_values))
    q_radiated = components["Q_radiated"]
    net_load = total_input - q_radiated

    input_axis = axes[0][0]
    _style_2d_axis(input_axis)
    input_axis.bar(
        input_labels,
        input_values,
        color=[NASA_LINE[i % len(NASA_LINE)] for i in range(len(input_labels))],
        edgecolor=NASA_TEXT,
        linewidth=0.8,
    )
    input_axis.set_title("HEATING INPUTS", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    input_axis.set_ylabel("Heat rate [W]", color=NASA_TEXT, fontsize=8)

    signed_axis = axes[0][1]
    _style_2d_axis(signed_axis)
    signed_labels = input_labels + [_thermal_component_label("Q_radiated")]
    signed_values = np.concatenate([input_values, np.array([-q_radiated], dtype=float)])
    signed_colors = [NASA_LINE[i % len(NASA_LINE)] for i in range(len(input_labels))] + ["#6b7280"]
    signed_axis.bar(signed_labels, signed_values, color=signed_colors, edgecolor=NASA_TEXT, linewidth=0.8)
    signed_axis.axhline(0.0, color=NASA_TEXT, linewidth=0.9)
    signed_axis.set_title("SIGNED CONTRIBUTIONS", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    signed_axis.set_ylabel("Heat rate [W]", color=NASA_TEXT, fontsize=8)

    balance_axis = axes[1][0]
    _style_2d_axis(balance_axis)
    balance_labels = ["Total in", "Radiated", "Net load"]
    balance_values = np.asarray([total_input, q_radiated, net_load], dtype=float)
    balance_axis.bar(balance_labels, balance_values, color=["#dd8452", "#6b7280", "#2f6db3"], edgecolor=NASA_TEXT, linewidth=0.8)
    balance_axis.set_title("THERMAL BALANCE", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    balance_axis.set_ylabel("Heat rate [W]", color=NASA_TEXT, fontsize=8)

    summary_axis = axes[1][1]
    summary_axis.set_facecolor(NASA_BG)
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.98,
        (
            f"A_rad = {_safe_float(state.geometry.A_rad):.3f} m^2\n"
            f"T_des = {_safe_float(state.thermal.T_des):.2f} K\n"
            f"alpha_body = {_safe_float(state.thermal.alpha_body):.3f}\n"
            f"alpha_solar = {_safe_float(state.thermal.alpha_solar):.3f}\n"
            f"eps_body = {_safe_float(state.thermal.epsilon_therm_body):.3f}\n"
            f"eps_solar = {_safe_float(state.thermal.epsilon_therm_solar):.3f}\n"
            f"eps_rad = {_safe_float(state.thermal.epsilon_therm_rad):.3f}"
        ),
        transform=summary_axis.transAxes,
        ha="left",
        va="top",
        color=NASA_TEXT,
        fontsize=10,
        fontfamily="Courier New",
    )
    summary_axis.set_title("THERMAL STATE", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")

    figure.patch.set_facecolor(NASA_BG)
    figure.suptitle(
        f"THERMAL CONTRIBUTIONS | Iteration {iteration if iteration is not None else 0}",
        color=NASA_TEXT,
        fontsize=14,
        fontfamily="Courier New",
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])


def draw_propulsion_overview(figure, state: SpacecraftState, iteration: int | None = None) -> None:
    """Render a stylized thruster silhouette and propulsion telemetry."""
    figure.clear()
    silhouette_axis, telemetry_axis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1.0]})

    power_required = _safe_float(getattr(state.thruster, "power", 0.0))
    thrust = _safe_float(getattr(state.thruster, "thrust", 0.0))
    mass_flow = _safe_float(getattr(state.thruster, "m_flow", 0.0))
    isp = _safe_float(getattr(state.thruster, "specific_impulse", 0.0))
    efficiency = _safe_float(getattr(state.thruster, "eff", 0.0))
    prop_area = _safe_float(getattr(state.geometry, "A_prop", 0.0))
    intake_area = _safe_float(getattr(state.geometry, "A_in", 0.0))
    orbit_density = _safe_float(getattr(state.orbit, "density", 0.0))
    orbit_velocity = _safe_float(getattr(state.orbit, "velocity", 0.0))
    exhaust_velocity = const.EARTH_GRAVITY * isp
    thrust_to_power = 0.0 if power_required <= 0.0 else 1.0e6 * thrust / power_required

    intake_radius = max(np.sqrt(max(intake_area, 1.0e-9) / np.pi), 0.16)
    throat_radius = max(np.sqrt(max(prop_area, 1.0e-9) / np.pi), 0.08)
    chamber_radius = max(throat_radius * 1.35, intake_radius * 0.62)
    exit_radius = max(throat_radius * (1.45 + 0.8 * efficiency), throat_radius * 1.6)
    intake_length = max(0.45, 0.45 + 0.30 * intake_radius)
    chamber_length = max(0.9, 0.55 + 0.00006 * power_required + 0.25 * chamber_radius)
    nozzle_length = max(1.0, 0.85 + 0.00002 * isp + 0.20 * exit_radius)

    x0 = 0.0
    x1 = x0 + intake_length
    x2 = x1 + chamber_length
    x3 = x2 + nozzle_length
    upper = np.array([intake_radius, chamber_radius, throat_radius, exit_radius])
    lower = -upper
    x_nodes = np.array([x0, x1, x2, x3])

    silhouette_axis.set_facecolor(NASA_BG)
    silhouette_axis.fill_between(x_nodes, upper, lower, color="#dde7ef", edgecolor=NASA_TEXT, linewidth=1.6)
    silhouette_axis.plot([x0, x3], [0.0, 0.0], color=NASA_TEXT, linewidth=1.0, linestyle="--")
    silhouette_axis.fill_between([x1, x2], [0.22 * chamber_radius, 0.22 * chamber_radius], [-0.22 * chamber_radius, -0.22 * chamber_radius], color="#c44e52", alpha=0.85)
    silhouette_axis.arrow(x0 - 0.55, 0.0, 0.35, 0.0, width=0.03, head_width=0.16, head_length=0.12, color="#2f6db3", length_includes_head=True)
    silhouette_axis.arrow(x3 + 0.08, 0.0, 0.55, 0.0, width=0.035, head_width=0.18, head_length=0.14, color="#c44e52", length_includes_head=True)
    silhouette_axis.text(x0 + 0.05, intake_radius + 0.12, "INTAKE", color=NASA_TEXT, fontsize=9, fontfamily="Courier New")
    silhouette_axis.text(x1 + 0.08, chamber_radius + 0.12, "THRUSTER", color=NASA_TEXT, fontsize=9, fontfamily="Courier New")
    silhouette_axis.text(x2 + 0.10, exit_radius + 0.12, "NOZZLE", color=NASA_TEXT, fontsize=9, fontfamily="Courier New")
    silhouette_axis.text(x0 - 0.60, -0.30, "FLOW", color="#2f6db3", fontsize=8, fontfamily="Courier New")
    silhouette_axis.text(x3 + 0.22, -0.30, "THRUST", color="#c44e52", fontsize=8, fontfamily="Courier New")
    silhouette_axis.set_xlim(x0 - 0.75, x3 + 0.85)
    silhouette_axis.set_ylim(-1.25 * max(exit_radius, chamber_radius, intake_radius), 1.25 * max(exit_radius, chamber_radius, intake_radius))
    silhouette_axis.set_aspect("equal", adjustable="box")
    silhouette_axis.axis("off")
    silhouette_axis.set_title("THRUSTER SILHOUETTE", color=NASA_TEXT, fontsize=11, fontfamily="Courier New")

    telemetry_axis.set_facecolor(NASA_PANEL)
    telemetry_axis.axis("off")
    lines = [
        "PROPULSION READOUT",
        "",
        f"Iteration                    : {iteration if iteration is not None else 0}",
        f"Thrust [N]                   : {thrust:.6e}",
        f"Isp [s]                      : {isp:.3f}",
        f"Exhaust Velocity [m/s]       : {exhaust_velocity:.3f}",
        f"Mass Flow [kg/s]             : {mass_flow:.6e}",
        f"Power Required [W]           : {power_required:.3f}",
        f"Thruster Efficiency [-]      : {efficiency:.4f}",
        f"T/P [mN/kW]                  : {thrust_to_power:.3f}",
        "",
        f"Propulsive Intake Area [m^2] : {prop_area:.6f}",
        f"Total Intake Area [m^2]      : {intake_area:.6f}",
        f"Drag Intake Area [m^2]       : {_safe_float(getattr(state.geometry, 'A_in_drag', 0.0)):.6f}",
        f"Orbit Density [kg/m^3]       : {orbit_density:.6e}",
        f"Orbit Velocity [m/s]         : {orbit_velocity:.3f}",
        f"Mission Altitude [km]        : {_safe_float(state.orbit.altitude):.3f}",
    ]
    telemetry_axis.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", color=NASA_TEXT, fontsize=10, fontfamily="Courier New", transform=telemetry_axis.transAxes)

    figure.patch.set_facecolor(NASA_BG)
    figure.suptitle(
        f"PROPULSION CONSOLE | Iteration {iteration if iteration is not None else 0}",
        color=NASA_TEXT,
        fontsize=14,
        fontfamily="Courier New",
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])


def _format_refueling_lines(state: SpacecraftState, iteration: int, history_length: int) -> list[str]:
    refuel_mass_flow = _safe_float(getattr(state.refueling, "m_flow", 0.0))
    return [
        "REFUELING READOUT",
        "",
        f"Iteration                    : {iteration} / {history_length - 1}",
        "",
        f"refueling.m_flow             : {refuel_mass_flow:.6e}",
    ]


# ---------------------------------------------------------------------------
# Tk history UI
# ---------------------------------------------------------------------------
class _HistoryPlotterUI:
    def __init__(
        self,
        history: list[SpacecraftState],
        series: dict[str, list[float]],
        default_specs: list[PlotSpec],
        window_title: str,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this environment.")

        self.history = history
        self.case_name = _spacecraft_case_name(history[0]) if history else ""
        self.series = series
        self.paths = list(series.keys())
        self.rows: list[dict[str, Any]] = []
        self.drag_state_cache: dict[int, SpacecraftState] = {}
        self.thermal_diagnostics_cache: dict[int, tuple[SpacecraftState, ThermalDiagnostics]] = {}
        self.atmosphere_profile_cache: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}

        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1520x920")
        self.root.configure(bg=NASA_BG)

        self.view_iteration = tk.IntVar(value=max(len(history) - 1, 0))
        self.status_vars = {
            "geometry": tk.StringVar(),
            "drag": tk.StringVar(),
            "thermal": tk.StringVar(),
            "atmosphere": tk.StringVar(),
        }
        self.figure_tabs: dict[str, dict[str, Any]] = {}
        self.text_tabs: dict[str, dict[str, Any]] = {}

        self._build_layout(default_specs)
        self.redraw()
        self._refresh_aux_tabs()

    def _build_layout(self, default_specs: list[PlotSpec]) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        controls = tk.Frame(self.root, bg=NASA_BG, padx=10, pady=10)
        controls.grid(row=0, column=0, sticky="ns")
        self._build_controls(controls)

        self.rows_frame = tk.Frame(controls, bg=NASA_BG)
        self.rows_frame.pack(fill="both", expand=True)

        self._configure_notebook_style()
        self.notebook = ttk.Notebook(self.root, style="ARISS.TNotebook")
        self.notebook.grid(row=0, column=1, sticky="nsew")

        self._create_plot_tab()
        self.figure_tabs["geometry"] = self._create_figure_tab("3D View", self.status_vars["geometry"])
        self.figure_tabs["drag"] = self._create_figure_tab("Drag/Propulsion", self.status_vars["drag"])
        self.figure_tabs["thermal"] = self._create_figure_tab("Thermal", self.status_vars["thermal"])
        self.figure_tabs["atmosphere"] = self._create_figure_tab("Atmosphere", self.status_vars["atmosphere"])

        for spec in default_specs:
            self.add_plot_row(*spec)

    def _build_controls(self, controls_parent) -> None:
        tk.Label(controls_parent, text="ARISS FLIGHT DATA BOARD", bg=NASA_BG, fg=NASA_TEXT, font=("Courier New", 11, "bold"), justify="left").pack(anchor="w", pady=(0, 8))
        if self.case_name:
            tk.Label(
                controls_parent,
                text=f"CASE: {self.case_name}",
                bg=NASA_BG,
                fg=NASA_TEXT,
                font=("Courier New", 9, "bold"),
                justify="left",
            ).pack(anchor="w", pady=(0, 8))
        tk.Label(
            controls_parent,
            text="SELECT ANY STATE CHANNEL\nMULTI-SERIES PER PLOT\nPLOTS + 3D + DRAG/PROP + THERMAL + ATMOSPHERE",
            bg=NASA_BG,
            fg=NASA_TEXT,
            font=("Courier New", 9),
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        button_row = tk.Frame(controls_parent, bg=NASA_BG)
        button_row.pack(fill="x", pady=(0, 10))
        tk.Button(button_row, text="Add Plot", command=self.add_plot_row, bg=NASA_PANEL, fg=NASA_TEXT, activebackground=NASA_GRID, activeforeground=NASA_TEXT, relief="ridge").pack(side="left", padx=(0, 6))
        tk.Button(button_row, text="Render", command=self.redraw, bg=NASA_PANEL, fg=NASA_TEXT, activebackground=NASA_GRID, activeforeground=NASA_TEXT, relief="ridge").pack(side="left")

    def _configure_notebook_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("ARISS.TNotebook", background=NASA_BG, borderwidth=0)
        style.configure("ARISS.TNotebook.Tab", background=NASA_PANEL, foreground=NASA_TEXT, padding=(10, 6))
        style.map(
            "ARISS.TNotebook.Tab",
            background=[("selected", "#e8dfcf")],
            foreground=[("selected", NASA_TEXT)],
        )

    def _create_iteration_controls(self, parent, status_var) -> None:
        controls = tk.Frame(parent, bg=NASA_BG, padx=10, pady=10)
        controls.pack(fill="x")
        tk.Label(controls, text="History Iteration", bg=NASA_BG, fg=NASA_TEXT, font=("Courier New", 10, "bold")).pack(anchor="w")
        tk.Scale(controls, from_=0, to=max(len(self.history) - 1, 0), orient="horizontal", variable=self.view_iteration, command=self._on_iteration_change, bg=NASA_BG, fg=NASA_TEXT, troughcolor=NASA_PANEL, activebackground=NASA_GRID, highlightthickness=0).pack(fill="x", pady=(4, 2))
        tk.Label(controls, textvariable=status_var, bg=NASA_BG, fg=NASA_TEXT, font=("Courier New", 9), justify="left").pack(anchor="w")

    def _create_plot_tab(self) -> None:
        plot_tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.notebook.add(plot_tab, text="History Plots")
        self.figure = plt.Figure(figsize=(11, 8), dpi=100, facecolor=NASA_BG)
        self.canvas, self.plot_toolbar = self._attach_figure_canvas(plot_tab, self.figure)

    def _create_figure_tab(self, tab_name: str, status_var) -> dict[str, Any]:
        tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.notebook.add(tab, text=tab_name)
        self._create_iteration_controls(tab, status_var)
        figure = plt.Figure(figsize=(11, 8), dpi=100, facecolor=NASA_BG)
        canvas, toolbar = self._attach_figure_canvas(tab, figure)
        return {"tab": tab, "figure": figure, "canvas": canvas, "toolbar": toolbar}

    def _attach_figure_canvas(self, parent, figure):
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(parent, bg=NASA_BG)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        return canvas, toolbar

    def _create_text_tab(self, tab_name: str, status_var) -> dict[str, Any]:
        tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.notebook.add(tab, text=tab_name)
        self._create_iteration_controls(tab, status_var)
        body = tk.Frame(tab, bg=NASA_BG, padx=10, pady=10)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        text_widget = tk.Text(body, bg=NASA_PANEL, fg=NASA_TEXT, font=("Courier New", 10), relief="flat", highlightthickness=1, highlightbackground=NASA_GRID, padx=10, pady=10, wrap="none")
        text_widget.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(body, orient="vertical", command=text_widget.yview, bg=NASA_PANEL, activebackground=NASA_GRID)
        scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scroll.set, state="disabled")
        return {"tab": tab, "text": text_widget}

    def _on_iteration_change(self, *_args) -> None:
        self._refresh_aux_tabs()

    def _current_index(self) -> int:
        return max(0, min(self.view_iteration.get(), len(self.history) - 1))

    def _refresh_aux_tabs(self) -> None:
        self.redraw_geometry()
        self.redraw_drag_test()
        self.redraw_thermal()
        self.redraw_atmosphere()

    def _get_drag_state(self, index: int) -> SpacecraftState:
        cached = self.drag_state_cache.get(index)
        if cached is not None:
            return cached
        with redirect_stdout(io.StringIO()):
            cached = compute_drag_diagnostics(self.history[index])
        self.drag_state_cache[index] = cached
        return cached

    def _get_thermal_diagnostics(self, index: int) -> tuple[SpacecraftState, ThermalDiagnostics]:
        cached = self.thermal_diagnostics_cache.get(index)
        if cached is not None:
            return cached
        state = deepcopy(self.history[index])
        diagnostics = thermal_model(state)
        cached = (state, diagnostics)
        self.thermal_diagnostics_cache[index] = cached
        return cached

    def _get_atmosphere_profile(self, state: SpacecraftState) -> dict[str, np.ndarray]:
        key = (
            str(getattr(state.orbit, "msis_date", "2000-01-01T00:00:00")),
            _safe_float(getattr(state.orbit, "msis_f107", 140.0)),
            _safe_float(getattr(state.orbit, "msis_ap", 15.0)),
        )
        cached = self.atmosphere_profile_cache.get(key)
        if cached is not None:
            return cached

        altitude_km = np.linspace(80.0, 1000.0, 600)
        total_density, _temperature, r_specific, o2_density, n2_density, o_density = atmos(
            altitude_km,
            msis_date=key[0],
            msis_f107=key[1],
            msis_ap=key[2],
        )
        cached = {
            "altitude_km": np.asarray(altitude_km, dtype=float),
            "total_density": np.maximum(np.asarray(total_density, dtype=float), 1.0e-30),
            "r_specific": np.asarray(r_specific, dtype=float),
            "o2_density": np.maximum(np.asarray(o2_density, dtype=float), 1.0e-30),
            "n2_density": np.maximum(np.asarray(n2_density, dtype=float), 1.0e-30),
            "o_density": np.maximum(np.asarray(o_density, dtype=float), 1.0e-30),
        }
        self.atmosphere_profile_cache[key] = cached
        return cached

    def _style_history_axis(self, axis) -> None:
        _style_2d_axis(axis)

    def _draw_empty_history_axis(self, axis, title: str) -> None:
        axis.set_title(title or "No series selected", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
        axis.text(0.5, 0.5, "Select one or more series", transform=axis.transAxes, ha="center", va="center", color=NASA_TEXT, fontsize=9, fontfamily="Courier New")
        axis.set_xticks([])
        axis.set_yticks([])

    def _plot_history_selection(self, axis, selected_paths: list[str], row: dict[str, Any], x_values: list[int]) -> None:
        all_positive = True
        for line_idx, path in enumerate(selected_paths):
            y_values = self.series.get(path, [])
            if not y_values or any(value <= 0.0 for value in y_values):
                all_positive = False
            axis.plot(x_values, y_values, color=_history_plot_color(line_idx), linewidth=1.9, label=path)

        axis.set_title(row["title"].get() or _default_title(selected_paths), color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
        axis.set_xlabel("Iteration", color=NASA_TEXT, fontsize=9)
        axis.set_ylabel(selected_paths[0] if len(selected_paths) == 1 else "Selected values", color=NASA_TEXT, fontsize=8)
        if row["log"].get():
            axis.set_yscale("log" if all_positive else "symlog", linthresh=1.0e-9)

        legend = axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)
        _style_legend_text(legend)

    def add_plot_row(
        self,
        path: str | Sequence[str] | None = None,
        title: str | None = None,
        log_scale: bool = False,
    ) -> None:
        selected_paths = _normalize_paths(path, self.paths)
        row_title = title or _default_title(selected_paths)
        row_frame = tk.Frame(self.rows_frame, bg=NASA_BG, highlightbackground=NASA_GRID, highlightthickness=1, padx=4, pady=4)
        row_frame.pack(fill="x", pady=3)
        row_frame.grid_columnconfigure(0, weight=1)
        title_var = tk.StringVar(value=row_title)
        log_var = tk.BooleanVar(value=log_scale)
        series_frame = tk.Frame(row_frame, bg=NASA_BG)
        series_frame.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        series_frame.grid_columnconfigure(0, weight=1)
        path_list = tk.Listbox(series_frame, selectmode=tk.MULTIPLE, exportselection=False, width=42, height=6, bg=NASA_PANEL, fg=NASA_TEXT, selectbackground="#d4dee8", selectforeground=NASA_TEXT, activestyle="none", relief="flat", highlightthickness=1, highlightbackground=NASA_GRID)
        path_list.grid(row=0, column=0, sticky="ew")
        scroll = tk.Scrollbar(series_frame, orient="vertical", command=path_list.yview, bg=NASA_PANEL, activebackground=NASA_GRID)
        scroll.grid(row=0, column=1, sticky="ns")
        path_list.configure(yscrollcommand=scroll.set)
        for option in self.paths:
            path_list.insert(tk.END, option)
        for selected_path in selected_paths:
            path_list.selection_set(self.paths.index(selected_path))
        tk.Entry(row_frame, textvariable=title_var, width=28, bg=NASA_PANEL, fg=NASA_TEXT, insertbackground=NASA_TEXT, relief="flat", highlightthickness=1, highlightbackground=NASA_GRID).grid(row=0, column=1, padx=(0, 6), sticky="w")
        tk.Checkbutton(row_frame, text="Log Y", variable=log_var, bg=NASA_BG, fg=NASA_TEXT, activebackground=NASA_BG, activeforeground=NASA_TEXT, selectcolor=NASA_PANEL).grid(row=0, column=2, padx=(0, 6), sticky="w")
        row_data = {"paths": path_list, "title": title_var, "log": log_var}

        def _remove() -> None:
            if row_data in self.rows:
                self.rows.remove(row_data)
            row_frame.destroy()
            self.redraw()

        tk.Button(row_frame, text="Remove", command=_remove, bg=NASA_PANEL, fg=NASA_TEXT, relief="ridge").grid(row=0, column=3, sticky="w")
        self.rows.append(row_data)

    def redraw(self) -> None:
        self.figure.clear()
        if not self.rows:
            self.canvas.draw_idle()
            self._refresh_aux_tabs()
            return

        plot_count = len(self.rows)
        cols = 1 if plot_count == 1 else 2
        row_count = ceil(plot_count / cols)
        axes = self.figure.subplots(row_count, cols, squeeze=False)
        x_values = list(range(len(self.history)))

        for idx, row in enumerate(self.rows):
            axis = axes[idx // cols][idx % cols]
            self._style_history_axis(axis)
            selected_paths = _selected_listbox_values(row["paths"])

            if not selected_paths:
                self._draw_empty_history_axis(axis, row["title"].get())
                continue

            self._plot_history_selection(axis, selected_paths, row, x_values)

        for idx in range(plot_count, row_count * cols):
            axes[idx // cols][idx % cols].axis("off")

        self.figure.patch.set_facecolor(NASA_BG)
        title = "ARISS FLIGHT DATA WALL"
        if self.case_name:
            title = f"{title} | {self.case_name}"
        self.figure.suptitle(title, color=NASA_TEXT, fontsize=14, fontfamily="Courier New", fontweight="bold")
        self.figure.tight_layout(rect=[0, 0, 1, 0.97])
        self.canvas.draw_idle()
        self._refresh_aux_tabs()

    def _draw_error_figure(self, figure, message: str) -> None:
        figure.clear()
        axis = figure.add_subplot(111)
        axis.set_facecolor(NASA_BG)
        axis.text(0.5, 0.5, message, ha="center", va="center", color=NASA_TEXT, fontsize=10, fontfamily="Courier New", transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(NASA_TEXT)

    def redraw_geometry(self) -> None:
        index = self._current_index()
        state = self.history[index]
        tab = self.figure_tabs["geometry"]
        figure = tab["figure"]
        figure.clear()
        axis = figure.add_subplot(111, projection="3d")
        draw_spacecraft_geometry(axis, state.geometry, iteration=index)
        self.status_vars["geometry"].set(
            f"Iteration {index} / {len(self.history) - 1} | Mass {_safe_float(state.mass.Mass_total):.2f} kg | Altitude {_safe_float(state.orbit.altitude):.2f} km"
        )
        figure.tight_layout()
        tab["canvas"].draw_idle()

    def redraw_drag_test(self) -> None:
        index = self._current_index()
        tab = self.figure_tabs["drag"]
        figure = tab["figure"]

        try:
            state = self._get_drag_state(index)
            draw_drag_distribution(figure, state, iteration=index)
            exchange_terms = _momentum_exchange_terms(state)
            drag_components = _drag_component_map(state)
            component_summary = " | ".join(f"{_drag_component_label(name)} {value:.3e} N" for name, value in drag_components.items())
            shown_drag_sum = float(np.sum(list(drag_components.values())))
            thrust_available = _safe_float(getattr(state.thruster, "thrust", 0.0))
            balance_residual = thrust_available - exchange_terms["total_load"]
            self.status_vars["drag"].set(
                f"Iteration {index} / {len(self.history) - 1}\n"
                f"{component_summary}\n"
                f"State total {exchange_terms['total_drag']:.3e} N | "
                f"Shown sum {shown_drag_sum:.3e} N | "
                f"Refuel exchange {exchange_terms['refueling_exchange']:.3e} N | "
                f"Propulsive exchange {exchange_terms['propulsive_exchange']:.3e} N | "
                f"Total load {exchange_terms['total_load']:.3e} N | "
                f"Thrust {thrust_available:.3e} N | "
                f"Residual {balance_residual:.3e} N"
            )
        except Exception as exc:
            self._draw_error_figure(figure, f"Drag diagnostics unavailable\n{exc}")
            self.status_vars["drag"].set(f"Iteration {index} / {len(self.history) - 1} | Drag diagnostics unavailable")

        tab["canvas"].draw_idle()

    def redraw_propulsion(self) -> None:
        index = self._current_index()
        state = self.history[index]
        tab = self.figure_tabs["propulsion"]
        draw_propulsion_overview(tab["figure"], state, iteration=index)
        self.status_vars["propulsion"].set(
            f"Iteration {index} / {len(self.history) - 1} | "
            f"Thrust {_safe_float(getattr(state.thruster, 'thrust', 0.0)):.3e} N | "
            f"Power {_safe_float(getattr(state.thruster, 'power', 0.0)):.3f} W | "
            f"m_flow {_safe_float(getattr(state.thruster, 'm_flow', 0.0)):.3e} kg/s"
        )
        tab["canvas"].draw_idle()

    def redraw_thermal(self) -> None:
        index = self._current_index()
        tab = self.figure_tabs["thermal"]
        figure = tab["figure"]

        try:
            state, diagnostics = self._get_thermal_diagnostics(index)
            draw_thermal_distribution(figure, state, diagnostics, iteration=index)
            components = _thermal_component_map(diagnostics)
            self.status_vars["thermal"].set(
                f"Iteration {index} / {len(self.history) - 1} | "
                f"Drag {components['Q_drag']:.3e} W | "
                f"Sun {components['Q_sun']:.3e} W | "
                f"Albedo {components['Q_albedo']:.3e} W | "
                f"IR {components['Q_ir']:.3e} W | "
                f"Internal {components['Q_internal']:.3e} W | "
                f"Radiated {components['Q_radiated']:.3e} W | "
                f"A_rad {_safe_float(state.geometry.A_rad):.3f} m^2"
            )
        except Exception as exc:
            self._draw_error_figure(figure, f"Thermal diagnostics unavailable\n{exc}")
            self.status_vars["thermal"].set(f"Iteration {index} / {len(self.history) - 1} | Thermal diagnostics unavailable")

        tab["canvas"].draw_idle()

    def _set_text_tab(self, key: str, lines: list[str]) -> None:
        text_widget = self.text_tabs[key]["text"]
        text_widget.configure(state="normal")
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", "\n".join(lines))
        text_widget.configure(state="disabled")

    def redraw_refueling(self) -> None:
        index = self._current_index()
        state = self.history[index]
        self.status_vars["refueling"].set(
            f"Iteration {index} / {len(self.history) - 1} | "
            f"refueling.m_flow = {_safe_float(getattr(state.refueling, 'm_flow', 0.0)):.6e}"
        )
        self._set_text_tab("refueling", _format_refueling_lines(state, index, len(self.history)))

    def redraw_atmosphere(self) -> None:
        index = self._current_index()
        state = self.history[index]
        altitude_km = _safe_float(state.orbit.altitude)
        tab = self.figure_tabs["atmosphere"]
        figure = tab["figure"]
        figure.clear()

        try:
            properties = atmosphere_properties_from_height(
                altitude_km,
                msis_date=state.orbit.msis_date,
                msis_f107=state.orbit.msis_f107,
                msis_ap=state.orbit.msis_ap,
            )
            profile = self._get_atmosphere_profile(state)
            axes = figure.subplots(2, 1, sharex=True, squeeze=True)
            composition_axis = axes[0]
            r_axis = axes[1]
            self._style_history_axis(composition_axis)
            self._style_history_axis(r_axis)

            altitude_profile = profile["altitude_km"]
            total_density_profile = profile["total_density"]
            o2_profile = profile["o2_density"]
            n2_profile = profile["n2_density"]
            o_profile = profile["o_density"]
            r_profile = profile["r_specific"]

            composition_axis.plot(altitude_profile, total_density_profile, color=NASA_TEXT, linewidth=1.8, linestyle="--", label="Total density")
            composition_axis.plot(altitude_profile, o2_profile, color="#4c72b0", linewidth=1.9, label="O2 density")
            composition_axis.plot(altitude_profile, n2_profile, color="#55a868", linewidth=1.9, label="N2 density")
            composition_axis.plot(altitude_profile, o_profile, color="#c44e52", linewidth=1.9, label="O density")
            composition_axis.axvline(altitude_km, color=NASA_TEXT, linestyle="--", linewidth=1.1, label=f"Current altitude = {altitude_km:.2f} km")
            model_density = max(_safe_float(properties["density"]), 1.0e-30)
            state_density = max(_safe_float(state.orbit.density, default=model_density), 1.0e-30)
            composition_axis.scatter([altitude_km], [model_density], color=NASA_TEXT, s=32, zorder=5, label=f"Model density = {model_density:.3e}")
            composition_axis.scatter([altitude_km], [state_density], color="#2f6db3", s=30, marker="D", zorder=5, label=f"State density = {state_density:.3e}")
            composition_axis.scatter([altitude_km], [max(_safe_float(properties["o2_density"]), 1.0e-30)], color="#4c72b0", s=30, zorder=5)
            composition_axis.scatter([altitude_km], [max(_safe_float(properties["n2_density"]), 1.0e-30)], color="#55a868", s=30, zorder=5)
            composition_axis.scatter([altitude_km], [max(_safe_float(properties["o_density"]), 1.0e-30)], color="#c44e52", s=30, zorder=5)
            composition_axis.set_yscale("log")
            composition_axis.set_ylabel("Density [kg/m^3]", color=NASA_TEXT, fontsize=8)
            composition_axis.set_title("COMPOSITION DENSITY VS ALTITUDE", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
            composition_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

            model_r = _safe_float(properties["specific_gas_constant"])
            state_r = _safe_float(getattr(state.orbit, "R_spec", model_r), default=model_r)
            r_axis.plot(altitude_profile, r_profile, color="#dd8452", linewidth=2.0, label="Model R_specific")
            r_axis.axvline(altitude_km, color=NASA_TEXT, linestyle="--", linewidth=1.1)
            r_axis.scatter([altitude_km], [model_r], color="#dd8452", s=34, zorder=5, label=f"Model @ altitude = {model_r:.2f}")
            r_axis.scatter([altitude_km], [state_r], color="#2f6db3", s=34, marker="D", zorder=5, label=f"State R_spec = {state_r:.2f}")
            r_axis.set_xlabel("Altitude [km]", color=NASA_TEXT, fontsize=9)
            r_axis.set_ylabel("R_specific [J/kg/K]", color=NASA_TEXT, fontsize=8)
            r_axis.set_title("R VALUES VS ALTITUDE", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
            r_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)
            x_min = min(float(np.min(altitude_profile)), altitude_km)
            x_max = max(float(np.max(altitude_profile)), altitude_km)
            composition_axis.set_xlim(x_min, x_max)

            self.status_vars["atmosphere"].set(
                f"Iteration {index} / {len(self.history) - 1} | Altitude {altitude_km:.2f} km | "
                f"R_model {model_r:.2f} J/kg/K | R_state {state_r:.2f} J/kg/K"
            )
            figure.patch.set_facecolor(NASA_BG)
            figure.suptitle(
                f"ATMOSPHERE DIAGNOSTICS | Iteration {index}",
                color=NASA_TEXT,
                fontsize=14,
                fontfamily="Courier New",
                fontweight="bold",
            )
            figure.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception as exc:
            self._draw_error_figure(figure, f"Atmosphere diagnostics unavailable\n{exc}")
            self.status_vars["atmosphere"].set(
                f"Iteration {index} / {len(self.history) - 1} | Atmosphere diagnostics unavailable"
            )

        tab["canvas"].draw_idle()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------
def launch_history_ui(
    sc: SpacecraftInput | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
    default_specs: list[PlotSpec] | None = None,
    window_title: str = "ARISS History Plotter",
    show: bool = True,
):
    """Run the sizing history and open the interactive visualization UI."""
    sc = _resolve_spacecraft_input(sc)
    case_name = _spacecraft_case_name(sc)
    if case_name:
        window_title = f"{window_title} | {case_name}"
    _, _, history = run_sizing_with_history(sc, max_iterations=max_iterations, mass_tolerance=mass_tolerance)
    series = _history_series(history)
    default_specs = default_specs or DEFAULT_HISTORY_SPECS

    if not show:
        return history, series

    try:
        app = _HistoryPlotterUI(history, series, default_specs, window_title)
    except Exception as exc:
        print(f"Visualization UI could not start: {exc}")
        return history, series

    app.run()
    return history, series


def _launch_with_specs(
    specs: list[PlotSpec],
    title: str,
    sc: SpacecraftInput | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        default_specs=specs,
        window_title=title,
        show=show,
    )


def plot_atmosphere_profiles(
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 600,
    show: bool = True,
):
    _ = (height_min_km, height_max_km, samples)
    return _launch_with_specs(ATMOSPHERE_SPECS, "ARISS Atmosphere / Orbit Console", show=show)


def plot_budgets_total(
    sc: SpacecraftInput | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1.0e-8,
    show: bool = True,
):
    return _launch_with_specs(
        BUDGET_SPECS,
        "ARISS Budget Console",
        sc=sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        show=show,
    )


def plot_dimension_evolution(
    sc: SpacecraftInput | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1.0e-8,
    show: bool = True,
):
    return _launch_with_specs(
        DIMENSION_SPECS,
        "ARISS Geometry Console",
        sc=sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        show=show,
    )


def plot_drag_diagnostics(
    sc: SpacecraftInput | None = None,
    n_points: int = 64,
    show: bool = True,
):
    _ = n_points
    return _launch_with_specs(DRAG_SPECS, "ARISS Drag Console", sc=sc, show=show)


def plot_power_diagnostics(
    sc: SpacecraftInput | None = None,
    efficiency: float = 0.2,
    alignment_deg: float = 0.0,
    baseline_power: float = 2000.0,
    show: bool = True,
):
    _ = (efficiency, alignment_deg, baseline_power)
    return _launch_with_specs(POWER_SPECS, "ARISS Power Console", sc=sc, show=show)


def plot_propulsion_diagnostics(
    sc: SpacecraftInput | None = None,
    baseline_drag: float = 0.2,
    show: bool = True,
):
    _ = baseline_drag
    return _launch_with_specs(PROPULSION_SPECS, "ARISS Propulsion Console", sc=sc, show=show)


def plot_simulation_budgets(
    sc: SpacecraftInput | None = None,
    max_iterations: int = 50,
    mass_tolerance: float = 1.0e-9,
    show: bool = True,
):
    return _launch_with_specs(
        SIM_BUDGET_SPECS,
        "ARISS Integrated Budget Console",
        sc=sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        show=show,
    )


def plot_simulation_history(
    sc: SpacecraftInput | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
    show: bool = True,
):
    return _launch_with_specs(
        DEFAULT_HISTORY_SPECS,
        "ARISS Mission History Console",
        sc=sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        show=show,
    )


__all__ = ["draw_spacecraft_drag_geometry", "draw_spacecraft_geometry", "launch_history_ui", "plot_atmosphere_profiles", "plot_budgets_total", "plot_dimension_evolution", "plot_drag_diagnostics", "plot_power_diagnostics", "plot_propulsion_diagnostics", "plot_simulation_budgets", "plot_simulation_history", "run_sizing_with_history"]


if __name__ == "__main__":
    try:
        plot_simulation_history()
    except Exception:
        traceback.print_exc()
