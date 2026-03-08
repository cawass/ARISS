"""Interactive history plots and geometry views for ``SpacecraftState``."""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from math import ceil, pi, sqrt
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

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import GeometryState, SpacecraftState
from ariss.modules.Drag import DragDiagnostics, drag_model
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmosphere_properties_from_height

NASA_BG = "#ffffff"
NASA_PANEL = "#f4efe2"
NASA_GRID = "#b9b1a2"
NASA_TEXT = "#1c2833"
NASA_LINE = [
    "#1f77b4",
    "#c44e52",
    "#dd8452",
    "#4c72b0",
    "#55a868",
    "#8172b2",
    "#937860",
]
GEOM_BODY = "#ffffff"
GEOM_INTAKE = "#ffffff"
GEOM_SOLAR = "#2f6db3"
GEOM_RAD = "#7c8da4"
GEOM_PROP = "#c44e52"
WAKE_CMAP = cm.get_cmap("RdYlGn_r")
WAKE_NORM = colors.Normalize(vmin=0.0, vmax=1.0)

PlotSpec = tuple[str | Sequence[str], str, bool]
DEFAULT_HISTORY_SPECS: list[PlotSpec] = [
    ("orbit.altitude", "ORBITAL HEIGHT", False),
    (
        [
            "power.Power_total",
            "power.Power_in",
            "power.Power_body",
            "power.Power_solar",
            "power.Power_rad",
            "power.Power_prop",
            "power.Power_ADCS",
            "power.Power_payload",
            "power.Power_refprop",
        ],
        "POWER BUDGETS",
        False,
    ),
    (
        [
            "mass.Mass_total",
            "mass.Mass_in",
            "mass.Mass_body",
            "mass.Mass_solar",
            "mass.Mass_rad",
            "mass.Mass_prop",
            "mass.Mass_ADCS",
            "mass.Mass_payload",
            "mass.Mass_refprop",
        ],
        "MASS BUDGETS",
        False,
    ),
    (
        [
            "geometry.A_body",
            "geometry.A_in",
            "geometry.A_in_drag",
            "geometry.A_prop",
            "geometry.A_solar",
            "geometry.L_body",
            "geometry.L_in",
        ],
        "KEY GEOMETRY",
        False,
    ),
]

ATMOSPHERE_SPECS: list[PlotSpec] = [
    ("orbit.altitude", "ALTITUDE", False),
    ("orbit.density", "DENSITY", True),
    ("orbit.temperature", "TEMPERATURE", False),
    ("orbit.molar_mass", "MOLAR MASS", True),
    ("orbit.velocity", "ORBITAL VELOCITY", False),
    ("drag.drag_total", "TOTAL DRAG", True),
]

BUDGET_SPECS: list[PlotSpec] = [
    ("mass.Mass_total", "TOTAL MASS", False),
    ("mass.Mass_in", "INLET MASS", False),
    ("mass.Mass_solar", "SOLAR ARRAY MASS", False),
    ("power.Power_total", "TOTAL POWER", False),
    ("power.Power_prop", "PROPULSION POWER", True),
    ("power.Power_solar", "SOLAR POWER", True),
]

DIMENSION_SPECS: list[PlotSpec] = [
    ("geometry.A_in", "INTAKE AREA", False),
    ("geometry.A_in_drag", "DRAG INTAKE AREA", False),
    ("geometry.A_prop", "PROPULSIVE AREA", False),
    ("geometry.A_solar", "SOLAR AREA", False),
    ("geometry.L_in", "INTAKE LENGTH", False),
    ("geometry.L_body", "BODY LENGTH", False),
    ("geometry.AR_in", "INTAKE ASPECT RATIO", False),
]

DRAG_SPECS: list[PlotSpec] = [
    ("drag.drag_total", "TOTAL DRAG", True),
    ("drag.drag_body_side", "BODY SIDE DRAG", True),
    ("drag.drag_inlet_side", "INLET SIDE DRAG", True),
    ("drag.drag_inlet_front", "INLET FRONT DRAG", True),
    ("drag.drag_solar", "SOLAR DRAG", True),
    ("geometry.A_in_drag", "DRAG INTAKE AREA", False),
    ("orbit.density", "DENSITY", True),
]

POWER_SPECS: list[PlotSpec] = [
    ("power.Power_total", "TOTAL POWER", False),
    ("power.Power_prop", "PROPULSION POWER", True),
    ("power.Power_solar", "SOLAR POWER", True),
    ("geometry.A_solar", "SOLAR ARRAY AREA", False),
    ("solar.eta_power", "POWER EFFICIENCY", False),
    ("solar.av_aligment", "ARRAY ALIGNMENT", False),
]

PROPULSION_SPECS: list[PlotSpec] = [
    ("geometry.A_prop", "REQUIRED PROPULSIVE AREA", False),
    ("geometry.A_in", "INTAKE AREA", False),
    ("geometry.A_in_drag", "DRAG INTAKE AREA", False),
    ("thruster.power_required", "POWER REQUIRED", True),
    ("thruster.thrust", "THRUST", True),
    ("thruster.m_flow", "PROPELLANT MASS FLOW", True),
    ("orbit.density", "INFERRED DENSITY", True),
]

SIM_BUDGET_SPECS: list[PlotSpec] = [
    ("mass.Mass_total", "TOTAL MASS", False),
    ("power.Power_total", "TOTAL POWER", False),
    ("drag.drag_total", "TOTAL DRAG", True),
    ("drag.drag_inlet_front", "FRONT INLET DRAG", True),
    ("geometry.A_prop", "PROPULSIVE AREA", False),
    ("orbit.altitude", "ALTITUDE", False),
    ("orbit.density", "DENSITY", True),
]


@dataclass(frozen=True)
class _SectionShape:
    width: float
    height: float
    semi_y: float
    semi_z: float
    is_square: bool


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(np.nan_to_num(value, nan=default, posinf=default, neginf=default))


def _positive(value: float, floor: float = 1.0e-6) -> float:
    return max(abs(float(value)), floor)


def _rect_dims(area: float, aspect_ratio: float) -> tuple[float, float]:
    area = _positive(area)
    aspect_ratio = _positive(aspect_ratio)
    width = sqrt(area * aspect_ratio)
    height = area / width
    return width, height


def _ellipse_radii(area: float, aspect_ratio: float) -> tuple[float, float]:
    area = _positive(area)
    aspect_ratio = _positive(aspect_ratio)
    semi_y = sqrt(area * aspect_ratio / pi)
    semi_z = area / (pi * semi_y)
    return semi_y, semi_z


def _is_square(shape: str) -> bool:
    return (shape or "").strip().lower().startswith("s")


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
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
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
    faces = [
        [f0, f1, f2, f3],
        [r0, r1, r2, r3],
        [f0, f1, r1, r0],
        [f1, f2, r2, r1],
        [f2, f3, r3, r2],
        [f3, f0, r0, r3],
    ]
    _add_poly3d(axis, faces, color, alpha, edgecolor, linewidth, zorder)


def _add_elliptic_tube(
    axis,
    x0: float,
    length: float,
    semi_y: float,
    semi_z: float,
    color: Any,
    alpha: float = 0.75,
    edgecolor: str = NASA_TEXT,
    linewidth: float = 0.3,
    zorder: float | None = None,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    x = np.array([x0, x0 + length])
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="ij")
    y_grid = semi_y * np.cos(theta_grid)
    z_grid = semi_z * np.sin(theta_grid)
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


def _set_equal_limits(axis, bounds: dict[str, list[float]]) -> None:
    x_min, x_max = min(bounds["x"]), max(bounds["x"])
    y_min, y_max = min(bounds["y"]), max(bounds["y"])
    z_min, z_max = min(bounds["z"]), max(bounds["z"])
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0) * 0.6
    axis.set_xlim(x_mid - radius, x_mid + radius)
    axis.set_ylim(y_mid - radius, y_mid + radius)
    axis.set_zlim(z_mid - radius, z_mid + radius)


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

    # Matplotlib keeps these private; we guard use for compatibility.
    for ax in (axis.xaxis, axis.yaxis, axis.zaxis):
        try:
            ax._axinfo["grid"]["color"] = grid_color
        except Exception:
            continue


def _drag_x_spacecraft_frame(total_length: float, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Diagnostics already provide axial coordinates in the spacecraft frame.
    # Keep that orientation and only clip/sort for robust interpolation/plotting.
    x_spacecraft = np.clip(np.asarray(x_values, dtype=float), 0.0, total_length)
    order = np.argsort(x_spacecraft)
    return x_spacecraft[order], order


def _wake_profile_arrays(total_length: float, diagnostics: DragDiagnostics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = np.asarray(diagnostics.x_array, dtype=float)
    x_drag, order = _drag_x_spacecraft_frame(total_length, x_raw)
    fy_profile = np.clip(np.nan_to_num(np.asarray(diagnostics.fy_array, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)[order]
    fz_profile = np.clip(np.nan_to_num(np.asarray(diagnostics.fz_array, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)[order]
    if x_drag.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([1.0])
    return x_drag, fy_profile, fz_profile


def _wake_fraction_at_x(
    x_geom: float,
    total_length: float,
    x_drag: np.ndarray,
    fy_profile: np.ndarray,
    fz_profile: np.ndarray,
) -> tuple[float, float, float]:
    if total_length <= 0.0:
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
    diagnostics: DragDiagnostics,
    max_segments: int = 80,
) -> np.ndarray:
    total_length = body_length + intake_length
    anchors = np.array([0.0, body_length, total_length], dtype=float)
    x_raw = np.asarray(diagnostics.x_array, dtype=float)
    if x_raw.size == 0 or total_length <= 0.0:
        return np.unique(anchors)

    x_geom, _ = _drag_x_spacecraft_frame(total_length, x_raw)
    samples = np.unique(np.concatenate([anchors, x_geom]))
    samples = samples[(samples >= 0.0) & (samples <= total_length)]
    samples.sort()

    if samples.size > max_segments + 1:
        selected = np.linspace(0, samples.size - 1, max_segments + 1, dtype=int)
        samples = np.unique(np.concatenate([anchors, samples[selected]]))
        samples.sort()

    return samples


def _render_spacecraft(
    axis,
    geometry: GeometryState,
    color_fn: Callable[[str, float, float], Any],
    x_samples: np.ndarray | None = None,
) -> dict[str, list[float]]:
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

    for x0, x1 in zip(x_samples[:-1], x_samples[1:]):
        if x1 <= x0:
            continue
        x_mid = 0.5 * (x0 + x1)
        shape_0 = _shape_at_x(x0, body_length, intake_length, body_shape, intake_shape)
        shape_1 = _shape_at_x(x1, body_length, intake_length, body_shape, intake_shape)
        segment_color = color_fn("core", x_mid, total_length)
        zorder = 6.0 if x_mid >= body_length else 5.0

        if body_shape.is_square and intake_shape.is_square:
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
                edgecolor="none",
                linewidth=0.0,
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
                edgecolor="none",
                linewidth=0.0,
                zorder=zorder,
            )

    max_width = max(body_shape.width, intake_shape.width)
    max_height = max(body_shape.height, intake_shape.height)
    _extend_bounds(bounds, [0.0, total_length], [-0.5 * max_width, 0.5 * max_width], [-0.5 * max_height, 0.5 * max_height])

    if geometry.A_prop > 0.0:
        prop_width, prop_height = _rect_dims(geometry.A_prop, geometry.AR_in)
        prop_x = body_length + 0.95 * intake_length
        _add_rectangle_outline(axis, prop_x, prop_width, prop_height, color_fn("prop", prop_x, total_length), zorder=40.0)
        _extend_bounds(bounds, [prop_x], [-0.5 * prop_width, 0.5 * prop_width], [-0.5 * prop_height, 0.5 * prop_height])

    if geometry.A_solar > 0.0:
        solar_area_each = 0.5 * geometry.A_solar
        solar_span, solar_chord = _rect_dims(solar_area_each, geometry.AR_solar)
        solar_thickness = max(0.06, 0.06 * min(body_shape.width, body_shape.height))
        solar_clearance = max(0.04, 1.25 * solar_thickness)
        solar_x0 = 0.5 * body_length - 0.5 * solar_chord
        solar_x1 = solar_x0 + solar_chord
        solar_color = color_fn("solar", 0.5 * (solar_x0 + solar_x1), total_length)
        starboard_y0 = 0.5 * body_shape.width + solar_clearance
        starboard_y1 = starboard_y0 + solar_span
        port_y1 = -0.5 * body_shape.width - solar_clearance
        port_y0 = port_y1 - solar_span

        _add_box(axis, solar_x0, solar_x1, starboard_y0, starboard_y1, -0.5 * solar_thickness, 0.5 * solar_thickness, solar_color, alpha=0.92, zorder=30.0)
        _add_box(axis, solar_x0, solar_x1, port_y0, port_y1, -0.5 * solar_thickness, 0.5 * solar_thickness, solar_color, alpha=0.92, zorder=30.0)
        _add_box_edges(axis, solar_x0, solar_x1, starboard_y0, starboard_y1, -0.5 * solar_thickness, 0.5 * solar_thickness, color=NASA_TEXT, linewidth=1.2, zorder=31.0)
        _add_box_edges(axis, solar_x0, solar_x1, port_y0, port_y1, -0.5 * solar_thickness, 0.5 * solar_thickness, color=NASA_TEXT, linewidth=1.2, zorder=31.0)
        _extend_bounds(bounds, [solar_x0, solar_x1], [port_y0, starboard_y1], [-0.5 * solar_thickness, 0.5 * solar_thickness])

    if geometry.A_rad > 0.0:
        rad_area_each = 0.5 * geometry.A_rad
        rad_length, rad_span = _rect_dims(rad_area_each, geometry.AR_rad)
        rad_thickness = max(0.02, 0.03 * min(body_shape.width, body_shape.height))
        rad_x0 = 0.55 * body_length - 0.5 * rad_length
        rad_x1 = rad_x0 + rad_length
        rad_color = color_fn("rad", 0.5 * (rad_x0 + rad_x1), total_length)
        _add_box(axis, rad_x0, rad_x1, -0.5 * rad_span, 0.5 * rad_span, 0.5 * body_shape.height, 0.5 * body_shape.height + rad_thickness, rad_color, alpha=0.78, zorder=20.0)
        _add_box(axis, rad_x0, rad_x1, -0.5 * rad_span, 0.5 * rad_span, -0.5 * body_shape.height - rad_thickness, -0.5 * body_shape.height, rad_color, alpha=0.78, zorder=20.0)
        _extend_bounds(
            bounds,
            [rad_x0, rad_x1],
            [-0.5 * rad_span, 0.5 * rad_span],
            [-0.5 * body_shape.height - rad_thickness, 0.5 * body_shape.height + rad_thickness],
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

    bounds = _render_spacecraft(axis, geometry, _color)
    if not bounds["x"]:
        _extend_bounds(bounds, [0.0, 1.0], [-0.5, 0.5], [-0.5, 0.5])
    _style_3d_axis(axis, f"SC Geometry | Iteration {iteration if iteration is not None else 0}")
    _set_equal_limits(axis, bounds)


def draw_spacecraft_drag_geometry(
    figure,
    state: SpacecraftState,
    diagnostics: DragDiagnostics,
    iteration: int | None = None,
) -> None:
    """Render spacecraft geometry colored by local wake ratio."""
    figure.clear()
    axis = figure.add_subplot(111, projection="3d")
    body_length = max(_safe_float(state.geometry.L_body), 0.0)
    intake_length = max(_safe_float(state.geometry.L_in), 0.0)
    total_length = body_length + intake_length

    x_drag, fy_profile, fz_profile = _wake_profile_arrays(total_length, diagnostics)
    x_samples = _drag_geometry_samples(body_length, intake_length, diagnostics)

    def _color(_component: str, x_mid: float, _total: float) -> Any:
        _, _, wake_mean = _wake_fraction_at_x(x_mid, total_length, x_drag, fy_profile, fz_profile)
        return _wake_color(wake_mean)

    bounds = _render_spacecraft(axis, state.geometry, _color, x_samples=x_samples)
    if not bounds["x"]:
        _extend_bounds(bounds, [0.0, 1.0], [-0.5, 0.5], [-0.5, 0.5])
    _style_3d_axis(axis, f"3D Drag Exposure | Iteration {iteration if iteration is not None else 0}")
    _set_equal_limits(axis, bounds)

    axis.text2D(
        0.02,
        0.02,
        "Color scale: green = 0.00, red = 1.00 | color = mean(fy, fz)",
        transform=axis.transAxes,
        color=NASA_TEXT,
        fontsize=9,
        fontfamily="Courier New",
    )

    scalar_map = cm.ScalarMappable(norm=WAKE_NORM, cmap=WAKE_CMAP)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=axis, fraction=0.045, pad=0.08)
    colorbar.set_label("Wake ratio [-]", color=NASA_TEXT, fontsize=9)
    colorbar.ax.tick_params(colors=NASA_TEXT, labelsize=8)
    colorbar.outline.set_edgecolor(NASA_TEXT)
    figure.patch.set_facecolor(NASA_BG)
    figure.tight_layout()


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


def _drag_component_map(state: SpacecraftState, diagnostics: DragDiagnostics) -> dict[str, float]:
    drag_values: dict[str, Any] = {}
    drag_state = state.drag

    if is_dataclass(drag_state):
        for field_info in fields(drag_state):
            drag_values[field_info.name] = getattr(drag_state, field_info.name)
    for name, value in vars(drag_state).items():
        drag_values[name] = value

    totals_to_attr = {
        "body": "drag_body_side",
        "inlet": "drag_inlet_side",
        "solar": "drag_solar",
        "rad": "drag_rad",
    }
    for key, value in getattr(diagnostics, "totals", {}).items():
        attr_name = totals_to_attr.get(key)
        if attr_name is not None:
            drag_values[attr_name] = value

    preferred_order = [
        "drag_body_side",
        "drag_inlet_side",
        "drag_inlet_front",
        "drag_solar",
        "drag_rad",
        "drag_body",
        "drag_inlet",
        "drag_inlet_normal",
    ]
    component_names = [name for name in preferred_order if name in drag_values]
    component_names.extend(
        sorted(
            name
            for name in drag_values
            if name.startswith("drag_") and name not in component_names and name != "drag_total"
        )
    )
    return {name: _safe_float(drag_values[name]) for name in component_names}


def draw_drag_distribution(
    figure,
    state: SpacecraftState,
    diagnostics: DragDiagnostics,
    iteration: int | None = None,
) -> None:
    """Render drag distribution diagnostics for one spacecraft state."""
    figure.clear()
    axes = figure.subplots(2, 2, squeeze=False)

    total_length = max(_safe_float(state.geometry.L_body + state.geometry.L_in), 0.0)
    x_raw = np.asarray(diagnostics.x_array, dtype=float)
    x_values, order = _drag_x_spacecraft_frame(total_length, x_raw)
    d_body = np.asarray(diagnostics.d_body_array, dtype=float)[order]
    d_in = np.asarray(diagnostics.d_in_array, dtype=float)[order]
    d_total = d_body + d_in

    if x_values.size >= 2:
        dx = np.diff(x_values)
        d_body_cum = np.concatenate(([0.0], np.cumsum(0.5 * (d_body[1:] + d_body[:-1]) * dx)))
        d_in_cum = np.concatenate(([0.0], np.cumsum(0.5 * (d_in[1:] + d_in[:-1]) * dx)))
    else:
        d_body_cum = np.zeros_like(d_body)
        d_in_cum = np.zeros_like(d_in)

    d_total_cum = d_body_cum + d_in_cum
    fy_values = np.asarray(diagnostics.fy_array, dtype=float)[order]
    fz_values = np.asarray(diagnostics.fz_array, dtype=float)[order]
    exchange_terms = _momentum_exchange_terms(state)

    def _style_axis(axis) -> None:
        axis.set_facecolor(NASA_BG)
        axis.grid(True, color=NASA_GRID, alpha=0.65, linestyle="--", linewidth=0.7)
        axis.tick_params(colors=NASA_TEXT, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(NASA_TEXT)

    def _shade_regions(axis) -> None:
        body_end = max(_safe_float(state.geometry.L_body), 0.0)
        inlet_end = max(_safe_float(state.geometry.L_body + state.geometry.L_in), body_end)
        axis.axvspan(0.0, body_end, color="#e8dfcf", alpha=0.35)
        axis.axvspan(body_end, inlet_end, color="#d7e3f0", alpha=0.35)

    local_axis = axes[0][0]
    _style_axis(local_axis)
    if x_values.size:
        _shade_regions(local_axis)
    local_axis.plot(x_values, d_body, color="#c44e52", linewidth=1.9, label="Body local")
    local_axis.plot(x_values, d_in, color="#4c72b0", linewidth=1.9, label="Intake local")
    local_axis.plot(x_values, d_total, color="#55a868", linewidth=1.6, linestyle="--", label="Total local")
    local_axis.set_title("LOCAL DRAG DENSITY", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    local_axis.set_xlabel("X from thruster plane [m]", color=NASA_TEXT, fontsize=9)
    local_axis.set_ylabel("Unscaled drag density", color=NASA_TEXT, fontsize=8)
    local_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

    cumulative_axis = axes[0][1]
    _style_axis(cumulative_axis)
    if x_values.size:
        _shade_regions(cumulative_axis)
    cumulative_axis.plot(x_values, d_body_cum, color="#c44e52", linewidth=1.9, label="Body cumulative")
    cumulative_axis.plot(x_values, d_in_cum, color="#4c72b0", linewidth=1.9, label="Intake cumulative")
    cumulative_axis.plot(x_values, d_total_cum, color="#55a868", linewidth=1.6, linestyle="--", label="Total cumulative")
    cumulative_axis.set_title("CUMULATIVE DRAG", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    cumulative_axis.set_xlabel("X from thruster plane [m]", color=NASA_TEXT, fontsize=9)
    cumulative_axis.set_ylabel("Integrated drag", color=NASA_TEXT, fontsize=8)
    cumulative_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

    capture_axis = axes[1][0]
    _style_axis(capture_axis)
    if x_values.size:
        _shade_regions(capture_axis)
    capture_axis.plot(x_values, fy_values, color="#4c72b0", linewidth=1.8, label="fy")
    capture_axis.plot(x_values, fz_values, color="#dd8452", linewidth=1.8, label="fz")
    capture_axis.set_ylim(-0.05, 1.05)
    capture_axis.set_title("CAPTURE FRACTION", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    capture_axis.set_xlabel("X from thruster plane [m]", color=NASA_TEXT, fontsize=9)
    capture_axis.set_ylabel("Capture fraction [-]", color=NASA_TEXT, fontsize=8)
    capture_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

    totals_axis = axes[1][1]
    _style_axis(totals_axis)
    drag_components = _drag_component_map(state, diagnostics)
    component_labels = [_drag_component_label(name) for name in drag_components]
    component_values = list(drag_components.values())
    component_labels.extend(["Refuel", "Propulsive"])
    component_values.extend([exchange_terms["refueling_exchange"], exchange_terms["propulsive_exchange"]])
    values = np.nan_to_num(np.asarray(component_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    totals_axis.bar(component_labels, values, color=[NASA_LINE[i % len(NASA_LINE)] for i in range(len(component_labels))], edgecolor=NASA_TEXT, linewidth=0.8)
    shown_drag_sum = float(np.sum(list(drag_components.values())))
    totals_axis.axhline(exchange_terms["total_drag"], color="#6b7280", linestyle="--", linewidth=1.1, label=f"State total drag = {exchange_terms['total_drag']:.3e} N")
    if not np.isclose(shown_drag_sum, exchange_terms["total_drag"], rtol=1.0e-6, atol=1.0e-12):
        totals_axis.axhline(shown_drag_sum, color="#2f6db3", linestyle="-.", linewidth=1.1, label=f"Shown drag sum = {shown_drag_sum:.3e} N")
    totals_axis.axhline(exchange_terms["total_load"], color="#dd8452", linestyle=":", linewidth=1.2, label=f"Drag + exchanges = {exchange_terms['total_load']:.3e} N")
    totals_axis.set_title("DRAG + EXCHANGE TERMS", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
    totals_axis.set_ylabel("Force [N]", color=NASA_TEXT, fontsize=8)
    totals_axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)

    figure.patch.set_facecolor(NASA_BG)
    figure.suptitle(f"DRAG DISTRIBUTION | Iteration {iteration if iteration is not None else 0}", color=NASA_TEXT, fontsize=14, fontfamily="Courier New", fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])


def draw_propulsion_overview(figure, state: SpacecraftState, iteration: int | None = None) -> None:
    """Render a stylized thruster silhouette and propulsion telemetry."""
    figure.clear()
    silhouette_axis, telemetry_axis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1.0]})

    power_required = _safe_float(getattr(state.thruster, "power_required", 0.0))
    thrust = _safe_float(getattr(state.thruster, "thrust", 0.0))
    mass_flow = _safe_float(getattr(state.thruster, "m_flow", 0.0))
    isp = _safe_float(getattr(state.thruster, "specific_impulse", 0.0))
    efficiency = _safe_float(getattr(state.thruster, "thruster_eff", 0.0))
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
    figure.suptitle(f"PROPULSION CONSOLE | Iteration {iteration if iteration is not None else 0}", color=NASA_TEXT, fontsize=14, fontfamily="Courier New", fontweight="bold")
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
        self.series = series
        self.paths = list(series.keys())
        self.rows: list[dict[str, Any]] = []
        self.drag_diagnostics_cache: dict[int, tuple[SpacecraftState, DragDiagnostics]] = {}

        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1520x920")
        self.root.configure(bg=NASA_BG)

        self.view_iteration = tk.IntVar(value=max(len(history) - 1, 0))
        self.status_vars = {
            "geometry": tk.StringVar(),
            "drag3d": tk.StringVar(),
            "drag": tk.StringVar(),
            "propulsion": tk.StringVar(),
            "refueling": tk.StringVar(),
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
        self.figure_tabs["drag3d"] = self._create_figure_tab("3D Drag", self.status_vars["drag3d"])
        self.figure_tabs["drag"] = self._create_figure_tab("Drag Test", self.status_vars["drag"])
        self.figure_tabs["propulsion"] = self._create_figure_tab("Propulsion", self.status_vars["propulsion"])
        self.text_tabs["refueling"] = self._create_text_tab("Refueling", self.status_vars["refueling"])
        self.text_tabs["atmosphere"] = self._create_text_tab("Atmosphere", self.status_vars["atmosphere"])

        for spec in default_specs:
            self.add_plot_row(*spec)

    def _build_controls(self, controls_parent) -> None:
        tk.Label(controls_parent, text="ARISS FLIGHT DATA BOARD", bg=NASA_BG, fg=NASA_TEXT, font=("Courier New", 11, "bold"), justify="left").pack(anchor="w", pady=(0, 8))
        tk.Label(
            controls_parent,
            text="SELECT ANY STATE CHANNEL\nMULTI-SERIES PER PLOT\nPLOTS + 3D + 3D DRAG + DRAG + PROP + REFUEL + ATMOSPHERE",
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
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(plot_tab, bg=NASA_BG)
        toolbar_frame.pack(fill="x")
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.pack(fill="x")

    def _create_figure_tab(self, tab_name: str, status_var) -> dict[str, Any]:
        tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.notebook.add(tab, text=tab_name)
        self._create_iteration_controls(tab, status_var)
        figure = plt.Figure(figsize=(11, 8), dpi=100, facecolor=NASA_BG)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(tab, bg=NASA_BG)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        return {"tab": tab, "figure": figure, "canvas": canvas, "toolbar": toolbar}

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
        self.redraw_drag_3d()
        self.redraw_drag_test()
        self.redraw_propulsion()
        self.redraw_refueling()
        self.redraw_atmosphere()

    def _get_drag_diagnostics(self, index: int) -> tuple[SpacecraftState, DragDiagnostics]:
        cached = self.drag_diagnostics_cache.get(index)
        if cached is not None:
            return cached
        state = deepcopy(self.history[index])
        with redirect_stdout(io.StringIO()):
            diagnostics = drag_model(state)
        cached = (state, diagnostics)
        self.drag_diagnostics_cache[index] = cached
        return cached

    def _style_history_axis(self, axis) -> None:
        axis.set_facecolor(NASA_BG)
        axis.grid(True, color=NASA_GRID, alpha=0.65, linestyle="--", linewidth=0.7)
        axis.tick_params(colors=NASA_TEXT, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color(NASA_TEXT)

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
            selected_indices = row["paths"].curselection()
            selected_paths = [row["paths"].get(i) for i in selected_indices]

            if not selected_paths:
                axis.set_title(row["title"].get() or "No series selected", color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
                axis.text(0.5, 0.5, "Select one or more series", transform=axis.transAxes, ha="center", va="center", color=NASA_TEXT, fontsize=9, fontfamily="Courier New")
                axis.set_xticks([])
                axis.set_yticks([])
                continue

            all_positive = True
            for line_idx, path in enumerate(selected_paths):
                y_values = self.series.get(path, [])
                if not y_values or any(value <= 0.0 for value in y_values):
                    all_positive = False
                axis.plot(x_values, y_values, color=NASA_LINE[line_idx % len(NASA_LINE)], linewidth=1.8, label=path)

            axis.set_title(row["title"].get() or _default_title(selected_paths), color=NASA_TEXT, fontsize=10, fontfamily="Courier New")
            axis.set_xlabel("Iteration", color=NASA_TEXT, fontsize=9)
            axis.set_ylabel(selected_paths[0] if len(selected_paths) == 1 else "Selected values", color=NASA_TEXT, fontsize=8)
            if row["log"].get():
                axis.set_yscale("log" if all_positive else "symlog", linthresh=1.0e-9)

            legend = axis.legend(loc="best", facecolor=NASA_PANEL, edgecolor=NASA_GRID, framealpha=1.0, fontsize=7)
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(NASA_TEXT)

        for idx in range(plot_count, row_count * cols):
            axes[idx // cols][idx % cols].axis("off")

        self.figure.patch.set_facecolor(NASA_BG)
        self.figure.suptitle("ARISS FLIGHT DATA WALL", color=NASA_TEXT, fontsize=14, fontfamily="Courier New", fontweight="bold")
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

    def redraw_drag_3d(self) -> None:
        index = self._current_index()
        tab = self.figure_tabs["drag3d"]
        figure = tab["figure"]

        try:
            state, diagnostics = self._get_drag_diagnostics(index)
            draw_spacecraft_drag_geometry(figure, state, diagnostics, iteration=index)
            fy_profile = np.asarray(diagnostics.fy_array, dtype=float)
            fz_profile = np.asarray(diagnostics.fz_array, dtype=float)
            wake_profile = 0.5 * (
                np.nan_to_num(fy_profile, nan=0.0, posinf=1.0, neginf=0.0)
                + np.nan_to_num(fz_profile, nan=0.0, posinf=1.0, neginf=0.0)
            )
            if wake_profile.size:
                wake_min, wake_mean, wake_max = float(np.min(wake_profile)), float(np.mean(wake_profile)), float(np.max(wake_profile))
            else:
                wake_min = wake_mean = wake_max = 0.0
            self.status_vars["drag3d"].set(
                f"Iteration {index} / {len(self.history) - 1} | Exposure min {wake_min:.3f} | mean {wake_mean:.3f} | max {wake_max:.3f}"
            )
        except Exception as exc:
            self._draw_error_figure(figure, f"3D drag view unavailable\n{exc}")
            self.status_vars["drag3d"].set(f"Iteration {index} / {len(self.history) - 1} | 3D drag view unavailable")

        tab["canvas"].draw_idle()

    def redraw_drag_test(self) -> None:
        index = self._current_index()
        tab = self.figure_tabs["drag"]
        figure = tab["figure"]

        try:
            state, diagnostics = self._get_drag_diagnostics(index)
            draw_drag_distribution(figure, state, diagnostics, iteration=index)
            exchange_terms = _momentum_exchange_terms(state)
            drag_components = _drag_component_map(state, diagnostics)
            component_summary = " | ".join(f"{_drag_component_label(name)} {value:.3e} N" for name, value in drag_components.items())
            shown_drag_sum = float(np.sum(list(drag_components.values())))
            self.status_vars["drag"].set(
                f"Iteration {index} / {len(self.history) - 1}\n"
                f"{component_summary}\n"
                f"State total {exchange_terms['total_drag']:.3e} N | "
                f"Shown sum {shown_drag_sum:.3e} N | "
                f"Refuel exchange {exchange_terms['refueling_exchange']:.3e} N | "
                f"Propulsive exchange {exchange_terms['propulsive_exchange']:.3e} N | "
                f"Total load {exchange_terms['total_load']:.3e} N"
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
            f"Power {_safe_float(getattr(state.thruster, 'power_required', 0.0)):.3f} W | "
            f"m_flow {_safe_float(getattr(state.thruster, 'm_flow', 0.0)):.3e} kg/s"
        )
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
        self.status_vars["atmosphere"].set(
            f"Iteration {index} / {len(self.history) - 1} | Altitude {_safe_float(state.orbit.altitude):.2f} km"
        )

        try:
            properties = atmosphere_properties_from_height(state.orbit.altitude)
            lines = [
                "ATMOSPHERIC SNAPSHOT",
                "",
                f"Altitude [km]               : {_safe_float(state.orbit.altitude):.6f}",
                f"State Density [kg/m^3]      : {_safe_float(state.orbit.density):.6e}",
                f"Model Density [kg/m^3]      : {_safe_float(properties['density']):.6e}",
                f"Temperature [K]             : {_safe_float(properties['temperature']):.6f}",
                f"Molar Mass [kg/mol]         : {_safe_float(properties['molar_mass']):.6e}",
                f"R_specific [J/kg/K]         : {_safe_float(properties['specific_gas_constant']):.6f}",
                f"Orbital Velocity [m/s]      : {_safe_float(properties['orbital_velocity']):.6f}",
                f"Dynamic Pressure [Pa]       : {_safe_float(properties['dynamic_pressure']):.6e}",
                "",
                "COMPOSITION",
                "",
                f"O2 Density [kg/m^3]         : {_safe_float(properties['o2_density']):.6e}",
                f"N2 Density [kg/m^3]         : {_safe_float(properties['n2_density']):.6e}",
                f"O Density [kg/m^3]          : {_safe_float(properties['o_density']):.6e}",
            ]
        except Exception as exc:
            lines = [
                "ATMOSPHERIC SNAPSHOT",
                "",
                f"Atmosphere helper unavailable: {exc}",
                "",
                f"Altitude [km]               : {_safe_float(state.orbit.altitude):.6f}",
                f"Density [kg/m^3]            : {_safe_float(state.orbit.density):.6e}",
                f"Temperature [K]             : {_safe_float(state.orbit.temperature):.6f}",
                f"Molar Mass [kg/mol]         : {_safe_float(state.orbit.molar_mass):.6e}",
                f"Orbital Velocity [m/s]      : {_safe_float(state.orbit.velocity):.6f}",
            ]

        self._set_text_tab("atmosphere", lines)

    def run(self) -> None:
        self.root.mainloop()


def launch_history_ui(
    sc: SpacecraftState | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
    default_specs: list[PlotSpec] | None = None,
    window_title: str = "ARISS History Plotter",
    show: bool = True,
):
    """Run the sizing history and open the interactive visualization UI."""
    sc = sc or SpacecraftState()
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
    sc: SpacecraftState | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-3,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc or SpacecraftState(),
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
    sc: SpacecraftState | None = None,
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
    sc: SpacecraftState | None = None,
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
    sc: SpacecraftState | None = None,
    n_points: int = 64,
    show: bool = True,
):
    _ = n_points
    return _launch_with_specs(DRAG_SPECS, "ARISS Drag Console", sc=sc, show=show)


def plot_power_diagnostics(
    sc: SpacecraftState | None = None,
    efficiency: float = 0.2,
    alignment_deg: float = 0.0,
    baseline_power: float = 2000.0,
    show: bool = True,
):
    _ = (efficiency, alignment_deg, baseline_power)
    return _launch_with_specs(POWER_SPECS, "ARISS Power Console", sc=sc, show=show)


def plot_propulsion_diagnostics(
    sc: SpacecraftState | None = None,
    baseline_drag: float = 0.2,
    show: bool = True,
):
    _ = baseline_drag
    return _launch_with_specs(PROPULSION_SPECS, "ARISS Propulsion Console", sc=sc, show=show)


def plot_simulation_budgets(
    sc: SpacecraftState | None = None,
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
    sc: SpacecraftState | None = None,
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


__all__ = [
    "draw_spacecraft_drag_geometry",
    "draw_spacecraft_geometry",
    "launch_history_ui",
    "plot_atmosphere_profiles",
    "plot_budgets_total",
    "plot_dimension_evolution",
    "plot_drag_diagnostics",
    "plot_power_diagnostics",
    "plot_propulsion_diagnostics",
    "plot_simulation_budgets",
    "plot_simulation_history",
    "run_sizing_with_history",
]


if __name__ == "__main__":
    try:
        plot_simulation_history()
    except Exception:
        traceback.print_exc()
