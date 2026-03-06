"""Interactive history plots and 3D geometry view for ``SpacecraftState``."""

from __future__ import annotations

import sys
import traceback
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from math import ceil, pi, sqrt
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None

from ariss.core.spacecraft import GeometryState, SpacecraftState
from ariss.modules.Budjects import sizing_model
from ariss.modules.Drag import drag_model
from ariss.modules.Power import power_model
from ariss.modules.Propulsion import propulsion_model
from ariss.utils.atmosphere import orbit_updates_from_height


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
            "geometry.A_prop",
            "geometry.A_solar",
            "geometry.L_body",
            "geometry.L_in",
        ],
        "KEY GEOMETRY",
        False,
    ),
]


def run_sizing_with_history(
    sc: SpacecraftState,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-8,
) -> tuple[SpacecraftState, bool, list[SpacecraftState]]:
    """Run the sizing loop and retain full state snapshots for plotting."""
    try:
        orbit_updates = orbit_updates_from_height(sc.mission_profile.mission_height)
        sc = replace(sc, orbit=replace(sc.orbit, **orbit_updates))
    except ImportError:
        pass

    history: list[SpacecraftState] = [deepcopy(sc)]
    converged = False

    for _ in range(max_iterations):
        drag_model(sc)
        propulsion_model(sc)
        sizing_model(sc)
        power_model(sc)
        sizing_model(sc)
        history.append(deepcopy(sc))

        residual = abs(history[-1].mass.Mass_total - history[-2].mass.Mass_total)
        if residual <= mass_tolerance:
            converged = True
            break

    return sc, converged, history


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if is_dataclass(value):
        for field_info in fields(value):
            child = getattr(value, field_info.name)
            child_prefix = f"{prefix}.{field_info.name}" if prefix else field_info.name
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
    for state in history:
        current: dict[str, float] = {}
        _flatten_numeric("", state, current)
        for key, value in current.items():
            series.setdefault(key, []).append(value)
    return {key: series[key] for key in sorted(series)}


def _normalize_paths(
    path: str | Sequence[str] | None,
    available_paths: list[str],
) -> list[str]:
    if path is None:
        default_path = "mass.Mass_total" if "mass.Mass_total" in available_paths else available_paths[0]
        return [default_path]

    if isinstance(path, str):
        candidates = [path]
    else:
        candidates = list(path)

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


def _add_box(
    axis,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    color: str,
    alpha: float = 0.7,
) -> None:
    collection = Poly3DCollection(
        _box_faces(x0, x1, y0, y1, z0, z1),
        facecolors=color,
        edgecolors=NASA_TEXT,
        linewidths=0.9,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def _add_tapered_box(
    axis,
    x0: float,
    x1: float,
    front_width: float,
    front_height: float,
    rear_width: float,
    rear_height: float,
    color: str,
    alpha: float = 0.7,
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
    collection = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=NASA_TEXT,
        linewidths=0.9,
        alpha=alpha,
    )
    axis.add_collection3d(collection)


def _add_elliptic_tube(
    axis,
    x0: float,
    length: float,
    semi_y: float,
    semi_z: float,
    color: str,
    alpha: float = 0.65,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    x = np.array([x0, x0 + length])
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="ij")
    y_grid = semi_y * np.cos(theta_grid)
    z_grid = semi_z * np.sin(theta_grid)

    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=0.3,
        edgecolor=NASA_TEXT,
        shade=False,
    )


def _add_tapered_elliptic_tube(
    axis,
    x0: float,
    length: float,
    front_semi_y: float,
    front_semi_z: float,
    rear_semi_y: float,
    rear_semi_z: float,
    color: str,
    alpha: float = 0.65,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    x = np.linspace(x0, x0 + length, 25)
    progress = np.linspace(0.0, 1.0, len(x))
    semi_y = front_semi_y + (rear_semi_y - front_semi_y) * progress
    semi_z = front_semi_z + (rear_semi_z - front_semi_z) * progress
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="ij")
    semi_y_grid = np.tile(semi_y, (len(theta), 1))
    semi_z_grid = np.tile(semi_z, (len(theta), 1))
    y_grid = semi_y_grid * np.cos(theta_grid)
    z_grid = semi_z_grid * np.sin(theta_grid)

    axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color=color,
        alpha=alpha,
        linewidth=0.3,
        edgecolor=NASA_TEXT,
        shade=False,
    )


def _add_rectangle_outline(
    axis,
    x: float,
    width: float,
    height: float,
    color: str,
) -> None:
    y0 = -0.5 * width
    y1 = 0.5 * width
    z0 = -0.5 * height
    z1 = 0.5 * height
    axis.plot(
        [x, x, x, x, x],
        [y0, y1, y1, y0, y0],
        [z0, z0, z1, z1, z0],
        color=color,
        linewidth=2.0,
    )


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


def draw_spacecraft_geometry(axis, geometry: GeometryState, iteration: int | None = None) -> None:
    """Render a simplified spacecraft geometry derived from ``GeometryState``."""
    axis.clear()
    axis.set_facecolor(NASA_BG)

    bounds = {"x": [], "y": [], "z": []}

    body_length = _positive(geometry.L_body, 0.1)
    intake_length = _positive(geometry.L_in, 0.1)

    if (geometry.S_body or "").strip().lower().startswith("s"):
        body_width, body_height = _rect_dims(geometry.A_body, geometry.AR_body)
        _add_box(
            axis,
            0.0,
            body_length,
            -0.5 * body_width,
            0.5 * body_width,
            -0.5 * body_height,
            0.5 * body_height,
            GEOM_BODY,
            alpha=0.95,
        )
    else:
        semi_y, semi_z = _ellipse_radii(geometry.A_body, geometry.AR_body)
        body_width = 2.0 * semi_y
        body_height = 2.0 * semi_z
        _add_elliptic_tube(axis, 0.0, body_length, semi_y, semi_z, GEOM_BODY, alpha=0.95)

    _extend_bounds(
        bounds,
        [0.0, body_length],
        [-0.5 * body_width, 0.5 * body_width],
        [-0.5 * body_height, 0.5 * body_height],
    )

    intake_is_square = (geometry.S_in or "").strip().lower().startswith("s")
    body_is_square = (geometry.S_body or "").strip().lower().startswith("s")

    if intake_is_square:
        intake_width, intake_height = _rect_dims(geometry.A_in, geometry.AR_in)
        intake_semi_y = 0.5 * intake_width
        intake_semi_z = 0.5 * intake_height
    else:
        intake_semi_y, intake_semi_z = _ellipse_radii(geometry.A_in, geometry.AR_in)
        intake_width = 2.0 * intake_semi_y
        intake_height = 2.0 * intake_semi_z

    body_semi_y = 0.5 * body_width
    body_semi_z = 0.5 * body_height

    intake_x0 = body_length
    intake_x1 = body_length + intake_length

    if intake_is_square and body_is_square:
        _add_tapered_box(
            axis,
            intake_x0,
            intake_x1,
            body_width,
            body_height,
            intake_width,
            intake_height,
            GEOM_INTAKE,
            alpha=0.9,
        )
    else:
        _add_tapered_elliptic_tube(
            axis,
            intake_x0,
            intake_length,
            body_semi_y,
            body_semi_z,
            intake_semi_y,
            intake_semi_z,
            GEOM_INTAKE,
            alpha=0.9,
        )

    _extend_bounds(
        bounds,
        [intake_x0, intake_x1],
        [-0.5 * max(intake_width, body_width), 0.5 * max(intake_width, body_width)],
        [-0.5 * max(intake_height, body_height), 0.5 * max(intake_height, body_height)],
    )

    if geometry.A_prop > 0.0:
        prop_width, prop_height = _rect_dims(geometry.A_prop, geometry.AR_in)
        prop_x = body_length + 0.95 * intake_length
        _add_rectangle_outline(axis, prop_x, prop_width, prop_height, GEOM_PROP)
        _extend_bounds(
            bounds,
            [prop_x],
            [-0.5 * prop_width, 0.5 * prop_width],
            [-0.5 * prop_height, 0.5 * prop_height],
        )

    if geometry.A_solar > 0.0:
        solar_area_each = 0.5 * geometry.A_solar
        solar_span, solar_chord = _rect_dims(solar_area_each, geometry.AR_solar)
        solar_thickness = max(0.02, 0.03 * min(body_width, body_height))
        solar_x0 = 0.5 * body_length - 0.5 * solar_chord
        solar_x1 = solar_x0 + solar_chord

        _add_box(
            axis,
            solar_x0,
            solar_x1,
            0.5 * body_width,
            0.5 * body_width + solar_span,
            -0.5 * solar_thickness,
            0.5 * solar_thickness,
            GEOM_SOLAR,
            alpha=0.75,
        )
        _add_box(
            axis,
            solar_x0,
            solar_x1,
            -0.5 * body_width - solar_span,
            -0.5 * body_width,
            -0.5 * solar_thickness,
            0.5 * solar_thickness,
            GEOM_SOLAR,
            alpha=0.75,
        )
        _extend_bounds(
            bounds,
            [solar_x0, solar_x1],
            [-0.5 * body_width - solar_span, 0.5 * body_width + solar_span],
            [-0.5 * solar_thickness, 0.5 * solar_thickness],
        )

    if geometry.A_rad > 0.0:
        rad_area_each = 0.5 * geometry.A_rad
        rad_length, rad_span = _rect_dims(rad_area_each, geometry.AR_rad)
        rad_thickness = max(0.02, 0.03 * min(body_width, body_height))
        rad_x0 = 0.55 * body_length - 0.5 * rad_length
        rad_x1 = rad_x0 + rad_length

        _add_box(
            axis,
            rad_x0,
            rad_x1,
            -0.5 * rad_span,
            0.5 * rad_span,
            0.5 * body_height,
            0.5 * body_height + rad_thickness,
            GEOM_RAD,
            alpha=0.65,
        )
        _add_box(
            axis,
            rad_x0,
            rad_x1,
            -0.5 * rad_span,
            0.5 * rad_span,
            -0.5 * body_height - rad_thickness,
            -0.5 * body_height,
            GEOM_RAD,
            alpha=0.65,
        )
        _extend_bounds(
            bounds,
            [rad_x0, rad_x1],
            [-0.5 * rad_span, 0.5 * rad_span],
            [-0.5 * body_height - rad_thickness, 0.5 * body_height + rad_thickness],
        )

    axis.set_title(
        f"SC Geometry | Iteration {iteration if iteration is not None else 0}",
        color=NASA_TEXT,
        fontsize=11,
        fontfamily="Courier New",
    )
    axis.set_xlabel("X [m]", color=NASA_TEXT, fontsize=9)
    axis.set_ylabel("Y [m]", color=NASA_TEXT, fontsize=9)
    axis.set_zlabel("Z [m]", color=NASA_TEXT, fontsize=9)
    axis.tick_params(colors=NASA_TEXT, labelsize=8)
    axis.view_init(elev=24, azim=-58)

    grid_color = (185 / 255, 177 / 255, 162 / 255, 0.4)
    axis.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axis.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axis.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axis.xaxis.pane.set_edgecolor(NASA_GRID)
    axis.yaxis.pane.set_edgecolor(NASA_GRID)
    axis.zaxis.pane.set_edgecolor(NASA_GRID)
    axis.xaxis._axinfo["grid"]["color"] = grid_color
    axis.yaxis._axinfo["grid"]["color"] = grid_color
    axis.zaxis._axinfo["grid"]["color"] = grid_color

    if not bounds["x"]:
        _extend_bounds(bounds, [0.0, 1.0], [-0.5, 0.5], [-0.5, 0.5])
    _set_equal_limits(axis, bounds)


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

        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry("1520x920")
        self.root.configure(bg=NASA_BG)

        self.view_iteration = tk.IntVar(value=max(len(history) - 1, 0))
        self.iteration_label_var = tk.StringVar()

        self._build_layout()
        for spec in default_specs:
            self.add_plot_row(*spec)
        self.redraw()
        self.redraw_geometry()

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.controls = tk.Frame(self.root, bg=NASA_BG, padx=10, pady=10)
        self.controls.grid(row=0, column=0, sticky="ns")

        title = tk.Label(
            self.controls,
            text="ARISS FLIGHT DATA BOARD",
            bg=NASA_BG,
            fg=NASA_TEXT,
            font=("Courier New", 11, "bold"),
            justify="left",
        )
        title.pack(anchor="w", pady=(0, 10))

        subtitle = tk.Label(
            self.controls,
            text="SELECT ANY STATE CHANNEL\nMULTI-SERIES PER PLOT\nPLOTS + 3D GEOMETRY VIEW",
            bg=NASA_BG,
            fg=NASA_TEXT,
            font=("Courier New", 9),
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        button_row = tk.Frame(self.controls, bg=NASA_BG)
        button_row.pack(fill="x", pady=(0, 10))

        tk.Button(
            button_row,
            text="Add Plot",
            command=self.add_plot_row,
            bg=NASA_PANEL,
            fg=NASA_TEXT,
            activebackground=NASA_GRID,
            activeforeground=NASA_TEXT,
            relief="ridge",
        ).pack(side="left", padx=(0, 6))

        tk.Button(
            button_row,
            text="Render",
            command=self.redraw,
            bg=NASA_PANEL,
            fg=NASA_TEXT,
            activebackground=NASA_GRID,
            activeforeground=NASA_TEXT,
            relief="ridge",
        ).pack(side="left")

        self.rows_frame = tk.Frame(self.controls, bg=NASA_BG)
        self.rows_frame.pack(fill="both", expand=True)

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

        self.notebook = ttk.Notebook(self.root, style="ARISS.TNotebook")
        self.notebook.grid(row=0, column=1, sticky="nsew")

        self.plot_tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.view3d_tab = tk.Frame(self.notebook, bg=NASA_BG)
        self.notebook.add(self.plot_tab, text="History Plots")
        self.notebook.add(self.view3d_tab, text="3D View")

        self.figure = plt.Figure(figsize=(11, 8), dpi=100, facecolor=NASA_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_toolbar_frame = tk.Frame(self.plot_tab, bg=NASA_BG)
        self.plot_toolbar_frame.pack(fill="x")
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_toolbar_frame, pack_toolbar=False)
        self.plot_toolbar.update()
        self.plot_toolbar.pack(fill="x")

        view_controls = tk.Frame(self.view3d_tab, bg=NASA_BG, padx=10, pady=10)
        view_controls.pack(fill="x")

        tk.Label(
            view_controls,
            text="History Iteration",
            bg=NASA_BG,
            fg=NASA_TEXT,
            font=("Courier New", 10, "bold"),
        ).pack(anchor="w")

        self.iteration_scale = tk.Scale(
            view_controls,
            from_=0,
            to=max(len(self.history) - 1, 0),
            orient="horizontal",
            variable=self.view_iteration,
            command=self._on_iteration_change,
            bg=NASA_BG,
            fg=NASA_TEXT,
            troughcolor=NASA_PANEL,
            activebackground=NASA_GRID,
            highlightthickness=0,
        )
        self.iteration_scale.pack(fill="x", pady=(4, 2))

        tk.Label(
            view_controls,
            textvariable=self.iteration_label_var,
            bg=NASA_BG,
            fg=NASA_TEXT,
            font=("Courier New", 9),
            justify="left",
        ).pack(anchor="w")

        self.geometry_figure = plt.Figure(figsize=(11, 8), dpi=100, facecolor=NASA_BG)
        self.geometry_canvas = FigureCanvasTkAgg(self.geometry_figure, master=self.view3d_tab)
        self.geometry_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.geometry_toolbar_frame = tk.Frame(self.view3d_tab, bg=NASA_BG)
        self.geometry_toolbar_frame.pack(fill="x")
        self.geometry_toolbar = NavigationToolbar2Tk(
            self.geometry_canvas,
            self.geometry_toolbar_frame,
            pack_toolbar=False,
        )
        self.geometry_toolbar.update()
        self.geometry_toolbar.pack(fill="x")

    def _on_iteration_change(self, *_args) -> None:
        self.redraw_geometry()

    def add_plot_row(
        self,
        path: str | Sequence[str] | None = None,
        title: str | None = None,
        log_scale: bool = False,
    ) -> None:
        selected_paths = _normalize_paths(path, self.paths)
        if title is None:
            title = _default_title(selected_paths)

        row_frame = tk.Frame(
            self.rows_frame,
            bg=NASA_BG,
            highlightbackground=NASA_GRID,
            highlightthickness=1,
            padx=4,
            pady=4,
        )
        row_frame.pack(fill="x", pady=3)
        row_frame.grid_columnconfigure(0, weight=1)

        title_var = tk.StringVar(value=title)
        log_var = tk.BooleanVar(value=log_scale)

        series_frame = tk.Frame(row_frame, bg=NASA_BG)
        series_frame.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        series_frame.grid_columnconfigure(0, weight=1)

        path_list = tk.Listbox(
            series_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            width=42,
            height=6,
            bg=NASA_PANEL,
            fg=NASA_TEXT,
            selectbackground="#d4dee8",
            selectforeground=NASA_TEXT,
            activestyle="none",
            relief="flat",
            highlightthickness=1,
            highlightbackground=NASA_GRID,
        )
        path_list.grid(row=0, column=0, sticky="ew")

        list_scroll = tk.Scrollbar(
            series_frame,
            orient="vertical",
            command=path_list.yview,
            bg=NASA_PANEL,
            activebackground=NASA_GRID,
        )
        list_scroll.grid(row=0, column=1, sticky="ns")
        path_list.configure(yscrollcommand=list_scroll.set)

        for option in self.paths:
            path_list.insert(tk.END, option)

        for selected_path in selected_paths:
            path_list.selection_set(self.paths.index(selected_path))

        entry = tk.Entry(
            row_frame,
            textvariable=title_var,
            width=28,
            bg=NASA_PANEL,
            fg=NASA_TEXT,
            insertbackground=NASA_TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=NASA_GRID,
        )
        entry.grid(row=0, column=1, padx=(0, 6), sticky="w")

        check = tk.Checkbutton(
            row_frame,
            text="Log Y",
            variable=log_var,
            bg=NASA_BG,
            fg=NASA_TEXT,
            activebackground=NASA_BG,
            activeforeground=NASA_TEXT,
            selectcolor=NASA_PANEL,
        )
        check.grid(row=0, column=2, padx=(0, 6), sticky="w")

        def _remove() -> None:
            self.rows.remove(row_data)
            row_frame.destroy()
            self.redraw()

        tk.Button(
            row_frame,
            text="Remove",
            command=_remove,
            bg=NASA_PANEL,
            fg=NASA_TEXT,
            relief="ridge",
        ).grid(row=0, column=3, sticky="w")

        row_data = {"paths": path_list, "title": title_var, "log": log_var}
        self.rows.append(row_data)

    def redraw(self) -> None:
        self.figure.clear()
        if not self.rows:
            self.canvas.draw_idle()
            return

        plot_count = len(self.rows)
        cols = 1 if plot_count == 1 else 2
        row_count = ceil(plot_count / cols)
        axes = self.figure.subplots(row_count, cols, squeeze=False)
        x_values = list(range(len(self.history)))

        for idx, row in enumerate(self.rows):
            axis = axes[idx // cols][idx % cols]
            selected_indices = row["paths"].curselection()
            selected_paths = [row["paths"].get(i) for i in selected_indices]

            axis.set_facecolor(NASA_BG)
            axis.grid(True, color=NASA_GRID, alpha=0.65, linestyle="--", linewidth=0.7)
            axis.tick_params(colors=NASA_TEXT, labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(NASA_TEXT)

            if not selected_paths:
                axis.set_title(
                    row["title"].get() or "No series selected",
                    color=NASA_TEXT,
                    fontsize=10,
                    fontfamily="Courier New",
                )
                axis.text(
                    0.5,
                    0.5,
                    "Select one or more series",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color=NASA_TEXT,
                    fontsize=9,
                    fontfamily="Courier New",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                continue

            all_positive = True
            for line_idx, path in enumerate(selected_paths):
                y_values = self.series.get(path, [])
                if not y_values or any(value <= 0.0 for value in y_values):
                    all_positive = False

                axis.plot(
                    x_values,
                    y_values,
                    color=NASA_LINE[line_idx % len(NASA_LINE)],
                    linewidth=1.8,
                    label=path,
                )

            axis.set_title(
                row["title"].get() or _default_title(selected_paths),
                color=NASA_TEXT,
                fontsize=10,
                fontfamily="Courier New",
            )
            axis.set_xlabel("Iteration", color=NASA_TEXT, fontsize=9)
            axis.set_ylabel(
                selected_paths[0] if len(selected_paths) == 1 else "Selected values",
                color=NASA_TEXT,
                fontsize=8,
            )

            if row["log"].get():
                if all_positive:
                    axis.set_yscale("log")
                else:
                    axis.set_yscale("symlog", linthresh=1.0e-9)

            legend = axis.legend(
                loc="best",
                facecolor=NASA_PANEL,
                edgecolor=NASA_GRID,
                framealpha=1.0,
                fontsize=7,
            )
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(NASA_TEXT)

        for idx in range(plot_count, row_count * cols):
            axes[idx // cols][idx % cols].axis("off")

        self.figure.patch.set_facecolor(NASA_BG)
        self.figure.suptitle(
            "ARISS FLIGHT DATA WALL",
            color=NASA_TEXT,
            fontsize=14,
            fontfamily="Courier New",
            fontweight="bold",
        )
        self.figure.tight_layout(rect=[0, 0, 1, 0.97])
        self.canvas.draw_idle()
        self.redraw_geometry()

    def redraw_geometry(self) -> None:
        self.geometry_figure.clear()
        axis = self.geometry_figure.add_subplot(111, projection="3d")
        index = max(0, min(self.view_iteration.get(), len(self.history) - 1))
        state = self.history[index]

        draw_spacecraft_geometry(axis, state.geometry, iteration=index)

        self.iteration_label_var.set(
            f"Iteration {index} / {len(self.history) - 1} | "
            f"Mass {state.mass.Mass_total:.2f} kg | Altitude {state.orbit.altitude:.2f} km"
        )
        self.geometry_figure.tight_layout()
        self.geometry_canvas.draw_idle()

    def run(self) -> None:
        self.root.mainloop()


def launch_history_ui(
    sc: SpacecraftState | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-8,
    default_specs: list[PlotSpec] | None = None,
    window_title: str = "ARISS History Plotter",
    show: bool = True,
):
    """Run the sizing history and open the interactive visualization UI."""
    if sc is None:
        sc = SpacecraftState()

    _, _, history = run_sizing_with_history(
        sc,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
    )
    series = _history_series(history)

    if default_specs is None:
        default_specs = DEFAULT_HISTORY_SPECS

    if not show:
        return history, series

    try:
        app = _HistoryPlotterUI(history, series, default_specs, window_title)
    except Exception as exc:
        print(f"Visualization UI could not start: {exc}")
        return history, series

    app.run()
    return history, series


def plot_atmosphere_profiles(
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 600,
    show: bool = True,
):
    _ = (height_min_km, height_max_km, samples)
    return launch_history_ui(
        sc=SpacecraftState(),
        default_specs=[
            ("orbit.altitude", "ALTITUDE", False),
            ("orbit.density", "DENSITY", True),
            ("orbit.temperature", "TEMPERATURE", False),
            ("orbit.molar_mass", "MOLAR MASS", True),
            ("orbit.velocity", "ORBITAL VELOCITY", False),
            ("drag.drag_total", "TOTAL DRAG", True),
        ],
        window_title="ARISS Atmosphere / Orbit Console",
        show=show,
    )


def plot_budgets_total(
    sc: SpacecraftState | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1.0e-8,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        default_specs=[
            ("mass.Mass_total", "TOTAL MASS", False),
            ("mass.Mass_in", "INLET MASS", False),
            ("mass.Mass_solar", "SOLAR ARRAY MASS", False),
            ("power.Power_total", "TOTAL POWER", False),
            ("power.Power_prop", "PROPULSION POWER", True),
            ("power.Power_solar", "SOLAR POWER", True),
        ],
        window_title="ARISS Budget Console",
        show=show,
    )


def plot_dimension_evolution(
    sc: SpacecraftState | None = None,
    max_iterations: int = 20,
    mass_tolerance: float = 1.0e-8,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        default_specs=[
            ("geometry.A_in", "INTAKE AREA", False),
            ("geometry.A_prop", "PROPULSIVE AREA", False),
            ("geometry.A_solar", "SOLAR AREA", False),
            ("geometry.L_in", "INTAKE LENGTH", False),
            ("geometry.L_body", "BODY LENGTH", False),
            ("geometry.AR_in", "INTAKE ASPECT RATIO", False),
        ],
        window_title="ARISS Geometry Console",
        show=show,
    )


def plot_drag_diagnostics(
    sc: SpacecraftState | None = None,
    n_points: int = 64,
    show: bool = True,
):
    _ = n_points
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        default_specs=[
            ("drag.drag_total", "TOTAL DRAG", True),
            ("drag.drag_body", "BODY DRAG", True),
            ("drag.drag_inlet", "INLET DRAG", True),
            ("drag.drag_solar", "SOLAR DRAG", True),
            ("geometry.A_in", "INTAKE AREA", False),
            ("orbit.density", "DENSITY", True),
        ],
        window_title="ARISS Drag Console",
        show=show,
    )


def plot_power_diagnostics(
    sc: SpacecraftState | None = None,
    efficiency: float = 0.2,
    alignment_deg: float = 0.0,
    baseline_power: float = 2000.0,
    show: bool = True,
):
    _ = (efficiency, alignment_deg, baseline_power)
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        default_specs=[
            ("power.Power_total", "TOTAL POWER", False),
            ("power.Power_prop", "PROPULSION POWER", True),
            ("power.Power_solar", "SOLAR POWER", True),
            ("geometry.A_solar", "SOLAR ARRAY AREA", False),
            ("solar.eta_power", "POWER EFFICIENCY", False),
            ("solar.av_aligment", "ARRAY ALIGNMENT", False),
        ],
        window_title="ARISS Power Console",
        show=show,
    )


def plot_propulsion_diagnostics(
    sc: SpacecraftState | None = None,
    baseline_drag: float = 0.2,
    show: bool = True,
):
    _ = baseline_drag
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        default_specs=[
            ("geometry.A_prop", "REQUIRED PROPULSIVE AREA", False),
            ("geometry.A_in", "INTAKE AREA", False),
            ("thruster.power_required", "POWER REQUIRED", True),
            ("thruster.thrust", "THRUST", True),
            ("thruster.propellant_mass", "PROPELLANT MASS FLOW", True),
            ("orbit.density", "INFERRED DENSITY", True),
        ],
        window_title="ARISS Propulsion Console",
        show=show,
    )


def plot_simulation_budgets(
    sc: SpacecraftState | None = None,
    max_iterations: int = 50,
    mass_tolerance: float = 1.0e-9,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        default_specs=[
            ("mass.Mass_total", "TOTAL MASS", False),
            ("power.Power_total", "TOTAL POWER", False),
            ("drag.drag_total", "TOTAL DRAG", True),
            ("geometry.A_prop", "PROPULSIVE AREA", False),
            ("orbit.altitude", "ALTITUDE", False),
            ("orbit.density", "DENSITY", True),
        ],
        window_title="ARISS Integrated Budget Console",
        show=show,
    )


def plot_simulation_history(
    sc: SpacecraftState | None = None,
    max_iterations: int = 200,
    mass_tolerance: float = 1.0e-8,
    show: bool = True,
):
    return launch_history_ui(
        sc=sc or SpacecraftState(),
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        default_specs=DEFAULT_HISTORY_SPECS,
        window_title="ARISS Mission History Console",
        show=show,
    )


__all__ = [
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
