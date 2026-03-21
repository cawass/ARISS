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
#      Reduced Crandall and Wirz (2022) validation driver. Drag and power plots
#      use the ARISS core drag model and paper reference geometries. The mission
#      storage plots (Fig. 19 and Fig. 20) are digitized paper recreations because
#      the conventional-EP tank packaging model is not part of ARISS core.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Validation.py
# ============================================================================== #

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _update_drag_outputs
from ariss.utils.atmosphere import orbit_updates_from_height
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend

CASE_3U_PATH = Path(__file__).with_name("CrandallWirz2022_3U.toml")
CASE_6U_PATH = Path(__file__).with_name("CrandallWirz2022_6U.toml")
FIG11_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig11.png")
TABLE1_TEXT_PATH = Path(__file__).with_name("crandall_wirz_2022_table1.txt")
TABLE1_PNG_PATH = Path(__file__).with_name("crandall_wirz_2022_table1.png")
FIG6_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig6.png")
FIG19_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig19.png")
FIG20_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig20.png")

REFERENCE_POWER_W = 96.0
TP_SWEEP_MN_KW = np.linspace(10.0, 30.0, 81, dtype=float)
FIG11_ALTITUDE_GRID_KM = np.linspace(150.0, 250.0, 401, dtype=float)
DRAG_ALTITUDE_GRID_KM = np.linspace(150.0, 250.0, 201, dtype=float)
TABLE1_ALTITUDES_KM = np.arange(150.0, 251.0, 10.0, dtype=float)
SOLAR_ACTIVITY_F107 = {
    "Solar Minimum": 62.0,
    "Mean Solar Activity": 114.0,
    "Solar Maximum": 200.0,
}
SOLAR_COLORS = {
    "Solar Minimum": PALETTE["l1_teal"],
    "Mean Solar Activity": PALETTE["sernn_pink"],
    "Solar Maximum": PALETTE["choice_mid"],
}
MISSION_COLORS = {
    "1 Year Mission": PALETTE["l1_teal"],
    "2 Year Mission": PALETTE["sernn_pink"],
    "3 Year Mission": PALETTE["choice_mid"],
    "4 Year Mission": PALETTE["cat_purple"],
}

# Digitized from the paper figure for a compact reference recreation.
FIG19_REFERENCE = {
    "1 Year Mission": {
        "altitude_km": np.asarray([184.0, 190.0, 198.0, 210.0, 220.0, 230.0, 240.0, 250.0, 265.0, 285.0, 300.0]),
        "payload_u": np.asarray([0.00, 0.52, 1.04, 1.50, 1.78, 2.00, 2.06, 2.11, 2.18, 2.30, 2.37]),
    },
    "2 Year Mission": {
        "altitude_km": np.asarray([205.0, 211.0, 220.0, 233.0, 243.0, 255.0, 265.0, 280.0, 300.0]),
        "payload_u": np.asarray([0.00, 0.46, 0.99, 1.46, 1.75, 1.98, 2.06, 2.18, 2.30]),
    },
    "3 Year Mission": {
        "altitude_km": np.asarray([218.0, 224.0, 233.0, 247.0, 258.0, 272.0, 282.0, 295.0, 300.0]),
        "payload_u": np.asarray([0.00, 0.44, 0.96, 1.43, 1.72, 1.95, 2.03, 2.14, 2.22]),
    },
    "4 Year Mission": {
        "altitude_km": np.asarray([228.0, 235.0, 245.0, 259.0, 270.0, 284.0, 292.0, 300.0]),
        "payload_u": np.asarray([0.00, 0.42, 0.94, 1.42, 1.70, 1.94, 2.01, 2.08]),
    },
}

# Digitized from the paper figure for a compact reference recreation.
FIG20_REFERENCE = {
    "1 Year Mission": {
        "altitude_km": np.asarray([185.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 280.0, 300.0]),
        "wet_mass_kg": np.asarray([10.25, 9.92, 8.98, 8.53, 8.26, 8.05, 7.94, 7.87, 7.82, 7.77, 7.74]),
    },
    "2 Year Mission": {
        "altitude_km": np.asarray([206.0, 215.0, 225.0, 235.0, 245.0, 255.0, 265.0, 280.0, 300.0]),
        "wet_mass_kg": np.asarray([10.30, 9.72, 9.23, 8.86, 8.56, 8.32, 8.14, 7.97, 7.90]),
    },
    "3 Year Mission": {
        "altitude_km": np.asarray([219.0, 228.0, 238.0, 248.0, 258.0, 268.0, 278.0, 290.0, 300.0]),
        "wet_mass_kg": np.asarray([10.28, 9.90, 9.49, 9.12, 8.80, 8.54, 8.31, 8.10, 7.99]),
    },
    "4 Year Mission": {
        "altitude_km": np.asarray([229.0, 238.0, 248.0, 258.0, 268.0, 278.0, 288.0, 300.0]),
        "wet_mass_kg": np.asarray([10.18, 9.80, 9.40, 9.03, 8.72, 8.47, 8.25, 8.08]),
    },
}


def load_spacecraft(case_path: Path):
    return load_spacecraft_from_base_config(case_path)


def _diameter_m(sc) -> float:
    return float(np.sqrt(max(sc.geometry.A_body, 0.0)))


def _solar_span_m(sc) -> float:
    return float(sc.geometry.A_solar / max(2.0 * sc.geometry.L_body, 1.0e-12))


def _apply_collection_efficiency_split(sc) -> None:
    if sc.geometry.use_intake_area_ratio:
        sc.geometry.A_in = sc.geometry.intake_area_ratio * sc.geometry.A_body

    sc.geometry.A_ref = 0.0
    sc.geometry.A_prop = sc.geometry.A_in * sc.refueling.coll_eff
    sc.geometry.A_in_drag = sc.geometry.A_in - sc.geometry.A_prop


def _evaluate_drag_state(spacecraft_template, altitude_km: float, f107: float | None = None):
    sc = deepcopy(spacecraft_template)
    sc.orbit.msis_f107 = float(sc.orbit.msis_f107 if f107 is None else f107)

    with redirect_stdout(io.StringIO()):
        orbit_updates = orbit_updates_from_height(
            altitude_km,
            msis_date=sc.orbit.msis_date,
            msis_f107=sc.orbit.msis_f107,
            msis_ap=sc.orbit.msis_ap,
            latitude=sc.orbit.latitude,
            longitude=sc.orbit.longitude,
            use_average=sc.orbit.use_average,
        )
    for key, value in orbit_updates.items():
        setattr(sc.orbit, key, value)

    _apply_collection_efficiency_split(sc)
    drag_model(sc)
    _update_drag_outputs(sc)
    return sc


def _required_load_n(sc) -> float:
    return float(sc.drag.drag_total + sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_prop)


def _inlet_total_load_n(sc) -> float:
    return float(sc.drag.drag_inlet_front + sc.drag.drag_inlet_side + sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_prop)


def build_fig11_curves(spacecraft_template) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}

    for label, f107 in SOLAR_ACTIVITY_F107.items():
        required_load_n = np.asarray(
            [_required_load_n(_evaluate_drag_state(spacecraft_template, float(altitude_km), f107)) for altitude_km in FIG11_ALTITUDE_GRID_KM],
            dtype=float,
        )
        target_thrust_n = 1.0e-6 * TP_SWEEP_MN_KW * REFERENCE_POWER_W
        feasible = (target_thrust_n >= float(required_load_n[-1])) & (target_thrust_n <= float(required_load_n[0]))
        altitude_km = np.interp(target_thrust_n[feasible], required_load_n[::-1], FIG11_ALTITUDE_GRID_KM[::-1])
        curves[label] = {
            "tp_mn_kw": TP_SWEEP_MN_KW[feasible],
            "altitude_km": np.asarray(altitude_km, dtype=float),
        }

    return curves


def plot_fig11(curves: dict[str, dict[str, np.ndarray]], save_path: Path = FIG11_OUTPUT_PATH, show: bool = True) -> Path:
    apply_validation_style()
    figure, axis = plt.subplots(figsize=(7.4, 5.5), dpi=150)

    for label, payload in curves.items():
        axis.plot(
            payload["tp_mn_kw"],
            payload["altitude_km"],
            color=SOLAR_COLORS[label],
            linewidth=2.0,
            label=label,
        )

    axis.set_xlim(10.0, 30.0)
    axis.set_ylim(150.0, 190.0)
    axis.set_xlabel("Thrust to Power [mN/kW]")
    axis.set_ylabel("Minimum Operating Altitude [km]")
    style_axis(axis)
    legend = axis.legend(loc="upper right")
    style_legend(legend)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def build_table1_rows(spacecraft_template) -> list[dict[str, str]]:
    baseline_state = _evaluate_drag_state(spacecraft_template, float(TABLE1_ALTITUDES_KM[0]))
    baseline_total_drag = max(_required_load_n(baseline_state), 1.0e-30)
    rows: list[dict[str, str]] = []

    for altitude_km in TABLE1_ALTITUDES_KM:
        state = _evaluate_drag_state(spacecraft_template, float(altitude_km))
        total_drag = max(_required_load_n(state), 1.0e-30)
        rows.append(
            {
                "Altitude [km]": f"{int(altitude_km)}",
                "Total": "1" if np.isclose(altitude_km, TABLE1_ALTITUDES_KM[0]) else f"{100.0 * total_drag / baseline_total_drag:.0f}%",
                "Inlet": f"{100.0 * _inlet_total_load_n(state) / total_drag:.0f}%",
                "SA Skin": f"{100.0 * float(state.drag.drag_solar) / total_drag:.0f}%",
                "Body Skin": f"{100.0 * float(state.drag.drag_body_side) / total_drag:.0f}%",
                "SA Frontal Area": f"{100.0 * float(state.drag.drag_solar_front) / total_drag:.0f}%",
            }
        )

    return rows


def save_table1_text(rows: list[dict[str, str]], save_path: Path = TABLE1_TEXT_PATH) -> Path:
    headers = ["Altitude [km]", "Total", "Inlet", "SA Skin", "Body Skin", "SA Frontal Area"]
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    lines = [
        "Crandall & Wirz (2022) | Table 1 recreation with ARISS",
        "",
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]

    for row in rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))

    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return save_path


def save_table1_png(rows: list[dict[str, str]], save_path: Path = TABLE1_PNG_PATH, show: bool = True) -> Path:
    headers = ["Altitude [km]", "Total", "Inlet", "SA Skin", "Body Skin", "SA Frontal Area"]
    cell_text = [[row[header] for header in headers] for row in rows]

    apply_validation_style()
    figure, axis = plt.subplots(figsize=(10.6, 4.8), dpi=150)
    axis.axis("off")
    axis.set_title(
        "Table 1  Percent drag contribution of each spacecraft surface for the 6U spacecraft",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    table = axis.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.55)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("black")
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(PALETTE["l1_teal_fill"])

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def build_drag_profiles(spacecraft_template, altitude_grid_km: np.ndarray = DRAG_ALTITUDE_GRID_KM) -> dict[str, np.ndarray]:
    total_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    frontal_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    solar_skin_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    body_skin_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    solar_front_drag_mn = np.empty_like(altitude_grid_km, dtype=float)

    for index, altitude_km in enumerate(altitude_grid_km):
        state = _evaluate_drag_state(spacecraft_template, float(altitude_km))
        total_drag_mn[index] = 1.0e3 * _required_load_n(state)
        frontal_drag_mn[index] = 1.0e3 * _inlet_total_load_n(state)
        solar_skin_drag_mn[index] = 1.0e3 * float(state.drag.drag_solar)
        body_skin_drag_mn[index] = 1.0e3 * float(state.drag.drag_body_side)
        solar_front_drag_mn[index] = 1.0e3 * float(state.drag.drag_solar_front)

    return {
        "altitude_km": np.asarray(altitude_grid_km, dtype=float),
        "total_drag_mn": total_drag_mn,
        "frontal_drag_mn": frontal_drag_mn,
        "solar_skin_drag_mn": solar_skin_drag_mn,
        "body_skin_drag_mn": body_skin_drag_mn,
        "solar_front_drag_mn": solar_front_drag_mn,
    }


def _plot_drag_panel(axis, profiles: dict[str, np.ndarray], title: str) -> None:
    altitude_km = profiles["altitude_km"]
    axis.plot(profiles["total_drag_mn"], altitude_km, color=PALETTE["l1_teal"], linewidth=1.8, label="Total Drag")
    axis.plot(profiles["frontal_drag_mn"], altitude_km, color=PALETTE["sernn_pink"], linewidth=1.6, label="Frontal Area Drag")
    axis.plot(profiles["solar_skin_drag_mn"], altitude_km, color=PALETTE["choice_mid"], linewidth=1.6, label="SA Skin Friction Drag")
    axis.plot(profiles["body_skin_drag_mn"], altitude_km, color=PALETTE["cat_purple"], linewidth=1.6, label="Body Skin Friction Drag")
    axis.plot(profiles["solar_front_drag_mn"], altitude_km, color=PALETTE["cat_green"], linewidth=1.6, label="SA Frontal Area Drag")
    axis.set_xscale("log")
    axis.set_xlim(1.0e-3, 1.0e1)
    axis.set_ylim(float(np.min(altitude_km)), float(np.max(altitude_km)))
    axis.set_xlabel("Drag [mN]")
    axis.set_title(title)
    style_axis(axis)
    axis.grid(True, which="both", color=PALETTE["light_grid"], linewidth=0.5, alpha=0.5)
    legend = axis.legend(loc="lower left", fontsize=8.5)
    style_legend(legend)


def plot_fig6_drag_comparison(
    profiles_3u: dict[str, np.ndarray],
    profiles_6u: dict[str, np.ndarray],
    save_path: Path = FIG6_OUTPUT_PATH,
    show: bool = True,
) -> Path:
    apply_validation_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150, sharey=True)
    _plot_drag_panel(axes[0], profiles_3u, "(a) 3U Drag")
    _plot_drag_panel(axes[1], profiles_6u, "(b) 6U Drag")
    axes[0].set_ylabel("Altitude [km]")
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def _smooth_digitized_curve(payload: dict[str, np.ndarray], x_key: str) -> tuple[np.ndarray, np.ndarray]:
    altitude_km = np.asarray(payload["altitude_km"], dtype=float)
    values = np.asarray(payload[x_key], dtype=float)
    interpolator = PchipInterpolator(altitude_km, values)
    altitude_fine = np.linspace(float(altitude_km[0]), float(altitude_km[-1]), 300, dtype=float)
    values_fine = np.asarray(interpolator(altitude_fine), dtype=float)
    return values_fine, altitude_fine


def plot_fig19_payload_volume(save_path: Path = FIG19_OUTPUT_PATH, show: bool = True) -> Path:
    apply_validation_style()
    figure, axis = plt.subplots(figsize=(6.1, 5.3), dpi=150)

    for label, payload in FIG19_REFERENCE.items():
        payload_u, altitude_km = _smooth_digitized_curve(payload, "payload_u")
        axis.plot(payload_u, altitude_km, color=MISSION_COLORS[label], linewidth=1.8, label=label)

    axis.set_xlim(0.0, 2.4)
    axis.set_ylim(180.0, 300.0)
    axis.set_xlabel("Payload Volume [U]")
    axis.set_ylabel("Altitude [km]")
    style_axis(axis)
    legend = axis.legend(loc="upper left", fontsize=10)
    style_legend(legend)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def plot_fig20_wet_mass(save_path: Path = FIG20_OUTPUT_PATH, show: bool = True) -> Path:
    apply_validation_style()
    figure, axis = plt.subplots(figsize=(6.1, 5.3), dpi=150)

    for label, payload in FIG20_REFERENCE.items():
        wet_mass_kg, altitude_km = _smooth_digitized_curve(payload, "wet_mass_kg")
        axis.plot(wet_mass_kg, altitude_km, color=MISSION_COLORS[label], linewidth=1.8, label=label)

    axis.set_xlim(7.5, 10.3)
    axis.set_ylim(180.0, 300.0)
    axis.set_xlabel("Mass [kg]")
    axis.set_ylabel("Operating Altitude [km]")
    style_axis(axis)
    legend = axis.legend(loc="upper right", fontsize=10)
    style_legend(legend)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def run_crandall_wirz_2022_validation(show: bool = True) -> Path:
    spacecraft_3u = load_spacecraft(CASE_3U_PATH)
    spacecraft_6u = load_spacecraft(CASE_6U_PATH)

    fig11_curves = build_fig11_curves(spacecraft_6u)
    table1_rows = build_table1_rows(spacecraft_6u)
    drag_profiles_3u = build_drag_profiles(spacecraft_3u)
    drag_profiles_6u = build_drag_profiles(spacecraft_6u)

    fig11_path = plot_fig11(fig11_curves, save_path=FIG11_OUTPUT_PATH, show=show)
    table1_text_path = save_table1_text(table1_rows, save_path=TABLE1_TEXT_PATH)
    table1_png_path = save_table1_png(table1_rows, save_path=TABLE1_PNG_PATH, show=show)
    fig6_path = plot_fig6_drag_comparison(drag_profiles_3u, drag_profiles_6u, save_path=FIG6_OUTPUT_PATH, show=show)
    fig19_path = plot_fig19_payload_volume(save_path=FIG19_OUTPUT_PATH, show=show)
    fig20_path = plot_fig20_wet_mass(save_path=FIG20_OUTPUT_PATH, show=show)

    print("Crandall & Wirz (2022) reduced validation suite")
    for case_path, spacecraft in ((CASE_3U_PATH, spacecraft_3u), (CASE_6U_PATH, spacecraft_6u)):
        diameter_m = _diameter_m(spacecraft)
        print(
            f"{case_path.name}: d = {diameter_m:.3f} m | "
            f"L/d = {spacecraft.geometry.L_body / max(diameter_m, 1.0e-12):.1f} | "
            f"s/d = {_solar_span_m(spacecraft) / max(diameter_m, 1.0e-12):.1f}"
        )
    print(f"Saved Fig. 11 recreation: {fig11_path}")
    print(f"Saved Table 1 text: {table1_text_path}")
    print(f"Saved Table 1 figure: {table1_png_path}")
    print(f"Saved Fig. 6 recreation: {fig6_path}")
    print(f"Saved Fig. 19 reference recreation: {fig19_path}")
    print(f"Saved Fig. 20 reference recreation: {fig20_path}")
    return fig11_path


if __name__ == "__main__":
    run_crandall_wirz_2022_validation(show=True)
