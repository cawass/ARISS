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
#      Manual Crandall and Wirz (2022) validation overlay for Fig. 11 using a
#      fixed-geometry ARISS drag solver with prescribed thrust-to-power.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Validation.py
# ============================================================================== #

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _update_drag_outputs
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmos, calculate_orbital_velocity, orbit_updates_from_height


CONFIG_PATH = Path(__file__).with_name("CrandallWirz2022.toml")
OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig11_overlay.png")
TABLE1_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_table1.txt")
DRAG_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_drag_breakdown.png")
FIG13_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig13_validation.png")
FIG14_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig14_validation.png")
FIG15_OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig15_validation.png")

REFERENCE_POWER_W = 96.0
ALTITUDE_SWEEP_KM = np.linspace(120.0, 260.0, 500, dtype=float)
TABLE1_ALTITUDES_KM = np.arange(150.0, 251.0, 10.0, dtype=float)
TP_SWEEP_MN_KW = np.linspace(10.0, 30.0, 41, dtype=float)
GEOMETRY_LD_VALUES = np.linspace(3.0, 10.0, 41, dtype=float)
GEOMETRY_SD_VALUES = np.linspace(1.0, 4.0, 31, dtype=float)
PARAMETER_ALTITUDE_SWEEP_KM = np.linspace(130.0, 300.0, 341, dtype=float)
FIGURE_BETA_DEG = [0.0, 30.0, 60.0, 90.0]
FIGURE_TP_VALUES = [10.0, 15.0, 20.0, 25.0, 30.0]
FIGURE_13_VB = 1500.0
FIGURE_15_VB = 500.0
PROPULSION_EFFICIENCY = 0.85
PROPULSION_GAMMA = 1.0
SOLAR_PANEL_THICKNESS_RATIO = 0.0083
ELEMENTARY_CHARGE = 1.602176634e-19
SOLAR_ACTIVITY_F107 = {"Solar Minimum": 62.0, "Mean Solar Activity": 114.0, "Solar Maximum": 200.0}
CURVE_COLORS = {"Solar Minimum": "#0b69c7", "Mean Solar Activity": "#e25822", "Solar Maximum": "#f2b01e"}

REFERENCE_TP = np.asarray([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0], dtype=float)
REFERENCE_CURVES = {
    "Solar Minimum": np.asarray([174.2, 169.4, 165.8, 162.4, 159.6, 157.0, 154.8, 152.9, 151.3, 150.8, 150.4], dtype=float),
    "Mean Solar Activity": np.asarray([178.8, 173.7, 169.4, 165.7, 162.5, 159.7, 157.2, 155.0, 153.5, 152.8, 152.3], dtype=float),
    "Solar Maximum": np.asarray([188.0, 182.7, 177.9, 173.7, 170.0, 166.8, 163.9, 161.2, 159.0, 157.5, 156.6], dtype=float),
}
SPECIES_PARTICLE_MASS = {
    "o": 15.999e-3 / const.AVOGADRO_NUMBER,
    "n2": 28.0134e-3 / const.AVOGADRO_NUMBER,
    "o2": 31.9988e-3 / const.AVOGADRO_NUMBER,
}


def load_reference_spacecraft(config_path: Path = CONFIG_PATH):
    return load_spacecraft_from_base_config(config_path)


def build_fig11_reference_curves(samples: int = 400) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    tp_fine = np.linspace(float(REFERENCE_TP[0]), float(REFERENCE_TP[-1]), samples, dtype=float)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for label, altitude_values in REFERENCE_CURVES.items():
        interpolator = PchipInterpolator(REFERENCE_TP, altitude_values)
        curves[label] = (tp_fine, np.asarray(interpolator(tp_fine), dtype=float))

    return curves


def _paper_diagonal_from_state(sc) -> float:
    return float(np.sqrt(max(sc.geometry.A_body, 0.0)))


def _apply_collection_efficiency_split(sc) -> None:
    if sc.geometry.use_intake_area_ratio:
        sc.geometry.A_in = sc.geometry.intake_area_ratio * sc.geometry.A_body

    sc.geometry.A_ref = 0.0
    sc.geometry.A_prop = sc.geometry.A_in * sc.refueling.coll_eff
    sc.geometry.A_in_drag = sc.geometry.A_in - sc.geometry.A_prop


def _evaluate_drag_state(reference_spacecraft, altitude_km: float, f107: float):
    sc = deepcopy(reference_spacecraft)
    sc.orbit.msis_f107 = float(f107)

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


def _required_propulsion_load(sc) -> float:
    return float(sc.drag.drag_total + sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_prop)


def _inlet_total_load(sc) -> float:
    return float(sc.drag.drag_inlet_front + sc.drag.drag_inlet_side + sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_prop)


def _reference_power_density(reference_spacecraft) -> float:
    return REFERENCE_POWER_W / max(float(reference_spacecraft.geometry.A_solar), 1.0e-12)


def _build_parameter_atmosphere_profile(reference_spacecraft, f107: float | None = None) -> dict[str, np.ndarray]:
    effective_f107 = float(reference_spacecraft.orbit.msis_f107 if f107 is None else f107)
    altitude_km = np.asarray(PARAMETER_ALTITUDE_SWEEP_KM, dtype=float)
    density, temperature, r_specific, o2_density, n2_density, o_density = atmos(
        altitude_km,
        msis_date=reference_spacecraft.orbit.msis_date,
        msis_f107=effective_f107,
        msis_ap=reference_spacecraft.orbit.msis_ap,
        latitude=reference_spacecraft.orbit.latitude,
        longitude=reference_spacecraft.orbit.longitude,
        use_average=reference_spacecraft.orbit.use_average,
    )
    velocity = calculate_orbital_velocity(altitude_km)
    q = 0.5 * density * velocity**2
    molar_mass = const.UNIVERSAL_GAS / np.maximum(np.asarray(r_specific, dtype=float), 1.0e-30)

    ref_state = deepcopy(reference_spacecraft)
    cd_solar = np.empty_like(altitude_km)
    cd_solar_front = np.empty_like(altitude_km)
    cd_body_side = np.empty_like(altitude_km)
    cd_inlet_front = np.empty_like(altitude_km)

    for index, altitude in enumerate(altitude_km):
        ref_state.orbit.altitude = float(altitude)
        ref_state.orbit.msis_f107 = effective_f107
        ref_state.orbit.density = float(density[index])
        ref_state.orbit.temperature = float(temperature[index])
        ref_state.orbit.molar_mass = float(molar_mass[index])
        ref_state.orbit.velocity = float(velocity[index])
        drag_model(ref_state)
        cd_solar[index] = float(ref_state.drag.cd_solar)
        cd_solar_front[index] = float(ref_state.drag.cd_solar_front)
        cd_body_side[index] = float(ref_state.drag.cd_body_side)
        cd_inlet_front[index] = float(ref_state.drag.cd_inlet_front)

    return {
        "altitude_km": altitude_km,
        "density": np.asarray(density, dtype=float),
        "temperature": np.asarray(temperature, dtype=float),
        "molar_mass": np.asarray(molar_mass, dtype=float),
        "o2_density": np.asarray(o2_density, dtype=float),
        "n2_density": np.asarray(n2_density, dtype=float),
        "o_density": np.asarray(o_density, dtype=float),
        "velocity": np.asarray(velocity, dtype=float),
        "q": np.asarray(q, dtype=float),
        "cd_solar": cd_solar,
        "cd_solar_front": cd_solar_front,
        "cd_body_side": cd_body_side,
        "cd_inlet_front": cd_inlet_front,
    }


def _mean_power_fraction(beta_deg: float, altitude_km: np.ndarray) -> np.ndarray:
    beta = np.radians(float(beta_deg))
    nu = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=float)
    psi = ((0.5 * np.pi - beta) * 0.5) * np.cos(2.0 * nu) + 0.5 * (0.5 * np.pi + beta)
    sunlight_factor = np.maximum(np.sin(psi), 0.0)

    orbital_radius = const.EARTH_RADIUS + np.asarray(altitude_km, dtype=float)
    numerator = np.sqrt(np.maximum(np.asarray(altitude_km, dtype=float) ** 2 + 2.0 * const.EARTH_RADIUS * np.asarray(altitude_km, dtype=float), 0.0))
    denominator = orbital_radius * max(np.cos(beta), 1.0e-12)
    argument = numerator / denominator
    eclipse_fraction = np.zeros_like(argument)
    eclipsed = argument < 1.0
    eclipse_fraction[eclipsed] = np.arccos(np.clip(argument[eclipsed], -1.0, 1.0)) / np.pi

    phase = np.abs(((nu[None, :] - np.pi + np.pi) % (2.0 * np.pi)) - np.pi)
    eclipse_half_width = np.pi * eclipse_fraction[:, None]
    eclipse_mask = phase <= eclipse_half_width
    mean_fraction = np.mean(np.where(eclipse_mask, 0.0, sunlight_factor[None, :]), axis=1)
    return np.asarray(mean_fraction, dtype=float)


def _geometry_sweep_arrays(reference_spacecraft) -> dict[str, np.ndarray]:
    d = _paper_diagonal_from_state(reference_spacecraft)
    ld_grid, sd_grid = np.meshgrid(GEOMETRY_LD_VALUES, GEOMETRY_SD_VALUES)
    area_body = d**2
    a_prop = area_body * reference_spacecraft.refueling.coll_eff
    a_in_drag = area_body - a_prop
    body_side_area = 4.0 * d * (ld_grid * d)
    solar_area = 2.0 * (sd_grid * d) * (ld_grid * d)
    solar_front_area = 2.0 * (sd_grid * d) * (SOLAR_PANEL_THICKNESS_RATIO * sd_grid * d)
    return {
        "ld_grid": ld_grid,
        "sd_grid": sd_grid,
        "a_prop": np.full_like(ld_grid, a_prop, dtype=float),
        "a_in_drag": np.full_like(ld_grid, a_in_drag, dtype=float),
        "body_side_area": body_side_area,
        "solar_area": solar_area,
        "solar_front_area": solar_front_area,
        "power_available_scale": _reference_power_density(reference_spacecraft) * solar_area,
        "a_in": np.full_like(ld_grid, area_body, dtype=float),
    }


def _thrust_profile_n(profile: dict[str, np.ndarray], geometry_payload: dict[str, np.ndarray], acceleration_voltage_v: float) -> np.ndarray:
    species_sum = (
        profile["o_density"] / np.sqrt(SPECIES_PARTICLE_MASS["o"])
        + profile["n2_density"] / np.sqrt(SPECIES_PARTICLE_MASS["n2"])
        + profile["o2_density"] / np.sqrt(SPECIES_PARTICLE_MASS["o2"])
    )
    thrust_n = (
        PROPULSION_EFFICIENCY
        * float(geometry_payload["a_in"][0, 0])
        * PROPULSION_GAMMA
        * float(geometry_payload["a_prop"][0, 0] / max(float(geometry_payload["a_in"][0, 0]), 1.0e-12))
        * profile["velocity"]
        * float(np.sqrt(2.0 * ELEMENTARY_CHARGE * acceleration_voltage_v))
        * species_sum
    )
    return np.asarray(thrust_n, dtype=float)


def _required_load_cube(profile: dict[str, np.ndarray], geometry_payload: dict[str, np.ndarray]) -> np.ndarray:
    q = profile["q"][:, None, None]
    return (
        q * profile["cd_solar"][:, None, None] * (2.0 * geometry_payload["solar_area"][None, :, :])
        + q * profile["cd_solar_front"][:, None, None] * geometry_payload["solar_front_area"][None, :, :]
        + q * profile["cd_body_side"][:, None, None] * geometry_payload["body_side_area"][None, :, :]
        + q * profile["cd_inlet_front"][:, None, None] * geometry_payload["a_in_drag"][None, :, :]
        + 2.0 * q * geometry_payload["a_prop"][None, :, :]
    )


def _first_feasible_map(altitude_km: np.ndarray, feasible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    has_solution = np.any(feasible, axis=0)
    first_index = np.argmax(feasible, axis=0)
    altitude_map = np.where(has_solution, altitude_km[first_index], np.nan)
    return altitude_map, first_index


def _build_parameter_sweep_case(reference_spacecraft, acceleration_voltage_v: float) -> dict[str, dict[tuple[float, float], np.ndarray]]:
    profile = _build_parameter_atmosphere_profile(reference_spacecraft)
    geometry_payload = _geometry_sweep_arrays(reference_spacecraft)
    load_cube = _required_load_cube(profile, geometry_payload)
    thrust_profile_n = _thrust_profile_n(profile, geometry_payload, acceleration_voltage_v)
    td_cube = thrust_profile_n[:, None, None] / np.maximum(load_cube, 1.0e-30)

    altitude_maps: dict[tuple[float, float], np.ndarray] = {}
    td_maps: dict[tuple[float, float], np.ndarray] = {}

    for beta_deg in FIGURE_BETA_DEG:
        mean_power_fraction = _mean_power_fraction(beta_deg, profile["altitude_km"])[:, None, None]
        power_available = geometry_payload["power_available_scale"][None, :, :] * mean_power_fraction

        for tp_mn_kw in FIGURE_TP_VALUES:
            tp_n_per_w = float(tp_mn_kw) * 1.0e-6
            power_required = load_cube / tp_n_per_w
            power_feasible = power_available >= power_required
            td_at_power_altitude, first_power_index = _first_feasible_map(profile["altitude_km"], power_feasible)
            td_map = np.where(
                np.any(power_feasible, axis=0),
                np.take_along_axis(td_cube, first_power_index[None, :, :], axis=0)[0],
                np.nan,
            )
            fully_feasible = power_feasible & (td_cube >= 1.0)
            altitude_map, _first_index = _first_feasible_map(profile["altitude_km"], fully_feasible)
            altitude_maps[(beta_deg, tp_mn_kw)] = altitude_map
            td_maps[(beta_deg, tp_mn_kw)] = td_map

    return {
        "altitude_maps": altitude_maps,
        "td_maps": td_maps,
        "ld_grid": geometry_payload["ld_grid"],
        "sd_grid": geometry_payload["sd_grid"],
        "f107": np.asarray([reference_spacecraft.orbit.msis_f107], dtype=float),
    }


def _plot_parameter_sweep_grid(
    data_maps: dict[tuple[float, float], np.ndarray],
    ld_grid: np.ndarray,
    sd_grid: np.ndarray,
    title: str,
    colorbar_label: str,
    save_path: Path,
    show: bool = True,
) -> Path:
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    figure, axes = plt.subplots(len(FIGURE_BETA_DEG), len(FIGURE_TP_VALUES), figsize=(13.0, 8.8), dpi=180, sharex=True, sharey=True)

    stacked = np.concatenate([values[np.isfinite(values)] for values in data_maps.values() if np.any(np.isfinite(values))])
    vmin = float(np.min(stacked))
    vmax = float(np.max(stacked))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#111111")
    mappable = None

    for row_index, beta_deg in enumerate(FIGURE_BETA_DEG):
        for col_index, tp_mn_kw in enumerate(FIGURE_TP_VALUES):
            axis = axes[row_index, col_index]
            payload = np.ma.masked_invalid(data_maps[(beta_deg, tp_mn_kw)])
            mappable = axis.contourf(ld_grid, sd_grid, payload, levels=16, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"T/P = {tp_mn_kw:.0f} mN/kW", fontsize=9)
            axis.grid(True, color="#c7c7c7", linewidth=0.4, alpha=0.35)
            if col_index == 0:
                axis.set_ylabel(f"beta = {beta_deg:.0f} deg\ns / d")
            if row_index == len(FIGURE_BETA_DEG) - 1:
                axis.set_xlabel("L / d")

    figure.suptitle(title, fontsize=13)
    if mappable is not None:
        colorbar = figure.colorbar(mappable, ax=axes, shrink=0.92, pad=0.02)
        colorbar.set_label(colorbar_label)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return save_path


def _build_drag_curve(reference_spacecraft, f107: float, altitude_grid_km: np.ndarray = ALTITUDE_SWEEP_KM) -> dict[str, np.ndarray]:
    drag_n = np.empty_like(altitude_grid_km, dtype=float)

    for index, altitude_km in enumerate(altitude_grid_km):
        sc = _evaluate_drag_state(reference_spacecraft, float(altitude_km), f107)
        drag_n[index] = _required_propulsion_load(sc)

    return {"altitude_km": np.asarray(altitude_grid_km, dtype=float), "drag_n": drag_n}


def solve_propulsion_power_point(reference_spacecraft, target_tp_mn_kw: float, f107: float, power_w: float = REFERENCE_POWER_W):
    drag_curve = _build_drag_curve(reference_spacecraft, f107)
    target_thrust_n = 1.0e-6 * float(target_tp_mn_kw) * float(power_w)

    drag_n = drag_curve["drag_n"]
    altitude_km = drag_curve["altitude_km"]
    drag_sorted = drag_n[::-1]
    altitude_sorted = altitude_km[::-1]

    if target_thrust_n < float(drag_sorted[0]) or target_thrust_n > float(drag_sorted[-1]):
        return None

    solved_altitude_km = float(np.interp(target_thrust_n, drag_sorted, altitude_sorted))
    solved_state = _evaluate_drag_state(reference_spacecraft, solved_altitude_km, f107)
    solved_state.thruster.power = float(power_w)
    solved_state.thruster.thrust = target_thrust_n
    solved_state.thruster.propellant_mass = 0.0
    solved_state.thruster.m_flow = 0.0
    solved_state.thruster.propulsive_ram_load = 0.0
    solved_state.thruster.refueling_ram_load = 0.0
    solved_state.thruster.required_load = _required_propulsion_load(solved_state)
    solved_state.thruster.force_residual = solved_state.thruster.thrust - solved_state.thruster.required_load

    return {
        "tp_mn_kw": float(target_tp_mn_kw),
        "power_w": float(power_w),
        "thrust_mn": 1.0e3 * target_thrust_n,
        "altitude_km": solved_altitude_km,
        "drag_n": float(solved_state.drag.drag_total),
        "required_load_n": float(solved_state.thruster.required_load),
        "force_residual_n": float(solved_state.thruster.force_residual),
    }


def run_custom_tp_sweep(reference_spacecraft):
    results: dict[str, dict[str, np.ndarray]] = {}

    for label, f107 in SOLAR_ACTIVITY_F107.items():
        tp_mn_kw: list[float] = []
        thrust_mn: list[float] = []
        altitude_km: list[float] = []
        drag_n: list[float] = []
        force_residual_n: list[float] = []

        for target_tp in TP_SWEEP_MN_KW:
            solution = solve_propulsion_power_point(reference_spacecraft, float(target_tp), float(f107))
            if solution is None:
                continue

            tp_mn_kw.append(solution["tp_mn_kw"])
            thrust_mn.append(solution["thrust_mn"])
            altitude_km.append(solution["altitude_km"])
            drag_n.append(solution["drag_n"])
            force_residual_n.append(solution["force_residual_n"])

        results[label] = {
            "tp_mn_kw": np.asarray(tp_mn_kw, dtype=float),
            "thrust_mn": np.asarray(thrust_mn, dtype=float),
            "altitude_km": np.asarray(altitude_km, dtype=float),
            "drag_n": np.asarray(drag_n, dtype=float),
            "force_residual_n": np.asarray(force_residual_n, dtype=float),
        }

    return results


def build_table1_rows(reference_spacecraft, f107: float | None = None) -> list[dict[str, str]]:
    effective_f107 = float(reference_spacecraft.orbit.msis_f107 if f107 is None else f107)
    baseline_state = _evaluate_drag_state(reference_spacecraft, float(TABLE1_ALTITUDES_KM[0]), effective_f107)
    baseline_total_drag = max(_required_propulsion_load(baseline_state), 1.0e-30)
    rows: list[dict[str, str]] = []

    for altitude_km in TABLE1_ALTITUDES_KM:
        state = _evaluate_drag_state(reference_spacecraft, float(altitude_km), effective_f107)
        total_drag = max(_required_propulsion_load(state), 1.0e-30)
        inlet_drag = _inlet_total_load(state)
        solar_skin_drag = float(state.drag.drag_solar)
        body_skin_drag = float(state.drag.drag_body_side)
        solar_front_drag = float(state.drag.drag_solar_front)

        rows.append(
            {
                "Altitude [km]": f"{int(altitude_km)}",
                "Total": "1" if np.isclose(altitude_km, TABLE1_ALTITUDES_KM[0]) else f"{100.0 * total_drag / baseline_total_drag:.0f}%",
                "Inlet": f"{100.0 * inlet_drag / total_drag:.0f}%",
                "SA Skin": f"{100.0 * solar_skin_drag / total_drag:.0f}%",
                "Body Skin": f"{100.0 * body_skin_drag / total_drag:.0f}%",
                "SA Frontal Area": f"{100.0 * solar_front_drag / total_drag:.0f}%",
            }
        )

    return rows


def save_table1_text(rows: list[dict[str, str]], save_path: Path = TABLE1_OUTPUT_PATH) -> Path:
    headers = ["Altitude [km]", "Total", "Inlet", "SA Skin", "Body Skin", "SA Frontal Area"]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }

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


def build_drag_breakdown_profiles(reference_spacecraft, altitude_grid_km: np.ndarray) -> dict[str, np.ndarray]:
    total_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    frontal_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    solar_skin_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    body_skin_drag_mn = np.empty_like(altitude_grid_km, dtype=float)
    solar_front_drag_mn = np.empty_like(altitude_grid_km, dtype=float)

    for index, altitude_km in enumerate(altitude_grid_km):
        state = _evaluate_drag_state(reference_spacecraft, float(altitude_km), float(reference_spacecraft.orbit.msis_f107))
        total_drag_mn[index] = 1.0e3 * _required_propulsion_load(state)
        frontal_drag_mn[index] = 1.0e3 * _inlet_total_load(state)
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


def plot_drag_breakdown(
    profiles: dict[str, np.ndarray],
    save_path: Path = DRAG_OUTPUT_PATH,
    show: bool = True,
) -> Path:
    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    figure, axis = plt.subplots(figsize=(6.2, 4.7), dpi=150)
    altitude_km = profiles["altitude_km"]

    axis.plot(profiles["total_drag_mn"], altitude_km, color="#0b69c7", linewidth=1.8, label="Total Drag")
    axis.plot(profiles["frontal_drag_mn"], altitude_km, color="#e25822", linewidth=1.6, label="Frontal Area Drag")
    axis.plot(profiles["solar_skin_drag_mn"], altitude_km, color="#f2b01e", linewidth=1.6, label="SA Skin Friction Drag")
    axis.plot(profiles["body_skin_drag_mn"], altitude_km, color="#7b3294", linewidth=1.6, label="Body Skin Friction Drag")
    axis.plot(profiles["solar_front_drag_mn"], altitude_km, color="#6dbb3c", linewidth=1.6, label="SA Frontal Area Drag")

    axis.set_xscale("log")
    axis.set_xlim(1.0e-3, 1.0e1)
    axis.set_ylim(float(np.min(altitude_km)), float(np.max(altitude_km)))
    axis.set_xlabel("Drag [mN]")
    axis.set_ylabel("Altitude [km]")
    axis.set_title("(b) 6U Drag")
    axis.grid(True, which="both", color="#bdbdbd", linewidth=0.5, alpha=0.35)
    axis.legend(loc="lower left", frameon=True, edgecolor="black", fancybox=False, fontsize=8.5)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return save_path


def plot_fig11_overlay(
    reference_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    ariss_results: dict[str, dict[str, np.ndarray]],
    save_path: Path = OUTPUT_PATH,
    show: bool = True,
) -> Path:
    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    figure, axis = plt.subplots(figsize=(7.4, 5.5), dpi=150)

    for label, (tp_values, altitude_values) in reference_curves.items():
        axis.plot(tp_values, altitude_values, color=CURVE_COLORS[label], linewidth=2.0, label=label)

    for label, payload in ariss_results.items():
        if payload["tp_mn_kw"].size == 0:
            continue
        axis.plot(
            payload["tp_mn_kw"],
            payload["altitude_km"],
            color=CURVE_COLORS[label],
            linestyle="--",
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            label=f"ARISS {label}",
        )

    reference_min_altitude = min(float(np.min(altitude_values)) for _, altitude_values in reference_curves.values())
    ariss_altitude_values = [
        float(np.min(payload["altitude_km"]))
        for payload in ariss_results.values()
        if payload["altitude_km"].size > 0
    ]
    ariss_min_altitude = min(ariss_altitude_values) if ariss_altitude_values else reference_min_altitude

    axis.set_xlim(10.0, 30.0)
    axis.set_ylim(min(reference_min_altitude, ariss_min_altitude) - 1.0, 190.0)
    axis.set_xlabel("Thrust to Power [mN/kW]")
    axis.set_ylabel("Minimum Operating Altitude [km]")
    axis.set_title("Crandall & Wirz 2022 | Fig. 11 with ARISS Overlay")
    axis.grid(True, color="#bdbdbd", linewidth=0.6, alpha=0.45)
    axis.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)

    return save_path


def plot_fig13_validation(reference_spacecraft, save_path: Path = FIG13_OUTPUT_PATH, show: bool = True) -> Path:
    payload = _build_parameter_sweep_case(reference_spacecraft, FIGURE_13_VB)
    return _plot_parameter_sweep_grid(
        payload["altitude_maps"],
        payload["ld_grid"],
        payload["sd_grid"],
        "Crandall & Wirz 2022 | Fig. 13 ARISS-style Minimum Operating Altitude | Vb = 1500 V",
        "Minimum operating altitude [km]",
        save_path,
        show=show,
    )


def plot_fig14_validation(reference_spacecraft, save_path: Path = FIG14_OUTPUT_PATH, show: bool = True) -> Path:
    payload = _build_parameter_sweep_case(reference_spacecraft, FIGURE_13_VB)
    return _plot_parameter_sweep_grid(
        payload["td_maps"],
        payload["ld_grid"],
        payload["sd_grid"],
        "Crandall & Wirz 2022 | Fig. 14 ARISS-style T/D | Vb = 1500 V",
        "Thrust-to-drag ratio [-]",
        save_path,
        show=show,
    )


def plot_fig15_validation(reference_spacecraft, save_path: Path = FIG15_OUTPUT_PATH, show: bool = True) -> Path:
    payload = _build_parameter_sweep_case(reference_spacecraft, FIGURE_15_VB)
    return _plot_parameter_sweep_grid(
        payload["altitude_maps"],
        payload["ld_grid"],
        payload["sd_grid"],
        "Crandall & Wirz 2022 | Fig. 15 ARISS-style Minimum Operating Altitude | Vb = 500 V",
        "Minimum operating altitude [km]",
        save_path,
        show=show,
    )


def run_crandall_wirz_2022_validation(show: bool = True) -> Path:
    reference_spacecraft = load_reference_spacecraft()
    reference_curves = build_fig11_reference_curves()
    ariss_results = run_custom_tp_sweep(reference_spacecraft)
    output_path = plot_fig11_overlay(reference_curves, ariss_results, save_path=OUTPUT_PATH, show=show)
    table1_rows = build_table1_rows(reference_spacecraft)
    table1_path = save_table1_text(table1_rows)
    drag_profiles = build_drag_breakdown_profiles(reference_spacecraft, TABLE1_ALTITUDES_KM)
    drag_path = plot_drag_breakdown(drag_profiles, save_path=DRAG_OUTPUT_PATH, show=show)
    fig13_path = plot_fig13_validation(reference_spacecraft, save_path=FIG13_OUTPUT_PATH, show=show)
    fig14_path = plot_fig14_validation(reference_spacecraft, save_path=FIG14_OUTPUT_PATH, show=show)
    fig15_path = plot_fig15_validation(reference_spacecraft, save_path=FIG15_OUTPUT_PATH, show=show)

    diameter_m = _paper_diagonal_from_state(reference_spacecraft)
    solar_span_m = reference_spacecraft.geometry.A_solar / (2.0 * max(reference_spacecraft.geometry.L_body, 1.0e-12))

    print("Crandall & Wirz (2022) | Fig. 11 ARISS custom overlay")
    print(
        "Reference spacecraft: "
        f"{reference_spacecraft.name} | d = {diameter_m:.3f} m | "
        f"L/d = {reference_spacecraft.geometry.L_body / max(diameter_m, 1.0e-12):.1f} | "
        f"s/d = {solar_span_m / max(diameter_m, 1.0e-12):.1f} | "
        f"eta_c = {reference_spacecraft.refueling.coll_eff:.2f}"
    )
    print(f"Prescribed power for T/P conversion: {REFERENCE_POWER_W:.1f} W")
    print("Reference curves are approximate read-offs from the published Fig. 11 for manual validation only.")
    for label, payload in ariss_results.items():
        if payload["tp_mn_kw"].size == 0:
            print(f"{label}: no solved ARISS custom sweep points")
            continue
        print(
            f"{label}: {payload['tp_mn_kw'].size} points | "
            f"T/P range = {float(np.min(payload['tp_mn_kw'])):.2f}-{float(np.max(payload['tp_mn_kw'])):.2f} mN/kW | "
            f"altitude range = {float(np.min(payload['altitude_km'])):.2f}-{float(np.max(payload['altitude_km'])):.2f} km | "
            f"max |force residual| = {float(np.max(np.abs(payload['force_residual_n']))):.3e} N"
        )
    print(f"Saved figure: {output_path}")
    print(f"Saved Table 1 recreation: {table1_path}")
    print(f"Saved drag breakdown figure: {drag_path}")
    print(f"Saved Fig. 13 validation: {fig13_path}")
    print(f"Saved Fig. 14 validation: {fig14_path}")
    print(f"Saved Fig. 15 validation: {fig15_path}")
    return output_path


if __name__ == "__main__":
    run_crandall_wirz_2022_validation(show=True)
