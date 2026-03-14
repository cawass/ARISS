# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS — Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Propulsion sizing model for drag compensation and optional refueling.
#
#  Project:        ARISS
#  Module:         Propulsion.py
#  Author:         Carlos Carrasco Requejo,
# ============================================================================

from dataclasses import dataclass
from operator import itemgetter

import numpy as np

from ariss.utils import constants as const
from ariss.utils.atmosphere import orbit_updates_from_density


def _is_round(shape_code: str) -> bool:
    shape = str(shape_code).strip().lower()
    return shape.startswith("e") or shape.startswith("c")


def _section_dims(area: float, aspect_ratio: float, shape_code: str) -> tuple[float, float]:
    if area <= 0.0 or aspect_ratio <= 0.0:
        return 0.0, 0.0
    if _is_round(shape_code):
        height = np.sqrt(4.0 * area / (np.pi * aspect_ratio))
    else:
        height = np.sqrt(area / aspect_ratio)
    width = aspect_ratio * height
    return width, height


def _section_perimeter(width: float, height: float, shape_code: str) -> float:
    if width <= 0.0 or height <= 0.0:
        return 0.0
    if _is_round(shape_code):
        semi_y = 0.5 * width
        semi_z = 0.5 * height
        h_term = ((semi_y - semi_z) ** 2) / ((semi_y + semi_z) ** 2)
        return np.pi * (semi_y + semi_z) * (1.0 + (3.0 * h_term) / (10.0 + np.sqrt(4.0 - 3.0 * h_term)))
    return 2.0 * (width + height)


@dataclass(frozen=True)
class PropulsionDiagnostics:
    rho: float
    s_ref: float
    v_inf: float
    isp: float
    g0: float
    exhaust_velocity: float
    mass_flow_rate: float
    required_thrust: float
    required_prop_area: float


def _side_areas(geometry) -> tuple[float, float]:
    # Inputs:
    #   geometry: spacecraft geometry state with areas, shapes, and lengths.
    #
    # Outputs:
    #   Body-side area and inlet-side area [m^2].

    w_in, h_in = _section_dims(geometry.A_in, geometry.AR_in, geometry.S_in)
    w_body, h_body = _section_dims(geometry.A_body, geometry.AR_body, geometry.S_body)
    body_perimeter = _section_perimeter(w_body, h_body, geometry.S_body)
    inlet_perimeter = _section_perimeter(w_in, h_in, geometry.S_in)
    body_side_area = body_perimeter * geometry.L_body
    inlet_side_area = 0.5 * (inlet_perimeter + body_perimeter) * geometry.L_in
    return body_side_area, inlet_side_area


def _drag_reference_area_sum(sc) -> float:
    # Inputs:
    #   sc: spacecraft state with geometry and drag coefficients.
    #
    # Outputs:
    #   Sum of Cd*A contributions seen by the propulsion balance [m^2].

    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    cd_s_solar = sc.drag.cd_solar * sc.geometry.A_solar
    cd_s_rad = sc.drag.cd_rad * sc.geometry.A_rad
    cd_s_body = sc.drag.cd_body_side * body_side_area
    cd_s_inlet_side = sc.drag.cd_inlet_side * inlet_side_area
    cd_s_inlet_front = sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    return cd_s_solar + cd_s_rad + cd_s_body + cd_s_inlet_side + cd_s_inlet_front


def _update_refueling_capture(sc, exhaust_velocity: float) -> None:
    # Inputs:
    #   sc: spacecraft state with mission, orbit, and refueling data.
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates refueling mass flow and refueling intake area in place.

    if sc.mission_profile.active_refueling:
        sc.mission_profile.required_fuel = sc.mass.Mass_total * (np.exp(sc.mission_profile.delta_v / exhaust_velocity) - 1)
        sc.refueling.m_flow = sc.mission_profile.required_fuel / sc.refueling.t_refuel
        sc.geometry.A_ref = sc.refueling.m_flow / (sc.orbit.density * sc.orbit.velocity)
    else:
        sc.geometry.A_ref = 0.0
        sc.refueling.m_flow = 0.0


def _solve_required_prop_area(sc, cd_s_total: float, exhaust_velocity: float) -> float:
    # Inputs:
    #   sc: spacecraft state with orbit and refueling geometry.
    #   cd_s_total: total drag reference area [m^2].
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Propulsive capture area needed to balance drag and refueling [m^2].

    return (0.5 * sc.orbit.velocity * cd_s_total + sc.orbit.velocity * sc.geometry.A_ref) / (exhaust_velocity - sc.orbit.velocity)


def _update_density_from_power(sc, exhaust_velocity: float) -> None:
    # Inputs:
    #   sc: spacecraft state with propulsion power and geometry.
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates atmospheric density inferred from the propulsion power closure.

    sc.orbit.density = (2.0 * sc.thruster.power * sc.thruster.eff) / (sc.orbit.velocity * sc.geometry.A_prop * (exhaust_velocity ** 2))


def _update_orbit_from_density(sc) -> None:
    # Inputs:
    #   sc: spacecraft state with density already updated.
    #
    # Outputs:
    #   Updates altitude, temperature, molar mass, and velocity from density.

    sc.orbit.altitude, sc.orbit.temperature, sc.orbit.molar_mass, sc.orbit.velocity = itemgetter("altitude", "temperature", "molar_mass", "velocity")(
        orbit_updates_from_density(
            sc.orbit.density,
            msis_date=sc.orbit.msis_date,
            msis_f107=sc.orbit.msis_f107,
            msis_ap=sc.orbit.msis_ap,
        )
    )


def _update_intake_split_from_collection_efficiency(sc) -> None:
    # Inputs:
    #   sc: spacecraft state with solved propulsive and refueling areas.
    #
    # Outputs:
    #   Updates total intake and drag-only intake from collection efficiency.

    sc.geometry.A_in_drag = (sc.geometry.A_ref + sc.geometry.A_prop) * (1 / sc.refueling.coll_eff - 1)
    sc.geometry.A_in = sc.geometry.A_prop + sc.geometry.A_ref + sc.geometry.A_in_drag


def _solve_fixed_body_ratio_mode(sc, cd_s_total: float, exhaust_velocity: float) -> float:
    # Inputs:
    #   sc: spacecraft state using fixed-body intake-area-ratio mode.
    #   cd_s_total: total drag reference area [m^2].
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates the fixed-body ratio solution in place and returns exhaust velocity [m/s].

    sc.geometry.A_in = sc.geometry.intake_area_ratio * sc.geometry.A_body
    _update_refueling_capture(sc, exhaust_velocity)
    sc.geometry.A_prop = sc.geometry.A_in * sc.refueling.coll_eff - sc.geometry.A_ref
    sc.geometry.A_in_drag = sc.geometry.A_in - sc.geometry.A_prop - sc.geometry.A_ref
    sc.thruster.specific_impulse = (sc.orbit.velocity * (sc.geometry.A_prop + sc.geometry.A_ref + 0.5 * cd_s_total)) / (const.EARTH_GRAVITY * sc.geometry.A_prop)
    exhaust_velocity = const.EARTH_GRAVITY * sc.thruster.specific_impulse
    sc.thruster.m_flow = sc.geometry.A_prop * sc.orbit.velocity * sc.orbit.density
    _update_density_from_power(sc, exhaust_velocity)
    _update_orbit_from_density(sc)
    sc.geometry.A_in_drag = sc.geometry.A_in - sc.geometry.A_prop - sc.geometry.A_ref
    return exhaust_velocity


def _solve_variable_body_ratio_mode(sc, cd_s_total: float, exhaust_velocity: float) -> float:
    # Inputs:
    #   sc: spacecraft state using free-body intake-area-ratio mode.
    #   cd_s_total: total drag reference area [m^2].
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates the free-body ratio solution in place and returns exhaust velocity [m/s].

    _update_refueling_capture(sc, exhaust_velocity)
    sc.geometry.A_prop = _solve_required_prop_area(sc, cd_s_total, exhaust_velocity)
    sc.thruster.m_flow = sc.geometry.A_prop * sc.orbit.velocity * sc.orbit.density
    _update_density_from_power(sc, exhaust_velocity)
    _update_orbit_from_density(sc)
    _update_intake_split_from_collection_efficiency(sc)
    sc.geometry.A_body = sc.geometry.A_in / sc.geometry.intake_area_ratio
    return exhaust_velocity


def _solve_free_intake_mode(sc, cd_s_total: float, exhaust_velocity: float) -> float:
    # Inputs:
    #   sc: spacecraft state using the collection-efficiency intake split.
    #   cd_s_total: total drag reference area [m^2].
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates the free-intake solution in place and returns exhaust velocity [m/s].

    _update_refueling_capture(sc, exhaust_velocity)
    sc.geometry.A_prop = _solve_required_prop_area(sc, cd_s_total, exhaust_velocity)
    sc.thruster.m_flow = sc.geometry.A_prop * sc.orbit.velocity * sc.orbit.density
    _update_density_from_power(sc, exhaust_velocity)
    _update_orbit_from_density(sc)
    _update_intake_split_from_collection_efficiency(sc)
    return exhaust_velocity


def _update_drag_outputs(sc) -> None:
    # Inputs:
    #   sc: spacecraft state with updated orbit, geometry, and drag coefficients.
    #
    # Outputs:
    #   Updates drag forces in place using the current dynamic pressure.

    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    q = 0.5 * sc.orbit.density * sc.orbit.velocity ** 2
    sc.drag.drag_solar = q * sc.drag.cd_solar * sc.geometry.A_solar
    sc.drag.drag_rad = q * sc.drag.cd_rad * sc.geometry.A_rad
    sc.drag.drag_body_side = q * sc.drag.cd_body_side * body_side_area
    sc.drag.drag_inlet_side = q * sc.drag.cd_inlet_side * inlet_side_area
    sc.drag.drag_inlet_front = q * sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    sc.drag.drag_total = sc.drag.drag_solar + sc.drag.drag_rad + sc.drag.drag_body_side + sc.drag.drag_inlet_side + sc.drag.drag_inlet_front


def _update_force_balance_outputs(sc, exhaust_velocity: float) -> None:
    # Inputs:
    #   sc: spacecraft state with updated drag, orbit, and intake geometry.
    #   exhaust_velocity: current thruster exhaust velocity [m/s].
    #
    # Outputs:
    #   Updates thrust, total propulsion load, and the final force residual.

    sc.thruster.propellant_mass = sc.orbit.density * sc.orbit.velocity * sc.geometry.A_prop
    sc.thruster.thrust = exhaust_velocity * sc.thruster.propellant_mass
    sc.thruster.propulsive_ram_load = sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_prop
    sc.thruster.refueling_ram_load = sc.orbit.density * sc.orbit.velocity ** 2 * sc.geometry.A_ref
    sc.thruster.required_load = sc.drag.drag_total + sc.thruster.propulsive_ram_load + sc.thruster.refueling_ram_load
    sc.thruster.force_residual = sc.thruster.thrust - sc.thruster.required_load


def propulsion_model(sc):
    # Inputs:
    #   sc: spacecraft state with orbit, drag, geometry, thruster, and mission data.
    #
    # Outputs:
    #   Updates propulsion, orbit, intake geometry, refueling flow, and drag terms in place.
    #
    # Equations used:
    #   v_e = g0 * Isp
    #   P_jet = 0.5 * m_dot * v_e^2
    #   m_dot = rho * V * A_prop
    #   T = m_dot * v_e
    #   rho = 2 * P_jet / (V * A_prop * v_e^2)
    #   m_fuel = m0 * (exp(delta_v / v_e) - 1)
    #   q = 0.5 * rho * V^2



    exhaust_velocity = const.EARTH_GRAVITY * sc.thruster.specific_impulse
    cd_s_total = _drag_reference_area_sum(sc)

    if sc.geometry.use_intake_area_ratio and sc.geometry.fixed_body:
        exhaust_velocity = _solve_fixed_body_ratio_mode(sc, cd_s_total, exhaust_velocity)
    elif sc.geometry.use_intake_area_ratio:
        exhaust_velocity = _solve_variable_body_ratio_mode(sc, cd_s_total, exhaust_velocity)
    else:
        exhaust_velocity = _solve_free_intake_mode(sc, cd_s_total, exhaust_velocity)

    _update_drag_outputs(sc)
    _update_force_balance_outputs(sc, exhaust_velocity)
