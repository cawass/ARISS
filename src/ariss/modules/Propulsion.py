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

    # Convert specific impulse into exhaust velocity for the intake-fed thruster.
    exhaust_velocity = const.EARTH_GRAVITY * sc.thruster.specific_impulse

    # Recover inlet and body dimensions from area and aspect ratio so all drag
    # terms can be combined as equivalent Cd*A contributions.
    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    h_body = np.sqrt(sc.geometry.A_body / sc.geometry.AR_body)
    w_body = sc.geometry.A_body / h_body

    body_side_area = (2.0 * w_body + 2.0 * h_body) * sc.geometry.L_body
    inlet_side_area = 0.5 * (2.0 * w_in + 2.0 * h_in + 2.0 * w_body + 2.0 * h_body) * sc.geometry.L_in

    # Build the total drag-reference area seen by the propulsion system.
    cd_s_solar = sc.drag.cd_solar * sc.geometry.A_solar
    cd_s_rad = sc.drag.cd_rad * sc.geometry.A_rad
    cd_s_body = sc.drag.cd_body_side * body_side_area
    cd_s_inlet_side = sc.drag.cd_inlet_side * inlet_side_area
    cd_s_inlet_front = sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    cd_s_total = cd_s_solar + cd_s_rad + cd_s_body + cd_s_inlet_side + cd_s_inlet_front

    # Solve the propulsion capture area needed to balance drag and, if active,
    # leave additional intake margin for the refueling stream.
    sc.geometry.A_prop = (0.5 * sc.orbit.velocity * cd_s_total + sc.geometry.A_ref) / (exhaust_velocity - sc.orbit.velocity)

    # Invert the propulsion power relation to infer the atmospheric density that
    # makes the chosen propulsion area feasible at the available jet power.
    sc.orbit.density = (2.0 * sc.thruster.power_required) / (sc.orbit.velocity * sc.geometry.A_prop * (exhaust_velocity ** 2))

    # Update the orbit state from the solved density with a single atmosphere lookup.
    sc.orbit.altitude, sc.orbit.temperature, sc.orbit.molar_mass, sc.orbit.velocity = itemgetter("altitude", "temperature", "molar_mass", "velocity")(orbit_updates_from_density(sc.orbit.density))

    # The propulsion intake mass flow follows directly from captured density flux.
    sc.thruster.m_flow = sc.geometry.A_prop * sc.orbit.velocity * sc.orbit.density

    if sc.mission_profile.active_refueling:
        # Use the rocket equation to convert the mission delta-v requirement into
        # a fuel mass target, then convert that target into a required refill rate.
        sc.mission_profile.required_fuel = sc.mass.Mass_total * (np.exp((sc.mission_profile.delta_v) / (exhaust_velocity)) - 1)
        sc.refueling.m_flow = sc.mission_profile.required_fuel / sc.refueling.t_refuel
        sc.geometry.A_ref = sc.refueling.m_flow / (sc.orbit.density * sc.orbit.velocity)
    else:
        sc.geometry.A_ref = 0.0
        sc.refueling.m_flow = 0.0

    # Split the total intake into useful propulsion flow, useful refueling flow,
    # and the extra capture area lost through imperfect collection efficiency.
    sc.geometry.A_in_drag = (sc.geometry.A_ref + sc.geometry.A_prop) * (1 / sc.refueling.coll_eff - 1)
    sc.geometry.A_in = sc.geometry.A_prop + sc.geometry.A_ref + sc.geometry.A_in_drag

    # Resize the inlet geometry and recompute drag forces using the updated intake.
    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    inlet_side_area = 0.5 * (2.0 * w_in + 2.0 * h_in + 2.0 * w_body + 2.0 * h_body) * sc.geometry.L_in
    q = 0.5 * sc.orbit.density * sc.orbit.velocity ** 2

    # Update each drag contribution with the current dynamic pressure.
    sc.drag.drag_solar = q * sc.drag.cd_solar * sc.geometry.A_solar
    sc.drag.drag_rad = q * sc.drag.cd_rad * sc.geometry.A_rad
    sc.drag.drag_body_side = q * sc.drag.cd_body_side * body_side_area
    sc.drag.drag_inlet_side = q * sc.drag.cd_inlet_side * inlet_side_area
    sc.drag.drag_inlet_front = q * sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    sc.drag.drag_total = (sc.drag.drag_solar + sc.drag.drag_rad + sc.drag.drag_body_side + sc.drag.drag_inlet_side + sc.drag.drag_inlet_front)

    # Final propulsion outputs are thrust and propellant throughput.
    sc.thruster.thrust = exhaust_velocity * sc.orbit.density * sc.orbit.velocity * sc.geometry.A_prop
    sc.thruster.propellant_mass = sc.orbit.density * sc.orbit.velocity * sc.geometry.A_prop
