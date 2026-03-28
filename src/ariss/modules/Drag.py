# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS  "Atmospheric Refueling Iterative System Solver"
# ============================================================================== #
#  Description:
#      Free-molecular drag coefficient model used inside the sizing loop.
#
#  Project:        ARISS
#  Module:         Drag.py
#  Author:         Carlos Carrasco Requejo
# ============================================================================ #

import numpy as np
from scipy.special import erf

from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const

# ============================================================================== #
#  HELPERS
# ============================================================================== #

def _drag_coefficient(speed_ratio: float, epsilon: float, alpha: float, orb_temp: float, wall_temp: float, multiplier: float) -> float:
    # Inputs:
    #   speed_ratio: free-molecular speed ratio.
    #   epsilon: accommodation coefficient of the evaluated surface.
    #   alpha: incidence angle [rad].
    #   orb_temp: freestream temperature [K].
    #   wall_temp: wall/reference temperature [K].
    #   multiplier: geometric multiplier used by the pressure term.
    #
    # Outputs:
    #   Drag coefficient contribution for the selected surface.
    #
    # Equations used:
    #   Cd = friction + pressure + thermal
    #   pressure term uses erf(speed_ratio * sin(alpha))

    sin_a = np.sin(alpha)
    cos_2a = np.cos(2.0 * alpha)
    return (
        (1.0 - epsilon * cos_2a) / (np.sqrt(np.pi) * speed_ratio) * np.exp(-(speed_ratio ** 2) * (sin_a ** 2)) * multiplier
        + sin_a / (speed_ratio ** 2) * (1.0 + 2.0 * speed_ratio ** 2 + epsilon * (1.0 - 2.0 * speed_ratio ** 2 * cos_2a)) * erf(speed_ratio * sin_a) 
        + (1.0 - epsilon) / speed_ratio * np.sqrt(np.pi) * (sin_a ** 2) * np.sqrt(wall_temp / orb_temp) 
    )


def _section_dims(area: float, aspect_ratio: float, shape_code: str) -> tuple[float, float]:
    # Inputs:
    #   area: section area [m^2].
    #   aspect_ratio: width/height ratio.
    #   shape_code: geometry code ("e"/"c" for elliptic-circular, "s"/"r" for rectangular).
    #
    # Outputs:
    #   width, height [m] consistent with area and aspect ratio.
    #
    # Equations used:
    #   Elliptic/circular: area = pi * width * height / 4
    #   Rectangular: area = width * height

    if area <= 0.0 or aspect_ratio <= 0.0:
        return 0.0, 0.0
    shape = str(shape_code).strip().lower()
    is_round = shape.startswith("e") or shape.startswith("c")
    height = np.sqrt(4.0 * area / (np.pi * aspect_ratio)) if is_round else np.sqrt(area / aspect_ratio)
    return aspect_ratio * height, height


def _panel_front_area(total_area: float, aspect_ratio: float, thickness: float) -> float:
    # Inputs:
    #   total_area: total two-panel planform area [m^2].
    #   aspect_ratio: panel span/chord ratio.
    #   thickness: panel thickness [m].
    #
    # Outputs:
    #   Total frontal-edge area of the two panels [m^2].

    if total_area <= 0.0 or aspect_ratio <= 0.0 or thickness <= 0.0:
        return 0.0
    area_each = 0.5 * total_area
    chord = np.sqrt(area_each / aspect_ratio)
    span = area_each / chord
    return 2.0 * span * thickness

# ============================================================================== #
#  CORE
# ============================================================================== #

def drag_model(sc: SpacecraftState) -> None:
    # Inputs:
    #   sc: spacecraft state containing orbit, geometry, thermal, and drag fields.
    #
    # Outputs:
    #   Updates drag coefficients in sc.drag in place.
    #
    # Equations used:
    #   S = V_orbit * sqrt(M / (2 * R_u * T))
    #   alpha_in = atan(taper_half_gap / L_in)
    #   cd_effective = cd_raw * wake_factor

    # Compute the free-molecular speed ratio from orbit state variables.
    speed_ratio = sc.orbit.velocity * np.sqrt(sc.orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * sc.orbit.temperature))

    # Recover inlet and body section dimensions from area/aspect-ratio/shape.
    w_in, h_in = _section_dims(sc.geometry.A_in, sc.geometry.AR_in, sc.geometry.S_in)
    w_body, h_body = _section_dims(sc.geometry.A_body, sc.geometry.AR_body, sc.geometry.S_body)

    # Estimate the inlet side incidence from the maximum section taper half-gap.
    width_gap = abs(w_body - w_in)
    height_gap = abs(h_body - h_in)
    taper_half_gap = 0.5 * (width_gap if width_gap >= height_gap else height_gap)
    if sc.geometry.A_in < sc.geometry.A_body and taper_half_gap > 0.0 and sc.geometry.L_in > 0.0:
        alpha_in = np.arctan(taper_half_gap / sc.geometry.L_in)
    else:
        alpha_in = 0.0

    # Apply wake factors directly from geometry settings to each drag channel if applicable.

    if sc.geometry.A_in > sc.geometry.A_body:
        sc.drag.cd_inlet_side = _drag_coefficient(speed_ratio, sc.geometry.epsilon_in, sc.orbit.alpha + alpha_in, sc.orbit.temperature, sc.thermal.T_des, 1.0) * sc.geometry.wake_in
        sc.drag.cd_inlet_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_in_norm, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
        sc.drag.cd_solar = _drag_coefficient(speed_ratio, sc.geometry.epsilon_solar, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2.0) * sc.geometry.wake_solar
        sc.drag.cd_solar_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_solar, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) * sc.geometry.wake_solar
        sc.drag.cd_rad = _drag_coefficient(speed_ratio, sc.geometry.epsilon_rad, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2.0) * sc.geometry.wake_radiator
        sc.drag.cd_rad_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_rad, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) * sc.geometry.wake_radiator
        sc.drag.cd_body_side = _drag_coefficient(speed_ratio, sc.geometry.epsilon_body, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 1.0) * sc.geometry.wake_body
    else:
        sc.drag.cd_inlet_side = _drag_coefficient(speed_ratio, sc.geometry.epsilon_in, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
        sc.drag.cd_inlet_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_in_norm, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
        sc.drag.cd_solar = _drag_coefficient(speed_ratio, sc.geometry.epsilon_solar, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2.0) 
        sc.drag.cd_solar_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_solar, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
        sc.drag.cd_rad = _drag_coefficient(speed_ratio, sc.geometry.epsilon_rad, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2.0) 
        sc.drag.cd_rad_front = _drag_coefficient(speed_ratio, sc.geometry.epsilon_rad, sc.orbit.alpha + 0.5 * np.pi, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
        sc.drag.cd_body_side = _drag_coefficient(speed_ratio, sc.geometry.epsilon_body, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 1.0) 
    

