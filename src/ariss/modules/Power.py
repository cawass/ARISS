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
#      Solar-array sizing model based on projected solar flux and the required
#      electrical power budget.
#
#  Project:        ARISS
#  Module:         Power.py
#  Author:         Carlos Carrasco Requejo
# ============================================================================

from dataclasses import dataclass

import numpy as np

from ariss.utils import constants as const


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


@dataclass(frozen=True)
class PowerDiagnostics:
    power_required: float
    efficiency: float
    alignment_deg: float
    projected_flux: float
    required_area: float
    fixed_top_area: float
    deployable_area: float


def power_model(sc):
    # Inputs:
    #   sc: spacecraft state with solar, power, and geometry data.
    #
    # Outputs:
    #   sc.geometry.A_solar: required deployable solar area [m^2].
    #
    # Equations used:
    #   q_solar = eta_solar * S_sun * cos(theta)
    #   A_required = P_total / q_solar
    #   Elliptic/circular section: A = pi * W * H / 4
    #   Rectangular section: A = W * H

    # Compute the usable solar power flux on the panel plane after accounting
    # for cell efficiency and the average Sun-pointing alignment angle.
    projected_flux = sc.solar.eta_solar * const.SOLAR_CONSTANT * np.cos(np.radians(sc.solar.av_aligment))

    # Convert the total spacecraft electrical demand into the solar collection
    # area needed to supply that power at the available projected flux.
    required_area = sc.power.Power_total / projected_flux

    # Recover intake and body widths from area/aspect-ratio/shape so the exposed
    # top surface already available for solar cells is geometry-consistent.
    w_in, _h_in = _section_dims(sc.geometry.A_in, sc.geometry.AR_in, sc.geometry.S_in)
    w_body, _h_body = _section_dims(sc.geometry.A_body, sc.geometry.AR_body, sc.geometry.S_body)

    # Fixed top area includes the full body top surface plus the tapered intake
    # top surface approximated with the average of body and intake widths.
    fixed_top_area = (w_in + w_body) * 0.5 * sc.geometry.L_in + w_body * sc.geometry.L_body

    # Only the power area not already covered by the fixed spacecraft top
    # surface must be added as deployable solar panels.
    deployable_area = required_area - fixed_top_area
    sc.geometry.A_solar = deployable_area if deployable_area > 0.0 else 0.0
