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
    #   H = sqrt(A / AR)
    #   W = A / H

    # Compute the usable solar power flux on the panel plane after accounting
    # for cell efficiency and the average Sun-pointing alignment angle.
    projected_flux = sc.solar.eta_solar * const.SOLAR_CONSTANT * np.cos(np.radians(sc.solar.av_aligment))

    # Convert the total spacecraft electrical demand into the solar collection
    # area needed to supply that power at the available projected flux.
    required_area = sc.power.Power_total / projected_flux

    # Recover intake and body widths from area and aspect ratio so the exposed
    # top surface already available for solar cells can be estimated.
    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    h_body = np.sqrt(sc.geometry.A_body / sc.geometry.AR_body)
    w_body = sc.geometry.A_body / h_body

    # Fixed top area includes the full body top surface plus the tapered intake
    # top surface approximated with the average of body and intake widths.
    fixed_top_area = (w_in + w_body) * 0.5 * sc.geometry.L_in + w_body * sc.geometry.L_body

    # Only the power area not already covered by the fixed spacecraft top
    # surface must be added as deployable solar panels.
    sc.geometry.A_solar = max(0.0, required_area - fixed_top_area)
