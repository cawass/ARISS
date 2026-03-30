# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
# ============================================================================== #
#
#  ARISS - Atmospheric Refueling Iterative System Solver
# ----------------------------------------------------------------------------
#  Description:
#      Atmospheric refueling power model based on intake mass flow and tank
#      compression work.
#
#  Project:        ARISS
#  Module:         Refueling.py
#  Author:         Lucas Calderon del Rio
# ============================================================================

import numpy as np

from ariss.core.spacecraft import SpacecraftState

def refueling_model(sc: SpacecraftState) -> float:
    # Skip the compression model entirely when the mission does not require
    # propellant replenishment.
    sc.power.Power_refprop = 0.0

    if sc.mission_profile.active_refueling:
        # Captured atmospheric mass flow routed to the refueling tanks.

        if sc.mission_profile.active_and_bypass:
            m_dot_b = sc.refueling.m_flow
        else:
            m_dot_b = sc.refueling.m_flow + sc.thruster.m_flow

        # Power required to compress the refueling mass flow to tank pressure (isothermal).
        sc.power.Power_refprop = 1 / sc.refueling.eta_refuel * m_dot_b * sc.orbit.R_spec * sc.thermal.T_des * np.log(sc.refueling.p_tank / sc.orbit.p_orb)

        # Ideal-gas storage volume at design temperature and tank pressure.
        sc.refueling.V_prop = sc.mass.Mass_prop * sc.orbit.R_spec * sc.thermal.T_des / sc.refueling.p_tank

    return sc.power.Power_refprop
