import os
import sys

from iteration.SpacecraftClass import SpacecraftClass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def refueling_model(sc: SpacecraftClass) -> None:

    # Calculate the intake area
    sc.S_dict['ref'] = sc.M_prop / sc._t_refuel / (sc.rho_orb * sc.V_orb * sc._epsilon)

    Tb = sc._T_eq  # sc.T_orb * beta ** (sc._gamma - 1)

    # Calculate the work done on the fluid
    m_dot_b = sc.rho_orb * sc.V_orb * sc.S_dict['ref'] * sc._epsilon # Mass flow rate after the intake

    m_dot_b += sc.m_dot_prop  # Add the propellant mass flow rate to the intake mass flow rate because there is no bypass
    P2 = 1 / sc._eta_refuel * 1 / (sc._gamma - 1) * m_dot_b * sc._R_spec * Tb * ((sc._p2 / sc._p1) ** ((sc._gamma - 1) / sc._gamma) - 1)
    sc.V_prop = sc.M_prop * sc._R_spec * sc._T_eq / sc._p2

    # Sum both powers
    sc.P_dict['ref'] = sc._P1 + P2
