import numpy as np
from scipy.special import erf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_friction_drag(S, epsilon, alpha, N_surf):
    return N_surf * (1 - epsilon * np.cos(2 * alpha)) / (np.sqrt(np.pi) * S) * np.exp(- S ** 2 * np.sin(alpha) ** 2)


def get_pressure_drag(S, epsilon, alpha):
    return np.sin(alpha) / S ** 2 * (1 + 2 * S ** 2 + epsilon * (1 - 2 * S ** 2 * np.cos(2 * alpha))) * erf(
        S * np.sin(alpha))


def get_thermal_drag(S, epsilon, alpha, T_orb, T_r):
    return (1 - epsilon) / S * np.sqrt(np.pi) * np.sin(alpha) ** 2 * np.sqrt(T_r / T_orb)


def get_drag_coefficient(S, epsilon, alpha, N_surf, T_orb, T_r):
    CD_friction = get_friction_drag(S, epsilon, alpha, N_surf)
    CD_pressure = get_pressure_drag(S, epsilon, alpha)
    CD_thermal = get_thermal_drag(S, epsilon, alpha, T_orb, T_r)
    return CD_friction + CD_pressure + CD_thermal


def drag_model(V_orb) -> None:
    S = sc.V_orb * np.sqrt(sc._M / (2 * sc._R * sc.T_orb))

    # solar panels, assuming a zero angle of attack
    sc.CD_dict['solar_extended'] = (1 - sc.w_solar) * get_drag_coefficient(
        S, sc._epsilon_specular, 0, 2, sc.T_orb, sc._T_reflected
    )

    # radiators, assuming a zero angle of attack
    sc.CD_dict['therm'] = (1 - sc.w_thermal) * get_drag_coefficient(
        S, sc._epsilon_specular, 0, 2, sc.T_orb, sc._T_reflected
    )

    # spacecraft surface, assuming a zero angle of attack
    sc.CD_dict['bus'] = (1 - sc.w_bus) * get_drag_coefficient(
        S, sc._epsilon_specular, 0, 1, sc.T_orb, sc._T_reflected
    )

    # refueling intake
    sc.CD_dict['ref'] = get_drag_coefficient(
        S, sc._epsilon_specular, np.pi / 2, 1, sc.T_orb, sc._T_reflected
    )

    # propulsion intake
    sc.CD_dict['prop'] = get_drag_coefficient(
        S, sc._epsilon_specular, np.pi / 2, 1, sc.T_orb, sc._T_reflected
    )
