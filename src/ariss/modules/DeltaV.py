import numpy as np
import scipy.special
import os
import sys

from iteration.SpacecraftClass import SpacecraftClass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def f(e) -> float:
    return 2 / (1 - e) * scipy.special.ellipk(4 * e / (1 + e) ** 2) + 2 / (1 + e) * scipy.special.ellipe(
        4 * e / (1 + e) ** 2)

def spiral_dv_tan(e, a_0, a_1, GM) -> float:
    return 2 * np.pi * np.abs(np.sqrt(GM / a_0) - np.sqrt(GM / a_1)) / (1 - e**2) / f(e)

def inclin_dv(di, r, GM) -> float:
    return np.pi / 2 * np.sqrt(GM / r) * di

def delta_v_model(sc: SpacecraftClass) -> None:
    dv_down = spiral_dv_tan(0, sc._proba_h + sc._R_Earth, sc.h_orb * 1e3 + sc._R_Earth, sc._GM)
    m_prop_down = (np.exp(dv_down / (sc._Isp_orb * sc._g0)) - 1) * (sc.M_dry_tot + sc._proba_mass)

    dv_inc = inclin_dv(np.radians(sc._assumed_inclination_change), sc._R_Earth + sc._proba_h, sc._GM)
    m_prop_inc = (np.exp(dv_inc / (sc._Isp_orb * sc._g0)) - 1) * (sc.M_dry_tot + m_prop_down)

    dv_up = spiral_dv_tan(0, sc.h_orb * 1e3 + sc._R_Earth, sc._proba_h + sc._R_Earth, sc._GM)
    m_prop_up = (np.exp(dv_up / (sc._Isp_orb * sc._g0)) - 1) * (sc.M_dry_tot + m_prop_down + m_prop_inc)

    m_prop_tot = m_prop_up + m_prop_down + m_prop_inc
    sc.deltaV = sc._delta_V_proba + sc._delta_V_empty
    sc.M_prop =(np.exp(sc._delta_V_proba / (sc._Isp_orb * sc._g0)) - 1) * (sc.M_dry_tot + sc._proba_mass)* sc._propellant_SF + \
               (np.exp(sc._delta_V_empty / (sc._Isp_orb * sc._g0)) - 1) * (sc.M_dry_tot)* sc._propellant_SF
