import os
import sys
import numpy as np

from iteration.SpacecraftClass import SpacecraftClass
from helpers.atmosphere import get_atmosphere_functions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def propulsion_model(sc: SpacecraftClass) -> None:
    CD_S: float = 0
    for key in sc.CD_dict:
        if key != 'prop':
            CD_S += sc.CD_dict[key] * sc.S_dict[key]

    sc.S_dict['prop'] = (1 / 2 * sc._V_assumed * CD_S + sc._epsilon * sc.S_dict['ref']) / \
                        (sc._epsilon* sc._Isp_atm * sc._g0 - 1 / 2 * sc._V_assumed * sc.CD_dict['prop'] - sc._V_assumed * sc._epsilon)

    sc.rho_orb = 2 * sc._eta_prop / (sc._g0 * sc._Isp_atm) * sc._P_prop \
                 / (sc.S_dict['prop'] * sc._V_assumed * sc._epsilon * sc._g0 * sc._Isp_atm)

    rho_func, v_func, _, temp_func = get_atmosphere_functions(PLOT=False)
    sc.h_orb = rho_func(sc.rho_orb)
    sc.V_orb = v_func(sc.h_orb)
    sc.T_orb = temp_func(sc.h_orb)

    sc.m_dot_prop = sc.S_dict['prop'] * sc.rho_orb * sc.V_orb * sc._epsilon
    sc.Thr = sc.m_dot_prop * sc._Isp_atm * sc._g0
    print("Thrust:", sc.Thr)
