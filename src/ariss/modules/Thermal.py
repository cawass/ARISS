"""
Thermal model for ARISS

This is a simple, 1-node thermal model which assumes a homogenous temperature across the spacecraft body.
"""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const

@dataclass(frozen=True)
class DragDiagnostics:
    """Detailed thermal outputs for post-processing.

    Attributes
    ----------
    Q_drag : float
        Drag heating [W]
    Q_sun : float
        Sun radiation heating [W]
    Q_albedo : float
        Earth albedo heating [W]
    Q_ir : float
        Earth infrared heating [W]
    Q_internal : float
        Internal heating [W]
    Q_radiated : float
        Heat radiated by spacecraft [W]
    T_max : float
        Maximum experienced temperature
    T_min : float
        Minimum experienced temperature
    """

    Q_drag: float = 0.0
    Q_sun: float = 0.0
    Q_albedo: float = 0.0
    Q_ir: float = 0.0
    Q_internal: float = 0.0
    Q_radiated: float = 0.0

    T_max: float = 0.0
    T_min: float = 0.0


def thermal_model(sc: SpacecraftClass) -> Tuple[Dict[str,float]]:
    """
    Thermal model for the spacecraft
    TODO Elaborate docstring
    """



    
def thermal_model(sc: SpacecraftClass) -> None:

    # Side area for earth heating
    S_side = sc.L * sc.H + (sc.H + sc.H_in) * sc.L_in / 2
    # Radiative area of bus - 1.5 factor for 0.5 intake emmisivity
    S_rad_bus = sc.H * sc.D * 1.5 + sc.L * sc.H * 2 + sc.L * sc.D + np.pi * (sc.D/2 + sc.D_in/2) * np.sqrt((sc.D_in/2 - sc.D/2) ** 2 + sc.L_in**2) *3/4
    # S_rad_bus = sc.H * sc.D + sc.L * sc.H * 1.5 + (sc.H_in+sc.H)/2*np.sqrt(sc.L_in**2 + ((sc.H_in-sc.H)/2)**2)*2 + sc.L * sc.D + (sc.D_in+sc.D)/2*np.sqrt(sc.L_in**2 + ((sc.D_in-sc.D)/2)**2)
    # Radiative area of solar panels
    S_rad_solar = sc.S_dict['pow'] - (sc.D_in+sc.D)/2*sc.L_in + np.pi * (sc.D/2 + sc.D_in/2) * np.sqrt((sc.D_in/2 - sc.D/2) ** 2 + sc.L_in**2) /4

    # Heat input
    #  Drag - it's assumed the incoming air transfers all its kinetic energy into heat and this is all the heating from drag
    sc.Q_dict['Q_d'] = 1 / 2 * sc.rho_orb * sc.V_orb ** 3 * (sc.S_dict['ref'] + sc.S_dict['prop'])
    #  Radiation - assuming the spacecraft is a box with taper with the top having the sun at 90 degrees and same for albedo and infrared
    sc.Q_dict['Q_s'] = sc._Sc * sc.S_dict['pow'] * (sc._alpha_dict['pow'] * (1 - sc._eta_solar))
    sc.Q_dict['Q_a'] = sc._alb_Earth * sc._Sc * S_side * sc._alpha_dict['bus']
    sc.Q_dict['Q_r'] = sc._IR_Earth * S_side * sc._epsilon_dict['bus'] * (sc._R_Earth / (sc.h_orb + sc._R_Earth)) ** 2
    #  Internal - due to devices on board
    P_bus = sc.P_tot - sc.P_dict['ref'] - sc._P_prop
    sc.Q_dict['Q_i'] = P_bus + sc._P_prop * (1 - 0.9) + sc.P_dict['ref'] * (1 - sc._eta_refuel)
    
    # Heat Output
    sc.Q_dict['Q_o'] = sc._kB * sc._T_eq ** 4 * (S_rad_solar * sc._epsilon_dict['pow'] + S_rad_bus * sc._epsilon_dict['bus'])

    # Final Area - assuming radiators don't absorb anything and back of solar panels are radiators
    sc.S_dict['therm'] = max(((sc.Q_dict['Q_d'] + sc.Q_dict['Q_s'] + sc.Q_dict['Q_a'] + sc.Q_dict['Q_r'] + sc.Q_dict['Q_i'] - sc.Q_dict['Q_o'])
                              / (sc._kB * sc._epsilon_dict['therm'] * sc._T_eq ** 4) - sc.S_dict['solar_extended']) / 2, 0)
