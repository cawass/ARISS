import os
import sys
import numpy as np

from iteration.SpacecraftClass import SpacecraftClass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
