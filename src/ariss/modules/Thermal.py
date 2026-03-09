"""
Thermal model for ARISS

This is a simple, 1-node thermal model which assumes a homogenous temperature across the spacecraft body.
"""
import numpy as np
from dataclasses import dataclass

from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const

@dataclass(frozen=True)
class ThermalDiagnostics:
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
        Heat radiated by spacecraft excluding radiators [W]
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


def thermal_model(sc: SpacecraftState) -> float:
    """
    Thermal model for the spacecraft
    TODO Elaborate docstring
    """

    # Geometry Calculations
    H_in = np.sqrt(sc.geometry.A_in /  sc.geometry.AR_in) # Intake height
    W_in =  sc.geometry.A_in / H_in # Intake width
    H_body = np.sqrt(sc.geometry.A_body /  sc.geometry.AR_body) # Body height
    W_body =  sc.geometry.A_body / H_body # Body width
    # Area of top of body
    A_body_top = W_body * sc.geometry.L_body
    # Area of side of body 
    A_body_side = H_body * sc.geometry.L_body
    # Projected area of the part of the body exposed to the Sun
    A_sun_body = 0.5 * (W_in + W_body) * sc.geometry.L_in + W_body * sc.geometry.L_body
    # Projected area of the part exposed to the Earth
    A_earth = 0.5 * (H_in + H_body) * sc.geometry.L_in  + H_body * sc.geometry.L_body
    # Area of top of intake
    A_in_top = 0.5 * (W_in + W_body) * np.sqrt(np.square(H_in - H_body) + np.square(sc.geometry.L_in))
    # Area of side of intake
    A_in_side = 0.5 * (H_in + H_body) * np.sqrt(np.square(W_in - W_body) + np.square(sc.geometry.L_in))
    # Total effective emissivity area
    Ae_total = (A_in_top + A_body_top + sc.geometry.A_solar) * sc.thermal.epsilon_therm_solar + sc.geometry.A_in * sc.thermal.epsilon_therm_in + (A_in_top + 2 * A_in_side + A_body_top + 2 * A_body_side + sc.geometry.A_body) * sc.thermal.epsilon_therm_body


    # Heat input
    #  Drag heating - it's assumed the incoming air transfers all its kinetic energy into heat and this is all the heating from drag
    Q_drag = 0.5 * sc.orbit.density * np.power(sc.orbit.velocity, 3) * sc.geometry.A_in
    #  Sun heating - assuming sun hits at 90 degrees and solar panels are producing
    # theoretically there can be 2 cases:
    # external solar panels: external area + projected body area + projected intake area <--- ASSUMED
    # no external solar: (projected body + projected intake)_solar*solar+(projected body + projected intake)_body*body 
    Q_sun = const.SOLAR_CONSTANT * (A_sun_body + sc.geometry.A_solar) * (sc.thermal.alpha_solar * (1 - sc.solar.eta_solar))
    #  Earth albedo heating - assuming side of the spacecraft is hit at 90 degrees
    Q_albedo = const.SOLAR_CONSTANT * const.EARTH_ALBEDO * A_earth * sc.thermal.alpha_body
    #  Earth infrared heating - assuming side of the spacecraft is hit at 90 degrees
    Q_ir = const.EARTH_IR_EMISSION * np.square((const.EARTH_RADIUS / (const.EARTH_RADIUS + sc.orbit.altitude))) * A_earth * sc.thermal.epsilon_therm_body
    #  Internal heating - due to devices on board
    Q_internal = sc.power.Power_total - sc.power.Power_prop * sc.thruster.thruster_eff - sc.power.Power_refprop * 0.5 # O.5 PLACEHOLDER UNTIL REFUELLINGSTATE IMPLEMENTED
    
    # Heat output at desired temperature excluding potential radiators
    Q_radiated = Ae_total * np.power(sc.thermal.T_des, 4) * const.STEFAN_BOLTZMANN

    # Final Area - assuming radiators don't absorb anything and back of solar panels are radiators
    return max(((Q_drag + Q_sun + Q_albedo + Q_ir + Q_internal - Q_radiated)/ (const.STEFAN_BOLTZMANN * np.power(sc.thermal.T_des, 4) * sc.thermal.epsilon_therm_body) - sc.geometry.A_solar)/2, 0)

