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
#      This is a simple, 1-node thermal model which assumes a homogenous temperature across the spacecraft body.
#
#  Project:        ARISS
#  Module:         Thermal.py
#  Author:         Jan
# ============================================================================


import numpy as np
from dataclasses import dataclass

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Power import _is_round, _section_dims
from ariss.utils import constants as const

@dataclass(frozen=True)
class ThermalDiagnostics:
    """Detailed thermal outputs for post-processing.

    Attributes
    ----------
    Ae_total: float
        Total area emissivity product excluding radiators [m2]
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
    """
    Ae_total: float = 0.0

    Q_drag: float = 0.0
    Q_sun: float = 0.0
    Q_albedo: float = 0.0
    Q_ir: float = 0.0
    Q_internal: float = 0.0
    Q_radiated: float = 0.0


def thermal_model(sc: SpacecraftState):
    """
    Thermal model for the spacecraft using a 1-node homogenous temperature assumption.

    This function calculates the heat balance for a spacecraft by considering all heat
    sources (solar radiation, Earth albedo, Earth infrared, internal dissipation, and
    drag heating) and heat sinks (thermal radiation). It computes the required radiator
    area to maintain the desired design temperature and provides detailed thermal
    diagnostics for post-processing and analysis.

    Parameters
    ----------
    sc : SpacecraftState
        The spacecraft state object containing orbital, geometric, thermal, power,
        and propulsion parameters.

    Returns
    -------
    ThermalDiagnostics
        A dataclass containing detailed thermal outputs

    Updates
    -----
    sc : SpacecraftState
        Updates the sc.geometry.A_rad with the required radiator area [m²]
    """

    # Geometry Calculations for universal all geometry types
    W_in, H_in = _section_dims(sc.geometry.A_in, sc.geometry.AR_in, sc.geometry.S_in)
    W_body, H_body = _section_dims(sc.geometry.A_body, sc.geometry.AR_body, sc.geometry.S_body)
    # Projected area of the part of the spacecraft exposed to the Sun
    A_sun_total = 0.5 * (W_in + W_body) * sc.geometry.L_in + W_body * sc.geometry.L_body + sc.geometry.A_solar
    # Projected area of the part exposed to the Earth
    A_in_earth = 0.5 * (H_in + H_body) * sc.geometry.L_in
    A_body_earth = H_body * sc.geometry.L_body

    # Specific geometry types calculation
    # Intake Surface caclulations
    if sc.geometry.S_in == "s":
        # From the side (assumed Earth direction) there is just body surface
        # Effective projected absorptivity area of side of intake
        Aa_in_earth = A_in_earth * sc.thermal.alpha_body 
        # Effective projected emissivity area of side of intake
        Ae_in_earth = A_in_earth * sc.thermal.epsilon_therm_body

        # Surface areas are different for other geometries unlike projected areas
        # Surface area of top of intake
        A_in_top = 0.5 * (W_in + W_body) * np.sqrt(np.square(H_in - H_body)/4 + np.square(sc.geometry.L_in))
        # Surface area of side of intake
        A_in_side = 0.5 * (H_in + H_body) * np.sqrt(np.square(W_in - W_body)/4 + np.square(sc.geometry.L_in))
        # Intake effective emissivity area 
        Ae_in = A_in_top * sc.thermal.epsilon_therm_solar + sc.geometry.A_in * sc.thermal.epsilon_therm_in + (A_in_top + 2 * A_in_side) * sc.thermal.epsilon_therm_body
    elif sc.geometry.S_in == "c":
        # From the side (assumed Earth direction) there half body surface half solar panel
        # Effective projected absorptivity area of side of intake
        Aa_in_earth = A_in_earth * (sc.thermal.alpha_body + sc.thermal.alpha_solar*(1 - sc.solar.eta_solar))/2
        # Effective projected emissivity area of side of intake
        Ae_in_earth = A_in_earth * (sc.thermal.epsilon_therm_body + sc.thermal.epsilon_therm_solar)/2

        # Surface areas are different for other geometries unlike projected areas
        # Surface area of intake
        if W_in > H_in:
            # if wide, the base of the intake will have the diamter of the height
            A_in = np.pi * (W_in + W_body)/2 * np.sqrt(np.square(H_in - H_body)/4 + np.square(sc.geometry.L_in)) 
        else:
            # if tall, the base of the intake will have the diamter of the width
            A_in = np.pi * (W_in + W_body)/2 * np.sqrt(np.square(W_in - W_body)/4 + np.square(sc.geometry.L_in)) 
        # Intake effective emissivity area 
        Ae_in = A_in/2 * sc.thermal.epsilon_therm_solar + sc.geometry.A_in * sc.thermal.epsilon_therm_in + A_in/2 * sc.thermal.epsilon_therm_body
    elif sc.geometry.S_in == "e":
        # From the side (assumed Earth direction) there half body surface half solar panel
        # Effective absorptivity area of side of intake
        Aa_in_earth = A_in_earth * (sc.thermal.alpha_body + sc.thermal.alpha_solar*(1 - sc.solar.eta_solar))/2
        # Effective absorptivity area of side of intake
        Ae_in_earth = A_in_earth * (sc.thermal.epsilon_therm_body + sc.thermal.epsilon_therm_solar)/2

        # Surface areas are different for other geometries unlike projected areas
        # Approximation for circumference of ellipses
        P_in = np.pi * (3*(W_in/2 + H_in/2) - np.sqrt((3*W_in/2 + H_in/2)*(W_in/2 + 3*H_in/2)))
        P_body = np.pi * (3*(W_body/2 + H_body/2) - np.sqrt((3*W_body/2 + H_body/2)*(W_body/2 + 3*H_body/2)))
        # Approximation for length
        if sc.geometry.AR_body < sc.geometry.AR_in:
            # If body is taller than intake
            # Surface area of top of intake
            A_in_top = (P_in + P_body) * np.sqrt(np.square(H_in - W_body/sc.geometry.AR_in)/4 + np.square(sc.geometry.L_in)) / 4
            # Surface area of side of intake
            A_in_side = (P_in + P_body) * np.sqrt(np.square(W_in - W_body)/4 + np.square(sc.geometry.L_in)) / 4 
        else:
            # If body is wider than intake
            # Surface area of top of intake
            A_in_top = (P_in + P_body) * np.sqrt(np.square(H_in - H_body)/4 + np.square(sc.geometry.L_in)) / 4
            # Surface area of side of intake
            A_in_side = (P_in + P_body) * np.sqrt(np.square(W_in - H_body*sc.geometry.AR_in)/4 + np.square(sc.geometry.L_in)) / 4
        # Intake effective emissivity area 
        Ae_in = A_in_top * sc.thermal.epsilon_therm_solar + sc.geometry.A_in * sc.thermal.epsilon_therm_in + A_in_side * sc.thermal.epsilon_therm_body
    else:
        raise ValueError(f"Invalid intake shape S_in: {sc.geometry.S_in}") 
    

    # Body surface calculations
    if sc.geometry.S_body == "s":
        # From the side (assumed Earth direction) there is just body surface
        # Effective projected absorptivity area of side of body
        Aa_body_earth = A_body_earth * sc.thermal.alpha_body 
        # Effective projected emissivity area of side of body
        Ae_body_earth = A_body_earth * sc.thermal.epsilon_therm_body

        # Surface areas are different for other geometries unlike projected areas
        # Surface area of top of body
        A_body_top = W_body * sc.geometry.L_body
        # Surface area of side of body
        A_body_side = H_body * sc.geometry.L_body
        # body effective emissivity area 
        Ae_body = A_body_top * sc.thermal.epsilon_therm_solar + sc.geometry.A_body * sc.thermal.epsilon_therm_body + (A_body_top + 2 * A_body_side) * sc.thermal.epsilon_therm_body
    elif sc.geometry.S_body == "c":
        # From the side (assumed Earth direction) there half body surface half solar panel
        # Effective projected absorptivity area of side of body
        Aa_body_earth = A_body_earth * (sc.thermal.alpha_body + sc.thermal.alpha_solar*(1 - sc.solar.eta_solar))/2
        # Effective projected emissivity area of side of body
        Ae_body_earth = A_body_earth * (sc.thermal.epsilon_therm_body + sc.thermal.epsilon_therm_solar)/2

        # Surface areas are different for other geometries unlike projected areas
        # Surface area of top of body
        A_body_top = np.pi * W_body * sc.geometry.L_body
        # Surface area of side of body
        A_body_side = np.pi * H_body * sc.geometry.L_body
        # body effective emissivity area 
        Ae_body = A_body_top * sc.thermal.epsilon_therm_solar + sc.geometry.A_body * sc.thermal.epsilon_therm_body + A_body_side * sc.thermal.epsilon_therm_body
    elif sc.geometry.S_body == "e":
        # From the side (assumed Earth direction) there half body surface half solar panel
        # Effective absorptivity area of side of body
        Aa_body_earth = A_body_earth * (sc.thermal.alpha_body + sc.thermal.alpha_solar*(1 - sc.solar.eta_solar))/2
        # Effective absorptivity area of side of body
        Ae_body_earth = A_body_earth * (sc.thermal.epsilon_therm_body + sc.thermal.epsilon_therm_solar)/2

        # Surface areas are different for other geometries unlike projected areas
        # Approximation for circumference of ellipses
        P_body = np.pi * (3*(W_body/2 + H_body/2) - np.sqrt((3*W_body/2 + H_body/2)*(W_body/2 + 3*H_body/2)))
        # Surface area of top of body
        A_body_top = P_body * sc.geometry.L_body
        # body effective emissivity area 
        Ae_body = A_body_top * sc.thermal.epsilon_therm_solar + sc.geometry.A_body * sc.thermal.epsilon_therm_body + A_body_top * sc.thermal.epsilon_therm_body
    else:
        raise ValueError(f"Invalid body shape S_body: {sc.geometry.S_body}") 
        
    # Total effective emissivity area
    Ae_total = Ae_body + Ae_in + sc.geometry.A_solar * sc.thermal.epsilon_therm_solar

    # Heat input
    #  Drag heating - it's assumed the incoming air transfers all its kinetic energy into heat and this is all the heating from drag
    Q_drag = 0.5 * sc.orbit.density * sc.orbit.velocity**3 * sc.geometry.A_in
    #  Sun heating - assuming sun hits at 90 degrees and solar panels are producing
    # theoretically there can be 2 cases:
    # external solar panels: external area + projected body area + projected intake area <--- ASSUMED
    # no external solar: (projected body + projected intake)_solar*solar+(projected body + projected intake)_body*body 
    Q_sun = const.SOLAR_CONSTANT * (A_sun_total) * (sc.thermal.alpha_solar * (1 - sc.solar.eta_solar))
    #  Earth albedo heating - assuming side of the spacecraft is hit at 90 degrees
    Q_albedo = const.SOLAR_CONSTANT * const.EARTH_ALBEDO * (Aa_in_earth + Aa_body_earth)
    #  Earth infrared heating - assuming side of the spacecraft is hit at 90 degrees
    Q_ir = const.EARTH_IR_EMISSION * np.square((const.EARTH_RADIUS / (const.EARTH_RADIUS + sc.orbit.altitude))) * (Ae_in_earth + Ae_body_earth)
    #  Internal heating - due to devices on board
    Q_internal = sc.power.Power_total - sc.power.Power_prop * sc.thruster.thermal_eff - sc.power.Power_refprop * sc.refueling.eta_refuel
    
    # Heat output at desired temperature excluding potential radiators
    Q_radiated = Ae_total * sc.thermal.T_des**4 * const.STEFAN_BOLTZMANN

    Q_in_total = Q_drag + Q_sun + Q_albedo + Q_ir + Q_internal
    # Final Area - assuming radiators don't absorb anything and back of solar panels are radiators
    sc.geometry.A_rad = max(((Q_in_total - Q_radiated)/ (const.STEFAN_BOLTZMANN * sc.thermal.T_des**4 * sc.thermal.epsilon_therm_rad) - sc.geometry.A_solar)/2, 0.0)
    
    diagnostics = ThermalDiagnostics(
        Ae_total=Ae_total,
        Q_drag=Q_drag,
        Q_sun=Q_sun,
        Q_albedo=Q_albedo,
        Q_ir=Q_ir,
        Q_internal=Q_internal,
        Q_radiated=Q_radiated,
    )
    return diagnostics  
