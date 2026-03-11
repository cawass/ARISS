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


def thermal_model(sc: SpacecraftState):
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
    # Projected area of the part of the spacecraft exposed to the Sun
    A_sun_sc = 0.5 * (W_in + W_body) * sc.geometry.L_in + W_body * sc.geometry.L_body
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
    Q_drag = 0.5 * sc.orbit.density * sc.orbit.velocity**3 * sc.geometry.A_in
    #  Sun heating - assuming sun hits at 90 degrees and solar panels are producing
    # theoretically there can be 2 cases:
    # external solar panels: external area + projected body area + projected intake area <--- ASSUMED
    # no external solar: (projected body + projected intake)_solar*solar+(projected body + projected intake)_body*body 
    Q_sun = const.SOLAR_CONSTANT * (A_sun_sc + sc.geometry.A_solar) * (sc.thermal.alpha_solar * (1 - sc.solar.eta_solar))
    #  Earth albedo heating - assuming side of the spacecraft is hit at 90 degrees
    Q_albedo = const.SOLAR_CONSTANT * const.EARTH_ALBEDO * A_earth * sc.thermal.alpha_body
    #  Earth infrared heating - assuming side of the spacecraft is hit at 90 degrees
    Q_ir = const.EARTH_IR_EMISSION * np.square((const.EARTH_RADIUS / (const.EARTH_RADIUS + sc.orbit.altitude))) * A_earth * sc.thermal.epsilon_therm_body
    #  Internal heating - due to devices on board
    Q_internal = sc.power.Power_total - sc.power.Power_prop * sc.thruster.thruster_thermal_eff - sc.power.Power_refprop * sc.refueling.eta_refuel
    
    # Heat output at desired temperature excluding potential radiators
    Q_radiated = Ae_total * sc.thermal.T_des**4 * const.STEFAN_BOLTZMANN

    Q_in_total = Q_drag + Q_sun + Q_albedo + Q_ir + Q_internal
    # Final Area - assuming radiators don't absorb anything and back of solar panels are radiators
    sc.geometry.A_rad = max(((Q_in_total - Q_radiated)/ (const.STEFAN_BOLTZMANN * sc.thermal.T_des**4 * sc.thermal.epsilon_therm_rad) - sc.geometry.A_solar)/2, 0.0)
    
    diagnostics = ThermalDiagnostics(
        Q_drag=Q_drag,
        Q_sun=Q_sun,
        Q_albedo=Q_albedo,
        Q_ir=Q_ir,
        Q_internal=Q_internal,
        Q_radiated=Q_radiated,
    )
    return diagnostics

def thermal_extremes(sc: SpacecraftState, diagnostics: ThermalDiagnostics, Ae_total: float, convergence_threshold: float = 0.1, max_iterations: int = 10000):
    """
    Calculates the minimum and maximum temperature for a given spacecraft.
    
    Parameters
    ----------
    sc : SpacecraftState
        The spacecraft state object
    diagnostics : ThermalDiagnostics
        Thermal diagnostics from thermal_model function
    Ae_total : float
        Total effective emissivity area [m^2]
    convergence_threshold : float, optional
        Temperature convergence criterion [K], default 0.1
    max_iterations : int, optional
        Maximum iterations for thermal simulation, default 10000
        
    Returns
    -------
    T_max : float
        Maximum temperature [K]
    T_min : float
        Minimum temperature [K]
    """
    Q_in_total = (diagnostics.Q_drag + diagnostics.Q_sun + diagnostics.Q_albedo + 
                  diagnostics.Q_ir + diagnostics.Q_internal)
    
    # Calculate equilibrium temperature
    if sc.geometry.A_rad == 0.0:
        # No external radiators needed, calculate actual equilibrium temperature
        T_eq = (Q_in_total / (Ae_total * const.STEFAN_BOLTZMANN)) ** (1/4) if Q_in_total > 0 else 0.0
        T_max = T_eq
    else:
        # Radiators are used, temperature is maintained at design temperature
        T_max = sc.thermal.T_des

    # Calculate cold-case temperature (eclipse conditions)
    if sc.orbit.max_eclipse_fraction > 0.0:
        time = 0
        dt = 0.1
        # Calculate orbital period and eclipse duration
        orbital_radius = const.EARTH_RADIUS + sc.orbit.altitude  # [m]
        orbital_period = 2 * np.pi * orbital_radius / sc.orbit.velocity  # [s] Orbital period
        # Calculate heat capacity
        heat_capacity = sc.thermal.specific_heat * sc.mass.Mass_total
        # Set starting temperature to hot case
        T_history = [T_max]

        # Convergence tracking for minimum temperature per orbit
        min_temps_per_orbit = []

        for iter in range(max_iterations):
            if (time % orbital_period) <= (sc.orbit.max_eclipse_fraction * orbital_period):
                Q_in = Q_in_total - diagnostics.Q_sun - diagnostics.Q_albedo
            else:
                Q_in = Q_in_total
            
            Q_radiated_total = (Ae_total + sc.geometry.A_rad * 2 * sc.thermal.epsilon_therm_rad) * T_history[-1]**4 * const.STEFAN_BOLTZMANN

            T_history.append(T_history[-1] + (Q_in - Q_radiated_total) / heat_capacity * dt)

            time += dt
            
            # Record minimum temperature when completing each orbit
            orbits_completed = int(time / orbital_period)
            if len(min_temps_per_orbit) < orbits_completed:
                # Find minimum temperature in the latest orbit
                start_idx = len(min_temps_per_orbit) * int(orbital_period / dt)
                orbit_min_temp = min(T_history[start_idx:])
                min_temps_per_orbit.append(orbit_min_temp)
                
                # Check convergence: compare last two orbits' minimum temperatures
                if len(min_temps_per_orbit) >= 3:
                    temp_change = abs(min_temps_per_orbit[-1] - min_temps_per_orbit[-2])
                    if temp_change < convergence_threshold:
                        T_min = min_temps_per_orbit[-1]
                        return T_max, T_min
        
        # If loop completes without convergence, use the last minimum
        T_min = min_temps_per_orbit[-1] if min_temps_per_orbit else T_max

    else:
        T_min = T_max
    
    return T_max, T_min
    
    
