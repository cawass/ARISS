from dataclasses import dataclass
from ariss.utils import constants as const
from ariss.utils.atmosphere import orbit_updates_from_density
import numpy as np

@dataclass(frozen=True)
class PropulsionDiagnostics:
    rho: float
    s_ref: float
    v_inf: float
    isp: float
    g0: float
    exhaust_velocity: float
    mass_flow_rate: float
    required_thrust: float
    required_prop_area: float


def propulsion_model(sc):

    exhaust_velocity = const.EARTH_GRAVITY * sc.thruster.specific_impulse
    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    h_body = np.sqrt(sc.geometry.A_body / sc.geometry.AR_body)
    w_body = sc.geometry.A_body / h_body

    body_side_area = (2.0 * w_body + 2.0 * h_body) * sc.geometry.L_body
    inlet_side_area = 0.5 * (2.0 * w_in + 2.0 * h_in + 2.0 * w_body + 2.0 * h_body) * sc.geometry.L_in

    cd_s_solar = sc.drag.cd_solar * sc.geometry.A_solar
    cd_s_rad = sc.drag.cd_rad * sc.geometry.A_rad
    cd_s_body = sc.drag.cd_body_side * body_side_area
    cd_s_inlet_side = sc.drag.cd_inlet_side * inlet_side_area
    cd_s_inlet_front = sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    cd_s_total = cd_s_solar + cd_s_rad + cd_s_body + cd_s_inlet_side + cd_s_inlet_front

    sc.geometry.A_prop = (0.5 * sc.orbit.velocity * cd_s_total  + sc.geometry.A_ref) / (exhaust_velocity - sc.orbit.velocity)
    sc.orbit.density = (2.0 * sc.thruster.power_required) / (sc.orbit.velocity * sc.geometry.A_prop * (exhaust_velocity ** 2))
    
    sc.orbit.altitude = orbit_updates_from_density(sc.orbit.density)["altitude"]
    sc.orbit.temperature = orbit_updates_from_density(sc.orbit.density)["temperature"]
    sc.orbit.molar_mass = orbit_updates_from_density(sc.orbit.density)["molar_mass"]
    sc.orbit.velocity = orbit_updates_from_density(sc.orbit.density)["velocity"]
    sc.thruster.m_flow = sc.geometry.A_prop*sc.orbit.velocity*sc.orbit.density

    if sc.mission_profile.active_refueling:
        sc.mission_profile.required_fuel = sc.mass.Mass_total*(np.exp((sc.mission_profile.delta_v)/(exhaust_velocity))-1)
        sc.refueling.m_flow = sc.mission_profile.required_fuel/sc.mission_profile.refueling_time
        sc.geometry.A_ref = sc.refueling.m_flow/(sc.orbit.density*sc.orbit.velocity)
    else:
        sc.geometry.A_ref = 0.0

    
    
    sc.geometry.A_in_drag = (sc.geometry.A_ref + sc.geometry.A_prop)*(1/sc.refueling.coll_eff - 1)
    sc.geometry.A_in = sc.geometry.A_prop + sc.geometry.A_ref + sc.geometry.A_in_drag

    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    inlet_side_area = 0.5 * (2.0 * w_in + 2.0 * h_in + 2.0 * w_body + 2.0 * h_body) * sc.geometry.L_in
    q = 0.5 * sc.orbit.density * sc.orbit.velocity**2

    sc.drag.drag_solar = q * sc.drag.cd_solar * sc.geometry.A_solar
    sc.drag.drag_rad = q * sc.drag.cd_rad * sc.geometry.A_rad
    sc.drag.drag_body_side = q * sc.drag.cd_body_side * body_side_area
    sc.drag.drag_inlet_side = q * sc.drag.cd_inlet_side * inlet_side_area
    sc.drag.drag_inlet_front = q * sc.drag.cd_inlet_front * sc.geometry.A_in_drag
    sc.drag.drag_total = (sc.drag.drag_solar + sc.drag.drag_rad + sc.drag.drag_body_side + sc.drag.drag_inlet_side+ sc.drag.drag_inlet_front)

   
    sc.thruster.thrust =  exhaust_velocity * sc.orbit.density * sc.orbit.velocity * sc.geometry.A_prop
    sc.thruster.propellant_mass =  sc.orbit.density * sc.orbit.velocity * sc.geometry.A_prop




    print(f"Drag Force: {sc.drag.drag_total:.6e} N")
    print(f"Required Propulsion Area: {sc.geometry.A_prop:.6f} m^2")
    print(f"Required Refueling Area: {sc.geometry.A_ref:.6f} m^2")
    print(f"Required Total Area: {sc.geometry.A_in:.6f} m^2")
    print(f"Orbit Updates - Altitude: {sc.orbit.altitude:.6f} km")
