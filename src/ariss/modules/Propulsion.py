from dataclasses import dataclass
from ariss.utils import constants as const
from ariss.utils.atmosphere import orbit_updates_from_density


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
    required_prop_area = (
        sc.drag.drag_total +  sc.geometry.A_ref * (sc.orbit.velocity ** 2)
    ) / (sc.orbit.velocity * (exhaust_velocity - sc.orbit.velocity))

    inferred_density = (
        2.0 * sc.thruster.power_required) / (sc.orbit.velocity * required_prop_area * (exhaust_velocity ** 2))
    orbit_updates = orbit_updates_from_density(inferred_density)
    print(f"Orbit Updates - Altitude: {orbit_updates['altitude']:.6f} km")
    sc.orbit.altitude = orbit_updates["altitude"]
    sc.orbit.density = inferred_density
    sc.orbit.temperature = orbit_updates["temperature"]
    sc.orbit.molar_mass = orbit_updates["molar_mass"]
    sc.orbit.velocity = orbit_updates["velocity"]

    mass_flow_rate = sc.orbit.density * sc.orbit.velocity * required_prop_area
    required_thrust = mass_flow_rate * exhaust_velocity

    sc.geometry.A_prop = required_prop_area
    print(f"Drag Force: {sc.drag.drag_total:.6e} N")
    print(f"Required Propulsion Area: {sc.geometry.A_prop:.6f} m^2")
    sc.geometry.A_in = required_prop_area + sc.geometry.A_ref
    sc.thruster.propellant_mass = mass_flow_rate
    sc.thruster.thrust = required_thrust

    
    diagnostics = PropulsionDiagnostics(
            rho=sc.orbit.density,
            s_ref=sc.geometry.A_ref,
            v_inf=sc.orbit.velocity,
            isp=sc.thruster.specific_impulse,
            g0=const.EARTH_GRAVITY,
            exhaust_velocity=exhaust_velocity,
            mass_flow_rate=mass_flow_rate,
            required_thrust=required_thrust,
            required_prop_area=required_prop_area
        )
    
    return diagnostics
