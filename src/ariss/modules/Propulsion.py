"""Minimal propulsion intake-area and power model."""

from dataclasses import dataclass

from ariss.utils import constants as const


@dataclass(frozen=True)
class PropulsionDiagnostics:
    drag_force: float
    rho: float
    s_ref: float
    v_inf: float
    isp: float
    g0: float
    exhaust_velocity: float
    mass_flow_rate: float
    required_thrust: float
    numerator: float
    denominator: float
    required_prop_area: float
    required_prop_power: float


def propulsion_model(sc, drag_force: float, return_diagnostics: bool = False):
    """Size intake area and propulsion power.

    Returns
    -------
    tuple
        ``(required_prop_area, required_prop_power)``.
        Diagnostics are appended as the final item when ``return_diagnostics`` is ``True``.
    """
    exhaust_velocity = const.EARTH_GRAVITY * sc.thruster.specific_impulse

    numerator = drag_force + sc.orbit.density * sc.geometry.A_ref * sc.orbit.velocity
    denominator = sc.orbit.density * sc.orbit.velocity * (exhaust_velocity - sc.orbit.velocity)
    required_prop_area = numerator / denominator
    mass_flow_rate = sc.orbit.density * sc.orbit.velocity * required_prop_area
    required_thrust = mass_flow_rate * exhaust_velocity
    required_prop_power = 1/2 * mass_flow_rate * (exhaust_velocity ** 2)

    if return_diagnostics:
        diagnostics = PropulsionDiagnostics(
            drag_force=drag_force,
            rho=sc.orbit.density,
            s_ref=sc.geometry.A_ref,
            v_inf=sc.orbit.velocity,
            isp=sc.thruster.specific_impulse,
            g0=const.EARTH_GRAVITY,
            exhaust_velocity=exhaust_velocity,
            mass_flow_rate=mass_flow_rate,
            required_thrust=required_thrust,
            numerator=numerator,
            denominator=denominator,
            required_prop_area=required_prop_area,
            required_prop_power=required_prop_power,
        )
        return required_prop_area, required_prop_power, diagnostics

    return required_prop_area, required_prop_power, required_thrust
