"""Delta-v post-processing using the rocket equation."""

import math

from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const


def delta_v_model(sc: SpacecraftState, last_mass_budgets: list[float]) -> dict[str, float]:
    """Estimate propellant and timing from ``delta_v = Isp * g0 * ln(m0 / mf)``."""
    eps = 1.0e-12

    target_delta_v = max(sc.mission_profile.delta_v, 0.0)
    exhaust_velocity = const.EARTH_GRAVITY * max(sc.thruster.specific_impulse, 0.0)
    m0 = max(last_mass_budgets[8] if len(last_mass_budgets) > 8 else sc.total_mass, 0.0)
    available_refueling_mass_flow = max(last_mass_budgets[18] if len(last_mass_budgets) > 18 else 0.0, 0.0)

    if target_delta_v <= 0.0 or exhaust_velocity <= 0.0 or m0 <= 0.0:
        return {
            "delta_v_orbital": 0.0,
            "delta_v_total": 0.0,
            "required_refueling_mass_flow": 0.0,
            "available_refueling_mass_flow": available_refueling_mass_flow,
            "required_propellant_mass": 0.0,
            "burn_time_required": 0.0,
            "refueling_time_required": 0.0,
        }

    required_propellant_mass = m0 * (1.0 - math.exp(-target_delta_v / exhaust_velocity))
    mf = max(m0 - required_propellant_mass, eps)
    achieved_delta_v = exhaust_velocity * math.log(m0 / mf)

    if sc.thruster.thrust > 0.0:
        required_refueling_mass_flow = sc.thruster.thrust / exhaust_velocity
        burn_time_required = required_propellant_mass / max(required_refueling_mass_flow, eps)
    else:
        required_refueling_mass_flow = 0.0
        # Without thrust, burn duration is not finite for a positive propellant request.
        burn_time_required = float("inf")

    if available_refueling_mass_flow > 0.0:
        refueling_time_required = required_propellant_mass / available_refueling_mass_flow
    else:
        # No captured refuel flow means refueling cannot complete in finite time.
        refueling_time_required = float("inf")

    return {
        "delta_v_orbital": 0.0,
        "delta_v_total": achieved_delta_v,
        "required_refueling_mass_flow": required_refueling_mass_flow,
        "available_refueling_mass_flow": available_refueling_mass_flow,
        "required_propellant_mass": required_propellant_mass,
        "burn_time_required": burn_time_required,
        "refueling_time_required": refueling_time_required,
    }
