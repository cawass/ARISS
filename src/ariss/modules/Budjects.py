"""Budget sizing model for subsystem mass and power."""

from enum import IntEnum

from ariss.core.spacecraft import SpacecraftState


class BudgetIdx(IntEnum):
    """Indices for the flat array returned by ``budjet_model``."""

    MASS_IN = 0
    MASS_BODY = 1
    MASS_SOLAR = 2
    MASS_RAD = 3
    MASS_PROP = 4
    MASS_ADCS = 5
    MASS_PAYLOAD = 6
    MASS_REFPROP = 7
    MASS_TOTAL = 8
    POWER_IN = 9
    POWER_BODY = 10
    POWER_SOLAR = 11
    POWER_RAD = 12
    POWER_PROP = 13
    POWER_ADCS = 14
    POWER_PAYLOAD = 15
    POWER_REFPROP = 16
    POWER_TOTAL = 17
    REFUEL_MASS_FLOW = 18


def budjet_model(sc: SpacecraftState) -> list[float]:
    """Return mass/power budgets as a flat array indexed by ``BudgetIdx``."""
    ref_mass_flow = sc.geometry.A_ref * sc.orbit.velocity * sc.orbit.density
    prop_mass_flow = sc.geometry.A_prop * sc.orbit.velocity * sc.orbit.density

    mass_in = (sc.geometry.A_in + sc.geometry.A_body) * sc.geometry.L_in * sc.budget.B_mass_volume_in / 2.0
    mass_body = sc.geometry.A_body * sc.geometry.L_body * sc.budget.B_mass_volume_body
    mass_solar = sc.geometry.A_solar * sc.budget.B_mass_surface_solar
    mass_rad = sc.geometry.A_rad * sc.budget.B_mass_surface_rad
    mass_prop_power = sc.thruster.power_required * sc.budget.B_mass_power_prop
    mass_prop_flow = sc.budget.B_mass_mflow_prop * prop_mass_flow
    mass_prop = mass_prop_power + mass_prop_flow
    mass_payload = sc.budget.B_mass_payload
    mass_ref = sc.budget.B_mass_mflow_ref * ref_mass_flow

    base_mass = mass_in + mass_body + mass_solar + mass_rad + mass_prop + mass_payload + mass_ref
    mass_total = base_mass
    for _ in range(10):
        mass_adcs = sc.budget.B_mass_mass_ADCS * mass_total
        mass_total = base_mass + mass_adcs

    power_in = 0.0
    power_body = 0.0
    power_rad = 0.0
    power_prop_drive = (sc.budget.B_power_power_prop + 1.0) * sc.thruster.power_required
    power_prop_flow = sc.budget.B_power_mflow_prop * prop_mass_flow
    power_prop = power_prop_drive + power_prop_flow
    power_adcs = sc.budget.B_power_mass_ADCS * mass_total
    power_payload = sc.budget.B_power_payload
    power_ref = sc.budget.B_power_mflow_ref * ref_mass_flow

    base_power = power_in + power_body + power_rad + power_prop + power_adcs + power_payload + power_ref
    power_total = base_power
    for _ in range(10):
        power_solar = sc.budget.B_power_power_solar * power_total
        power_total = base_power + power_solar

    return [
        mass_in,
        mass_body,
        mass_solar,
        mass_rad,
        mass_prop,
        mass_adcs,
        mass_payload,
        mass_ref,
        mass_total,
        power_in,
        power_body,
        power_solar,
        power_rad,
        power_prop,
        power_adcs,
        power_payload,
        power_ref,
        power_total,
        ref_mass_flow,
    ]
