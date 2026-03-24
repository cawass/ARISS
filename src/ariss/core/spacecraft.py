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
#      Spacecraft state definitions and serialization helpers for ARISS.
#
#  Project:        ARISS
#  Module:         spacecraft.py
#  Author:         Lucas Calderon del Rio, Carlos Carrasco Requejo, Jan
# ============================================================================

import tomllib
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any

try:
    import tomli_w
except ImportError:  # pragma: no cover - optional dependency for TOML writing only
    tomli_w = None


def _coerce_dataclass(dataclass_type: type[Any], payload: dict[str, Any]) -> Any:
    # Inputs:
    #   dataclass_type: destination dataclass type.
    #   payload: field mapping used to build the destination object.
    #
    # Output:
    #   Dataclass instance with nested dictionaries converted recursively.

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping for {dataclass_type.__name__}, got {type(payload).__name__}"
        )

    field_map = {field_info.name: field_info for field_info in fields(dataclass_type)}

    unknown_keys = sorted(set(payload) - set(field_map))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise KeyError(f"Unknown keys for {dataclass_type.__name__}: {joined}")

    coerced: dict[str, Any] = {}

    for key, value in payload.items():
        field_info = field_map[key]
        field_type = field_info.type

        if isinstance(value, dict) and isinstance(field_type, type) and is_dataclass(field_type):
            coerced[key] = _coerce_dataclass(field_type, value)
        else:
            coerced[key] = value

    return dataclass_type(**coerced)


def _load_toml_file(filepath: str | PathLike[str]) -> dict[str, Any]:
    # Inputs:
    #   filepath: path to TOML file.
    #
    # Output:
    #   Parsed TOML mapping. Accepts UTF-8 with or without BOM.

    raw = Path(filepath).read_bytes()
    return tomllib.loads(raw.decode("utf-8-sig"))

@dataclass(frozen=False)
class MissionProfileState:
    """Mission-level targets and refueling activation flags."""

    active_refueling: bool = False  # [bool] Enables the mission branch that collects atmospheric propellant.
    delta_v: float = 1157.8  # [m/s] Mission delta-v requirement used in the rocket equation.
    required_fuel: float = 0  # [kg] Derived: fuel mass required to satisfy the mission delta-v target.

    def update(self, **kwargs: Any) -> "MissionProfileState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class OrbitState:
    """Orbital environment inputs and atmosphere-derived state."""

    altitude: float = 0  # [km] Input/derived: orbital altitude above Earth.
    velocity: float = 0  # [m/s] Derived: orbital velocity at the current altitude.
    density: float = 0  # [kg/m^3] Derived: local atmospheric density.
    p_orb: float = 1e-5  # [Pa] Input: local orbital pressure used by refueling compression.
    temperature: float = 0  # [K] Derived: local atmospheric temperature.
    molar_mass: float = 0  # [kg/mol] Derived: local atmospheric molar mass.
    alpha: float = 0  # [rad] Input: angle of attack used by drag and thermal models.
    gamma: float = 1.4  # [-] Input: specific heat ratio of the captured gas.
    R_spec: float = 287.0  # [J/kg/K] Input/derived: specific gas constant for the local gas mix.
    msis_date: str = "2000-01-01T00:00:00"  # [UTC ISO-8601] Input: atmosphere model timestamp.
    msis_f107: float = 140.0  # [sfu] Input: solar radio flux index for the atmosphere model.
    msis_ap: float = 15.0  # [-] Input: geomagnetic activity index for the atmosphere model.
    latitude: float = 0.0  # [deg] Input: latitude used for point-sampled atmosphere queries.
    longitude: float = 0.0  # [deg] Input: longitude used for point-sampled atmosphere queries.
    use_average: bool = False  # [bool] Input: if true, use a latitude/longitude-averaged atmosphere.


@dataclass(frozen=False)
class GeometryState:
    """Spacecraft geometry inputs and solved capture-area allocation."""

    S_in: str = "c"  # [code] Input: intake cross-section shape code.
    S_body: str = "s"  # [code] Input: body cross-section shape code.
    use_intake_area_ratio: bool = False  # [bool] Input: enforces A_in = intake_area_ratio * A_body when enabled.
    fixed_body: bool = False  # [bool] Input: keeps the body area fixed while solving in ratio mode.
    intake_area_ratio: float = 1.0  # [-] Input: total intake-to-body frontal area ratio.

    AR_in: float = 1.0  # [-] Input: intake section aspect ratio.
    AR_body: float = 1.0  # [-] Input: body section aspect ratio.
    AR_solar: float = 5  # [-] Input: solar array planform aspect ratio.
    AR_rad: float = 5  # [-] Input: radiator planform aspect ratio.

    epsilon_in: float = 0.1  # [-] Input: inlet wall accommodation coefficient for side drag.
    epsilon_body: float = 0.1  # [-] Input: body wall accommodation coefficient for drag.
    epsilon_solar: float = 0.1  # [-] Input: solar array accommodation coefficient for drag.
    epsilon_rad: float = 0.1  # [-] Input: radiator accommodation coefficient for drag.
    epsilon_in_norm: float = 0.9  # [-] Input: inlet normal-face accommodation coefficient.

    wake_in: float = 1  # [-] Input: inlet wake factor applied to side drag.
    wake_body: float = 1  # [-] Input: body wake factor applied to side drag.
    wake_solar: float = 1  # [-] Input: solar array wake factor.
    wake_radiator: float = 1  # [-] Input: radiator wake factor.

    A_in: float = 4.0387  # [m^2] Input/derived: total intake frontal area.
    A_ref: float = 2  # [m^2] Derived: intake area assigned to refueling capture.
    A_prop: float = 2  # [m^2] Derived: intake area assigned to propulsion capture.
    A_in_drag: float = 2  # [m^2] Derived: intake area exposed only to drag.
    A_body: float = 0.5  # [m^2] Input/derived: spacecraft body frontal area.
    A_solar: float = 5  # [m^2] Input: total deployed solar array area.
    A_rad: float = 0.0  # [m^2] Input/derived: total radiator area.
    t_solar: float = 0.0  # [m] Input: solar panel thickness used for frontal-edge drag.
    t_rad: float = 0.0  # [m] Input: radiator panel thickness used for frontal-edge drag.

    L_in: float = 2.5  # [m] Input: intake length along the spacecraft axis.
    L_body: float = 2.5  # [m] Input: body length along the spacecraft axis.
    X_solar: float = 2.0  # [m] Input: solar array mounting center from the aft reference.
    X_rad: float = 0.5  # [m] Input: radiator mounting center from the aft reference.

    def update(self, **kwargs: Any) -> "GeometryState":
        return replace(self, **kwargs)

@dataclass(frozen=True)
class RateState:
    """Structural mass-rate coefficients used by sizing relations."""

    R_mass_volume_in: float = 10  # [kg/m^3] Input: inlet structural mass density rate.
    R_mass_volume_body: float = 10  # [kg/m^3] Input: body structural mass density rate.
    R_mass_surface_solar: float = 5  # [kg/m^2] Input: solar array areal mass rate.
    R_mass_surface_rad: float = 5  # [kg/m^2] Input: radiator areal mass rate.

    def update(self, **kwargs: Any) -> "RateState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class MassState:
    """Subsystem and total mass bookkeeping."""

    Mass_in: float = 0.0  # [kg] Derived: intake structural mass.
    Mass_body: float = 0.0  # [kg] Derived: body structural mass.
    Mass_solar: float = 0.0  # [kg] Derived: solar array mass.
    Mass_rad: float = 0.0  # [kg] Derived: radiator mass.
    Mass_prop: float = 61  # [kg] Input/derived: propulsion subsystem dry mass.
    Mass_ADCS: float = 20  # [kg] Input: ADCS subsystem mass.
    Mass_payload: float = 24  # [kg] Input: payload mass.
    Mass_refprop: float = 700  # [kg] Input/derived: refueling and propellant storage subsystem mass.
    Mass_total: float = 0.0  # [kg] Derived: total spacecraft mass.

    def update(self, **kwargs: Any) -> "MassState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class PowerState:
    """Subsystem and total power bookkeeping."""

    Power_in: float = 0.0  # [W] Derived: intake subsystem power.
    Power_body: float = 0.0  # [W] Derived: body subsystem power.
    Power_solar: float = 0.0  # [W] Derived: solar power generation or allocation.
    Power_rad: float = 0.0  # [W] Derived: radiator subsystem power.
    Power_prop: float = 0.0  # [W] Derived: propulsion subsystem power.
    Power_ADCS: float = 2000.0  # [W] Input: ADCS power demand.
    Power_payload: float = 0.0  # [W] Input: payload power demand.
    Power_refprop: float = 0.0  # [W] Derived: refueling and propellant processing power.
    Power_total: float = 0.0  # [W] Derived: total spacecraft power demand.

    def update(self, **kwargs: Any) -> "PowerState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class ThrusterState:
    """Thruster operating point, efficiency, and flow state."""

    thrust: float = 0.1039  # [N] Derived: thrust produced by the intake-fed thruster.
    specific_impulse: float = 4500  # [s] Input/derived: thruster specific impulse.
    eff: float = 0.5  # [-] Input: thruster efficiency.
    thermal_eff: float = 0.8 # [-] Input: thruster thermal efficiency
    power: float = 5000.0  # [W] Input: electrical power available to the thruster.
    propellant_mass: float = 0.0  # [kg/s] Derived: propellant throughput inferred from intake capture.
    m_flow: float = 1e-3  # [kg/s] Derived: thruster propellant mass flow rate.
    propulsive_ram_load: float = 0.0  # [N] Derived: momentum-exchange load from propulsion capture.
    refueling_ram_load: float = 0.0  # [N] Derived: momentum-exchange load from refueling capture.
    required_load: float = 0.0  # [N] Derived: total propulsion load including aero and ram terms.
    force_residual: float = 0.0  # [N] Derived: thrust minus total required propulsion load.

    def update(self, **kwargs: Any) -> "ThrusterState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class RefuelingState:
    """Refueling performance inputs and solved refill flow state."""

    coll_eff: float = 0.61  # [-] Input: intake collection efficiency for useful captured flow.
    t_refuel: float = 140 * 24 * 3600  # [s] Input: time allowed to complete the refueling target.
    eta_refuel: float = 0.1  # [-] Input: refueling compression efficiency.
    m_flow: float = 1e-3  # [kg/s] Derived: refueling mass flow rate.
    p_tank: float = 100000  # [Pa] Input: storage tank pressure.
    V_prop: float = 0.7  # [m^3] Input: propellant storage volume.

    def update(self, **kwargs: Any) -> "RefuelingState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SolarState:
    """Solar conversion and pointing efficiency inputs."""

    av_aligment: float = 60  # [deg] Input: average solar array alignment angle.
    eta_solar: float = 0.3  # [-] Input: solar conversion efficiency.
    eta_power: float = 0.9  # [-] Input: power conditioning efficiency.

    def update(self, **kwargs: Any) -> "SolarState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class DragState:
    """Drag coefficients and force outputs for each exposed surface."""

    cd_solar: float = 0.2  # [-] Derived: solar array drag coefficient.
    cd_solar_front: float = 0.2  # [-] Derived: solar array frontal-edge drag coefficient.
    cd_rad: float = 0.2  # [-] Derived: radiator drag coefficient.
    cd_rad_front: float = 0.2  # [-] Derived: radiator frontal-edge drag coefficient.
    cd_body_side: float = 0.2  # [-] Derived: body-side drag coefficient.
    cd_inlet_side: float = 0.2  # [-] Derived: inlet-side drag coefficient.
    cd_inlet_front: float = 0.2  # [-] Derived: inlet-front drag coefficient.

    drag_total: float = 1  # [N] Derived: total aerodynamic drag force.
    drag_solar: float = 0.2  # [N] Derived: solar array drag force.
    drag_solar_front: float = 0.2  # [N] Derived: solar array frontal-edge drag force.
    drag_rad: float = 0.2  # [N] Derived: radiator drag force.
    drag_rad_front: float = 0.2  # [N] Derived: radiator frontal-edge drag force.
    drag_body_side: float = 0.2  # [N] Derived: body-side drag force.
    drag_inlet_side: float = 0.2  # [N] Derived: inlet-side drag force.
    drag_inlet_front: float = 0.2  # [N] Derived: inlet-front drag force.


@dataclass(frozen=True)
class ThermalState:
    """Thermal design targets and surface optical properties."""

    T_des: float = 300.0  # [K] Input: design temperature target.
    alpha_body: float = 0.1  # [-] Input: body solar absorptivity.
    alpha_solar: float = 0.9  # [-] Input: solar array absorptivity.

    epsilon_therm_in: float = 0.5  # [-] Input: intake thermal emissivity.
    epsilon_therm_body: float = 0.9  # [-] Input: body thermal emissivity.
    epsilon_therm_solar: float = 0.85  # [-] Input: solar array thermal emissivity.
    epsilon_therm_rad: float = 0.9  # [-] Input: radiator thermal emissivity.

    def update(self, **kwargs: Any) -> "ThermalState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SpacecraftState:
    """Top-level container that groups every spacecraft subsystem state."""

    name: str = "ARISS Case"  # [text] Input: human-readable case name.

    orbit: OrbitState = field(default_factory=OrbitState)  # [state] Input/derived: orbital environment state bundle.
    geometry: GeometryState = field(default_factory=GeometryState)  # [state] Input/derived: geometry definition and solved areas.
    thruster: ThrusterState = field(default_factory=ThrusterState)  # [state] Input/derived: propulsion operating point bundle.
    rate: RateState = field(default_factory=RateState)  # [state] Input: structural mass rate bundle.
    mass: MassState = field(default_factory=MassState)  # [state] Derived: subsystem and total mass bundle.
    power: PowerState = field(default_factory=PowerState)  # [state] Derived: subsystem and total power bundle.
    solar: SolarState = field(default_factory=SolarState)  # [state] Input: solar performance parameter bundle.
    thermal: ThermalState = field(default_factory=ThermalState)  # [state] Input: thermal property bundle.
    drag: DragState = field(default_factory=DragState)  # [state] Derived: drag coefficient and force bundle.
    refueling: RefuelingState = field(default_factory=RefuelingState)  # [state] Input/derived: refueling performance bundle.
    mission_profile: MissionProfileState = field(default_factory=MissionProfileState)  # [state] Input/derived: mission target bundle.

    def check_bounds(self) -> None:
        """
        Validate all spacecraft variables against explicit bounds.

        Raises:
            ValueError: if any variable is out of bounds or has the wrong type.
        """

        values = {
            "name": self.name,

            "orbit.altitude": self.orbit.altitude,
            "orbit.velocity": self.orbit.velocity,
            "orbit.density": self.orbit.density,
            "orbit.p_orb": self.orbit.p_orb,
            "orbit.temperature": self.orbit.temperature,
            "orbit.molar_mass": self.orbit.molar_mass,
            "orbit.alpha": self.orbit.alpha,
            "orbit.gamma": self.orbit.gamma,
            "orbit.R_spec": self.orbit.R_spec,
            "orbit.msis_date": self.orbit.msis_date,
            "orbit.msis_f107": self.orbit.msis_f107,
            "orbit.msis_ap": self.orbit.msis_ap,
            "orbit.latitude": self.orbit.latitude,
            "orbit.longitude": self.orbit.longitude,
            "orbit.use_average": self.orbit.use_average,

            "geometry.S_in": self.geometry.S_in,
            "geometry.S_body": self.geometry.S_body,
            "geometry.use_intake_area_ratio": self.geometry.use_intake_area_ratio,
            "geometry.fixed_body": self.geometry.fixed_body,
            "geometry.intake_area_ratio": self.geometry.intake_area_ratio,
            "geometry.AR_in": self.geometry.AR_in,
            "geometry.AR_body": self.geometry.AR_body,
            "geometry.AR_solar": self.geometry.AR_solar,
            "geometry.AR_rad": self.geometry.AR_rad,
            "geometry.epsilon_in": self.geometry.epsilon_in,
            "geometry.epsilon_body": self.geometry.epsilon_body,
            "geometry.epsilon_solar": self.geometry.epsilon_solar,
            "geometry.epsilon_rad": self.geometry.epsilon_rad,
            "geometry.epsilon_in_norm": self.geometry.epsilon_in_norm,
            "geometry.wake_in": self.geometry.wake_in,
            "geometry.wake_body": self.geometry.wake_body,
            "geometry.wake_solar": self.geometry.wake_solar,
            "geometry.wake_radiator": self.geometry.wake_radiator,
            "geometry.A_in": self.geometry.A_in,
            "geometry.A_ref": self.geometry.A_ref,
            "geometry.A_prop": self.geometry.A_prop,
            "geometry.A_in_drag": self.geometry.A_in_drag,
            "geometry.A_body": self.geometry.A_body,
            "geometry.A_solar": self.geometry.A_solar,
            "geometry.A_rad": self.geometry.A_rad,
            "geometry.t_solar": self.geometry.t_solar,
            "geometry.t_rad": self.geometry.t_rad,
            "geometry.L_in": self.geometry.L_in,
            "geometry.L_body": self.geometry.L_body,
            "geometry.X_solar": self.geometry.X_solar,
            "geometry.X_rad": self.geometry.X_rad,

            "rate.R_mass_volume_in": self.rate.R_mass_volume_in,
            "rate.R_mass_volume_body": self.rate.R_mass_volume_body,
            "rate.R_mass_surface_solar": self.rate.R_mass_surface_solar,
            "rate.R_mass_surface_rad": self.rate.R_mass_surface_rad,

            "mass.Mass_in": self.mass.Mass_in,
            "mass.Mass_body": self.mass.Mass_body,
            "mass.Mass_solar": self.mass.Mass_solar,
            "mass.Mass_rad": self.mass.Mass_rad,
            "mass.Mass_prop": self.mass.Mass_prop,
            "mass.Mass_ADCS": self.mass.Mass_ADCS,
            "mass.Mass_payload": self.mass.Mass_payload,
            "mass.Mass_refprop": self.mass.Mass_refprop,
            "mass.Mass_total": self.mass.Mass_total,

            "power.Power_in": self.power.Power_in,
            "power.Power_body": self.power.Power_body,
            "power.Power_solar": self.power.Power_solar,
            "power.Power_rad": self.power.Power_rad,
            "power.Power_prop": self.power.Power_prop,
            "power.Power_ADCS": self.power.Power_ADCS,
            "power.Power_payload": self.power.Power_payload,
            "power.Power_refprop": self.power.Power_refprop,
            "power.Power_total": self.power.Power_total,

            "thruster.thrust": self.thruster.thrust,
            "thruster.specific_impulse": self.thruster.specific_impulse,
            "thruster.eff": self.thruster.eff,
            "thruster.thermal_eff": self.thruster.thermal_eff,
            "thruster.power": self.thruster.power,
            "thruster.propellant_mass": self.thruster.propellant_mass,
            "thruster.m_flow": self.thruster.m_flow,
            "thruster.propulsive_ram_load": self.thruster.propulsive_ram_load,
            "thruster.refueling_ram_load": self.thruster.refueling_ram_load,
            "thruster.required_load": self.thruster.required_load,
            "thruster.force_residual": self.thruster.force_residual,

            "refueling.coll_eff": self.refueling.coll_eff,
            "refueling.t_refuel": self.refueling.t_refuel,
            "refueling.eta_refuel": self.refueling.eta_refuel,
            "refueling.m_flow": self.refueling.m_flow,
            "refueling.p_tank": self.refueling.p_tank,
            "refueling.V_prop": self.refueling.V_prop,

            "solar.av_aligment": self.solar.av_aligment,
            "solar.eta_solar": self.solar.eta_solar,
            "solar.eta_power": self.solar.eta_power,

            "drag.cd_solar": self.drag.cd_solar,
            "drag.cd_solar_front": self.drag.cd_solar_front,
            "drag.cd_rad": self.drag.cd_rad,
            "drag.cd_rad_front": self.drag.cd_rad_front,
            "drag.cd_body_side": self.drag.cd_body_side,
            "drag.cd_inlet_side": self.drag.cd_inlet_side,
            "drag.cd_inlet_front": self.drag.cd_inlet_front,
            "drag.drag_total": self.drag.drag_total,
            "drag.drag_solar": self.drag.drag_solar,
            "drag.drag_solar_front": self.drag.drag_solar_front,
            "drag.drag_rad": self.drag.drag_rad,
            "drag.drag_rad_front": self.drag.drag_rad_front,
            "drag.drag_body_side": self.drag.drag_body_side,
            "drag.drag_inlet_side": self.drag.drag_inlet_side,
            "drag.drag_inlet_front": self.drag.drag_inlet_front,

            "thermal.T_des": self.thermal.T_des,
            "thermal.alpha_body": self.thermal.alpha_body,
            "thermal.alpha_solar": self.thermal.alpha_solar,
            "thermal.epsilon_therm_in": self.thermal.epsilon_therm_in,
            "thermal.epsilon_therm_body": self.thermal.epsilon_therm_body,
            "thermal.epsilon_therm_solar": self.thermal.epsilon_therm_solar,
            "thermal.epsilon_therm_rad": self.thermal.epsilon_therm_rad,

            "mission_profile.active_refueling": self.mission_profile.active_refueling,
            "mission_profile.delta_v": self.mission_profile.delta_v,
            "mission_profile.required_fuel": self.mission_profile.required_fuel,
        }

        specs = {
            "name": {"kind": "str"},

            "orbit.altitude": {"min": 0.0, "max": 2000.0},
            "orbit.velocity": {"min": 0.0, "max": 20000.0},
            "orbit.density": {"min": 0.0, "max": 1.0},
            "orbit.p_orb": {"min": 0.0, "max": 1.0e9},
            "orbit.temperature": {"min": 0.0, "max": 10000.0},
            "orbit.molar_mass": {"min": 0.0, "max": 1.0},
            "orbit.alpha": {"min": 0.0, "max": 3.141592653589793},
            "orbit.gamma": {"min": 0.0, "max": 10.0},
            "orbit.R_spec": {"min": 0.0, "max": 1.0e5},
            "orbit.msis_date": {"kind": "str"},
            "orbit.msis_f107": {"min": 0.0, "max": 1000.0},
            "orbit.msis_ap": {"min": 0.0, "max": 1000.0},
            "orbit.latitude": {"min": -90.0, "max": 90.0},
            "orbit.longitude": {"min": -180.0, "max": 180.0},
            "orbit.use_average": {"kind": "bool"},

            "geometry.S_in": {"kind": "choice", "allowed": {"c", "s", "e", "r"}},
            "geometry.S_body": {"kind": "choice", "allowed": {"c", "s", "e", "r"}},
            "geometry.use_intake_area_ratio": {"kind": "bool"},
            "geometry.fixed_body": {"kind": "bool"},
            "geometry.intake_area_ratio": {"min": 0.0, "max": 100.0},
            "geometry.AR_in": {"min": 0.0, "max": 100.0},
            "geometry.AR_body": {"min": 0.0, "max": 100.0},
            "geometry.AR_solar": {"min": 0.0, "max": 100.0},
            "geometry.AR_rad": {"min": 0.0, "max": 100.0},
            "geometry.epsilon_in": {"min": 0.0, "max": 1.0},
            "geometry.epsilon_body": {"min": 0.0, "max": 1.0},
            "geometry.epsilon_solar": {"min": 0.0, "max": 1.0},
            "geometry.epsilon_rad": {"min": 0.0, "max": 1.0},
            "geometry.epsilon_in_norm": {"min": 0.0, "max": 1.0},
            "geometry.wake_in": {"min": 0.0, "max": 10.0},
            "geometry.wake_body": {"min": 0.0, "max": 10.0},
            "geometry.wake_solar": {"min": 0.0, "max": 10.0},
            "geometry.wake_radiator": {"min": 0.0, "max": 10.0},
            "geometry.A_in": {"min": 0.0, "max": 1.0e5},
            "geometry.A_ref": {"min": 0.0, "max": 1.0e5},
            "geometry.A_prop": {"min": 0.0, "max": 1.0e5},
            "geometry.A_in_drag": {"min": 0.0, "max": 1.0e5},
            "geometry.A_body": {"min": 0.0, "max": 1.0e5},
            "geometry.A_solar": {"min": 0.0, "max": 1.0e5},
            "geometry.A_rad": {"min": 0.0, "max": 1.0e5},
            "geometry.t_solar": {"min": 0.0, "max": 100.0},
            "geometry.t_rad": {"min": 0.0, "max": 100.0},
            "geometry.L_in": {"min": 0.0, "max": 1.0e5},
            "geometry.L_body": {"min": 0.0, "max": 1.0e5},
            "geometry.X_solar": {"min": 0.0, "max": 1.0e5},
            "geometry.X_rad": {"min": 0.0, "max": 1.0e5},

            "rate.R_mass_volume_in": {"min": 0.0, "max": 1.0e6},
            "rate.R_mass_volume_body": {"min": 0.0, "max": 1.0e6},
            "rate.R_mass_surface_solar": {"min": 0.0, "max": 1.0e6},
            "rate.R_mass_surface_rad": {"min": 0.0, "max": 1.0e6},

            "mass.Mass_in": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_body": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_solar": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_rad": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_prop": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_ADCS": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_payload": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_refprop": {"min": 0.0, "max": 1.0e9},
            "mass.Mass_total": {"min": 0.0, "max": 1.0e9},

            "power.Power_in": {"min": 0.0, "max": 1.0e9},
            "power.Power_body": {"min": 0.0, "max": 1.0e9},
            "power.Power_solar": {"min": 0.0, "max": 1.0e9},
            "power.Power_rad": {"min": 0.0, "max": 1.0e9},
            "power.Power_prop": {"min": 0.0, "max": 1.0e9},
            "power.Power_ADCS": {"min": 0.0, "max": 1.0e9},
            "power.Power_payload": {"min": 0.0, "max": 1.0e9},
            "power.Power_refprop": {"min": 0.0, "max": 1.0e9},
            "power.Power_total": {"min": 0.0, "max": 1.0e9},

            "thruster.thrust": {"min": 0.0, "max": 1.0e9},
            "thruster.specific_impulse": {"min": 0.0, "max": 1.0e6},
            "thruster.eff": {"min": 0.0, "max": 1.0},
            "thruster.thermal_eff": {"min": 0.0, "max": 1.0},
            "thruster.power": {"min": 0.0, "max": 1.0e9},
            "thruster.propellant_mass": {"min": 0.0, "max": 1.0e6},
            "thruster.m_flow": {"min": 0.0, "max": 1.0e6},
            "thruster.propulsive_ram_load": {"min": 0.0, "max": 1.0e9},
            "thruster.refueling_ram_load": {"min": 0.0, "max": 1.0e9},
            "thruster.required_load": {"min": 0.0, "max": 1.0e9},
            "thruster.force_residual": {"min": 0.0, "max": 1.0e9},

            "refueling.coll_eff": {"min": 0.0, "max": 1.0},
            "refueling.t_refuel": {"min": 0.0, "max": 1.0e10},
            "refueling.eta_refuel": {"min": 0.0, "max": 1.0},
            "refueling.m_flow": {"min": 0.0, "max": 1.0e6},
            "refueling.p_tank": {"min": 0.0, "max": 1.0e10},
            "refueling.V_prop": {"min": 0.0, "max": 1.0e6},

            "solar.av_aligment": {"min": 0.0, "max": 180.0},
            "solar.eta_solar": {"min": 0.0, "max": 1.0},
            "solar.eta_power": {"min": 0.0, "max": 1.0},

            "drag.cd_solar": {"min": 0.0, "max": 100.0},
            "drag.cd_solar_front": {"min": 0.0, "max": 100.0},
            "drag.cd_rad": {"min": 0.0, "max": 100.0},
            "drag.cd_rad_front": {"min": 0.0, "max": 100.0},
            "drag.cd_body_side": {"min": 0.0, "max": 100.0},
            "drag.cd_inlet_side": {"min": 0.0, "max": 100.0},
            "drag.cd_inlet_front": {"min": 0.0, "max": 100.0},
            "drag.drag_total": {"min": 0.0, "max": 1.0e9},
            "drag.drag_solar": {"min": 0.0, "max": 1.0e9},
            "drag.drag_solar_front": {"min": 0.0, "max": 1.0e9},
            "drag.drag_rad": {"min": 0.0, "max": 1.0e9},
            "drag.drag_rad_front": {"min": 0.0, "max": 1.0e9},
            "drag.drag_body_side": {"min": 0.0, "max": 1.0e9},
            "drag.drag_inlet_side": {"min": 0.0, "max": 1.0e9},
            "drag.drag_inlet_front": {"min": 0.0, "max": 1.0e9},

            "thermal.T_des": {"min": 0.0, "max": 10000.0},
            "thermal.alpha_body": {"min": 0.0, "max": 1.0},
            "thermal.alpha_solar": {"min": 0.0, "max": 1.0},
            "thermal.epsilon_therm_in": {"min": 0.0, "max": 1.0},
            "thermal.epsilon_therm_body": {"min": 0.0, "max": 1.0},
            "thermal.epsilon_therm_solar": {"min": 0.0, "max": 1.0},
            "thermal.epsilon_therm_rad": {"min": 0.0, "max": 1.0},

            "mission_profile.active_refueling": {"kind": "bool"},
            "mission_profile.delta_v": {"min": 0.0, "max": 1.0e7},
            "mission_profile.required_fuel": {"min": 0.0, "max": 1.0e9},
        }

        errors: list[str] = []

        for path, spec in specs.items():
            value = values[path]
            kind = spec.get("kind", "number")

            if kind == "bool":
                if not isinstance(value, bool):
                    errors.append(f"{path} must be a bool, got {type(value).__name__}")
                continue

            if kind == "str":
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{path} must be a non-empty string")
                continue

            if kind == "choice":
                if value not in spec["allowed"]:
                    errors.append(f"{path} must be one of {sorted(spec['allowed'])}, got {value!r}")
                continue

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{path} must be a number, got {type(value).__name__}")
                continue

            if not math.isfinite(float(value)):
                errors.append(f"{path} must be finite, got {value!r}")
                continue

            min_value = spec["min"]
            max_value = spec["max"]

            if value < min_value or value > max_value:
                errors.append(
                    f"{path} must be between {min_value} and {max_value}, got {value}"
                )

        if (not self.geometry.use_intake_area_ratio) and (not self.geometry.fixed_body):
            errors.append(
                "Invalid geometry mode: geometry.use_intake_area_ratio and "
                "geometry.fixed_body cannot both be false."
            )

        if errors:
            raise ValueError("SpacecraftState bound check failed:\n - " + "\n - ".join(errors))

    @classmethod
    def from_toml(cls, filepath: str | PathLike[str]) -> "SpacecraftState":
        # Inputs:
        #   filepath: path to a TOML spacecraft definition file.
        #
        # Output:
        #   SpacecraftState loaded from TOML.

        data = _load_toml_file(filepath)

        return _coerce_dataclass(cls, data)

    def to_toml(self, filepath: str | PathLike[str]) -> None:
        # Inputs:
        #   filepath: output path for the TOML spacecraft file.
        #
        # Output:
        #   Writes the current spacecraft state to TOML.

        if tomli_w is None:
            raise ImportError(
                "tomli_w is required to write TOML files. Install it with: pip install tomli-w"
            )

        with open(Path(filepath), "wb") as handle:
            tomli_w.dump(asdict(self), handle)

    def update_from_toml(self, filepath: str | PathLike[str]) -> "SpacecraftState":
        # Inputs:
        #   filepath: TOML file with updated spacecraft parameters.
        #
        # Output:
        #   Updates the spacecraft state using TOML values while preserving
        #   existing values for keys not present in the override file.

        data = _load_toml_file(filepath)

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected mapping for override at <root>, got {type(data).__name__}"
            )

        def apply(target: Any, payload: dict[str, Any], prefix: str = "") -> None:
            for key, value in payload.items():
                dotted = f"{prefix}.{key}" if prefix else key

                if not hasattr(target, key):
                    raise KeyError(f"Unknown keys for override: {dotted}")

                current = getattr(target, key)

                if isinstance(value, dict):
                    if not is_dataclass(current):
                        raise TypeError(
                            f"Cannot apply nested override to non-dataclass field: {dotted}"
                        )
                    apply(current, value, dotted)
                else:
                    object.__setattr__(target, key, value)

        apply(self, data)

        return self
