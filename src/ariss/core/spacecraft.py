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
