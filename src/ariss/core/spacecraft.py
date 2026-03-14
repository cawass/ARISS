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
    active_refueling: bool = False  # [bool] Enables the mission branch that collects atmospheric propellant.
    delta_v: float = 1157.8  # [m/s] Mission delta-v requirement used in the rocket equation.
    required_fuel: float = 0  # [kg] Fuel mass required to satisfy the mission delta-v target.

    def update(self, **kwargs: Any) -> "MissionProfileState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class OrbitState:
    altitude: float = 0
    velocity: float = 0
    density: float = 0
    p_orb: float = 1e-5
    temperature: float = 0
    molar_mass: float = 0
    alpha: float = 0
    gamma: float = 1.4
    R_spec: float = 287.0
    msis_date: str = "2000-01-01T00:00:00"
    msis_f107: float = 140.0
    msis_ap: float = 15.0


@dataclass(frozen=False)
class GeometryState:
    S_in: str = "c"
    S_body: str = "s"
    use_intake_area_ratio: bool = False
    intake_area_ratio: float = 1.0

    AR_in: float = 1.0
    AR_body: float = 1.0
    AR_solar: float = 5
    AR_rad: float = 5

    epsilon_in: float = 0.1
    epsilon_body: float = 0.1
    epsilon_solar: float = 0.1
    epsilon_rad: float = 0.1
    epsilon_in_norm: float = 0.9

    wake_in: float = 1
    wake_body: float = 1
    wake_solar: float = 1
    wake_radiator: float = 1

    A_in: float = 4.0387
    A_ref: float = 2
    A_prop: float = 2
    A_in_drag: float = 2
    A_body: float = 0.5
    A_solar: float = 5
    A_rad: float = 0.0

    L_in: float = 2.5
    L_body: float = 2.5
    X_solar: float = 2.0
    X_rad: float = 0.5

    def update(self, **kwargs: Any) -> "GeometryState":
        return replace(self, **kwargs)

@dataclass(frozen=True)
class RateState:
    R_mass_volume_in: float = 10
    R_mass_volume_body: float = 10
    R_mass_surface_solar: float = 5
    R_mass_surface_rad: float = 5

    def update(self, **kwargs: Any) -> "RateState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class MassState:
    Mass_in: float = 0.0
    Mass_body: float = 0.0
    Mass_solar: float = 0.0
    Mass_rad: float = 0.0
    Mass_prop: float = 61
    Mass_ADCS: float = 20
    Mass_payload: float = 24
    Mass_refprop: float = 700
    Mass_total: float = 0.0

    def update(self, **kwargs: Any) -> "MassState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class PowerState:
    Power_in: float = 0.0
    Power_body: float = 0.0
    Power_solar: float = 0.0
    Power_rad: float = 0.0
    Power_prop: float = 0.0
    Power_ADCS: float = 2000.0
    Power_payload: float = 0.0
    Power_refprop: float = 0.0
    Power_total: float = 0.0

    def update(self, **kwargs: Any) -> "PowerState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class ThrusterState:
    thrust: float = 0.1039
    specific_impulse: float = 4500
    eff: float = 0.5
    power: float = 5000.0
    propellant_mass: float = 0.0
    m_flow: float = 1e-3

    def update(self, **kwargs: Any) -> "ThrusterState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class RefuelingState:
    coll_eff: float = 0.61
    t_refuel: float = 140 * 24 * 3600
    eta_refuel: float = 0.1
    m_flow: float = 1e-3
    p_tank: float = 100000
    V_prop: float = 0.7

    def update(self, **kwargs: Any) -> "RefuelingState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SolarState:
    av_aligment: float = 60
    eta_solar: float = 0.3
    eta_power: float = 0.9

    def update(self, **kwargs: Any) -> "SolarState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class DragState:
    cd_solar: float = 0.2
    cd_rad: float = 0.2
    cd_body_side: float = 0.2
    cd_inlet_side: float = 0.2
    cd_inlet_front: float = 0.2

    drag_total: float = 1
    drag_solar: float = 0.2
    drag_rad: float = 0.2
    drag_body_side: float = 0.2
    drag_inlet_side: float = 0.2
    drag_inlet_front: float = 0.2


@dataclass(frozen=True)
class ThermalState:
    T_des: float = 300.0
    alpha_body: float = 0.1
    alpha_solar: float = 0.9

    epsilon_therm_in: float = 0.5
    epsilon_therm_body: float = 0.9
    epsilon_therm_solar: float = 0.85
    epsilon_therm_rad: float = 0.9

    def update(self, **kwargs: Any) -> "ThermalState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SpacecraftState:
    name: str = "ARISS Case"

    orbit: OrbitState = field(default_factory=OrbitState)
    geometry: GeometryState = field(default_factory=GeometryState)
    thruster: ThrusterState = field(default_factory=ThrusterState)
    rate: RateState = field(default_factory=RateState)
    mass: MassState = field(default_factory=MassState)
    power: PowerState = field(default_factory=PowerState)
    solar: SolarState = field(default_factory=SolarState)
    thermal: ThermalState = field(default_factory=ThermalState)
    drag: DragState = field(default_factory=DragState)
    refueling: RefuelingState = field(default_factory=RefuelingState)
    mission_profile: MissionProfileState = field(default_factory=MissionProfileState)

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
