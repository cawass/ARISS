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

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any

try:
    import tomli_w
except ImportError:  # pragma: no cover - optional dependency for TOML writing only
    tomli_w = None

def _is_dataclass_type(value: Any) -> bool:
    # Inputs:
    #   value: object or type to inspect.
    #
    # Output:
    #   True if value is a dataclass type, else False.
    return isinstance(value, type) and is_dataclass(value)

def _coerce_dataclass(dataclass_type: type[Any], payload: dict[str, Any]) -> Any:
    # Inputs:
    #   dataclass_type: destination dataclass type.
    #   payload: field mapping used to build the destination object.
    #
    # Output:
    #   Dataclass instance with nested dictionaries converted recursively.
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping for {dataclass_type.__name__}, got {type(payload).__name__}")

    field_map = {field_info.name: field_info for field_info in fields(dataclass_type)}
    unknown_keys = sorted(set(payload) - set(field_map))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise KeyError(f"Unknown keys for {dataclass_type.__name__}: {joined}")

    coerced: dict[str, Any] = {}
    for key, value in payload.items():
        field_info = field_map[key]
        if isinstance(value, dict) and _is_dataclass_type(field_info.type):
            coerced[key] = _coerce_dataclass(field_info.type, value)
        else:
            coerced[key] = value
    return dataclass_type(**coerced)

@dataclass(frozen=False)
class MissionProfileState:
    active_refueling: bool = True  # [bool] Enables the mission branch that collects atmospheric propellant.
    delta_v: float = 1157.8  # [m/s] Mission delta-v requirement used in the rocket equation.
    required_fuel: float = 0  # [kg] Fuel mass required to satisfy the mission delta-v target.

    def update(self, **kwargs: Any) -> "MissionProfileState":
        return replace(self, **kwargs)

@dataclass(frozen=False)
class OrbitState:
    altitude: float = 0  # [km] Orbital altitude above Earth used by the atmosphere model.
    velocity: float = 0  # [m/s] Circular orbital velocity at the selected altitude.
    density: float = 0  # [kg/m^3] Atmospheric density at orbital altitude.
    p_orb: float = 1e-5  # [Pa] Ambient atmospheric pressure at orbital altitude.
    temperature: float = 0  # [K] Atmospheric temperature at orbital altitude.
    molar_mass: float = 0  # [kg/mol] Effective molar mass of the local atmosphere.
    alpha: float = 0  # [rad] Flow incidence angle used by the drag model.
    gamma: float = 1.4  # [-] Ratio of specific heats of the atmosphere.
    R_spec: float = 287.0  # [J/(kg*K)] Specific gas constant of the local atmosphere.

@dataclass(frozen=False)
class GeometryState:
    S_in: str = "s"  # [str] Intake shape code; values starting with "s" are treated as square, "e" as elliptical.
    S_body: str = "s"  # [str] Body shape code; values starting with "s" are treated as square, "e" as elliptical.

    AR_in: float = 1.0  # [-] Intake cross-section aspect ratio, width/height.
    AR_body: float = 1.0  # [-] Body cross-section aspect ratio, width/height.
    AR_solar: float = 0.5  # [-] Solar-panel planform aspect ratio.
    AR_rad: float = 0.3  # [-] Radiator planform aspect ratio.

    epsilon_in: float = 0.1  # [-] Free-molecular accommodation coefficient of inlet side surfaces.
    epsilon_body: float = 0.1  # [-] Free-molecular accommodation coefficient of body side surfaces.
    epsilon_solar: float = 0.1  # [-] Free-molecular accommodation coefficient of solar-panel surfaces.
    epsilon_rad: float = 0.1  # [-] Free-molecular accommodation coefficient of radiator surfaces.
    epsilon_in_norm: float = 0.1  # [-] Free-molecular accommodation coefficient of the inlet front face.

    A_in: float = 4.0387  # [m^2] Total inlet area exposed to incoming atmospheric flow.
    A_ref: float = 2  # [m^2] Intake area reserved for the refueling stream that fills the tanks(Perfect collection efficiency).
    A_prop: float = 2  # [m^2] Intake area feeding the propulsion stream for drag compensation(Perfect collection efficiency).
    A_in_drag: float = 2  # [m^2] Intake area that creates drag but does not become useful captured flow(0 collection efficiency).
    A_body: float = 0.5  # [m^2] Body cross-sectional area.
    A_solar: float = 5  # [m^2] Deployable solar-array area beyond the fixed top surface.
    A_rad: float = 0.0  # [m^2] Radiator area.

    L_in: float = 2.5  # [m] Intake length along the spacecraft axis.
    L_body: float = 2.5  # [m] Main body length along the spacecraft axis.
    L_solar: float = 2.0  # [m] Characteristic solar-panel length.
    L_rad: float = 0.5  # [m] Characteristic radiator length.

    def update(self, **kwargs: Any) -> "GeometryState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class RateState:
    R_mass_volume_in: float = 10  # [kg/m^3] Intake structural mass density used in volume-based sizing.
    R_mass_volume_body: float = 10  # [kg/m^3] Body structural mass density used in volume-based sizing.
    R_mass_surface_solar: float = 5  # [kg/m^2] Solar-array areal mass density.
    R_mass_surface_rad: float = 5  # [kg/m^2] Radiator areal mass density.

    def update(self, **kwargs: Any) -> "RateState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class MassState:
    Mass_in: float = 0.0  # [kg] Intake structure mass.
    Mass_body: float = 0.0  # [kg] Main body structure mass.
    Mass_solar: float = 0.0  # [kg] Solar-array mass.
    Mass_rad: float = 0.0  # [kg] Radiator mass.
    Mass_prop: float = 61  # [kg] Stored propulsion propellant mass used for tank sizing.
    Mass_ADCS: float = 20  # [kg] Attitude determination and control subsystem mass.
    Mass_payload: float = 24  # [kg] Payload mass.
    Mass_refprop: float = 700  # [kg] Refueling and propellant-processing subsystem mass allocation.
    Mass_total: float = 0.0  # [kg] Total spacecraft mass used by the sizing loop.

    def update(self, **kwargs: Any) -> "MassState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class PowerState:
    Power_in: float = 0.0  # [W] Inlet subsystem electrical load.
    Power_body: float = 0.0  # [W] Main body or spacecraft-bus electrical load.
    Power_solar: float = 0.0  # [W] Solar power-chain overhead required by conversion inefficiency.
    Power_rad: float = 0.0  # [W] Thermal-control or radiator electrical load.
    Power_prop: float = 0.0  # [W] Thruster electrical bus power demand.
    Power_ADCS: float = 2000.0  # [W] ADCS electrical load.
    Power_payload: float = 0.0  # [W] Payload electrical load.
    Power_refprop: float = 0.0  # [W] Refueling and propellant-processing electrical load.
    Power_total: float = 0.0  # [W] Total spacecraft electrical power demand.

    def update(self, **kwargs: Any) -> "PowerState":
        return replace(self, **kwargs)

@dataclass(frozen=False)
class ThrusterState:
    thrust: float = 0.1039  # [N] Thruster force generated by the propulsion stream.
    specific_impulse: float = 4500  # [s] Thruster specific impulse.
    thruster_eff: float = 0.5  # [-] Electrical-to-jet efficiency of the thruster.
    power_supplied: float = 5000.0  # [W] Electrical power supplied to the thruster.
    power_required: float = power_supplied * thruster_eff  # [W] Effective jet power used by the propulsion sizing equations.
    propellant_mass: float = 0.0  # [kg/s] Propellant mass flow through the thruster; field name kept for legacy compatibility.
    m_flow: float = 1e-3  # [kg/s] Thruster mass flow rate.

    def update(self, **kwargs: Any) -> "ThrusterState":
        return replace(self, **kwargs)
    
@dataclass(frozen=False)
class RefuelingState:
    coll_eff: float = 0.61  # [-] Intake collection efficiency into useful captured flow.
    t_refuel: float = 120 * 24 * 3600  # [s] Nominal refueling duration parameter stored with the subsystem.
    eta_refuel: float = 0.1  # [-] Efficiency of the refueling compression process.
    m_flow: float = 1e-3  # [kg/s] Captured mass flow routed into the storage tanks.
    p_tank: float = 100000  # [Pa] Storage tank pressure.
    V_prop: float = 0.7  # [m^3] Stored propellant tank volume.

    def update(self, **kwargs: Any) -> "RefuelingState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SolarState:
    av_aligment: float = 60  # [deg] Average array alignment angle relative to the Sun vector.
    eta_solar: float = 0.3  # [-] Solar-cell conversion efficiency.
    eta_power: float = 0.9  # [-] End-to-end electrical power-chain efficiency.

    def update(self, **kwargs: Any) -> "SolarState":
        return replace(self, **kwargs)

@dataclass(frozen=False)
class DragState:
    cd_solar: float = 0.2  # [-] Drag coefficient of the solar-panel side surfaces.
    cd_rad: float = 0.2  # [-] Drag coefficient of the radiator surfaces.
    cd_body_side: float = 0.2  # [-] Drag coefficient of the spacecraft body side surfaces.
    cd_inlet_side: float = 0.2  # [-] Drag coefficient of the inlet side surfaces.
    cd_inlet_front: float = 0.2  # [-] Drag coefficient of the front inlet face.

    drag_total: float = 1  # [N] Total aerodynamic drag force.
    drag_solar: float = 0.2  # [N] Drag contribution from the solar panels.
    drag_rad: float = 0.2  # [N] Drag contribution from the radiators.
    drag_body_side: float = 0.2  # [N] Drag contribution from the spacecraft body sides.
    drag_inlet_side: float = 0.2  # [N] Drag contribution from the inlet side walls.
    drag_inlet_front: float = 0.2  # [N] Drag contribution from the front inlet face.

@dataclass(frozen=True)
class ThermalState:
    T_des: float = 320.0  # [K] Design temperature used by thermal and refueling calculations.
    alpha_body: float = 0.1  # [-] Solar absorptivity of the body surface.
    alpha_solar: float = 0.9  # [-] Solar absorptivity of the solar-panel surface.

    epsilon_therm_in: float = 0.5  # [-] Thermal emissivity of the intake surfaces.
    epsilon_therm_body: float = 0.9  # [-] Thermal emissivity of the body surfaces.
    epsilon_therm_solar: float = 0.85  # [-] Thermal emissivity of the solar-panel surfaces.
    epsilon_therm_rad: float = 0.9  # [-] Thermal emissivity of the radiator surfaces.

    def update(self, **kwargs: Any) -> "ThermalState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SpacecraftState:
    orbit: OrbitState = field(default_factory=OrbitState)  # Orbit and atmosphere state.
    geometry: GeometryState = field(default_factory=GeometryState)  # Spacecraft geometry and surface properties.
    thruster: ThrusterState = field(default_factory=ThrusterState)  # Thruster performance and flow state.
    rate: RateState = field(default_factory=RateState)  # Mass-scaling coefficients for sizing.
    mass: MassState = field(default_factory=MassState)  # Spacecraft mass breakdown.
    power: PowerState = field(default_factory=PowerState)  # Spacecraft power breakdown.
    solar: SolarState = field(default_factory=SolarState)  # Solar-generation parameters.
    thermal: ThermalState = field(default_factory=ThermalState)  # Thermal and radiative properties.
    drag: DragState = field(default_factory=DragState)  # Aerodynamic drag coefficients and forces.
    refueling: RefuelingState = field(default_factory=RefuelingState)  # Atmospheric refueling state.
    mission_profile: MissionProfileState = field(default_factory=MissionProfileState)  # Mission-level requirements.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpacecraftState":
        # Inputs:
        #   data: nested dictionary containing spacecraft-state fields.
        #
        # Output:
        #   SpacecraftState built from the provided dictionary.
        return _coerce_dataclass(cls, data)

    @classmethod
    def from_json(cls, filepath: str | PathLike[str]) -> "SpacecraftState":
        # Inputs:
        #   filepath: path to a JSON spacecraft definition file.
        #
        # Output:
        #   SpacecraftState loaded from JSON.
        with open(Path(filepath), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, filepath: str | PathLike[str]) -> "SpacecraftState":
        # Inputs:
        #   filepath: path to a TOML or JSON spacecraft definition file.
        #
        # Output:
        #   SpacecraftState loaded from the detected file format.
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix == ".toml":
            return cls.from_toml(path)
        if suffix == ".json":
            return cls.from_json(path)
        raise ValueError(f"Unsupported spacecraft file format: {path.suffix or '<no extension>'}")

    def to_json(self, filepath: str | PathLike[str]) -> None:
        # Inputs:
        #   filepath: output path for the JSON spacecraft file.
        #
        # Output:
        #   Writes the current spacecraft state to JSON.
        with open(Path(filepath), "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=4)

    @classmethod
    def from_toml(cls, filepath: str | PathLike[str]) -> "SpacecraftState":
        # Inputs:
        #   filepath: path to a TOML spacecraft definition file.
        #
        # Output:
        #   SpacecraftState loaded from TOML.
        with open(Path(filepath), "rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data)

    def to_toml(self, filepath: str | PathLike[str]) -> None:
        # Inputs:
        #   filepath: output path for the TOML spacecraft file.
        #
        # Output:
        #   Writes the current spacecraft state to TOML.
        if tomli_w is None:
            raise ImportError("tomli_w is required to write TOML files. Install it with: pip install tomli-w")
        with open(Path(filepath), "wb") as handle:
            tomli_w.dump(asdict(self), handle)


def load_spacecraft(source: SpacecraftState | str | PathLike[str]) -> SpacecraftState:
    # Inputs:
    #   source: existing SpacecraftState or path to a TOML/JSON file.
    #
    # Output:
    #   Ready-to-use SpacecraftState instance.
    if isinstance(source, SpacecraftState):
        return source
    return SpacecraftState.from_file(source)
