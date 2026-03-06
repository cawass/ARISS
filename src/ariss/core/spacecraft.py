import json
import tomllib
import tomli_w
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any

@dataclass(frozen=True)
class MissionProfileState:
    delta_v: float = 2000.0
    refueling_area: float = 1.0
    mission_height: float = 120

    def update(self, **kwargs: Any) -> "MissionProfileState":
        return replace(self, **kwargs)


@dataclass(frozen=False)
class OrbitState:
    altitude: float = 0
    velocity: float = 0
    density: float = 0
    temperature: float = 0
    molar_mass: float = 0
    alpha: float = 0

@dataclass(frozen=False)
class DragState:
    drag_total: float = 0.0
    drag_solar: float = 0.0
    drag_rad: float = 0.0
    drag_body: float = 0.0
    drag_inlet: float = 0.0

@dataclass(frozen=False)
class GeometryState:
    S_in: str = "s"
    S_body: str = "s"

    AR_in: float = 1.0
    AR_body: float = 1.0
    AR_solar: float = 0.5
    AR_rad: float = 0.3

    epsilon_in: float = 0.6
    epsilon_body: float = 0.6
    epsilon_solar: float = 0.6
    epsilon_rad: float = 0.6

    A_in: float = 4.0387
    A_body: float = 1.21
    A_solar: float = 0
    A_rad: float = 0.0
    A_ref: float = 2.2
    A_prop: float = 1.0

    L_in: float = 2.26
    L_body: float = 2.80
    L_solar: float = 3.0
    L_rad: float = 0.5

    def update(self, **kwargs: Any) -> "GeometryState":
        return replace(self, **kwargs)

@dataclass(frozen=False)
class ThrusterState:
    thrust: float = 0.0
    specific_impulse: float = 5500 
    thruster_eff: float = 0.53
    power_required: float = 5000*thruster_eff
    propellant_mass: float = 0.0
    def update(self, **kwargs: Any) -> "ThrusterState":
        return replace(self, **kwargs)
    
@dataclass(frozen=True)
class RateState:
    R_mass_volume_in: float = 10
    R_mass_volume_body: float = 10
    R_mass_surface_solar: float = 10
    R_mass_surface_rad: float = 10

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
    Mass_refprop: float = 300
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
    Power_ADCS: float =  160.0
    Power_payload: float = 300.0
    Power_refprop: float = 300.0
    Power_total: float = 300.0

    def update(self, **kwargs: Any) -> "PowerState":
        return replace(self, **kwargs)

@dataclass(frozen=True)
class SolarState:
    av_aligment: float = 60
    eta_solar: float = 0.3
    eta_power: float = 0.95

    def update(self, **kwargs: Any) -> "SolarState":
        return replace(self, **kwargs)

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
    orbit: OrbitState = field(default_factory=OrbitState)
    geometry: GeometryState = field(default_factory=GeometryState)
    thruster: ThrusterState = field(default_factory=ThrusterState)
    rate: RateState = field(default_factory=RateState)
    mass: MassState = field(default_factory=MassState)
    power: PowerState = field(default_factory=PowerState)
    solar: SolarState = field(default_factory=SolarState)
    thermal: ThermalState = field(default_factory=ThermalState)
    drag: DragState = field(default_factory=DragState)
    mission_profile: MissionProfileState = field(default_factory=MissionProfileState)

    @classmethod
    def from_json(cls, filepath: str) -> "SpacecraftState":
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    def to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=4)

    @classmethod
    def from_toml(cls, filepath: str) -> "SpacecraftState":
        with open(filepath, "rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data)

    def to_toml(self, filepath: str) -> None:
        with open(filepath, "wb") as handle:
            tomli_w.dump(asdict(self), handle)
