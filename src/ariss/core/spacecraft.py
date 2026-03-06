import json
import tomllib
import tomli_w
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict


_ORBIT_KEY_ALIASES = {
    "h_orb": "altitude",
    "V_orb": "velocity",
    "rho_orb": "density",
    "T_orb": "temperature",
}


def _normalize_orbit_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in kwargs.items():
        normalized[_ORBIT_KEY_ALIASES.get(key, key)] = value
    return normalized


def _filter_dataclass_kwargs(dataclass_type, payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dataclass_type)}
    return {key: value for key, value in payload.items() if key in allowed}


def _legacy_mass_budget_to_mass_state(mass_budget_data: Dict[str, Any]) -> Dict[str, float]:
    structure = float(mass_budget_data.get("structure", 0.0))
    payload = float(mass_budget_data.get("payload", 0.0))
    power_system = float(mass_budget_data.get("power_system", 0.0))
    thermal = float(mass_budget_data.get("thermal", 0.0))
    adcs = float(mass_budget_data.get("adcs", 0.0))
    communications = float(mass_budget_data.get("communications", 0.0))
    total = structure + payload + power_system + thermal + adcs + communications
    return {
        "Mass_body": structure,
        "Mass_payload": payload,
        "Mass_solar": power_system,
        "Mass_rad": thermal,
        "Mass_ADCS": adcs,
        "Mass_refprop": communications,
        "Mass_total": total,
    }


@dataclass(frozen=True)
class OrbitState:
    altitude: float = 188.0151e3
    velocity: float = 7791.44
    density: float = 2.09e-10
    temperature: float = 1000.0
    molar_mass: float = 0.02897
    alpha: float = 0.0

    @property
    def h_orb(self) -> float:
        return self.altitude

    @property
    def V_orb(self) -> float:
        return self.velocity

    @property
    def rho_orb(self) -> float:
        return self.density

    @property
    def T_orb(self) -> float:
        return self.temperature

    def update(self, **kwargs: Any) -> "OrbitState":
        return replace(self, **_normalize_orbit_kwargs(dict(kwargs)))


@dataclass(frozen=True)
class GeometryState:
    S_in: str = "s"
    S_body: str = "s"

    AR_in: float = 1.0
    AR_body: float = 1.0
    AR_solar: float = 0.5
    AR_rad: float = 0.3

    epsilon_in: float = 0.6
    epsilon_body: float = 0.6
    epsilon_solar: float = 0.9
    epsilon_rad: float = 0.6

    A_in: float = 4.0387
    A_body: float = 1.21
    A_solar: float = 0
    A_rad: float = 0.0
    A_ref: float = 2.0661
    A_prop: float = 1.0

    L_in: float = 2.26
    L_body: float = 2.80
    L_solar: float = 3.0
    L_rad: float = 0.5

    def update(self, **kwargs: Any) -> "GeometryState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class BudgetState:
    B_mass_total: float = 0.0
    B_mass_volume_in: float = 10
    B_mass_volume_body: float = 10
    B_mass_surface_solar: float = 0.1
    B_mass_surface_rad: float = 0.1
    B_mass_power_prop: float = 0.05
    B_mass_mass_ADCS: float = 0.01
    B_mass_payload: float = 150
    B_mass_mflow_prop: float = 1.0
    B_mass_mflow_ref: float = 1.0

    B_power_power_solar: float = 0.1
    B_power_power_prop: float = 0.5
    B_power_mass_ADCS: float = 0.1
    B_power_payload: float = 100
    B_power_mflow_prop: float = 0.1
    B_power_mflow_ref: float = 0.1

    def update(self, **kwargs: Any) -> "BudgetState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ThrusterState:
    thrust: float = 0.0
    specific_impulse: float = 3500.0
    power_required: float = 0.0
    propellant_mass: float = 0.0

    def update(self, **kwargs: Any) -> "ThrusterState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class MassState:
    Mass_in: float = 0.0
    Mass_body: float = 0.0
    Mass_solar: float = 0.0
    Mass_rad: float = 0.0
    Mass_prop: float = 0.0
    Mass_ADCS: float = 0.0
    Mass_payload: float = 0.0
    Mass_refprop: float = 0.0
    Mass_total: float = 0.0

    def update(self, **kwargs: Any) -> "MassState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class PowerState:
    eff: float = 0.2
    alignment: float = 0.0

    Power_in: float = 0.0
    Power_body: float = 0.0
    Power_solar: float = 0.0
    Power_rad: float = 0.0
    Power_prop: float = 0.0
    Power_ADCS: float = 0.0
    Power_payload: float = 0.0
    Power_refprop: float = 0.0
    Power_total: float = 0.0

    @property
    def aligment(self) -> float:  # Legacy typo compatibility.
        return self.alignment

    def update(self, **kwargs: Any) -> "PowerState":
        payload = dict(kwargs)
        if "aligment" in payload and "alignment" not in payload:
            payload["alignment"] = payload.pop("aligment")
        return replace(self, **payload)


@dataclass(frozen=True)
class RefuelingState:
    active_refuel: bool = False
    eta_refuel: float = 0.7

    def update(self, **kwargs: Any) -> "RefuelingState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ThermalState:
    T_des: float = 300.0
    alpha_body: float = 0.1
    alpha_solar: float = 0.9
    epsilon_in: float = 0.5
    epsilon_body: float = 0.9
    epsilon_solar: float = 0.85
    epsilon_rad: float = 0.9

    def update(self, **kwargs: Any) -> "ThermalState":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class MissionProfileState:
    delta_v: float = 2000.0
    refueling_area: float = 1.0
    mission_height: float = 100.0

    def update(self, **kwargs: Any) -> "MissionProfileState":
        payload = dict(kwargs)
        if "orbital_height" in payload and "mission_height" not in payload:
            payload["mission_height"] = payload.pop("orbital_height")
        return replace(self, **payload)


@dataclass(frozen=True)
class SpacecraftState:
    orbit: OrbitState = field(default_factory=OrbitState)
    geometry: GeometryState = field(default_factory=GeometryState)
    thruster: ThrusterState = field(default_factory=ThrusterState)
    budget: BudgetState = field(default_factory=BudgetState)
    mass: MassState = field(default_factory=MassState)
    power: PowerState = field(default_factory=PowerState)
    refueling: RefuelingState = field(default_factory=RefuelingState)
    thermal: ThermalState = field(default_factory=ThermalState)
    mission_profile: MissionProfileState = field(default_factory=MissionProfileState)

    @property
    def total_mass(self) -> float:
        return self.mass.Mass_total

    @property
    def temperature_sc(self) -> float:
        return self.thermal.T_des

    def update(self, **kwargs: Any) -> "SpacecraftState":
        payload = dict(kwargs)
        changes: Dict[str, Any] = {}

        mass_patch: Dict[str, Any] = {}
        if "total_mass" in payload:
            mass_patch["Mass_total"] = payload.pop("total_mass")

        thermal_patch: Dict[str, Any] = {}
        if "temperature_sc" in payload:
            thermal_patch["T_des"] = payload.pop("temperature_sc")

        if "mass_budget" in payload:
            legacy = payload.pop("mass_budget")
            if isinstance(legacy, dict):
                mapped = _legacy_mass_budget_to_mass_state(legacy)
                mass_patch.update({k: v for k, v in mapped.items() if k != "Mass_total"})
                if "Mass_total" not in mass_patch:
                    mass_patch["Mass_total"] = mapped["Mass_total"]

        for key, value in payload.items():
            if not hasattr(self, key):
                raise ValueError(f"'{key}' is not a valid attribute of {self.__class__.__name__}")
            current_value = getattr(self, key)
            if is_dataclass(current_value) and isinstance(value, dict):
                changes[key] = current_value.update(**value)
            else:
                changes[key] = value

        if mass_patch:
            mass_current = changes.get("mass", self.mass)
            changes["mass"] = mass_current.update(**mass_patch)
        if thermal_patch:
            thermal_current = changes.get("thermal", self.thermal)
            changes["thermal"] = thermal_current.update(**thermal_patch)

        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpacecraftState":
        payload = dict(data)

        orbit_data = _normalize_orbit_kwargs(payload.pop("orbit", {}))
        geometry_data = payload.pop("geometry", {})
        thruster_data = payload.pop("thruster", {})
        budget_data = payload.pop("budget", {})
        mass_data = payload.pop("mass", {})
        power_data = payload.pop("power", {})
        refueling_data = payload.pop("refueling", {})
        thermal_data = payload.pop("thermal", {})
        mission_profile_data = payload.pop("mission_profile", {})
        if "orbital_height" in mission_profile_data and "mission_height" not in mission_profile_data:
            mission_profile_data["mission_height"] = mission_profile_data.pop("orbital_height")

        if "aligment" in power_data and "alignment" not in power_data:
            power_data["alignment"] = power_data.pop("aligment")

        legacy_mass_budget = payload.pop("mass_budget", None)
        if isinstance(legacy_mass_budget, dict):
            merged_mass_data = _legacy_mass_budget_to_mass_state(legacy_mass_budget)
            merged_mass_data.update(mass_data)
            mass_data = merged_mass_data

        if "total_mass" in payload and "Mass_total" not in mass_data:
            mass_data["Mass_total"] = float(payload.pop("total_mass"))
        if "temperature_sc" in payload and "T_des" not in thermal_data:
            thermal_data["T_des"] = float(payload.pop("temperature_sc"))

        payload.pop("drag_coeff", None)

        top_level = _filter_dataclass_kwargs(cls, payload)

        return cls(
            **top_level,
            orbit=OrbitState(**_filter_dataclass_kwargs(OrbitState, orbit_data)),
            geometry=GeometryState(**_filter_dataclass_kwargs(GeometryState, geometry_data)),
            thruster=ThrusterState(**_filter_dataclass_kwargs(ThrusterState, thruster_data)),
            budget=BudgetState(**_filter_dataclass_kwargs(BudgetState, budget_data)),
            mass=MassState(**_filter_dataclass_kwargs(MassState, mass_data)),
            power=PowerState(**_filter_dataclass_kwargs(PowerState, power_data)),
            refueling=RefuelingState(**_filter_dataclass_kwargs(RefuelingState, refueling_data)),
            thermal=ThermalState(**_filter_dataclass_kwargs(ThermalState, thermal_data)),
            mission_profile=MissionProfileState(**_filter_dataclass_kwargs(MissionProfileState, mission_profile_data)),
        )

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
