import json
import tomllib  # Built-in in Python 3.11+
import tomli_w  # Needed for writing TOML files
from dataclasses import dataclass, field, replace, is_dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class OrbitState:
    """Represents the orbital parameters and environment of the spacecraft."""
    h_orb: float = 400e3       # Altitude in meters
    V_orb: float = 7800.0      # Orbital velocity in m/s
    rho_orb: float = 0.0          # Atmospheric density
    T_orb: float = 0.0          # Orbital temperature
    
    def update(self, **kwargs: Any) -> 'OrbitState':
        return replace(self, **kwargs)


@dataclass(frozen=True)
class GeometryState:
    """Holds the geometric parameters and aerodynamic properties."""
    drag_coeff: float = 2.2       # Coefficient of drag (CD)
    reference_area: float = 1.0   # Area for drag/solar calculations
    diameter: float = 1.0         # Main body diameter
    length: float = 2.0           # Main body length
    
    def update(self, **kwargs: Any) -> 'GeometryState':
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ThrusterState:
    """Defines the propulsion system parameters."""
    thrust: float = 0.0           # Thrust produced
    specific_impulse: float = 0.0 # Isp
    power_required: float = 0.0   # Power drawn by thrusters
    propellant_mass: float = 0.0  # Available propellant
    
    def update(self, **kwargs: Any) -> 'ThrusterState':
        return replace(self, **kwargs)


@dataclass(frozen=True)
class MassBudgetState:
    """Tracks sub-system mass budgets."""
    structure: float = 0.0
    payload: float = 0.0
    power_system: float = 0.0
    thermal: float = 0.0
    adcs: float = 0.0
    communications: float = 0.0
    
    @property
    def dry_mass(self) -> float:
        return (self.structure + self.payload + self.power_system + 
                self.thermal + self.adcs + self.communications)

    def update(self, **kwargs: Any) -> 'MassBudgetState':
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SpacecraftState:
    """
    Immutable representation of a spacecraft's state at a given iteration.
    
    Use the `update()` method to create a new instance with the new values.
    """
    total_mass: float = 0.0
    drag_coeff: float = 0.0 # Convenience property mapped at top level
    
    # Nested subsystems
    orbit: OrbitState = field(default_factory=OrbitState)
    geometry: GeometryState = field(default_factory=GeometryState)
    thruster: ThrusterState = field(default_factory=ThrusterState)
    mass_budget: MassBudgetState = field(default_factory=MassBudgetState)

    def update(self, **kwargs: Any) -> 'SpacecraftState':
        """
        Creates a new instance with updated properties.
        Supports updating nested components via dictionaries.
        
        Example:
            new_sc = sc.update(
                total_mass=150.0,
                geometry={'drag_coeff': 2.5}
            )
        """
        changes = {}
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"'{key}' is not a valid attribute of {self.__class__.__name__}")

            current_value = getattr(self, key)
            
            # If the attribute is a nested dataclass and a dict was passed alongside it,
            # recursively call its update method to replace the interior fields.
            if is_dataclass(current_value) and isinstance(value, dict):
                changes[key] = current_value.update(**value)
            else:
                changes[key] = value
                
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpacecraftState':
        """
        Instantiates a SpacecraftState object from a dictionary
        """
        data_copy = data.copy()
        
        # Extract sub-components. Default to empty dictionaries if not present
        orbit_data = data_copy.pop('orbit', {})
        geometry_data = data_copy.pop('geometry', {})
        thruster_data = data_copy.pop('thruster', {})
        mass_budget_data = data_copy.pop('mass_budget', {})
        
        return cls(
            **data_copy,
            orbit=OrbitState(**orbit_data),
            geometry=GeometryState(**geometry_data),
            thruster=ThrusterState(**thruster_data),
            mass_budget=MassBudgetState(**mass_budget_data)
        )
        
    @classmethod
    def from_json(cls, filepath: str) -> 'SpacecraftState':
        """Loads a spacecraft state cleanly from a JSON configuration file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
    def to_json(self, filepath: str) -> None:
        """Saves the entire structure exactly as-is to JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
            
    @classmethod
    def from_toml(cls, filepath: str) -> 'SpacecraftState':
        """Loads a spacecraft state from a highly readable TOML template."""
        with open(filepath, 'rb') as f:
            data = tomllib.load(f)
        return cls.from_dict(data)
        
    def to_toml(self, filepath: str) -> None:
        """Exports the entire spacecraft state back to a TOML file."""
        with open(filepath, 'wb') as f:
            tomli_w.dump(asdict(self), f)