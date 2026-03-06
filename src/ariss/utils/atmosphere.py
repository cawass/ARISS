"""Atmosphere and orbital helpers backed by ``pymsis``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from ariss.utils import constants as const

try:
    import pymsis as msis  # type: ignore
except Exception:  # pragma: no cover - optional import path
    msis = None

def _require_pymsis() -> None:
    if msis is None:
        raise ImportError(
            "pymsis is required for atmosphere calculations. "
            "Install it in your environment: pip install pymsis"
        )

def _as_1d_float_array(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array

@dataclass(frozen=True)
class AtmosphereSample:
    """Atmospheric and orbital properties at one altitude."""

    height_km: float
    density: float
    temperature: float
    specific_gas_constant: float
    molar_mass: float
    o2_density: float
    n2_density: float
    o_density: float
    orbital_velocity: float
    dynamic_pressure: float

def atmos(height_array_km: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``rho, T, R_specific, O2, N2, O`` for altitudes in km."""
    _require_pymsis()
    heights_km = _as_1d_float_array(height_array_km)
    n = len(heights_km)
    et = np.array([np.datetime64("2020-01-01T00:00:00") for _ in range(n)])
    lons = np.zeros(n)
    lats = np.zeros(n)

    composition = msis.calculate(et, lons, lats, heights_km)
    composition = np.nan_to_num(composition)

    rho = composition[:, 0]
    n2 = composition[:, 1]
    o2 = composition[:, 2]
    o = composition[:, 3]
    temperature = composition[:, 10]

    total_mass = np.maximum(o + n2 + o2, 1.0e-30)
    molar_mass = (
        (o / total_mass) * 15.999e-3
        + (n2 / total_mass) * 28.0134e-3
        + (o2 / total_mass) * 31.9988e-3
    )
    r_specific = const.UNIVERSAL_GAS / np.maximum(molar_mass, 1.0e-30)
    return rho, temperature, r_specific, o2, n2, o

def calculate_orbital_velocity(height_array_km: np.ndarray | float) -> np.ndarray:
    """Circular-orbit velocity at altitude in km."""
    heights_km = _as_1d_float_array(height_array_km)
    return np.sqrt(const.EARTH_MU / (const.EARTH_RADIUS + heights_km * 1000.0))

def get_atmosphere_functions(
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 10000,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return interpolation functions for density, velocity, dynamic pressure, temperature."""
    height_array = np.linspace(height_min_km, height_max_km, samples)
    density, temperature, _, _, _, _ = atmos(height_array)
    velocity = calculate_orbital_velocity(height_array)
    dynamic_pressure = 0.5 * density * velocity**2

    dynamic_pressure_func = interp1d(dynamic_pressure, height_array, bounds_error=False, fill_value="extrapolate")
    velocity_func = interp1d(height_array, velocity, bounds_error=False, fill_value="extrapolate")
    density_func = interp1d(height_array, density, bounds_error=False, fill_value="extrapolate")
    temperature_func = interp1d(height_array, temperature, bounds_error=False, fill_value="extrapolate")
    return density_func, velocity_func, dynamic_pressure_func, temperature_func

def sample_atmosphere_at_height(height_km: float) -> AtmosphereSample:
    """Return atmosphere and orbit properties for one altitude in km."""
    density, temperature, r_specific, o2, n2, o = atmos(height_km)
    velocity = calculate_orbital_velocity(height_km)

    rho = float(density[0])
    temp = float(temperature[0])
    r_spec = float(r_specific[0])
    molar_mass = const.UNIVERSAL_GAS / max(r_spec, 1.0e-30)
    v_orb = float(velocity[0])
    q = 0.5 * rho * v_orb * v_orb

    return AtmosphereSample(
        height_km=float(height_km),
        density=rho,
        temperature=temp,
        specific_gas_constant=r_spec,
        molar_mass=molar_mass,
        o2_density=float(o2[0]),
        n2_density=float(n2[0]),
        o_density=float(o[0]),
        orbital_velocity=v_orb,
        dynamic_pressure=q,
    )

def orbit_updates_from_height(height_km: float) -> dict[str, float]:
    """Build orbit-state update payload from mission height in km."""
    sample = sample_atmosphere_at_height(height_km)
    return {
        "altitude": sample.height_km * 1000.0,
        "density": sample.density,
        "temperature": sample.temperature,
        "molar_mass": sample.molar_mass,
        "velocity": sample.orbital_velocity,
    }
