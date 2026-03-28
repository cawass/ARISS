"""Atmosphere and orbital helpers backed by ``pymsis``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from ariss.utils import constants as const

try:
    import pymsis as msis  # type: ignore
except Exception:  # pragma: no cover - optional import path
    msis = None

MSIS_REFERENCE_DATE = np.datetime64("2000-01-01T00:00:00")
MSIS_F107 = 140.0
MSIS_AP = 15.0
MSIS_LATITUDE = 0.0
MSIS_LONGITUDE = 0.0
MSIS_AVERAGE_LATITUDES = np.linspace(-90.0, 90.0, 5, dtype=float)
MSIS_AVERAGE_LONGITUDES = np.linspace(-180.0, 180.0, 5, endpoint=False, dtype=float)

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


def _species_mass_density(number_density: np.ndarray, molar_mass: float) -> np.ndarray:
    molecule_mass = molar_mass / const.AVOGADRO_NUMBER
    return np.asarray(number_density, dtype=float) * molecule_mass


def _resolve_msis_date(msis_date: str | np.datetime64 | None) -> np.datetime64:
    if msis_date is None:
        return MSIS_REFERENCE_DATE
    try:
        return np.datetime64(msis_date)
    except Exception:
        return MSIS_REFERENCE_DATE


def _resolve_msis_scalar(value: float | None, default: float) -> float:
    if value is None:
        return default
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if np.isfinite(resolved) else default


def _resolve_msis_latitude(value: float | None) -> float:
    return float(np.clip(_resolve_msis_scalar(value, MSIS_LATITUDE), -90.0, 90.0))


def _resolve_msis_longitude(value: float | None) -> float:
    resolved = _resolve_msis_scalar(value, MSIS_LONGITUDE)
    return float(((resolved + 180.0) % 360.0) - 180.0)


def _resolve_msis_bool(value: bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolved_msis_inputs(
    msis_date: str | np.datetime64 | None,
    msis_f107: float | None,
    msis_ap: float | None,
    latitude: float | None,
    longitude: float | None,
    use_average: bool | None,
) -> tuple[str, float, float, float, float, bool]:
    return (
        str(_resolve_msis_date(msis_date)),
        _resolve_msis_scalar(msis_f107, MSIS_F107),
        _resolve_msis_scalar(msis_ap, MSIS_AP),
        _resolve_msis_latitude(latitude),
        _resolve_msis_longitude(longitude),
        _resolve_msis_bool(use_average, False),
    )

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

    def to_orbit_updates(self) -> dict[str, float]:
        return {
            "altitude": self.height_km,
            "density": self.density,
            "temperature": self.temperature,
            "molar_mass": self.molar_mass,
            "velocity": self.orbital_velocity,
            "R_spec": self.specific_gas_constant,
        }

    def to_properties(self) -> dict[str, float]:
        return {
            "altitude_km": self.height_km,
            "density": self.density,
            "temperature": self.temperature,
            "specific_gas_constant": self.specific_gas_constant,
            "molar_mass": self.molar_mass,
            "o2_density": self.o2_density,
            "n2_density": self.n2_density,
            "o_density": self.o_density,
            "orbital_velocity": self.orbital_velocity,
            "dynamic_pressure": self.dynamic_pressure,
        }


@lru_cache(maxsize=32)
def _cached_profile(
    height_min_km: float,
    height_max_km: float,
    samples: int,
    msis_date: str,
    msis_f107: float,
    msis_ap: float,
    latitude: float,
    longitude: float,
    use_average: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height_array = np.linspace(height_min_km, height_max_km, int(samples), dtype=float)
    density, temperature, r_specific, o2_density, n2_density, o_density = atmos(
        height_array,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    return (
        np.asarray(height_array, dtype=float),
        np.asarray(density, dtype=float),
        np.asarray(temperature, dtype=float),
        np.asarray(r_specific, dtype=float),
        np.asarray(o2_density, dtype=float),
        np.asarray(n2_density, dtype=float),
        np.asarray(o_density, dtype=float),
    )


def _profile_arrays(
    height_min_km: float,
    height_max_km: float,
    samples: int,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved = _resolved_msis_inputs(msis_date, msis_f107, msis_ap, latitude, longitude, use_average)
    return _cached_profile(
        float(height_min_km),
        float(height_max_km),
        int(samples),
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
        resolved[5],
    )


def _sample_from_profile(
    height_km: float,
    height_array: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    r_specific: np.ndarray,
    o2_density: np.ndarray,
    n2_density: np.ndarray,
    o_density: np.ndarray,
) -> AtmosphereSample:
    rho = float(np.interp(height_km, height_array, density))
    temp = float(np.interp(height_km, height_array, temperature))
    r_spec = float(np.interp(height_km, height_array, r_specific))
    o2 = float(np.interp(height_km, height_array, o2_density))
    n2 = float(np.interp(height_km, height_array, n2_density))
    o = float(np.interp(height_km, height_array, o_density))
    molar_mass = const.UNIVERSAL_GAS / max(r_spec, 1.0e-30)
    v_orb = float(calculate_orbital_velocity(height_km)[0])
    q = 0.5 * rho * v_orb * v_orb
    return AtmosphereSample(
        height_km=float(height_km),
        density=rho,
        temperature=temp,
        specific_gas_constant=r_spec,
        molar_mass=molar_mass,
        o2_density=o2,
        n2_density=n2,
        o_density=o,
        orbital_velocity=v_orb,
        dynamic_pressure=q,
    )


def atmos(
    height_array_km: np.ndarray | float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``rho, T, R_specific, O2, N2, O`` with species densities in ``kg/m^3``."""
    _require_pymsis()
    heights_km = _as_1d_float_array(height_array_km)
    n = len(heights_km)
    date_value = _resolve_msis_date(msis_date)
    f107_value = _resolve_msis_scalar(msis_f107, MSIS_F107)
    ap_value = _resolve_msis_scalar(msis_ap, MSIS_AP)
    latitude_value = _resolve_msis_latitude(latitude)
    longitude_value = _resolve_msis_longitude(longitude)
    use_average_value = _resolve_msis_bool(use_average, False)

    if use_average_value:
        lat_grid, lon_grid = np.meshgrid(MSIS_AVERAGE_LATITUDES, MSIS_AVERAGE_LONGITUDES, indexing="ij")
        lat_samples = lat_grid.ravel()
        lon_samples = lon_grid.ravel()
        sample_count = len(lat_samples)
        weights = np.repeat(np.cos(np.deg2rad(MSIS_AVERAGE_LATITUDES)), len(MSIS_AVERAGE_LONGITUDES))
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.sum(weights)

        total = n * sample_count
        et = np.array([date_value for _ in range(total)])
        lats = np.tile(lat_samples, n)
        lons = np.tile(lon_samples, n)
        heights = np.repeat(heights_km, sample_count)
        f107 = np.full(total, f107_value, dtype=float)
        f107a = np.full(total, f107_value, dtype=float)
        ap = np.full((total, 7), ap_value, dtype=float)

        print(
            f"[ARISS] Running pymsis average atmosphere: altitudes={n}, "
            f"lat_samples={len(MSIS_AVERAGE_LATITUDES)}, lon_samples={len(MSIS_AVERAGE_LONGITUDES)}, "
            f"F10.7={f107_value:.1f}, Ap={ap_value:.1f}"
        )
        composition = msis.calculate(et, lons, lats, heights, f107s=f107, f107as=f107a, aps=ap)
        composition = np.nan_to_num(composition).reshape(n, sample_count, -1)
        composition = np.sum(composition * weights[None, :, None], axis=1)
    else:
        et = np.array([date_value for _ in range(n)])
        lons = np.full(n, longitude_value, dtype=float)
        lats = np.full(n, latitude_value, dtype=float)
        f107 = np.full(n, f107_value, dtype=float)
        f107a = np.full(n, f107_value, dtype=float)
        ap = np.full((n, 7), ap_value, dtype=float)
        print(
            f"[ARISS] Running pymsis point atmosphere: altitudes={n}, "
            f"lat={latitude_value:.3f}, lon={longitude_value:.3f}, "
            f"F10.7={f107_value:.1f}, Ap={ap_value:.1f}"
        )
        composition = msis.calculate(et, lons, lats, heights_km, f107s=f107, f107as=f107a, aps=ap)
        composition = np.nan_to_num(composition)

    rho = composition[:, msis.Variable.MASS_DENSITY]
    n2_number_density = composition[:, msis.Variable.N2]
    o2_number_density = composition[:, msis.Variable.O2]
    o_number_density = composition[:, msis.Variable.O]
    temperature = composition[:, msis.Variable.TEMPERATURE]

    total_number_density = np.maximum(o_number_density + n2_number_density + o2_number_density, 1.0e-30)
    molar_mass = (
        (o_number_density * 15.999e-3)
        + (n2_number_density * 28.0134e-3)
        + (o2_number_density * 31.9988e-3)
    ) / total_number_density
    o2_density = _species_mass_density(o2_number_density, 31.9988e-3)
    n2_density = _species_mass_density(n2_number_density, 28.0134e-3)
    o_density = _species_mass_density(o_number_density, 15.999e-3)
    r_specific = const.UNIVERSAL_GAS / np.maximum(molar_mass, 1.0e-30)
    return rho, temperature, r_specific, o2_density, n2_density, o_density

def calculate_orbital_velocity(height_array_km: np.ndarray | float) -> np.ndarray:
    """Circular-orbit velocity at altitude in km."""
    heights_km = _as_1d_float_array(height_array_km)
    return np.sqrt(const.EARTH_MU / (const.EARTH_RADIUS * 1000.0 + heights_km * 1000.0))

def get_atmosphere_functions(
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 10000,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return interpolation functions for density, velocity, dynamic pressure, temperature."""
    height_array, density, temperature, _, _, _, _ = _profile_arrays(
        height_min_km,
        height_max_km,
        samples,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    velocity = calculate_orbital_velocity(height_array)
    dynamic_pressure = 0.5 * density * velocity**2

    dynamic_pressure_func = interp1d(dynamic_pressure, height_array, bounds_error=False, fill_value="extrapolate")
    velocity_func = interp1d(height_array, velocity, bounds_error=False, fill_value="extrapolate")
    density_func = interp1d(height_array, density, bounds_error=False, fill_value="extrapolate")
    temperature_func = interp1d(height_array, temperature, bounds_error=False, fill_value="extrapolate")
    return density_func, velocity_func, dynamic_pressure_func, temperature_func

def sample_atmosphere_at_height(
    height_km: float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> AtmosphereSample:
    """Return atmosphere and orbit properties for one altitude in km."""
    density, temperature, r_specific, o2, n2, o = atmos(
        height_km,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
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

def orbit_updates_from_height(
    height_km: float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> dict[str, float]:
    """Build orbit-state update payload from mission height in km."""
    sample = sample_atmosphere_at_height(
        height_km,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    return sample.to_orbit_updates()


def atmosphere_properties_from_height(
    height_km: float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> dict[str, float]:
    """Return a serializable full-property atmosphere payload for one altitude in km."""
    sample = sample_atmosphere_at_height(
        height_km,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    return sample.to_properties()


def height_from_density(
    target_density: float,
    height_min_km: float = 80.0,
    height_max_km: float = 1000.0,
    samples: int = 5000,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> float:
    """Estimate altitude [km] for a target density [kg/m^3] via interpolation."""
    _require_pymsis()
    target = float(np.nan_to_num(float(target_density), nan=1.0e-30, posinf=1.0e30, neginf=1.0e-30))
    target = max(target, 1.0e-30)

    height_array, density, _, _, _, _, _ = _profile_arrays(
        height_min_km,
        height_max_km,
        samples,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    density = np.maximum(np.asarray(density, dtype=float), 1.0e-30)

    # Build monotonic interpolation domain in log-density space.
    sort_idx = np.argsort(density)
    density_sorted = density[sort_idx]
    height_sorted = height_array[sort_idx]

    target = float(np.clip(target, density_sorted[0], density_sorted[-1]))
    return float(
        np.interp(
            np.log(target),
            np.log(density_sorted),
            height_sorted,
        )
    )


def orbit_updates_from_density(
    target_density: float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> dict[str, float]:
    """Build orbit-state update payload from target density [kg/m^3]."""
    height_km = height_from_density(
        target_density,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    height_array, density, temperature, r_specific, o2_density, n2_density, o_density = _profile_arrays(
        80.0,
        1000.0,
        5000,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    sample = _sample_from_profile(
        height_km,
        height_array,
        density,
        temperature,
        r_specific,
        o2_density,
        n2_density,
        o_density,
    )
    updates = sample.to_orbit_updates()
    target = float(target_density)
    if np.isfinite(target) and target > 0.0:
        updates["density"] = target
    return updates


def atmosphere_properties_from_density(
    target_density: float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
) -> dict[str, float]:
    """Return a full-property atmosphere payload for a target density [kg/m^3]."""
    height_km = height_from_density(
        target_density,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    height_array, density, temperature, r_specific, o2_density, n2_density, o_density = _profile_arrays(
        80.0,
        1000.0,
        5000,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    sample = _sample_from_profile(
        height_km,
        height_array,
        density,
        temperature,
        r_specific,
        o2_density,
        n2_density,
        o_density,
    )
    properties = sample.to_properties()
    properties["model_density"] = properties["density"]
    properties["density"] = float(target_density)
    return properties
