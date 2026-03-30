# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS - Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Atmosphere and orbital helper functions backed by ``pymsis``.
#
#  Project:        ARISS
#  Module:         atmosphere.py
#  Author:         Carlos Carrasco Requejo, Lucas Calderon del Rio
# ============================================================================

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pymsis as msis

from ariss.utils import constants as const


# --------------------------------------------------------------------------------------
# MSIS defaults
# --------------------------------------------------------------------------------------
MSIS_REFERENCE_DATE = np.datetime64("2000-01-01T00:00:00")
MSIS_F107 = 140.0
MSIS_AP = 15.0
MSIS_LATITUDE = 0.0
MSIS_LONGITUDE = 0.0
MSIS_AVERAGE_LATITUDES = np.linspace(-90.0, 90.0, 5, dtype=float)
MSIS_AVERAGE_LONGITUDES = np.linspace(-180.0, 180.0, 5, endpoint=False, dtype=float)

MSISResolvedInputs = tuple[np.datetime64, float, float, float, float, bool]
AtmosphericProperties = dict[str, np.ndarray | float]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------
def _run_msis(heights_km: np.ndarray, resolved: MSISResolvedInputs, notprint: bool = True) -> np.ndarray:
    msis_date, f107_value, ap_value, latitude_value, longitude_value, use_average_value = resolved
    use_pymsis_default_activity = np.isclose(f107_value, 0.0) and np.isclose(ap_value, 0.0)

    def activity_kwargs(count: int) -> dict[str, np.ndarray]:
        return {
            "f107s": np.full(count, f107_value, dtype=float),
            "f107as": np.full(count, f107_value, dtype=float),
            "aps": np.full((count, 7), ap_value, dtype=float),
        }

    if use_average_value:
        n = len(heights_km)
        lat_grid, lon_grid = np.meshgrid(MSIS_AVERAGE_LATITUDES, MSIS_AVERAGE_LONGITUDES, indexing="ij")
        lat_samples = lat_grid.ravel()
        lon_samples = lon_grid.ravel()
        sample_count = len(lat_samples)

        weights = np.repeat(np.cos(np.deg2rad(MSIS_AVERAGE_LATITUDES)), len(MSIS_AVERAGE_LONGITUDES))
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.sum(weights)

        total = n * sample_count
        et = np.full(total, msis_date)
        lats = np.tile(lat_samples, n)
        lons = np.tile(lon_samples, n)
        heights = np.repeat(heights_km, sample_count)

        if use_pymsis_default_activity:
            if not notprint:
                print(
                    f"[ARISS] Running pymsis average atmosphere: altitudes={n}, "
                    f"lat_samples={len(MSIS_AVERAGE_LATITUDES)}, lon_samples={len(MSIS_AVERAGE_LONGITUDES)}, "
                    "F10.7/Ap=pymsis-default"
                )
            composition = msis.calculate(et, lons, lats, heights)
        else:
            if not notprint:
                print(
                    f"[ARISS] Running pymsis average atmosphere: altitudes={n}, "
                    f"lat_samples={len(MSIS_AVERAGE_LATITUDES)}, lon_samples={len(MSIS_AVERAGE_LONGITUDES)}, "
                    f"F10.7={f107_value:.1f}, Ap={ap_value:.1f}"
                )
            composition = msis.calculate(et, lons, lats, heights, **activity_kwargs(total))

        composition = np.nan_to_num(composition).reshape(n, sample_count, -1)
        return np.sum(composition * weights[None, :, None], axis=1)

    n = len(heights_km)
    et = np.full(n, msis_date)
    lats = np.full(n, latitude_value, dtype=float)
    lons = np.full(n, longitude_value, dtype=float)

    if use_pymsis_default_activity:
        if not notprint:
            print(
                f"[ARISS] Running pymsis point atmosphere: altitudes={n}, "
                f"lat={latitude_value:.3f}, lon={longitude_value:.3f}, "
                "F10.7/Ap=pymsis-default"
            )
        composition = msis.calculate(et, lons, lats, heights_km)
    else:
        if not notprint:
            print(
                f"[ARISS] Running pymsis point atmosphere: altitudes={n}, "
                f"lat={latitude_value:.3f}, lon={longitude_value:.3f}, "
                f"F10.7={f107_value:.1f}, Ap={ap_value:.1f}"
            )
        composition = msis.calculate(et, lons, lats, heights_km, **activity_kwargs(n))

    return np.nan_to_num(composition)


def _to_scalar_if_needed(properties: dict[str, np.ndarray], is_scalar_input: bool) -> AtmosphericProperties:
    if not is_scalar_input:
        return properties
    return {key: float(value[0]) for key, value in properties.items()}


@lru_cache(maxsize=32)
def _cached_density_profile(
    height_min_km: float,
    height_max_km: float,
    samples: int,
    msis_date: str,
    msis_f107: float,
    msis_ap: float,
    latitude: float,
    longitude: float,
    use_average: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height_profile = np.linspace(height_min_km, height_max_km, int(samples), dtype=float)
    properties = atmospheric_properties_from_height(
        height_profile,
        msis_date=msis_date,
        msis_f107=msis_f107,
        msis_ap=msis_ap,
        latitude=latitude,
        longitude=longitude,
        use_average=use_average,
    )
    return (
        np.asarray(properties["altitude_km"], dtype=float),
        np.maximum(np.asarray(properties["density"], dtype=float), 1.0e-30),
    )


# --------------------------------------------------------------------------------------
# Public atmosphere API
# --------------------------------------------------------------------------------------
def atmospheric_properties_from_height(
    height_array_km: np.ndarray | float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
    notprint: bool = True,
) -> AtmosphericProperties:
    heights_km = np.asarray(height_array_km, dtype=float)
    is_scalar_input = heights_km.ndim == 0
    if is_scalar_input:
        heights_km = heights_km.reshape(1)

    # Inputs are validated upstream (SpacecraftState.check_bounds).
    date_value = np.datetime64(msis_date) if msis_date is not None else MSIS_REFERENCE_DATE
    f107_value = float(MSIS_F107 if msis_f107 is None else msis_f107)
    ap_value = float(MSIS_AP if msis_ap is None else msis_ap)
    latitude_value = float(MSIS_LATITUDE if latitude is None else latitude)
    longitude_value = float(MSIS_LONGITUDE if longitude is None else longitude)
    use_average_value = bool(False if use_average is None else use_average)

    composition = _run_msis(
        heights_km,
        (date_value, f107_value, ap_value, latitude_value, longitude_value, use_average_value),
        notprint=bool(notprint),
    )

    density = np.asarray(composition[:, msis.Variable.MASS_DENSITY], dtype=float)
    n2_number_density = np.asarray(composition[:, msis.Variable.N2], dtype=float)
    o2_number_density = np.asarray(composition[:, msis.Variable.O2], dtype=float)
    o_number_density = np.asarray(composition[:, msis.Variable.O], dtype=float)
    temperature = np.asarray(composition[:, msis.Variable.TEMPERATURE], dtype=float)

    total_number_density = np.maximum(o_number_density + n2_number_density + o2_number_density, 1.0e-30)
    molar_mass = (
        (o_number_density * 15.999e-3)
        + (n2_number_density * 28.0134e-3)
        + (o2_number_density * 31.9988e-3)
    ) / total_number_density

    o2_density = o2_number_density * (31.9988e-3 / const.AVOGADRO_NUMBER)
    n2_density = n2_number_density * (28.0134e-3 / const.AVOGADRO_NUMBER)
    o_density = o_number_density * (15.999e-3 / const.AVOGADRO_NUMBER)
    specific_gas_constant = const.UNIVERSAL_GAS / np.maximum(molar_mass, 1.0e-30)
    orbital_velocity = np.sqrt(const.EARTH_MU / (const.EARTH_RADIUS * 1000.0 + heights_km * 1000.0))
    dynamic_pressure = 0.5 * density * orbital_velocity * orbital_velocity
    pressure = density * specific_gas_constant * temperature

    properties = {
        "altitude_km": np.asarray(heights_km, dtype=float),
        "altitude": np.asarray(heights_km, dtype=float),
        "density": np.asarray(density, dtype=float),
        "temperature": np.asarray(temperature, dtype=float),
        "specific_gas_constant": np.asarray(specific_gas_constant, dtype=float),
        "R_spec": np.asarray(specific_gas_constant, dtype=float),
        "molar_mass": np.asarray(molar_mass, dtype=float),
        "o2_density": np.asarray(o2_density, dtype=float),
        "n2_density": np.asarray(n2_density, dtype=float),
        "o_density": np.asarray(o_density, dtype=float),
        "orbital_velocity": np.asarray(orbital_velocity, dtype=float),
        "velocity": np.asarray(orbital_velocity, dtype=float),
        "dynamic_pressure": np.asarray(dynamic_pressure, dtype=float),
        "pressure": np.asarray(pressure, dtype=float),
    }
    return _to_scalar_if_needed(properties, is_scalar_input)


def atmosphere_properties_from_density(
    target_density: np.ndarray | float,
    msis_date: str | np.datetime64 | None = None,
    msis_f107: float | None = None,
    msis_ap: float | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    use_average: bool | None = None,
    notprint: bool = True,
) -> AtmosphericProperties:
    target_density_array = np.asarray(target_density, dtype=float)
    is_scalar_input = target_density_array.ndim == 0
    if is_scalar_input:
        target_density_array = target_density_array.reshape(1)

    target_density_safe = np.nan_to_num(
        target_density_array,
        nan=1.0e-30,
        posinf=1.0e30,
        neginf=1.0e-30,
    )
    target_density_safe = np.maximum(target_density_safe, 1.0e-30)

    # Inputs are validated upstream (SpacecraftState.check_bounds).
    date_value = np.datetime64(msis_date) if msis_date is not None else MSIS_REFERENCE_DATE
    f107_value = float(MSIS_F107 if msis_f107 is None else msis_f107)
    ap_value = float(MSIS_AP if msis_ap is None else msis_ap)
    latitude_value = float(MSIS_LATITUDE if latitude is None else latitude)
    longitude_value = float(MSIS_LONGITUDE if longitude is None else longitude)
    use_average_value = bool(False if use_average is None else use_average)

    height_profile, density_profile = _cached_density_profile(
        80.0,
        1000.0,
        5000,
        str(date_value),
        f107_value,
        ap_value,
        latitude_value,
        longitude_value,
        use_average_value,
    )

    sort_idx = np.argsort(density_profile)
    density_sorted = density_profile[sort_idx]
    height_sorted = height_profile[sort_idx]
    target_density_clipped = np.clip(target_density_safe, density_sorted[0], density_sorted[-1])
    height_from_density = np.interp(np.log(target_density_clipped), np.log(density_sorted), height_sorted)

    properties = atmospheric_properties_from_height(
        np.asarray(height_from_density, dtype=float),
        msis_date=str(date_value),
        msis_f107=f107_value,
        msis_ap=ap_value,
        latitude=latitude_value,
        longitude=longitude_value,
        use_average=use_average_value,
        notprint=notprint,
    )

    properties_arrays = {key: np.atleast_1d(np.asarray(value, dtype=float)) for key, value in properties.items()}
    properties_arrays["model_density"] = np.asarray(properties_arrays["density"], dtype=float).copy()
    properties_arrays["density"] = np.asarray(target_density_array, dtype=float)

    return _to_scalar_if_needed(properties_arrays, is_scalar_input)
