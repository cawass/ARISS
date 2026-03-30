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
#      Drag-only Fig. 6 recreation for Crandall and Wirz (2022) using 3U/6U cases
#      and digitized CSV datasets, styled with the shared validation plot format.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Validation.py
# ============================================================================== #

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np


# ------------------------------------------------------------------------------ #
# Path setup
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for p in (SRC, VALIDATION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _panel_front_area, _side_areas
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmospheric_properties_from_height
from ariss.utils.ploting import plot_validation_crandall_fig6_drag
from csv_helper import load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats, minimum_finite


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CASE_SPECS = [
    {
        "title": "1 x 3U CubeSat",
        "config": HERE / "CrandallWirz2022_3U.toml",
        "dataset": HERE / "3U Drag.csv",
    },
    {
        "title": "1 x 6U CubeSat",
        "config": HERE / "CrandallWirz2022_6U.toml",
        "dataset": HERE / "6U Drag.csv",
    },
]

OUTPUT_PATH = HERE / "crandall_wirz_2022_fig6_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

ALTITUDE_SAMPLES = 260
NEWTON_TO_MILLINEWTON = 1.0e3

COMPONENT_SPECS = [
    ("Total Drag", "drag_total", PALETTE["primary_text"], "o"),
    ("Frontal Area Drag", "drag_inlet_front", PALETTE["choice_mid"], "s"),
    ("SA Skin Friction Drag", "drag_solar", PALETTE["l1_teal"], "^"),
    ("Body Skin Friction Drag", "drag_body_side", PALETTE["sernn_pink"], "D"),
    ("SA Frontal Area Drag", "drag_solar_front", PALETTE["goal_dark"], "v"),
]


# ------------------------------------------------------------------------------ #
# Dataset loading
# ------------------------------------------------------------------------------ #

def _load_digitized_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return load_wide_xy_csv(
        path,
        sort_by="y",
        min_rows=3,
        label_transform=lambda label: (
            "SA Skin Friction Drag" if str(label).strip() == "SA Skin Friction" else str(label).strip()
        ),
    )


def _normalize_altitude_orientation(
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], bool]:
    if "Total Drag" not in dataset:
        return dataset, False

    x_total, y_total = dataset["Total Drag"]
    if len(x_total) < 3:
        return dataset, False

    correlation = float(np.corrcoef(x_total, y_total)[0, 1])
    if not np.isfinite(correlation) or correlation <= 0.0:
        return dataset, False

    y_all = np.concatenate([values[1] for values in dataset.values()])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (x_vals, y_vals) in dataset.items():
        y_flip = y_min + y_max - y_vals
        order = np.argsort(y_flip)
        normalized[label] = (x_vals[order], y_flip[order])
    return normalized, True


def _dataset_altitude_bounds(dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    all_y = []
    for label, _, _, _ in COMPONENT_SPECS:
        if label in dataset:
            all_y.extend(dataset[label][1].tolist())
    if not all_y:
        raise ValueError("No altitude data found in dataset for expected components.")
    return float(np.min(all_y)), float(np.max(all_y))


# ------------------------------------------------------------------------------ #
# Drag-only sweep
# ------------------------------------------------------------------------------ #

def _set_drag_force_outputs(sc) -> None:
    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    solar_front_area = _panel_front_area(sc.geometry.A_solar, sc.geometry.AR_solar, sc.geometry.t_solar)
    rad_front_area = _panel_front_area(sc.geometry.A_rad, sc.geometry.AR_rad, sc.geometry.t_rad)
    q_inf = 0.5 * sc.orbit.density * sc.orbit.velocity**2

    sc.drag.drag_solar = q_inf * sc.drag.cd_solar * sc.geometry.A_solar
    sc.drag.drag_solar_front = q_inf * sc.drag.cd_solar_front * solar_front_area
    sc.drag.drag_rad = q_inf * sc.drag.cd_rad * sc.geometry.A_rad
    sc.drag.drag_rad_front = q_inf * sc.drag.cd_rad_front * rad_front_area
    sc.drag.drag_body_side = q_inf * sc.drag.cd_body_side * body_side_area
    sc.drag.drag_inlet_side = q_inf * sc.drag.cd_inlet_side * inlet_side_area
    sc.drag.drag_inlet_front = q_inf * sc.drag.cd_inlet_front * sc.geometry.A_in_drag

    sc.drag.drag_total = (
        sc.drag.drag_solar
        + sc.drag.drag_solar_front
        + sc.drag.drag_rad
        + sc.drag.drag_rad_front
        + sc.drag.drag_body_side
        + sc.drag.drag_inlet_side
        + sc.drag.drag_inlet_front
    )


def _run_drag_sweep(config_path: Path, altitudes_km: np.ndarray) -> dict[str, np.ndarray]:
    base_sc = load_spacecraft_from_base_config(config_path)
    base_sc.geometry.A_in_drag = base_sc.geometry.A_in

    properties = atmospheric_properties_from_height(
        altitudes_km,
        msis_date=base_sc.orbit.msis_date,
        msis_f107=base_sc.orbit.msis_f107,
        msis_ap=base_sc.orbit.msis_ap,
        latitude=base_sc.orbit.latitude,
        longitude=base_sc.orbit.longitude,
        use_average=base_sc.orbit.use_average,
    )
    density = np.asarray(properties["density"], dtype=float)
    temperature = np.asarray(properties["temperature"], dtype=float)
    velocity = np.asarray(properties["orbital_velocity"], dtype=float)
    molar_mass = np.asarray(properties["molar_mass"], dtype=float)

    output = {
        "altitude": np.asarray(altitudes_km, dtype=float),
        "drag_total": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_inlet_front": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_solar": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_body_side": np.full_like(altitudes_km, np.nan, dtype=float),
        "drag_solar_front": np.full_like(altitudes_km, np.nan, dtype=float),
    }

    for i, altitude in enumerate(altitudes_km):
        sc = deepcopy(base_sc)
        sc.orbit.altitude = float(altitude)
        sc.orbit.density = float(density[i])
        sc.orbit.temperature = float(temperature[i])
        sc.orbit.molar_mass = float(molar_mass[i])
        sc.orbit.velocity = float(velocity[i])

        drag_model(sc)
        _set_drag_force_outputs(sc)

        output["drag_total"][i] = float(sc.drag.drag_total) * NEWTON_TO_MILLINEWTON
        output["drag_inlet_front"][i] = float(sc.drag.drag_inlet_front) * NEWTON_TO_MILLINEWTON
        output["drag_solar"][i] = float(sc.drag.drag_solar) * NEWTON_TO_MILLINEWTON
        output["drag_body_side"][i] = float(sc.drag.drag_body_side) * NEWTON_TO_MILLINEWTON
        output["drag_solar_front"][i] = float(sc.drag.drag_solar_front) * NEWTON_TO_MILLINEWTON

    return output


# ------------------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------------------ #


def run_crandall_wirz_validation(show: bool = True) -> Path:
    datasets: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for spec in CASE_SPECS:
        dataset = _load_digitized_dataset(spec["dataset"])
        dataset, flipped = _normalize_altitude_orientation(dataset)
        if flipped:
            print(f"[info] Mirrored altitude axis detected and corrected for dataset: {spec['dataset'].name}")
        datasets.append(dataset)
    altitude_min = min(_dataset_altitude_bounds(dataset)[0] for dataset in datasets)
    altitude_max = max(_dataset_altitude_bounds(dataset)[1] for dataset in datasets)
    altitudes_km = np.linspace(altitude_min, altitude_max, ALTITUDE_SAMPLES)

    models = [_run_drag_sweep(spec["config"], altitudes_km) for spec in CASE_SPECS]
    for spec, model, dataset in zip(CASE_SPECS, models, datasets):
        metrics: dict[str, tuple[float, float, int, int, float, int]] = {}
        for label, key, _, _ in COMPONENT_SPECS:
            if label not in dataset:
                continue
            x_ref, y_ref = dataset[label]
            stats_for_component = datapoint_relative_and_corr_stats(
                np.asarray(model["altitude"], dtype=float),
                np.asarray(model[key], dtype=float),
                np.asarray(y_ref, dtype=float),
                np.asarray(x_ref, dtype=float),
            )
            if stats_for_component is not None:
                metrics[label] = stats_for_component

        print(f"\n{spec['title']} relative-error and correlation by component:")
        pearson_values: list[float] = []
        for label, _, _, _ in COMPONENT_SPECS:
            if label in metrics:
                max_rel_error, mean_rel_error, line_max_rel, n_rel, pearson_r, n_corr = metrics[label]
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  {label:<24} "
                    f"max_relative_error={max_rel_error:10.6f} ({100.0 * max_rel_error:7.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel_error:10.6f} ({100.0 * mean_rel_error:7.3f}%), "
                    f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values.append(pearson_r)
        min_pearson = minimum_finite(pearson_values)
        if min_pearson is not None:
            print(f"  Minimum Pearson correlation coefficient: {min_pearson:.6f}")
        else:
            print("  Minimum Pearson correlation coefficient: n/a")

    output = plot_validation_crandall_fig6_drag(
        models,
        datasets,
        case_specs=CASE_SPECS,
        component_specs=COMPONENT_SPECS,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        output_path=OUTPUT_PATH,
        page_figsize=PAGE_FIGSIZE,
        show=show,
    )
    print(f"\nSaved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_validation(show=True)

