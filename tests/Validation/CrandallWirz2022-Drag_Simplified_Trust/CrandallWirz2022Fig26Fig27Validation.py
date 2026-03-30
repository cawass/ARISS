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
#      Side-by-side recreation of Crandall & Wirz (2022) Fig. 26 and Fig. 27.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Fig26Fig27Validation.py
# ============================================================================== #

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import replace
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

from ariss.core.simulation import load_spacecraft_from_base_config, logger as simulation_logger, run_sizing_loop
from ariss.utils import constants as const
from ariss.utils.ploting import plot_validation_crandall_fig26_fig27
from csv_helper import load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats_xy, minimum_finite


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CONFIG_PATH = HERE / "CrandallWirz2022_6U.toml"
DATASET_SOLAR_EFF_PATH = HERE / "Solar Cell Eff.csv"
DATASET_ACC_COEFF_PATH = HERE / "Acc Coeff.csv"
OUTPUT_PATH = HERE / "crandall_wirz_2022_fig26_fig27_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

TP_TARGET_MN_PER_KW = 10.0
TP_TARGET_N_PER_W = TP_TARGET_MN_PER_KW * 1.0e-6

MAX_ITERATIONS = 260
MASS_TOLERANCE = 1.0e-3

SOLAR_CASES = [
    {"label": "Solar Maximum", "f107": 200.0, "color": PALETTE["choice_mid"], "marker": "s"},
    {"label": "Mean Solar Activity", "f107": 114.0, "color": PALETTE["l1_teal"], "marker": "^"},
    {"label": "Solar Minimum", "f107": 62.0, "color": PALETTE["sernn_pink"], "marker": "D"},
]


# ------------------------------------------------------------------------------ #
# Dataset loading
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return load_wide_xy_csv(path, sort_by="x", min_rows=3)


def _remap_accommodation_dataset_to_local_convention(
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert paper x-values to local convention: 0=diffusive, 1=specular."""
    remapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, (x_vals, y_vals) in dataset.items():
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        if not np.any(finite):
            remapped[label] = (x_arr.copy(), y_arr.copy())
            continue
        x_local = 1.0 - x_arr[finite]
        y_local = y_arr[finite]
        order = np.argsort(x_local)
        remapped[label] = (x_local[order], y_local[order])
    return remapped


def _case_x_grid_from_dataset(
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
    label: str,
) -> np.ndarray:
    """Return sorted unique x-points for a case, matching paper points exactly."""
    if label not in dataset:
        raise ValueError(f"Missing paper dataset series for case: {label}")

    x_vals, _ = dataset[label]
    finite = np.isfinite(x_vals)
    if not np.any(finite):
        raise ValueError(f"No finite x-values found in paper dataset series: {label}")

    return np.unique(np.sort(np.asarray(x_vals[finite], dtype=float)))


# ------------------------------------------------------------------------------ #
# Model runs
# ------------------------------------------------------------------------------ #

def _solve_point(spacecraft, *, f107: float, solar_eff: float | None = None, acc_coeff: float | None = None) -> tuple[float, float]:
    spacecraft.orbit.msis_f107 = float(f107)
    spacecraft.geometry.use_intake_area_ratio = False

    # Keep T/P fixed at the paper condition for Fig. 26/27.
    spacecraft.thruster.specific_impulse = 2.0 * spacecraft.thruster.eff / (const.EARTH_GRAVITY * TP_TARGET_N_PER_W)
    if solar_eff is not None:
        spacecraft = replace(spacecraft, solar=replace(spacecraft.solar, eta_solar=float(solar_eff)))

    if acc_coeff is not None:
        # Use the project convention directly:
        #   0 -> diffusive, 1 -> specular.
        epsilon_kernel = float(np.clip(acc_coeff, 0.0, 1.0))
        spacecraft.geometry.epsilon_body = epsilon_kernel
        spacecraft.geometry.epsilon_solar = epsilon_kernel
        spacecraft.geometry.epsilon_rad = epsilon_kernel
        spacecraft.geometry.epsilon_in = epsilon_kernel
        spacecraft.geometry.epsilon_in_norm = epsilon_kernel

    with redirect_stdout(io.StringIO()):
        final_sc, _, _ = run_sizing_loop(
            spacecraft,
            max_iterations=MAX_ITERATIONS,
            mass_tolerance=MASS_TOLERANCE,
        )

    tp_value = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)
    altitude_value = float(final_sc.orbit.altitude)
    return tp_value, altitude_value


def run_solar_efficiency_sweep(dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in SOLAR_CASES:
        sc = load_spacecraft_from_base_config(CONFIG_PATH)
        x_grid = _case_x_grid_from_dataset(dataset, case["label"])
        y_values = []
        for eta_solar in x_grid:
            _, altitude = _solve_point(sc, f107=case["f107"], solar_eff=float(eta_solar))
            y_values.append(altitude)
        results[case["label"]] = (x_grid.copy(), np.asarray(y_values, dtype=float))
    return results


def run_accommodation_sweep(dataset: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in SOLAR_CASES:
        sc = load_spacecraft_from_base_config(CONFIG_PATH)
        x_grid = _case_x_grid_from_dataset(dataset, case["label"])
        y_values = []
        for acc in x_grid:
            _, altitude = _solve_point(sc, f107=case["f107"], acc_coeff=float(acc))
            y_values.append(altitude)
        results[case["label"]] = (x_grid.copy(), np.asarray(y_values, dtype=float))
    return results


def run_crandall_wirz_fig26_fig27_validation(show: bool = True) -> Path:
    fig26_dataset = load_wide_xy_dataset(DATASET_SOLAR_EFF_PATH)
    fig27_dataset = _remap_accommodation_dataset_to_local_convention(
        load_wide_xy_dataset(DATASET_ACC_COEFF_PATH)
    )

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)
    try:
        fig26_model = run_solar_efficiency_sweep(fig26_dataset)
        fig27_model = run_accommodation_sweep(fig27_dataset)
    finally:
        simulation_logger.setLevel(old_level)

    print("Datapoint relative-error and correlation against digitized datasets:")
    pearson_values_fig26: list[float] = []
    pearson_values_fig27: list[float] = []
    for case in SOLAR_CASES:
        label = case["label"]
        if label in fig26_dataset and label in fig26_model:
            stats_26 = datapoint_relative_and_corr_stats_xy(fig26_model[label], fig26_dataset[label])
            if stats_26 is not None:
                max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats_26
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  Fig26 {label:<20} "
                    f"max_relative_error={max_rel:9.6f} ({100.0 * max_rel:6.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel:9.6f} ({100.0 * mean_rel:6.3f}%), "
                    f"pearson_r={pearson_r:8.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values_fig26.append(pearson_r)
            else:
                print(f"  Fig26 {label:<20} n/a")
        if label in fig27_dataset and label in fig27_model:
            stats_27 = datapoint_relative_and_corr_stats_xy(fig27_model[label], fig27_dataset[label])
            if stats_27 is not None:
                max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats_27
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  Fig27 {label:<20} "
                    f"max_relative_error={max_rel:9.6f} ({100.0 * max_rel:6.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel:9.6f} ({100.0 * mean_rel:6.3f}%), "
                    f"pearson_r={pearson_r:8.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values_fig27.append(pearson_r)
            else:
                print(f"  Fig27 {label:<20} n/a")

    min_pearson_26 = minimum_finite(pearson_values_fig26)
    if min_pearson_26 is not None:
        print(f"  Fig26 minimum Pearson correlation coefficient: {min_pearson_26:.6f}")
    else:
        print("  Fig26 minimum Pearson correlation coefficient: n/a")
    min_pearson_27 = minimum_finite(pearson_values_fig27)
    if min_pearson_27 is not None:
        print(f"  Fig27 minimum Pearson correlation coefficient: {min_pearson_27:.6f}")
    else:
        print("  Fig27 minimum Pearson correlation coefficient: n/a")

    output = plot_validation_crandall_fig26_fig27(
        fig26_model,
        fig27_model,
        fig26_dataset,
        fig27_dataset,
        solar_cases=SOLAR_CASES,
        output_path=OUTPUT_PATH,
        page_figsize=PAGE_FIGSIZE,
        show=show,
    )
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_fig26_fig27_validation(show=True)

