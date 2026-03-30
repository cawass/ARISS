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
#      Fig. 11 recreation for Crandall and Wirz (2022) using full sizing loop
#      runs while sweeping thruster efficiency.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022Fig11Validation.py
# ============================================================================== #

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
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
from ariss.utils.ploting import plot_validation_crandall_fig11
from csv_helper import load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats, minimum_finite


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent

CONFIG_PATH = HERE / "CrandallWirz2022_6U.toml"
DATASET_PATH = HERE / "Fig 11.csv"
OUTPUT_PATH = HERE / "crandall_wirz_2022_fig11_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

EFFICIENCY_GRID = np.linspace(0.38, 1.00, 18)

MAX_ITERATIONS = 260
MASS_TOLERANCE = 1.0e-3

SOLAR_CASES = [
    {"label": "Solar Maximum", "f107": 200.0, "color": PALETTE["choice_mid"], "marker": "s"},
    {"label": "Mean Solar Activity", "f107": 114.0, "color": PALETTE["l1_teal"], "marker": "^"},
    {"label": "Solar Minimum", "f107": 62.0, "color": PALETTE["sernn_pink"], "marker": "D"},
]


# ------------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------------ #

def load_fig11_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return load_wide_xy_csv(path, sort_by="x", min_rows=3)


# ------------------------------------------------------------------------------ #
# Full-loop sweep
# ------------------------------------------------------------------------------ #

def run_case_for_solar_flux(f107_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tp_vals: list[float] = []
    alt_vals: list[float] = []
    isp_vals: list[float] = []
    converged_flags: list[bool] = []

    for efficiency in EFFICIENCY_GRID:
        spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)
        spacecraft.orbit.msis_f107 = float(f107_value)
        spacecraft.thruster.eff = float(efficiency)

        with redirect_stdout(io.StringIO()):
            final_sc, converged, _ = run_sizing_loop(
                spacecraft,
                max_iterations=MAX_ITERATIONS,
                mass_tolerance=MASS_TOLERANCE,
            )

        if final_sc.thruster.power == 0.0:
            continue

        tp_mn_per_kw = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)

        if np.isfinite(tp_mn_per_kw) and np.isfinite(final_sc.orbit.altitude):
            tp_vals.append(tp_mn_per_kw)
            alt_vals.append(float(final_sc.orbit.altitude))
            isp_vals.append(float(final_sc.thruster.specific_impulse))
            converged_flags.append(bool(converged))

    if not tp_vals:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)

    tp_array = np.asarray(tp_vals, dtype=float)
    alt_array = np.asarray(alt_vals, dtype=float)
    isp_array = np.asarray(isp_vals, dtype=float)
    converged_array = np.asarray(converged_flags, dtype=bool)
    order = np.argsort(tp_array)
    return tp_array[order], alt_array[order], isp_array[order], converged_array[order]


def run_fig11_sweep() -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)
    try:
        for spec in SOLAR_CASES:
            tp, altitude, isp, converged = run_case_for_solar_flux(spec["f107"])
            results[spec["label"]] = {
                "tp": tp,
                "altitude": altitude,
                "isp": isp,
                "converged": converged,
            }
    finally:
        simulation_logger.setLevel(old_level)

    return results


def run_crandall_wirz_fig11_validation(show: bool = True) -> Path:
    dataset = load_fig11_dataset(DATASET_PATH)
    results = run_fig11_sweep()

    print("Altitude relative-error and correlation against digitized Fig. 11 curves:")
    pearson_values: list[float] = []
    for spec in SOLAR_CASES:
        label = spec["label"]
        if label not in dataset:
            continue
        tp_model = np.asarray(results[label]["tp"], dtype=float)
        alt_model = np.asarray(results[label]["altitude"], dtype=float)
        isp_model = np.asarray(results[label]["isp"], dtype=float)
        tp_ref, alt_ref = dataset[label]
        stats = datapoint_relative_and_corr_stats(tp_model, alt_model, tp_ref, alt_ref)
        if stats is not None:
            max_relative_error, mean_relative_error, line_max_rel, n_rel, pearson_r, n_corr = stats
            line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
            print(
                f"  {label:<20} "
                f"max_relative_error={max_relative_error:8.6f} ({100.0 * max_relative_error:6.3f}%) (line {line_text}), "
                f"mean_relative_error={mean_relative_error:8.6f} ({100.0 * mean_relative_error:6.3f}%), "
                f"pearson_r={pearson_r:8.6f}, n_rel={n_rel}, n_corr={n_corr}"
            )
            if np.isfinite(pearson_r):
                pearson_values.append(pearson_r)
        else:
            print(f"  {label:<20} n/a (no T/P overlap with dataset)")
        if len(isp_model) > 0:
            print(f"    Isp range [s]: {float(np.min(isp_model)):.2f} to {float(np.max(isp_model)):.2f}")

    min_pearson = minimum_finite(pearson_values)
    if min_pearson is not None:
        print(f"  Minimum Pearson correlation coefficient: {min_pearson:.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    output = plot_validation_crandall_fig11(
        results,
        dataset,
        solar_cases=SOLAR_CASES,
        output_path=OUTPUT_PATH,
        page_figsize=PAGE_FIGSIZE,
        show=show,
    )
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_crandall_wirz_fig11_validation(show=True)

