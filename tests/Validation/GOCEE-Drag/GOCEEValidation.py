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
#      Recreation of Fig. 5 (GOCE drag coefficient comparison) from
#      1-s2.0-S009457652100607X-main.pdf using digitized CSV data.
#
#  Project:        ARISS
#  Module:         GOCEEValidation.py
# ============================================================================== #

from __future__ import annotations

import sys
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

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import _side_areas
from ariss.utils import constants as const
from ariss.utils.atmosphere import atmospheric_properties_from_height
from ariss.utils.ploting import plot_validation_gocee_fig5
from csv_helper import load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats, minimum_finite


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE / "Gocee.csv"
OUTPUT_PATH = HERE / "gocee_fig5_validation.png"
GOCE_CONFIG_PATH = HERE / "GOCEDrag.toml"
PAGE_FIGSIZE = (13.2, 5.4)


# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return load_wide_xy_csv(path, sort_by="x", min_rows=3)


# ------------------------------------------------------------------------------ #
# ARISS curve
# ------------------------------------------------------------------------------ #

def compute_ariss_body_cd_curve(
    x_values: np.ndarray,
    config_path: Path = GOCE_CONFIG_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    sc = load_spacecraft_from_base_config(config_path)

    # Use a GOCE-like reference atmosphere state (single point), then map S0 -> V.
    properties = atmospheric_properties_from_height(
        sc.orbit.altitude,
        msis_date=sc.orbit.msis_date,
        msis_f107=sc.orbit.msis_f107,
        msis_ap=sc.orbit.msis_ap,
        latitude=sc.orbit.latitude,
        longitude=sc.orbit.longitude,
        use_average=sc.orbit.use_average,
    )
    sc.orbit.temperature = float(properties["temperature"])
    sc.orbit.molar_mass = float(properties["molar_mass"])

    # Fig. 5 compares body free-molecular drag coefficient referenced to GOCE
    body_side_area, inlet_side_area = _side_areas(sc.geometry)
    a_ref = float(sc.geometry.A_in) if float(sc.geometry.A_in) > 0.0 else 1.0

    x = np.asarray(x_values, dtype=float)
    y = np.zeros_like(x)
    speed_scale = np.sqrt(sc.orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * sc.orbit.temperature))

    for i, s0 in enumerate(x):
        sc.orbit.velocity = float(s0 / speed_scale)
        drag_model(sc)
        y[i] = (
            float(sc.drag.cd_inlet_front) * float(sc.geometry.A_in_drag)
            + float(sc.drag.cd_inlet_side) * inlet_side_area
            + float(sc.drag.cd_body_side) * body_side_area
            + float(sc.drag.cd_solar) * float(sc.geometry.A_solar)
            + float(sc.drag.cd_rad) * float(sc.geometry.A_rad)
        ) / a_ref

    return x, y


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def run_gocee_fig5_validation(show: bool = True) -> Path:
    dataset = load_wide_xy_dataset(DATASET_PATH)

    x_ref: list[float] = []
    for key in ("Mansur", "Koppenwallner"):
        if key in dataset:
            x_ref.extend(np.asarray(dataset[key][0], dtype=float).tolist())
    x_min = float(np.nanmin(x_ref)) if x_ref else 8.0
    x_max = float(np.nanmax(x_ref)) if x_ref else 13.0
    x_ariss = np.linspace(x_min, x_max, 140)
    x_ariss, y_ariss = compute_ariss_body_cd_curve(x_ariss)

    print("Datapoint relative-error and correlation metrics against digitized GOCE curves:")
    pearson_values: list[float] = []
    for label in ("Mansur", "Koppenwallner"):
        if label not in dataset:
            continue

        x_ref_vals, y_ref_vals = dataset[label]
        stats = datapoint_relative_and_corr_stats(x_ariss, y_ariss, x_ref_vals, y_ref_vals)
        if stats is None:
            print(f"  {label:<14} n/a")
            continue

        max_rel_error, mean_rel_error, max_rel_line, rel_count, pearson_r, corr_count = stats
        max_rel_line_text = str(max_rel_line) if max_rel_line > 0 else "n/a"
        print(
            f"  {label:<14} "
            f"max_relative_error={max_rel_error:.6f} ({100.0 * max_rel_error:.3f}%) (line {max_rel_line_text}), "
            f"mean_relative_error={mean_rel_error:.6f} ({100.0 * mean_rel_error:.3f}%), "
            f"pearson_r={pearson_r:.6f}, n_rel={rel_count}, n_corr={corr_count}"
        )
        if np.isfinite(pearson_r):
            pearson_values.append(pearson_r)

    min_pearson = minimum_finite(pearson_values)
    if min_pearson is not None:
        print(f"  Minimum Pearson correlation coefficient: {min_pearson:.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    output = plot_validation_gocee_fig5(
        dataset,
        compute_ariss_body_cd_curve=compute_ariss_body_cd_curve,
        output_path=OUTPUT_PATH,
        page_figsize=PAGE_FIGSIZE,
        show=show,
    )
    print(f"Saved figure: {output}")
    return output


if __name__ == "__main__":
    run_gocee_fig5_validation(show=True)

