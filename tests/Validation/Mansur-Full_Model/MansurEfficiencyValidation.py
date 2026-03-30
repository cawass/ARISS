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
#      Efficiency sweep utility for Mansur-style validation curves.
#
#  Project:        ARISS
#  Module:         MansurEfficiencyVerification.py
# ============================================================================== #
import sys
import io
import logging
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for path in (SRC, VALIDATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ------------------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from ariss.utils.ploting import plot_validation_mansur_efficiency
from csv_helper import extract_value_after_token, load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("MansurValidation3000W.toml")
DATASET_PATH = Path(__file__).with_name("Isp Altitude Dataset.csv")

OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.png")
PAGE_FIGSIZE = (15.84, 5.4)

COLLECTION_EFFICIENCIES = (0.35, 0.40, 0.45)
ISP = np.linspace(2000, 6000, 40)

PLOT_COLORS = [
    PALETTE["l1_teal"],
    PALETTE["sernn_pink"],
    PALETTE["choice_mid"],
]

# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #

def sweep_mansur_efficiencies():
    base = load_spacecraft_from_base_config(CONFIG_PATH)
    results = {}

    prev_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for eff in COLLECTION_EFFICIENCIES:
            alt, isp_vals = [], []

            for isp in ISP:
                sc = deepcopy(base)
                sc.refueling.coll_eff = eff
                sc.thruster.specific_impulse = float(isp)

                with redirect_stdout(io.StringIO()):
                    final, ok, _ = run_sizing_loop(sc)

                if ok:
                    alt.append(final.orbit.altitude)
                    isp_vals.append(final.thruster.specific_impulse)

            results[eff] = (np.array(alt), np.array(isp_vals))

    finally:
        simulation_logger.setLevel(prev_level)

    return results


def load_mansur_paper_dataset():
    return load_wide_xy_csv(
        DATASET_PATH,
        label_parser=extract_value_after_token,
        sort_by="none",
        min_rows=2,
    )


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def plot_results(results, paper=None, save_path=OUTPUT_PATH, show=True):
    return plot_validation_mansur_efficiency(
        results,
        paper,
        collection_efficiencies=COLLECTION_EFFICIENCIES,
        plot_colors=PLOT_COLORS,
        output_path=save_path,
        page_figsize=PAGE_FIGSIZE,
        show=show,  
        relative_and_corr_stats_fn=datapoint_relative_and_corr_stats,
    )

# ------------------------------------------------------------------------------ #
# Entry
# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    results = sweep_mansur_efficiencies()
    paper = load_mansur_paper_dataset()
    path = plot_results(results, paper)
    print(f"Saved: {path}")


