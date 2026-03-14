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

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------------------ #
# ARISS simulation imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.core.simulation import run_sizing_loop


# ------------------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("MansurVerification.toml")

COLLECTION_EFFICIENCIES = (0.35, 0.40, 0.45)

POWER_GRID = np.linspace(200, 30000, 40)  # W


# ------------------------------------------------------------------------------ #
# Sweep the efficiencies and store converged results
# ------------------------------------------------------------------------------ #

def sweep_mansur_efficiencies():

    base_spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)

    results = {}

    for efficiency in COLLECTION_EFFICIENCIES:

        converged_altitudes = []
        converged_isp = []

        for power in POWER_GRID:

            print(f"Running Power: {power:.1f} W")

            spacecraft = deepcopy(base_spacecraft)

            spacecraft.refueling.coll_eff = efficiency
            spacecraft.thruster.power = power

            final_sc, converged, _ = run_sizing_loop(spacecraft)

            if converged:
                converged_altitudes.append(final_sc.orbit.altitude)
                converged_isp.append(final_sc.thruster.specific_impulse)

        results[efficiency] = (
            np.array(converged_altitudes),
            np.array(converged_isp),
        )

    return results


# ------------------------------------------------------------------------------ #
# Plot the Mansur verification curves
# ------------------------------------------------------------------------------ #

def plot_results(results):

    figure, axis = plt.subplots(figsize=(8, 5))

    colors = ["#1f77b4", "#d95f02", "#e6ab02"]

    for color, efficiency in zip(colors, COLLECTION_EFFICIENCIES):

        altitude_km, isp_s = results.get(efficiency, (np.array([]), np.array([])))

        if altitude_km.size == 0:
            continue

        axis.plot(
            altitude_km,
            isp_s,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=3.0,
            label=f"Isp for ηc = {efficiency:.2f}",
        )

        solution_limit = float(np.min(altitude_km))

        axis.axvline(
            solution_limit,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"Solution limit at {solution_limit:.1f} km",
        )

    axis.set_xlabel("Converged altitude (km)")
    axis.set_ylabel("Isp (s)")
    axis.set_title("Mansur Verification Sweep")

    axis.set_xlim(140, 200)
    axis.set_ylim(1500, 7000)

    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    plt.show()


# ------------------------------------------------------------------------------ #
# Run script
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":

    results = sweep_mansur_efficiencies()

    plot_results(results)