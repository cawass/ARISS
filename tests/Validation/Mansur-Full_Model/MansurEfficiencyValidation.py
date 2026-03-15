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
from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop


# ------------------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("MansurValidation.toml")

COLLECTION_EFFICIENCIES = (0.35, 0.40, 0.45)

ISP = np.linspace(200, 10000, 40)  # [s]


# ------------------------------------------------------------------------------ #
# Sweep the efficiencies and store converged results
# ------------------------------------------------------------------------------ #

def sweep_mansur_efficiencies():

    base_spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)

    results = {}
    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for efficiency in COLLECTION_EFFICIENCIES:

            converged_altitudes = []
            converged_isp = []

            for isp in ISP:

                print(f"Running Isp: {isp:.1f} s")

                spacecraft = deepcopy(base_spacecraft)

                spacecraft.refueling.coll_eff = efficiency
                spacecraft.thruster.specific_impulse = float(isp)

                with redirect_stdout(io.StringIO()):
                    final_sc, converged, _ = run_sizing_loop(spacecraft)

                if converged:
                    converged_altitudes.append(final_sc.orbit.altitude)
                    converged_isp.append(final_sc.thruster.specific_impulse)

            results[efficiency] = (np.array(converged_altitudes), np.array(converged_isp))
    finally:
        simulation_logger.setLevel(previous_level)

    return results


# ------------------------------------------------------------------------------ #
# Plot the Mansur verification curves
# ------------------------------------------------------------------------------ #

def _plot_sweep_curve(axis, altitude_km: np.ndarray, isp_s: np.ndarray, color: str, efficiency: float) -> None:

    # Inputs:
    #   axis: matplotlib axis used for plotting.
    #   altitude_km: converged altitudes in the same order as the Isp sweep.
    #   isp_s: converged specific impulse values in sweep order.
    #   color: curve color.
    #   efficiency: collection efficiency used for the sweep.
    #
    # Outputs:
    #   Plots the sweep curve without folding the descending and ascending
    #   altitude branches onto each other.

    turning_index = int(np.argmin(altitude_km))
    first_branch = slice(0, turning_index + 1)
    second_branch = slice(turning_index, altitude_km.size)

    axis.plot(
        altitude_km[first_branch],
        isp_s[first_branch],
        color=color,
        linewidth=1.6,
        marker="o",
        markersize=3.0,
        label=f"Isp for eta_c = {efficiency:.2f}",
    )

    if second_branch.stop - second_branch.start > 1:
        axis.plot(
            altitude_km[second_branch],
            isp_s[second_branch],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=3.0,
        )


def plot_results(results, show: bool = True):

    if show and "agg" in plt.get_backend().lower():
        try:
            plt.switch_backend("TkAgg")
        except Exception:
            pass

    figure, axis = plt.subplots(figsize=(8, 5))

    colors = ["#1f77b4", "#d95f02", "#e6ab02"]

    for color, efficiency in zip(colors, COLLECTION_EFFICIENCIES):

        altitude_km, isp_s = results.get(efficiency, (np.array([]), np.array([])))

        if altitude_km.size == 0:
            continue

        _plot_sweep_curve(axis, altitude_km, isp_s, color, efficiency)

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

    if show:
        plt.show()


# ------------------------------------------------------------------------------ #
# Run script
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":

    results = sweep_mansur_efficiencies()

    plot_results(results)
