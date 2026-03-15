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
#      Solver-level validation sweep against Gunaltay et al. (IEPC 2025).
#
#  Project:        ARISS
#  Module:         IEPC2025PaperValidation.py
# ============================================================================== #

import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------------------ #
# ARISS imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config, logger as simulation_logger, run_sizing_loop


# ------------------------------------------------------------------------------ #
# Paper-derived sweep configuration
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("IEPC2025Paper.toml")
OUTPUT_PATH = Path(__file__).with_name("iepc2025_paper_efficiency_sweep.png")

COLLECTION_EFFICIENCIES = np.asarray([0.35, 0.40, 0.45, 0.50, 0.55, 0.60], dtype=float)
PAPER_MIN_EFFICIENCY = 0.50
PAPER_MEAN_ALTITUDE = 255.0

MAX_ITERATIONS = 160
MASS_TOLERANCE = 1.0e-3


def run_efficiency_sweep(show: bool = True) -> Path:
    # Inputs:
    #   show: whether to display the figure interactively.
    #
    # Outputs:
    #   Saved path to the validation figure.

    results: list[tuple[float, bool, float]] = []

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)
    try:
        for coll_eff in COLLECTION_EFFICIENCIES:
            spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)
            spacecraft.refueling.coll_eff = float(coll_eff)

            with redirect_stdout(io.StringIO()):
                final_sc, converged, _history = run_sizing_loop(
                    spacecraft,
                    max_iterations=MAX_ITERATIONS,
                    mass_tolerance=MASS_TOLERANCE,
                )

            altitude = float(final_sc.orbit.altitude) if converged else np.nan
            results.append((coll_eff, converged, altitude))
    finally:
        simulation_logger.setLevel(previous_level)

    efficiency = np.asarray([item[0] for item in results], dtype=float)
    converged_mask = np.asarray([item[1] for item in results], dtype=bool)
    altitude = np.asarray([item[2] for item in results], dtype=float)

    fig, axis = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    axis.axvline(PAPER_MIN_EFFICIENCY, color="#c44e52", linestyle="--", linewidth=1.4, label="Paper threshold at eta_c = 0.50")
    axis.axhline(PAPER_MEAN_ALTITUDE, color="#6c757d", linestyle=":", linewidth=1.3, label="Paper mean altitude = 255 km")

    if np.any(converged_mask):
        axis.plot(
            efficiency[converged_mask],
            altitude[converged_mask],
            color="#1f77b4",
            marker="o",
            linewidth=2.0,
            label="ARISS steady-state result",
        )

    if np.any(~converged_mask):
        fail_altitude = np.nanmin(altitude[converged_mask]) - 5.0 if np.any(converged_mask) else PAPER_MEAN_ALTITUDE - 5.0
        axis.scatter(
            efficiency[~converged_mask],
            np.full(np.count_nonzero(~converged_mask), fail_altitude),
            color="#d97a3a",
            marker="x",
            s=52,
            linewidths=1.6,
            label="ARISS not converged",
        )

    axis.set_xlim(0.33, 0.62)
    altitude_floor = np.nanmin(altitude[converged_mask]) if np.any(converged_mask) else PAPER_MEAN_ALTITUDE
    altitude_ceiling = np.nanmax(altitude[converged_mask]) if np.any(converged_mask) else PAPER_MEAN_ALTITUDE
    axis.set_ylim(min(altitude_floor, PAPER_MEAN_ALTITUDE) - 10.0, max(altitude_ceiling, PAPER_MEAN_ALTITUDE) + 10.0)
    axis.set_xlabel("Collection efficiency eta_c [-]")
    axis.set_ylabel("Converged altitude [km]")
    axis.set_title("IEPC 2025 Paper | EULO Comparison Sweep")
    axis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axis.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

    print("IEPC 2025 paper sweep results")
    for coll_eff, converged, result_altitude in results:
        altitude_text = f"{result_altitude:.3f} km" if converged else "not converged"
        print(f"eta_c = {coll_eff:.2f} | converged = {converged} | altitude = {altitude_text}")
    print(f"Saved figure: {OUTPUT_PATH}")

    return OUTPUT_PATH


if __name__ == "__main__":
    run_efficiency_sweep(show=True)
