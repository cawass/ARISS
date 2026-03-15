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
#      Four-panel envelope utility for Mansur-style validation curves.
#
#  Project:        ARISS
#  Module:         MansurEnvelopeVerification.py
# ============================================================================== #

from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ------------------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ------------------------------------------------------------------------------
# ARISS imports
# ------------------------------------------------------------------------------

from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _tp_efficiency(tp_mn_kw: float, isp_s: float) -> float:

    tp_n_w = tp_mn_kw * 1e-6

    return 0.5 * tp_n_w * const.EARTH_GRAVITY * isp_s


def _thruster_mdot_pt_1kw(eta_t: float, isp_s: float) -> float:

    exhaust_velocity = const.EARTH_GRAVITY * isp_s

    m_dot_kg_s = 2.0 * eta_t * 1000.0 / (exhaust_velocity ** 2)

    return m_dot_kg_s * 1e6


def _contour_levels(values: np.ndarray, preferred: list[float], fallback_count: int = 8):

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return np.array([])

    low = float(np.min(finite))
    high = float(np.max(finite))

    selected = [v for v in preferred if low <= v <= high]

    if len(selected) >= 2:
        return np.array(selected)

    if np.isclose(low, high):
        return np.array([])

    return np.linspace(low, high, fallback_count)


# ------------------------------------------------------------------------------
# Envelope sweep
# ------------------------------------------------------------------------------

def sweep_mansur_envelope(
    config_path: str | Path | None = None,
    tp_values: np.ndarray | None = None,
    isp_values: np.ndarray | None = None,
    max_iterations: int = 240,
    mass_tolerance: float = 1e-3,
    thruster_power_w: float = 1000.0,
    force_legacy_intake_mode: bool = True,
):

    base_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("MansurValidation.toml")
    )

    base_state = SpacecraftState.from_toml(base_path)

    tp_grid = np.asarray(tp_values if tp_values is not None else np.linspace(6, 60, 28))
    isp_grid = np.asarray(isp_values if isp_values is not None else np.linspace(2800, 6000, 26))

    altitude = np.full((isp_grid.size, tp_grid.size), np.nan)
    eta_t = np.full_like(altitude, np.nan)
    mdot = np.full_like(altitude, np.nan)
    intake_area = np.full_like(altitude, np.nan)
    number_density = np.full_like(altitude, np.nan)

    prev_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:

        for i, isp in enumerate(isp_grid):

            print(f"Isp row {i+1}/{isp_grid.size}")

            for j, tp in enumerate(tp_grid):

                eta = _tp_efficiency(tp, isp)

                eta_t[i, j] = eta

                if eta <= 0 or eta > 1:
                    continue

                mdot[i, j] = _thruster_mdot_pt_1kw(eta, isp)

                sc = deepcopy(base_state)

                if force_legacy_intake_mode:
                    sc.geometry.use_intake_area_ratio = False

                sc.thruster.specific_impulse = isp
                sc.thruster.eff = eta
                sc.thruster.power = thruster_power_w

                with redirect_stdout(io.StringIO()):

                    final_sc, converged, _ = run_sizing_loop(
                        sc,
                        max_iterations=max_iterations,
                        mass_tolerance=mass_tolerance,
                    )

                if not converged:
                    continue

                altitude[i, j] = final_sc.orbit.altitude

                if final_sc.orbit.density > 0 and final_sc.orbit.velocity > 0:

                    m_dot = mdot[i, j] * 1e-6

                    intake_area[i, j] = m_dot / (
                        final_sc.orbit.density * final_sc.orbit.velocity
                    )

                if final_sc.orbit.molar_mass > 0:

                    number_density[i, j] = (
                        final_sc.orbit.density
                        * const.AVOGADRO_NUMBER
                        / final_sc.orbit.molar_mass
                    )

    finally:
        simulation_logger.setLevel(prev_level)

    return dict(
        tp_mn_kw=tp_grid,
        isp_s=isp_grid,
        altitude_km=altitude,
        eta_t=eta_t,
        m_dot_mg_s_pt1kw=mdot,
        a_i_m2_pt1kw=intake_area,
        number_density_m3=number_density,
    )


# ------------------------------------------------------------------------------
# Plot envelope
# ------------------------------------------------------------------------------

def plot_mansur_envelope_verification():

    results = sweep_mansur_envelope()

    tp = results["tp_mn_kw"]
    isp = results["isp_s"]

    altitude = results["altitude_km"]
    eta_t = results["eta_t"]
    mdot = results["m_dot_mg_s_pt1kw"]
    intake_area = results["a_i_m2_pt1kw"]
    number_density = results["number_density_m3"]

    tp_mesh, isp_mesh = np.meshgrid(tp, isp)

    alt_levels = _contour_levels(altitude, [130, 140, 150, 160, 170, 180, 190, 200])
    eta_levels = _contour_levels(eta_t, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    mdot_levels = _contour_levels(mdot, [0.35, 0.4, 0.5, 0.6, 0.8, 1.0])
    ai_levels = _contour_levels(intake_area, [0.2, 0.3, 0.5, 0.8, 1.2])

    fig, ax = plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    ax = ax.ravel()

    for a in ax:
        a.grid(True,linestyle="--",alpha=0.4)
        a.set_xlabel("T/P (mN/kW)")
        a.set_ylabel("Isp (s)")

    if alt_levels.size>1:
        c=ax[0].contour(tp_mesh,isp_mesh,altitude,levels=alt_levels,colors="#2f4bff")
        ax[0].clabel(c,fmt="%d")

    ax[0].set_title("(a) Feasible altitude")

    if eta_levels.size>1:
        c=ax[1].contour(tp_mesh,isp_mesh,eta_t,levels=eta_levels,colors="black")
        ax[1].clabel(c,fmt="%.1f")

    ax[1].set_title("(b) Thruster efficiency")

    if mdot_levels.size>1:
        c=ax[2].contour(tp_mesh,isp_mesh,mdot,levels=mdot_levels,colors="black")
        ax[2].clabel(c,fmt="%.2f")

    ax[2].set_title("(c) Thruster mass flow [mg/s]")

    if ai_levels.size>1:
        c=ax[3].contour(tp_mesh,isp_mesh,intake_area,levels=ai_levels,colors="black")
        ax[3].clabel(c,fmt="%.2f")

    ax[3].set_title("(d) Intake area [m²]")

    fig.suptitle("Mansur Operating Envelope (ARISS)")
    fig.tight_layout()

    plt.show()


# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    plot_mansur_envelope_verification()
