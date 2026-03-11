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


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const


def _tp_efficiency(tp_mn_kw: float, isp_s: float) -> float:
    # Inputs:
    #   tp_mn_kw: thrust-to-power ratio [mN/kW].
    #   isp_s: thruster specific impulse [s].
    #
    # Outputs:
    #   Electrical-to-jet thruster efficiency [-].
    #
    # Equations used:
    #   T/P [N/W] = 2 * eta_T / (g0 * Isp)
    #   eta_T = 0.5 * (T/P) * g0 * Isp
    tp_n_w = tp_mn_kw * 1.0e-6
    return 0.5 * tp_n_w * const.EARTH_GRAVITY * isp_s


def _thruster_mdot_pt_1kw(eta_t: float, isp_s: float) -> float:
    # Inputs:
    #   eta_t: electrical-to-jet thruster efficiency [-].
    #   isp_s: thruster specific impulse [s].
    #
    # Outputs:
    #   Thruster mass flow at Pt = 1 kW [mg/s].
    #
    # Equations used:
    #   v_e = g0 * Isp
    #   Pt = 0.5 * m_dot * v_e^2 / eta_T
    #   m_dot = 2 * eta_T * Pt / v_e^2
    exhaust_velocity = const.EARTH_GRAVITY * isp_s
    m_dot_kg_s = 2.0 * eta_t * 1000.0 / (exhaust_velocity ** 2)
    return m_dot_kg_s * 1.0e6


def _contour_levels(values: np.ndarray, preferred: list[float], fallback_count: int = 8) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([], dtype=float)

    low = float(np.min(finite))
    high = float(np.max(finite))
    selected = np.asarray([level for level in preferred if low <= level <= high], dtype=float)
    if selected.size >= 2:
        return selected

    if np.isclose(low, high):
        return np.array([], dtype=float)
    return np.linspace(low, high, fallback_count)


def sweep_mansur_envelope(
    config_path: str | Path | None = None,
    tp_values: np.ndarray | None = None,
    isp_values: np.ndarray | None = None,
    max_iterations: int = 240,
    mass_tolerance: float = 1.0e-3,
    thruster_power_w: float = 1000.0,
    force_legacy_intake_mode: bool = True,
) -> dict[str, np.ndarray]:
    base_path = Path(config_path) if config_path is not None else Path(__file__).with_name("MansurVerification.toml")
    base_state = SpacecraftState.from_toml(base_path)

    tp_grid = np.asarray(tp_values if tp_values is not None else np.linspace(6.0, 60.0, 28), dtype=float)
    isp_grid = np.asarray(isp_values if isp_values is not None else np.linspace(2800.0, 6000.0, 26), dtype=float)

    altitude_km = np.full((isp_grid.size, tp_grid.size), np.nan, dtype=float)
    eta_t = np.full_like(altitude_km, np.nan, dtype=float)
    m_dot_mg_s = np.full_like(altitude_km, np.nan, dtype=float)
    a_i_m2_pt1kw = np.full_like(altitude_km, np.nan, dtype=float)
    number_density_m3 = np.full_like(altitude_km, np.nan, dtype=float)

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)
    try:
        for isp_idx, isp in enumerate(isp_grid):
            print(f"Envelope ISP row {isp_idx + 1}/{isp_grid.size}: Isp={isp:.1f} s")
            for tp_idx, tp in enumerate(tp_grid):
                eta_value = _tp_efficiency(float(tp), float(isp))
                eta_t[isp_idx, tp_idx] = eta_value

                if eta_value <= 0.0 or eta_value > 1.0:
                    continue

                m_dot_mg_s[isp_idx, tp_idx] = _thruster_mdot_pt_1kw(eta_value, float(isp))

                sc = deepcopy(base_state)
                if force_legacy_intake_mode:
                    # In ratio mode intake/drag sizing decouples from collection
                    # behavior and compresses altitude sensitivity vs Isp.
                    sc.geometry.use_intake_area_ratio = False
                sc.thruster.specific_impulse = float(isp)
                sc.thruster.eff = eta_value
                sc.thruster.power = float(thruster_power_w)

                with redirect_stdout(io.StringIO()):
                    final_sc, converged, _history = run_sizing_loop(
                        sc,
                        max_iterations=max_iterations,
                        mass_tolerance=mass_tolerance,
                    )

                if not converged:
                    continue

                altitude_km[isp_idx, tp_idx] = final_sc.orbit.altitude
                if final_sc.orbit.density > 0.0 and final_sc.orbit.velocity > 0.0:
                    m_dot_kg_s = m_dot_mg_s[isp_idx, tp_idx] * 1.0e-6
                    a_i_m2_pt1kw[isp_idx, tp_idx] = m_dot_kg_s / (final_sc.orbit.density * final_sc.orbit.velocity)

                if final_sc.orbit.molar_mass > 0.0:
                    number_density_m3[isp_idx, tp_idx] = final_sc.orbit.density * const.AVOGADRO_NUMBER / final_sc.orbit.molar_mass
    finally:
        simulation_logger.setLevel(previous_level)

    return {
        "tp_mn_kw": tp_grid,
        "isp_s": isp_grid,
        "altitude_km": altitude_km,
        "eta_t": eta_t,
        "m_dot_mg_s_pt1kw": m_dot_mg_s,
        "a_i_m2_pt1kw": a_i_m2_pt1kw,
        "number_density_m3": number_density_m3,
    }


def plot_mansur_envelope_verification(
    config_path: str | Path | None = None,
    tp_values: np.ndarray | None = None,
    isp_values: np.ndarray | None = None,
    max_iterations: int = 240,
    mass_tolerance: float = 1.0e-3,
    thruster_power_w: float = 1000.0,
    force_legacy_intake_mode: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
):
    results = sweep_mansur_envelope(
        config_path=config_path,
        tp_values=tp_values,
        isp_values=isp_values,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
        thruster_power_w=thruster_power_w,
        force_legacy_intake_mode=force_legacy_intake_mode,
    )

    tp_grid = results["tp_mn_kw"]
    isp_grid = results["isp_s"]
    altitude_km = results["altitude_km"]
    eta_t = results["eta_t"]
    m_dot_mg_s = results["m_dot_mg_s_pt1kw"]
    a_i_m2 = results["a_i_m2_pt1kw"]
    number_density = results["number_density_m3"]

    tp_mesh, isp_mesh = np.meshgrid(tp_grid, isp_grid)

    altitude_levels = _contour_levels(
        altitude_km,
        preferred=[130, 140, 150, 155, 160, 165, 170, 180, 190, 200],
        fallback_count=9,
    )
    eta_levels = _contour_levels(eta_t, preferred=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fallback_count=7)
    mdot_levels = _contour_levels(m_dot_mg_s, preferred=[0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2], fallback_count=7)
    ai_levels = _contour_levels(a_i_m2, preferred=[0.16, 0.2, 0.3, 0.5, 0.8, 1.2], fallback_count=7)

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for axis in axes_flat:
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        axis.set_xlim(float(tp_grid.min()), float(tp_grid.max()))
        axis.set_ylim(float(isp_grid.min()), float(isp_grid.max()))
        axis.set_xlabel("T/P (mN/kW)")
        axis.set_ylabel("Isp (s)")

    no_solution_mask = np.isnan(altitude_km).astype(float)

    # (a) Feasible altitude
    if np.any(no_solution_mask > 0.5):
        axes_flat[0].contourf(
            tp_mesh,
            isp_mesh,
            no_solution_mask,
            levels=[0.5, 1.5],
            colors=["#efefef"],
            alpha=0.55,
        )
        axes_flat[0].text(
            0.065,
            0.07,
            "NO SOLUTION",
            transform=axes_flat[0].transAxes,
            fontsize=13,
            color="#1f1f1f",
            ha="left",
            va="center",
        )

    if altitude_levels.size > 1:
        contour_alt_a = axes_flat[0].contour(
            tp_mesh,
            isp_mesh,
            altitude_km,
            levels=altitude_levels,
            colors="#2f4bff",
            linestyles="--",
            linewidths=1.0,
        )
        axes_flat[0].clabel(contour_alt_a, inline=True, fmt="%d", fontsize=9, colors="#2f4bff")

    finite_density = number_density[np.isfinite(number_density)]
    if finite_density.size > 0 and float(np.min(finite_density)) <= 1.0e18 <= float(np.max(finite_density)):
        contour_n = axes_flat[0].contour(
            tp_mesh,
            isp_mesh,
            number_density,
            levels=[1.0e18],
            colors="#d62728",
            linestyles="--",
            linewidths=1.2,
        )
        axes_flat[0].clabel(contour_n, inline=True, fmt={1.0e18: r"$n=10^{18}\,\mathrm{m^{-3}}$"}, fontsize=10)

    axes_flat[0].set_title("(a) Feasible altitude vs Isp and T/P")
    axes_flat[0].legend(
        handles=[
            Line2D([0], [0], color="#2f4bff", linestyle="--", linewidth=1.2, label="Feasible alt (km)"),
            Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.2, label=r"$n=10^{18}\,\mathrm{m^{-3}}$"),
        ],
        loc="upper right",
        fontsize=9,
    )

    # (b) Thruster efficiency
    if altitude_levels.size > 1:
        contour_alt_b = axes_flat[1].contour(
            tp_mesh,
            isp_mesh,
            altitude_km,
            levels=altitude_levels,
            colors="#2f4bff",
            linestyles="--",
            linewidths=0.95,
        )
        axes_flat[1].clabel(contour_alt_b, inline=True, fmt="%d", fontsize=8, colors="#2f4bff")
    if eta_levels.size > 1:
        contour_eta = axes_flat[1].contour(
            tp_mesh,
            isp_mesh,
            eta_t,
            levels=eta_levels,
            colors="#2b2b2b",
            linewidths=0.9,
        )
        axes_flat[1].clabel(contour_eta, inline=True, fmt="%.1f", fontsize=8)
    axes_flat[1].set_title(r"(b) Thruster $\eta_T$ with feasible altitude")
    axes_flat[1].legend(
        handles=[
            Line2D([0], [0], color="#2f4bff", linestyle="--", linewidth=1.2, label="Feasible alt (km)"),
            Line2D([0], [0], color="#2b2b2b", linewidth=1.0, label=r"$\eta_T$"),
        ],
        loc="upper right",
        fontsize=9,
    )

    # (c) Thruster mass flow at Pt = 1kW
    if altitude_levels.size > 1:
        contour_alt_c = axes_flat[2].contour(
            tp_mesh,
            isp_mesh,
            altitude_km,
            levels=altitude_levels,
            colors="#2f4bff",
            linestyles="--",
            linewidths=0.95,
        )
        axes_flat[2].clabel(contour_alt_c, inline=True, fmt="%d", fontsize=8, colors="#2f4bff")
    if mdot_levels.size > 1:
        contour_mdot = axes_flat[2].contour(
            tp_mesh,
            isp_mesh,
            m_dot_mg_s,
            levels=mdot_levels,
            colors="#2b2b2b",
            linewidths=0.9,
        )
        axes_flat[2].clabel(contour_mdot, inline=True, fmt="%.2f", fontsize=8)
    axes_flat[2].set_title(r"(c) Feasible $\dot{m}_i$ [mg/s] @ $P_t=1\,\mathrm{kW}$")
    axes_flat[2].legend(
        handles=[
            Line2D([0], [0], color="#2f4bff", linestyle="--", linewidth=1.2, label="Feasible alt (km)"),
            Line2D([0], [0], color="#2b2b2b", linewidth=1.0, label=r"$\dot{m}_i$ @ Pt=1kW"),
        ],
        loc="upper right",
        fontsize=9,
    )

    # (d) Intake area at Pt = 1kW
    if altitude_levels.size > 1:
        contour_alt_d = axes_flat[3].contour(
            tp_mesh,
            isp_mesh,
            altitude_km,
            levels=altitude_levels,
            colors="#2f4bff",
            linestyles="--",
            linewidths=0.95,
        )
        axes_flat[3].clabel(contour_alt_d, inline=True, fmt="%d", fontsize=8, colors="#2f4bff")
    if ai_levels.size > 1:
        contour_ai = axes_flat[3].contour(
            tp_mesh,
            isp_mesh,
            a_i_m2,
            levels=ai_levels,
            colors="#2b2b2b",
            linewidths=0.9,
        )
        axes_flat[3].clabel(contour_ai, inline=True, fmt="%.2f", fontsize=8)
    axes_flat[3].set_title(r"(d) Feasible $A_i$ [m$^2$] @ $P_t=1\,\mathrm{kW}$")
    axes_flat[3].legend(
        handles=[
            Line2D([0], [0], color="#2f4bff", linestyle="--", linewidth=1.2, label="Feasible alt (km)"),
            Line2D([0], [0], color="#2b2b2b", linewidth=1.0, label=r"$A_i$ @ Pt=1kW"),
        ],
        loc="upper right",
        fontsize=9,
    )

    figure.suptitle("Mansur-Style Operating Envelope (ARISS)", fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        figure.savefig(Path(save_path), dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axes, results


if __name__ == "__main__":
    plot_mansur_envelope_verification()
