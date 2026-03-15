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
#      Recreate Crandall and Wirz (2022) Fig. 27 with the ARISS core drag model
#      for the 6U reference geometry.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022_Fig27_Accommodation.py
# ============================================================================== #

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CrandallWirz2022Validation import (
    CASE_6U_PATH,
    SOLAR_ACTIVITY_F107,
    SOLAR_COLORS,
    _evaluate_drag_state,
    _required_load_n,
    load_spacecraft,
)


OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig27.png")
ALTITUDE_GRID_KM = np.linspace(150.0, 250.0, 401, dtype=float)
ACCOMMODATION_SWEEP = np.linspace(0.0, 1.0, 101, dtype=float)
REFERENCE_POWER_W = 96.0
TP_MN_KW = 10.0


def _with_accommodation_sigma(spacecraft_template, sigma: float):
    sc = deepcopy(spacecraft_template)
    epsilon = 1.0 - float(sigma)
    sc.geometry.epsilon_body = epsilon
    sc.geometry.epsilon_in = epsilon
    sc.geometry.epsilon_in_norm = epsilon
    sc.geometry.epsilon_solar = epsilon
    sc.geometry.epsilon_rad = epsilon
    return sc


def _solve_minimum_altitude(spacecraft_template, target_thrust_n: float, f107: float) -> float:
    required_load_n = np.asarray(
        [_required_load_n(_evaluate_drag_state(spacecraft_template, float(altitude_km), f107)) for altitude_km in ALTITUDE_GRID_KM],
        dtype=float,
    )
    if target_thrust_n < float(required_load_n[-1]) or target_thrust_n > float(required_load_n[0]):
        return float("nan")
    return float(np.interp(target_thrust_n, required_load_n[::-1], ALTITUDE_GRID_KM[::-1]))


def build_fig27_curves() -> dict[str, dict[str, np.ndarray]]:
    spacecraft = load_spacecraft(CASE_6U_PATH)
    target_thrust_n = 1.0e-6 * TP_MN_KW * REFERENCE_POWER_W
    curves: dict[str, dict[str, np.ndarray]] = {}

    for label, f107 in SOLAR_ACTIVITY_F107.items():
        altitude_values = []
        for sigma in ACCOMMODATION_SWEEP:
            modified_spacecraft = _with_accommodation_sigma(spacecraft, float(sigma))
            altitude_values.append(_solve_minimum_altitude(modified_spacecraft, target_thrust_n, float(f107)))
        curves[label] = {
            "sigma": np.asarray(ACCOMMODATION_SWEEP, dtype=float),
            "altitude_km": np.asarray(altitude_values, dtype=float),
        }

    return curves


def plot_fig27(curves: dict[str, dict[str, np.ndarray]], save_path: Path = OUTPUT_PATH, show: bool = True) -> Path:
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    figure, axis = plt.subplots(figsize=(7.2, 5.3), dpi=150)

    for label, payload in curves.items():
        axis.plot(
            payload["sigma"],
            payload["altitude_km"],
            color=SOLAR_COLORS[label],
            linewidth=2.0,
            label=label,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(150.0, 190.0)
    axis.set_xlabel("Accommodation Coefficient")
    axis.set_ylabel("Minimum Operating Altitude [km]")
    axis.grid(True, color="#bdbdbd", linewidth=0.6, alpha=0.45)
    axis.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def main(show: bool = True) -> Path:
    curves = build_fig27_curves()
    output_path = plot_fig27(curves, save_path=OUTPUT_PATH, show=show)
    print(f"Saved Fig. 27 recreation: {output_path}")
    return output_path


if __name__ == "__main__":
    main(show=True)
