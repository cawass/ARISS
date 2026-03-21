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
#      Recreate Crandall and Wirz (2022) Fig. 26 with the ARISS core drag model
#      for the 6U reference geometry.
#
#  Project:        ARISS
#  Module:         CrandallWirz2022_Fig26_SolarEfficiency.py
# ============================================================================== #

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = ROOT / "tests" / "Validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from plot_style import apply_validation_style, style_axis, style_legend
from CrandallWirz2022Validation import (
    CASE_6U_PATH,
    SOLAR_ACTIVITY_F107,
    SOLAR_COLORS,
    _evaluate_drag_state,
    _required_load_n,
    load_spacecraft,
)


OUTPUT_PATH = Path(__file__).with_name("crandall_wirz_2022_fig26.png")
ALTITUDE_GRID_KM = np.linspace(150.0, 250.0, 401, dtype=float)
SOLAR_EFFICIENCY_SWEEP = np.linspace(0.25, 0.50, 101, dtype=float)
REFERENCE_POWER_W = 96.0
REFERENCE_SOLAR_CELL_EFFICIENCY = 0.30
TP_MN_KW = 10.0


def _solve_minimum_altitude(spacecraft_template, target_thrust_n: float, f107: float) -> float:
    required_load_n = np.asarray(
        [_required_load_n(_evaluate_drag_state(spacecraft_template, float(altitude_km), f107)) for altitude_km in ALTITUDE_GRID_KM],
        dtype=float,
    )
    if target_thrust_n < float(required_load_n[-1]) or target_thrust_n > float(required_load_n[0]):
        return float("nan")
    return float(np.interp(target_thrust_n, required_load_n[::-1], ALTITUDE_GRID_KM[::-1]))


def build_fig26_curves() -> dict[str, dict[str, np.ndarray]]:
    spacecraft = load_spacecraft(CASE_6U_PATH)
    curves: dict[str, dict[str, np.ndarray]] = {}

    for label, f107 in SOLAR_ACTIVITY_F107.items():
        altitude_values = []
        for eta_solar in SOLAR_EFFICIENCY_SWEEP:
            scaled_power_w = REFERENCE_POWER_W * float(eta_solar / REFERENCE_SOLAR_CELL_EFFICIENCY)
            target_thrust_n = 1.0e-6 * TP_MN_KW * scaled_power_w
            altitude_values.append(_solve_minimum_altitude(spacecraft, target_thrust_n, float(f107)))
        curves[label] = {
            "eta_solar": np.asarray(SOLAR_EFFICIENCY_SWEEP, dtype=float),
            "altitude_km": np.asarray(altitude_values, dtype=float),
        }

    return curves


def plot_fig26(curves: dict[str, dict[str, np.ndarray]], save_path: Path = OUTPUT_PATH, show: bool = True) -> Path:
    apply_validation_style()
    figure, axis = plt.subplots(figsize=(7.2, 5.3), dpi=150)

    for label, payload in curves.items():
        axis.plot(
            payload["eta_solar"],
            payload["altitude_km"],
            color=SOLAR_COLORS[label],
            linewidth=2.0,
            label=label,
        )

    axis.set_xlim(0.25, 0.50)
    axis.set_ylim(160.0, 195.0)
    axis.set_xlabel("Solar Cell Efficiency")
    axis.set_ylabel("Minimum Operating Altitude [km]")
    style_axis(axis)
    legend = axis.legend(loc="upper right")
    style_legend(legend)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(figure)
    return save_path


def main(show: bool = True) -> Path:
    curves = build_fig26_curves()
    output_path = plot_fig26(curves, save_path=OUTPUT_PATH, show=show)
    print(f"Saved Fig. 26 recreation: {output_path}")
    return output_path


if __name__ == "__main__":
    main(show=True)
