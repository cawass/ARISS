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
#      Validation of the thermal model using sources on GOCE
#       sources used:
#        
#
#  Project:        ARISS
#  Module:         GOCEEThermalValidation.py
# ============================================================================== #

import sys
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
# ARISS drag kernel
# ------------------------------------------------------------------------------ #

from ariss.modules.Thermal import thermal_model


# ------------------------------------------------------------------------------ #
# GOCE-like validation inputs from Tisaev et al. (2022), Fig. 5
# ------------------------------------------------------------------------------ #

AREA_RATIO = 28.2
ORBIT_TEMP = 1.0
WALL_TEMP = 0.3
FRONT_ALPHA = 0.5 * np.pi
SIDE_ALPHA = 0.0
DRAG_KERNEL_EPSILON = 0.0

OUTPUT_PATH = Path(__file__).with_name("gocee_drag_validation.png")

# The paper does not tabulate the Koppenwallner DSMC data. These points are
# approximate read-offs from Fig. 5 and are used only to reproduce the same plot.
REFERENCE_SPEED_RATIO = np.asarray([9.0, 10.0, 11.0], dtype=float)
REFERENCE_CD = np.asarray([3.79, 3.67, 3.51], dtype=float)


# ------------------------------------------------------------------------------ #
# GOCE-like free-molecular body drag
# ------------------------------------------------------------------------------ #

def gocee_total_cd(speed_ratio: np.ndarray | float) -> np.ndarray:
    # Inputs:
    #   speed_ratio: molecular speed ratio S_inf [-].
    #
    # Outputs:
    #   Total body drag coefficient referenced to front area [-].
    #
    # Equations used:
    #   Cd_total = Cd_front + Cd_side * (A_par / A_front)

    speed_ratio = np.asarray(speed_ratio, dtype=float)
    cd_front = _drag_coefficient(speed_ratio, DRAG_KERNEL_EPSILON, FRONT_ALPHA, ORBIT_TEMP, WALL_TEMP, 1.0)
    cd_side = _drag_coefficient(speed_ratio, DRAG_KERNEL_EPSILON, SIDE_ALPHA, ORBIT_TEMP, WALL_TEMP, 1.0)
    return cd_front + AREA_RATIO * cd_side


def plot_gocee_validation(save_path: Path = OUTPUT_PATH, show: bool = True) -> Path:
    # Inputs:
    #   save_path: PNG output path.
    #   show: whether to display the figure interactively.
    #
    # Outputs:
    #   Saved path of the validation figure.

    speed_ratio = np.linspace(8.0, 13.0, 400, dtype=float)
    total_cd = gocee_total_cd(speed_ratio)

    fig, axis = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    axis.plot(speed_ratio, total_cd, color="#4c9ed9", linewidth=2.0, label=r"$C_D$ with $A_{par}/A_{front} = 28.2$")
    axis.scatter(REFERENCE_SPEED_RATIO, REFERENCE_CD, color="#d97a3a", marker="x", s=42, linewidths=1.6, label="Koppenwallner GOCE $C_D$")

    axis.set_xlim(8.0, 13.0)
    axis.set_ylim(3.2, 4.2)
    axis.set_xlabel(r"Speed ratio $S_{\infty}$")
    axis.set_ylabel(r"Drag coefficient $C_D$")
    axis.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axis.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show and plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)

    return save_path


def run_gocee_validation(show: bool = True) -> Path:
    # Inputs:
    #   show: whether to display the validation figure.
    #
    # Outputs:
    #   Path to the saved validation figure.

    analytical_cd = gocee_total_cd(REFERENCE_SPEED_RATIO)
    max_relative_error = np.max(np.abs(analytical_cd - REFERENCE_CD) / REFERENCE_CD)

    print(f"GOCE-like area ratio A_par/A_front = {AREA_RATIO:.1f}")
    print(f"Reference points: {len(REFERENCE_SPEED_RATIO)}")
    print(f"Maximum relative difference to Fig. 5 points: {100.0 * max_relative_error:.2f}%")

    output_path = plot_gocee_validation(save_path=OUTPUT_PATH, show=show)
    print(f"Saved figure: {output_path}")
    return output_path


if __name__ == "__main__":
    run_gocee_validation(show=True)
