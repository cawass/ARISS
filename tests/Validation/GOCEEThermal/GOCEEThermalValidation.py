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

import numpy as np


# ------------------------------------------------------------------------------ #
# Path setup so the ARISS source can be imported
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Thermal import thermal_model
from validation_metrics import mse_summary

CONFIG_PATH = Path(__file__).with_name("GOCEThermal.toml")

def run_gocee_thermal_validation(show: bool = True) -> Path:
    spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)
    diagnostics = thermal_model(spacecraft)
    steady_temp_c = float(diagnostics.T_max - 273.15)
    expected_temp_c = 50
    mse = (steady_temp_c - expected_temp_c) ** 2
    mse_min, mse_max, mse_avg, mse_n = mse_summary([mse])

    print(f"Steady-state temperature: {steady_temp_c:.6f} C")
    print(f"Expected steady-state temperature: {expected_temp_c:.6f} C")
    print("Datapoint MSE (thermal reference):")
    print(f"  min={mse_min:.6f} (line 1), max={mse_max:.6f} (line 1), avg={mse_avg:.6f}, n={mse_n}")

if __name__ == "__main__":
    run_gocee_thermal_validation(show=True)

