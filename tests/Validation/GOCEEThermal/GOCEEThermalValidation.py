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

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.modules.Thermal import thermal_model

CONFIG_PATH = Path(__file__).with_name("GOCEThermal.toml")

def run_gocee_thermal_validation(show: bool = True) -> Path:
    spacecraft = load_spacecraft_from_base_config(CONFIG_PATH)
    diagnostics = thermal_model(spacecraft)
    print(f"Steady-state temperature:{diagnostics.T_max-273.15} C")
    print(f"Expected steady-state temperature: 100 C")
    print(f"Deviation: {273.15+50-diagnostics.T_max} C")

if __name__ == "__main__":
    run_gocee_thermal_validation(show=True)