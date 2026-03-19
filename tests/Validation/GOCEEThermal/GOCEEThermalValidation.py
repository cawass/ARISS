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

