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
#      Value checks for thermal-model outputs across drag verification configs.
#
#  Project:        ARISS
#  Module:         test_value_thermal.py
# ============================================================================== #

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.spacecraft import SpacecraftState
from ariss.modules.Thermal import thermal_model
from ariss.utils.atmosphere import orbit_updates_from_height


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.toml"))


def _build_state(config_path: Path) -> SpacecraftState:
    sc = SpacecraftState.from_toml(config_path)
    try:
        updates = orbit_updates_from_height(
            sc.orbit.altitude,
            msis_date=sc.orbit.msis_date,
            msis_f107=sc.orbit.msis_f107,
            msis_ap=sc.orbit.msis_ap,
        )
    except ImportError:
        pytest.skip("pymsis is required for thermal value tests.")
    for key, value in updates.items():
        setattr(sc.orbit, key, value)
    return sc


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_thermal_value_ranges_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Ensures thermal_model returns bounded, physically plausible heat terms.

    sc = _build_state(config_path)
    sc.power.Power_total = 3000.0
    sc.power.Power_prop = 1500.0
    sc.power.Power_refprop = 200.0

    diagnostics = thermal_model(sc)

    _assert_in_range("Q_drag", diagnostics.Q_drag, 1.0, 50000.0)
    _assert_in_range("Q_sun", diagnostics.Q_sun, 1.0, 50000.0)
    _assert_in_range("Q_albedo", diagnostics.Q_albedo, 1.0, 50000.0)
    _assert_in_range("Q_ir", diagnostics.Q_ir, 1.0, 50000.0)
    _assert_in_range("Q_internal", diagnostics.Q_internal, 1.0, 50000.0)
    _assert_in_range("Q_radiated", diagnostics.Q_radiated, 1.0, 50000.0)
    _assert_in_range("A_rad", sc.geometry.A_rad, 0.0, 100)

    assert diagnostics.T_max == 0.0
    assert diagnostics.T_min == 0.0
