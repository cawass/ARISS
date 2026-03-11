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
#      Value checks for propulsion-model outputs across drag verification configs.
#
#  Project:        ARISS
#  Module:         test_value_propulsion.py
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
from ariss.modules.Drag import drag_model
from ariss.modules.Propulsion import propulsion_model
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
        pytest.skip("pymsis is required for propulsion value tests.")
    for key, value in updates.items():
        setattr(sc.orbit, key, value)
    return sc


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_propulsion_value_ranges_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Ensures propulsion_model returns bounded, physically plausible values for
    #   all geometry/wake verification cases.

    sc = _build_state(config_path)

    drag_model(sc)
    propulsion_model(sc)

    _assert_in_range("A_prop", sc.geometry.A_prop, 0.15, 1.0)
    _assert_in_range("A_in", sc.geometry.A_in, 0.5, 3.0)
    _assert_in_range("A_in_drag", sc.geometry.A_in_drag, 0.3, 2.0)
    _assert_in_range("thrust", sc.thruster.thrust, 0.03, 0.05)
    _assert_in_range("m_flow", sc.thruster.m_flow, 6.0e-7, 8.0e-7)
    _assert_in_range("altitude", sc.orbit.altitude, 170.0, 230.0)
    _assert_in_range("density", sc.orbit.density, 1.0e-10, 5.0e-10)
    _assert_in_range("drag_total", sc.drag.drag_total, 9.0e-3, 4.0e-2)

    assert sc.geometry.A_in > sc.geometry.A_prop, "A_in must exceed A_prop due to drag/refuel intake terms."
    assert sc.geometry.A_in_drag > 0.0, "A_in_drag must be positive."
