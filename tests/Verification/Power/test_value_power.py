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
#      Value checks for power-model outputs across drag verification configs.
#
#  Project:        ARISS
#  Module:         test_value_power.py
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
from ariss.modules.Power import power_model
from ariss.utils.atmosphere import orbit_updates_from_height
from tests.Verification._cases import (
    build_spacecraft_from_case,
    verification_case_paths,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = verification_case_paths(CONFIG_DIR)


def _build_state(config_path: Path) -> SpacecraftState:
    sc = build_spacecraft_from_case(config_path)
    try:
        updates = orbit_updates_from_height(
            sc.orbit.altitude,
            msis_date=sc.orbit.msis_date,
            msis_f107=sc.orbit.msis_f107,
            msis_ap=sc.orbit.msis_ap,
        )
    except ImportError:
        pytest.skip("pymsis is required for power value tests.")
    for key, value in updates.items():
        setattr(sc.orbit, key, value)
    return sc


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_power_value_ranges_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Ensures power_model returns a physically consistent solar area for each
    #   geometry/wake verification case.

    sc = _build_state(config_path)
    sc.power.Power_total = 10000.0

    power_model(sc)

    _assert_in_range("A_solar", sc.geometry.A_solar, 20.0, 35.0)
    assert sc.geometry.A_solar >= 0.0, "A_solar must be non-negative."

