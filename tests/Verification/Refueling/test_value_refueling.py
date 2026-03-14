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
#      Value checks for the refueling compression model.
#
#  Project:        ARISS
#  Module:         test_value_refueling.py
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
from ariss.modules.Refueling import refueling_model
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
        pytest.skip("pymsis is required for refueling value tests.")
    for key, value in updates.items():
        setattr(sc.orbit, key, value)
    return sc


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_refueling_inactive_returns_zero_power(config_path: Path) -> None:
    sc = _build_state(config_path)
    sc.mission_profile.active_refueling = False

    power_refprop = refueling_model(sc)

    assert power_refprop == 0.0
    assert sc.power.Power_refprop == 0.0


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_refueling_active_value_range(config_path: Path) -> None:
    # Inputs:
    #   Controlled refueling state for each verification config.
    #
    # Outputs:
    #   Ensures refueling_model returns bounded compression power and tank volume.

    sc = _build_state(config_path)
    sc.mission_profile.active_refueling = True
    sc.refueling.m_flow = 1.0e-4
    sc.thruster.m_flow = 1.0e-6
    sc.orbit.p_orb = 5.0e-2
    sc.mass.Mass_prop = 5.0
    sc.refueling.eta_refuel = 0.2

    power_refprop = refueling_model(sc)

    _assert_in_range("Power_refprop", power_refprop, 1, 10.0e5)
    _assert_in_range("V_prop", sc.refueling.V_prop, 0.1, 5.0)
    assert sc.power.Power_refprop == pytest.approx(power_refprop, rel=1.0e-12, abs=1.0e-12)

