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
#      Unit tests for deterministic refueling-model behavior.
#
#  Project:        ARISS
#  Module:         test_refueling.py
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


def test_refueling_model_resets_power_to_zero_when_inactive() -> None:
    # Inputs:
    #   Refueling-inactive spacecraft with stale non-zero refueling power.
    #
    # Outputs:
    #   Ensures the refueling power term is reset to zero.

    sc = SpacecraftState()
    sc.mission_profile.active_refueling = False
    sc.power.Power_refprop = 123.0

    power_refprop = refueling_model(sc)

    assert power_refprop == pytest.approx(0.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.power.Power_refprop == pytest.approx(0.0, rel=1.0e-12, abs=1.0e-12)


def test_refueling_model_matches_compression_power_and_tank_volume_formula() -> None:
    # Inputs:
    #   Controlled refueling and thruster mass flows with known thermodynamic inputs.
    #
    # Outputs:
    #   Ensures refueling_model matches the implemented compression-power and
    #   ideal-gas storage-volume equations exactly.

    sc = SpacecraftState()
    sc.mission_profile.active_refueling = True
    sc.refueling.m_flow = 2.0e-4
    sc.thruster.m_flow = 1.0e-4
    sc.refueling.eta_refuel = 0.25
    sc.refueling.p_tank = 1.0e5
    sc.orbit.gamma = 1.4
    sc.orbit.R_spec = 300.0
    sc.orbit.p_orb = 50.0
    sc.mass.Mass_prop = 6.0
    object.__setattr__(sc.thermal, "T_des", 320.0)

    expected_power = (
        (sc.refueling.m_flow + sc.thruster.m_flow)
        * sc.orbit.R_spec
        * sc.thermal.T_des
        * ((sc.refueling.p_tank / sc.orbit.p_orb) ** ((sc.orbit.gamma - 1.0) / sc.orbit.gamma) - 1.0)
        / (sc.refueling.eta_refuel * (sc.orbit.gamma - 1.0))
    )
    expected_volume = sc.mass.Mass_prop * sc.orbit.R_spec * sc.thermal.T_des / sc.refueling.p_tank

    power_refprop = refueling_model(sc)

    assert power_refprop == pytest.approx(expected_power, rel=1.0e-12, abs=1.0e-12)
    assert sc.power.Power_refprop == pytest.approx(expected_power, rel=1.0e-12, abs=1.0e-12)
    assert sc.refueling.V_prop == pytest.approx(expected_volume, rel=1.0e-12, abs=1.0e-12)
