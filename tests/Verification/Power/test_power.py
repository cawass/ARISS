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
#      Unit tests for deterministic power-model area sizing behavior.
#
#  Project:        ARISS
#  Module:         test_power.py
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
from ariss.utils import constants as const


def _build_power_state() -> SpacecraftState:
    sc = SpacecraftState()
    sc.geometry.S_in = "s"
    sc.geometry.S_body = "s"
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 1.0
    sc.geometry.A_body = 4.0
    sc.geometry.L_in = 1.0
    sc.geometry.L_body = 2.0
    object.__setattr__(sc, "solar", sc.solar.update(eta_solar=0.25, av_aligment=0.0))
    return sc


def test_power_model_sets_deployable_area_from_power_gap() -> None:
    # Inputs:
    #   Simple rectangular geometry with analytically known top area.
    #
    # Outputs:
    #   Ensures A_solar equals required area minus fixed top area.

    sc = _build_power_state()
    projected_flux = sc.solar.eta_solar * const.SOLAR_CONSTANT
    fixed_top_area = 5.5
    required_area = 10.0
    sc.power.Power_total = projected_flux * required_area

    power_model(sc)

    assert sc.geometry.A_solar == pytest.approx(required_area - fixed_top_area, rel=1.0e-12, abs=1.0e-12)


def test_power_model_clips_negative_deployable_area_to_zero() -> None:
    # Inputs:
    #   Simple rectangular geometry with power demand below fixed top generation.
    #
    # Outputs:
    #   Ensures A_solar does not become negative.

    sc = _build_power_state()
    projected_flux = sc.solar.eta_solar * const.SOLAR_CONSTANT
    sc.power.Power_total = projected_flux * 5.0

    power_model(sc)

    assert sc.geometry.A_solar == pytest.approx(0.0, rel=1.0e-12, abs=1.0e-12)
