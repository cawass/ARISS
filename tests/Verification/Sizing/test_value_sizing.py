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
#      Value checks for the sizing (mass/power budget) model across the
#      verification configuration matrix.
#
#  Project:        ARISS
#  Module:         test_value_sizing.py
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
from ariss.modules.Budgets import sizing_model
from tests.Verification._cases import (
    build_spacecraft_from_case,
    verification_case_paths,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = verification_case_paths(CONFIG_DIR)


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_sizing_value_ranges_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Ensures sizing_model returns bounded, physically plausible mass and
    #   power budgets for all geometry/wake verification cases.

    sc = build_spacecraft_from_case(config_path)

    # Exercise power-chain overhead and power-budget closure with non-zero loads.
    sc.power.Power_in = 100.0
    sc.power.Power_body = 50.0
    sc.power.Power_rad = 20.0
    sc.power.Power_payload = 30.0
    sc.power.Power_refprop = 40.0
    sc.power.Power_total = 1200.0
    sizing_model(sc)

    # Mass ranges for the two area families in the config set.
    _assert_in_range("Mass_in", sc.mass.Mass_in, 0, 1000)
    _assert_in_range("Mass_body", sc.mass.Mass_body, 0, 1000)
    _assert_in_range("Mass_total", sc.mass.Mass_total, 0, 10000)
    
    _assert_in_range("Mass_solar", sc.mass.Mass_solar, 0, 10000)
    _assert_in_range("Mass_rad", sc.mass.Mass_rad, 0, 10000)

    _assert_in_range("Power_solar", sc.power.Power_solar, 0, 100000)
    _assert_in_range("Power_prop", sc.power.Power_prop, 0, 100000)
    _assert_in_range("Power_total", sc.power.Power_total, 0, 100000)

    mass_sum = (
        sc.mass.Mass_in
        + sc.mass.Mass_body
        + sc.mass.Mass_solar
        + sc.mass.Mass_rad
        + sc.mass.Mass_prop
        + sc.mass.Mass_ADCS
        + sc.mass.Mass_payload
        + sc.mass.Mass_refprop
    )
    assert sc.mass.Mass_total == pytest.approx(mass_sum, rel=1.0e-12, abs=1.0e-12)

    power_sum = (
        sc.power.Power_in
        + sc.power.Power_body
        + sc.power.Power_solar
        + sc.power.Power_rad
        + sc.power.Power_prop
        + sc.power.Power_ADCS
        + sc.power.Power_payload
        + sc.power.Power_refprop
    )
    assert sc.power.Power_total == pytest.approx(power_sum, rel=1.0e-12, abs=1.0e-12)

