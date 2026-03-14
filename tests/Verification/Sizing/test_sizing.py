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
#      Unit tests for deterministic sizing-model mass and power closure.
#
#  Project:        ARISS
#  Module:         test_sizing.py
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


def test_sizing_model_matches_mass_component_formulas() -> None:
    # Inputs:
    #   Spacecraft geometry and rate values with analytically known masses.
    #
    # Outputs:
    #   Ensures sizing_model reproduces the implemented mass-scaling relations.

    sc = SpacecraftState()
    sc.geometry.A_in = 2.0
    sc.geometry.A_body = 4.0
    sc.geometry.A_solar = 6.0
    sc.geometry.A_rad = 3.0
    sc.geometry.L_in = 5.0
    sc.geometry.L_body = 7.0
    object.__setattr__(sc.rate, "R_mass_volume_in", 10.0)
    object.__setattr__(sc.rate, "R_mass_volume_body", 20.0)
    object.__setattr__(sc.rate, "R_mass_surface_solar", 2.0)
    object.__setattr__(sc.rate, "R_mass_surface_rad", 4.0)
    sc.mass.Mass_prop = 11.0
    sc.mass.Mass_ADCS = 13.0
    sc.mass.Mass_payload = 17.0
    sc.mass.Mass_refprop = 19.0

    sizing_model(sc)

    assert sc.mass.Mass_in == pytest.approx(150.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.mass.Mass_body == pytest.approx(560.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.mass.Mass_solar == pytest.approx(12.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.mass.Mass_rad == pytest.approx(12.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.mass.Mass_total == pytest.approx(794.0, rel=1.0e-12, abs=1.0e-12)


def test_sizing_model_rebuilds_power_budget_from_losses_and_thruster_power() -> None:
    # Inputs:
    #   Pre-sizing subsystem power loads and a known power-chain efficiency.
    #
    # Outputs:
    #   Ensures sizing_model recomputes Power_solar, Power_prop, and Power_total
    #   from the implemented closure equations.

    sc = SpacecraftState()
    object.__setattr__(sc.solar, "eta_power", 0.8)
    sc.thruster.power = 500.0
    sc.power.Power_total = 1000.0
    sc.power.Power_in = 10.0
    sc.power.Power_body = 20.0
    sc.power.Power_rad = 30.0
    sc.power.Power_ADCS = 40.0
    sc.power.Power_payload = 50.0
    sc.power.Power_refprop = 60.0

    sizing_model(sc)

    assert sc.power.Power_prop == pytest.approx(500.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.power.Power_solar == pytest.approx(250.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.power.Power_total == pytest.approx(960.0, rel=1.0e-12, abs=1.0e-12)
