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
from ariss.utils.atmosphere import atmospheric_properties_from_height
from tests.Verification._cases import (
    build_spacecraft_from_case,
    verification_case_paths,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = verification_case_paths(CONFIG_DIR)


def _build_state(config_path: Path) -> SpacecraftState:
    sc = build_spacecraft_from_case(config_path)
    try:
        properties = atmospheric_properties_from_height(
            sc.orbit.altitude,
            msis_date=sc.orbit.msis_date,
            msis_f107=sc.orbit.msis_f107,
            msis_ap=sc.orbit.msis_ap,
        )
        updates = {
            "altitude": float(properties["altitude_km"]),
            "density": float(properties["density"]),
            "temperature": float(properties["temperature"]),
            "molar_mass": float(properties["molar_mass"]),
            "velocity": float(properties["orbital_velocity"]),
            "R_spec": float(properties["specific_gas_constant"]),
        }
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

    _assert_in_range("T_max", diagnostics.T_max, 250.0, 350.0)

