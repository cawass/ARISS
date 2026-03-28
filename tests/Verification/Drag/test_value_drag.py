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
#      Value checks for drag coefficients and drag-force terms across the
#      drag verification configuration matrix after running drag and
#      propulsion models.
#
#  Project:        ARISS
#  Module:         test_value_drag.py
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
from ariss.utils.atmosphere import atmospheric_properties_from_height
from tests.Verification._cases import (
    build_spacecraft_from_case,
    verification_case_paths,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = verification_case_paths(CONFIG_DIR)


def _build_state(config_path: Path) -> SpacecraftState:
    # Inputs:
    #   config_path: path to a spacecraft TOML.
    #
    # Outputs:
    #   Spacecraft state initialized from the selected TOML with
    #   atmosphere-dependent orbit values filled from altitude.
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
        pytest.skip("pymsis is required for drag/propulsion value tests.")

    for key, value in updates.items():
        setattr(sc.orbit, key, value)
    return sc


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_drag_values_after_drag_and_propulsion_models_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Validates that drag coefficients and drag force components remain inside
    #   expected physical ranges for each drag verification geometry/wake case.
    #
    # Equations used:
    #   drag_model() updates cd terms
    #   propulsion_model() updates q-based drag forces and drag_total

    sc = _build_state(config_path)

    drag_model(sc)
    _assert_in_range("cd_solar", sc.drag.cd_solar, 1.0e-2, 2.0e-1)
    _assert_in_range("cd_rad", sc.drag.cd_rad, 1.0e-2, 2.0e-1)
    _assert_in_range("cd_body_side", sc.drag.cd_body_side, 3.0e-2, 1)
    _assert_in_range("cd_inlet_side", sc.drag.cd_inlet_side, 1.0e-2, 2)
    _assert_in_range("cd_inlet_front", sc.drag.cd_inlet_front, 0.5, 3)

    propulsion_model(sc)

    _assert_in_range("drag_solar", sc.drag.drag_solar, 0, 1)
    _assert_in_range("drag_body_side", sc.drag.drag_body_side, 5.0e-4, 1)
    _assert_in_range("drag_inlet_side", sc.drag.drag_inlet_side, 5.0e-4, 1)
    _assert_in_range("drag_inlet_front", sc.drag.drag_inlet_front, 5.0e-4, 1)
    _assert_in_range("drag_total", sc.drag.drag_total, 5.0e-4, 1)

    assert sc.drag.drag_rad >= 0.0, "drag_rad must be non-negative."
    assert sc.drag.drag_body_side > 0.0, "drag_body_side must be strictly positive."
    assert sc.drag.drag_inlet_side > 0.0, "drag_inlet_side must be strictly positive."
    assert sc.drag.drag_inlet_front > 0.0, "drag_inlet_front must be strictly positive."
    component_sum = (
        sc.drag.drag_solar
        + sc.drag.drag_rad
        + sc.drag.drag_body_side
        + sc.drag.drag_inlet_side
        + sc.drag.drag_inlet_front
    )
    assert sc.drag.drag_total == pytest.approx(component_sum, rel=1.0e-12, abs=1.0e-12)

