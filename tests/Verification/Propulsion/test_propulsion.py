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
#      Unit tests for deterministic propulsion-model branch behavior.
#
#  Project:        ARISS
#  Module:         test_propulsion.py
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
from ariss.modules import Propulsion as propulsion_module
from ariss.modules.Propulsion import propulsion_model
from ariss.utils import constants as const


def _stub_orbit_updates_from_density(_density: float, **_kwargs) -> dict[str, float]:
    return {
        "altitude": 150.0,
        "temperature": 300.0,
        "molar_mass": 0.028,
        "velocity": 100.0,
        "R_spec": 287.0,
    }


def _build_propulsion_state() -> SpacecraftState:
    sc = SpacecraftState()
    sc.orbit.velocity = 100.0
    sc.orbit.density = 2.0
    sc.orbit.temperature = 300.0
    sc.orbit.molar_mass = 0.028

    sc.geometry.S_in = "s"
    sc.geometry.S_body = "s"
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.L_in = 0.0
    sc.geometry.L_body = 0.0
    sc.geometry.A_ref = 0.0
    sc.geometry.A_in_drag = 0.0
    sc.geometry.A_rad = 0.0
    sc.geometry.A_solar = 1.0

    sc.drag.cd_solar = 2.0
    sc.drag.cd_rad = 0.0
    sc.drag.cd_body_side = 0.0
    sc.drag.cd_inlet_side = 0.0
    sc.drag.cd_inlet_front = 0.0

    sc.thruster.specific_impulse = 200.0 / const.EARTH_GRAVITY
    sc.thruster.power = 20000.0
    sc.thruster.eff = 0.5

    sc.refueling.coll_eff = 0.25
    sc.mission_profile.active_refueling = False
    return sc


def test_fixed_body_ratio_mode_preserves_body_area_and_imposes_intake_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inputs:
    #   Fixed-body ratio-mode spacecraft with zero refueling.
    #
    # Outputs:
    #   Ensures ratio mode keeps A_body fixed and splits A_in by collection efficiency.

    monkeypatch.setattr(propulsion_module, "orbit_updates_from_density", _stub_orbit_updates_from_density)

    sc = _build_propulsion_state()
    sc.geometry.use_intake_area_ratio = True
    sc.geometry.fixed_body = True
    sc.geometry.intake_area_ratio = 1.5
    sc.geometry.A_body = 2.0

    propulsion_model(sc)

    assert sc.geometry.A_body == pytest.approx(2.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in == pytest.approx(3.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_prop == pytest.approx(0.75, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in_drag == pytest.approx(2.25, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_ref == pytest.approx(0.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.orbit.altitude == pytest.approx(150.0, rel=1.0e-12, abs=1.0e-12)


def test_variable_body_ratio_mode_rebuilds_body_area_from_solved_intake(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inputs:
    #   Ratio-mode spacecraft with free body area and zero refueling.
    #
    # Outputs:
    #   Ensures solved A_in updates A_body through the configured intake-area ratio.

    monkeypatch.setattr(propulsion_module, "orbit_updates_from_density", _stub_orbit_updates_from_density)

    sc = _build_propulsion_state()
    sc.geometry.use_intake_area_ratio = True
    sc.geometry.fixed_body = False
    sc.geometry.intake_area_ratio = 2.0
    sc.geometry.A_body = 10.0

    propulsion_model(sc)

    assert sc.geometry.A_prop == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in_drag == pytest.approx(3.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in == pytest.approx(4.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_body == pytest.approx(2.0, rel=1.0e-12, abs=1.0e-12)


def test_free_intake_mode_keeps_body_area_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inputs:
    #   Free-intake spacecraft with zero refueling.
    #
    # Outputs:
    #   Ensures the legacy collection-efficiency split does not modify A_body.

    monkeypatch.setattr(propulsion_module, "orbit_updates_from_density", _stub_orbit_updates_from_density)

    sc = _build_propulsion_state()
    sc.geometry.use_intake_area_ratio = False
    sc.geometry.fixed_body = False
    sc.geometry.A_body = 10.0

    propulsion_model(sc)

    assert sc.geometry.A_prop == pytest.approx(1.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in_drag == pytest.approx(3.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_in == pytest.approx(4.0, rel=1.0e-12, abs=1.0e-12)
    assert sc.geometry.A_body == pytest.approx(10.0, rel=1.0e-12, abs=1.0e-12)
