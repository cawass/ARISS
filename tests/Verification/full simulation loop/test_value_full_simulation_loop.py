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
#      Value-range checks for the fully coupled sizing loop across the
#      verification spacecraft configuration matrix.
#
#  Project:        ARISS
#  Module:         test_value_full_simulation_loop.py
# ============================================================================== #

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState
from tests.Verification._cases import (
    build_spacecraft_from_case,
    verification_case_paths,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = verification_case_paths(CONFIG_DIR)
RUN_UI_TEST = os.environ.get("ARISS_RUN_UI_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}


def _assert_in_range(name: str, value: float, low: float, high: float) -> None:
    assert low <= value <= high, f"{name}={value:.6e} is outside [{low:.6e}, {high:.6e}]"


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=lambda path: path.name)
def test_full_simulation_loop_value_ranges_for_all_configs(config_path: Path) -> None:
    # Inputs:
    #   tests/Verification/configs/*.toml
    #
    # Outputs:
    #   Ensures the full iterative loop converges and returns bounded final-state
    #   values for all verification spacecraft configurations.

    sc = build_spacecraft_from_case(config_path)

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)
    try:
        try:
            with redirect_stdout(io.StringIO()):
                final_sc, converged, history = run_sizing_loop(
                    sc,
                    max_iterations=120,
                    mass_tolerance=1.0e-3,
                )
        except ImportError:
            pytest.skip("pymsis is required for full-loop value tests.")
    finally:
        simulation_logger.setLevel(previous_level)

    assert converged, f"Sizing loop did not converge for {config_path.name}"
    _assert_in_range("iterations", float(len(history)), 1.0, 119.0)

    _assert_in_range("altitude", final_sc.orbit.altitude, 130.0, 400.0)
    _assert_in_range("density", final_sc.orbit.density, 7.0e-12, 8.2e-9)
    _assert_in_range("Mass_total", final_sc.mass.Mass_total, 0, 2000)
    _assert_in_range("thrust", final_sc.thruster.thrust, 0, 1)
    _assert_in_range("drag_total", final_sc.drag.drag_total, 0, 1)
    _assert_in_range("A_prop", final_sc.geometry.A_prop, 0, 10)
    _assert_in_range("A_in", final_sc.geometry.A_in, 0, 10)
    _assert_in_range("A_solar", final_sc.geometry.A_solar, 0, 50)
    _assert_in_range("A_rad", final_sc.geometry.A_rad, 0, 20)

    assert final_sc.thruster.thrust > final_sc.drag.drag_total, "Final thrust must exceed final drag."
    assert final_sc.geometry.A_in > final_sc.geometry.A_prop, "Final intake area must exceed propulsive intake area."


def test_run_ui_test() -> None:
    # Inputs:
    #   RUN_UI_TEST boolean toggle (via ARISS_RUN_UI_TEST env var).
    #
    # Outputs:
    #   When enabled, launches the history UI for each verification config for
    #   manual visualization.
    #
    # Usage:
    #   PowerShell:
    #     $env:ARISS_RUN_UI_TEST='1'
    #     python -m pytest "tests\\Verification\\full simulation loop\\test_value_full_simulation_loop.py" -k run_ui_test -s

    if not RUN_UI_TEST:
        pytest.skip("UI test disabled. Set ARISS_RUN_UI_TEST=1 to run manual visualization.")

    try:
        from ariss.core.simulation_ui import plot_simulation_history
    except Exception as exc:
        pytest.skip(f"UI dependencies unavailable: {exc}")

    # Preflight Tk once so missing Tcl/Tk installations skip cleanly.
    try:
        import tkinter as tk

        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except Exception as exc:
        pytest.skip(f"Tk UI backend unavailable: {exc}")

    for config_path in CONFIG_PATHS:
        sc = build_spacecraft_from_case(config_path)
        plot_simulation_history(
            sc,
            max_iterations=120,
            mass_tolerance=1.0e-3,
            show=True,
        )

