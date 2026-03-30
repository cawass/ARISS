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
#      Verification tests for spacecraft loader behavior in core simulation.
#
#  Project:        ARISS
#  Module:         test_simulation_loader.py
# ============================================================================== #

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.core.spacecraft import SpacecraftState
from ariss.core import spacecraft as spacecraft_module


BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"


def test_loader_matches_direct_spacecraft_loading_for_default_base_case() -> None:
    # Inputs:
    #   No case override and default base_config.toml.
    #
    # Outputs:
    #   Ensures loader returns the same state as direct SpacecraftState.from_toml.

    loaded = load_spacecraft_from_base_config()
    expected = SpacecraftState.from_toml(BASE_CONFIG_PATH)

    assert loaded == expected


def test_loader_applies_nested_case_overrides(tmp_path: Path) -> None:
    # Inputs:
    #   Existing verification override with nested sections.
    #
    # Outputs:
    #   Ensures nested values are applied on top of the base config.

    case_path = tmp_path / "nested_override.toml"
    case_path.write_text(
        (
            'name = "Loader nested override case"\n\n'
            "[orbit]\n"
            "altitude = 245.5\n\n"
            "[geometry]\n"
            'S_in = "s"\n'
            'S_body = "c"\n\n'
            "[mission_profile]\n"
            "active_refueling = true\n"
            "active_and_bypass = false\n"
        ),
        encoding="utf-8",
    )
    loaded = load_spacecraft_from_base_config(case_path=case_path)

    assert loaded.name == "Loader nested override case"
    assert loaded.orbit.altitude == pytest.approx(245.5, rel=1.0e-12, abs=1.0e-12)
    assert loaded.geometry.S_in == "s"
    assert loaded.geometry.S_body == "c"
    assert loaded.mission_profile.active_refueling is True
    assert loaded.mission_profile.active_and_bypass is False


def test_loader_raises_on_unknown_override_key(tmp_path: Path) -> None:
    # Inputs:
    #   Override TOML containing a non-existent nested key.
    #
    # Outputs:
    #   Ensures loader fails fast with a KeyError and dotted path context.

    case_path = tmp_path / "bad_override.toml"
    case_path.write_text(
        "[orbit]\nunknown_field = 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError) as exc_info:
        load_spacecraft_from_base_config(case_path=case_path)

    assert "Unknown key in spacecraft override: orbit.unknown_field" in str(exc_info.value)


def test_loader_uses_explicit_base_config_path(tmp_path: Path) -> None:
    # Inputs:
    #   Custom base_config path with a modified case name.
    #
    # Outputs:
    #   Ensures loader reads from the provided base_config_path argument.

    original_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")
    custom_text = original_text.replace(
        'name = "Drag Base Case"',
        'name = "Custom loader base case"',
        1,
    )

    custom_base = tmp_path / "custom_base.toml"
    custom_base.write_text(custom_text, encoding="utf-8")

    loaded = load_spacecraft_from_base_config(base_config_path=custom_base)

    assert loaded.name == "Custom loader base case"


def test_spacecraft_can_be_saved_to_toml_and_loaded_back(tmp_path: Path) -> None:
    # Inputs:
    #   Spacecraft state with edited values and writable output path.
    #
    # Outputs:
    #   Ensures SpacecraftState.to_toml writes a valid file that round-trips
    #   through SpacecraftState.from_toml.

    if spacecraft_module.tomli_w is None:
        pytest.skip("tomli_w is not installed; to_toml writer path unavailable.")

    sc = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    object.__setattr__(sc, "name", "Roundtrip TOML case")
    object.__setattr__(sc.orbit, "altitude", 312.25)
    object.__setattr__(sc.geometry, "A_in", 2.75)
    object.__setattr__(sc.mission_profile, "active_refueling", True)
    object.__setattr__(sc.mission_profile, "active_and_bypass", False)

    output_path = tmp_path / "roundtrip_case.toml"
    sc.to_toml(output_path)

    loaded = SpacecraftState.from_toml(output_path)

    assert loaded == sc
