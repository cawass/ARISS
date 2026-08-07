from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Sequence

from ariss.core.spacecraft import SpacecraftState

__version__ = "0.1.0"

def load_spacecraft(source: SpacecraftState | str | PathLike[str]) -> SpacecraftState:
    # Inputs:
    #   source: SpacecraftState instance or TOML path.
    #
    # Output:
    #   Spacecraft loaded from base config with optional case overrides.

    if isinstance(source, SpacecraftState):
        return source

    path = Path(source)
    if path.suffix.lower() == ".toml":
        from ariss.core.simulation import load_spacecraft_from_base_config

        return load_spacecraft_from_base_config(path)

    raise ValueError(f"Unsupported spacecraft file format: {path.suffix or '<no extension>'}")


def launch_history_ui(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.core.simulation_ui import launch_history_ui as _launch_history_ui

    return _launch_history_ui(sc=sc, **kwargs)


def plot_simulation_history(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.core.simulation_ui import plot_simulation_history as _plot_simulation_history

    return _plot_simulation_history(sc=sc, **kwargs)


def run_simulation(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.core.simulation import load_spacecraft_from_base_config, run_sizing_loop

    spacecraft = load_spacecraft_from_base_config() if sc is None else load_spacecraft(sc)
    return run_sizing_loop(spacecraft, **kwargs)


def run_sensitivity(*args: Any, **kwargs: Any):
    from ariss.core.sensitivity import run_sensitivity_analysis as _run_sensitivity

    return _run_sensitivity(*args, **kwargs)


def run_core_sensitivity(*args: Any, **kwargs: Any):
    from ariss.core.sensitivity import core_sensitivity as _run_core_sensitivity

    return _run_core_sensitivity(*args, **kwargs)


def run_sensitivity_ranking(*args: Any, **kwargs: Any):
    from ariss.core.sensitivity import run_sensitivity_ranking as _run_sensitivity_ranking

    return _run_sensitivity_ranking(*args, **kwargs)


def run_efficiency_sensitivity_ranking(
    output_paths: str | Sequence[str],
    *,
    perturbation: float = 0.10,
    epsilon_path: str = "geometry.epsilon_body",
    case_path=None,
    base_config_path=None,
    max_iterations: int = 200,
):
    from ariss.core.sensitivity import (
        run_sensitivity_ranking as _run_sensitivity_ranking,
    )

    parameters = [
        {"label": "Thruster efficiency", "variable_path": "thruster.eff"},
        {"label": "Collection efficiency", "variable_path": "refueling.coll_eff"},
        {"label": "Specular reflection fraction", "variable_path": epsilon_path},
        {"label": "Solar-cell efficiency", "variable_path": "solar.eta_solar"},
    ]
    return _run_sensitivity_ranking(
        parameters=parameters,
        output_paths=output_paths,
        perturbation=perturbation,
        case_path=case_path,
        base_config_path=base_config_path,
        max_iterations=max_iterations,
    )


__all__ = [
    "SpacecraftState",
    "launch_history_ui",
    "load_spacecraft",
    "plot_simulation_history",
    "run_efficiency_sensitivity_ranking",
    "run_core_sensitivity",
    "run_sensitivity",
    "run_sensitivity_ranking",
    "run_simulation",
]
