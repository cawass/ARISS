from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from ariss.core.spacecraft import SpacecraftState


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


__all__ = [
    "SpacecraftState",
    "launch_history_ui",
    "load_spacecraft",
    "plot_simulation_history",
    "run_simulation",
]
