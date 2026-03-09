from __future__ import annotations

from os import PathLike
from typing import Any

from ariss.core.spacecraft import SpacecraftState, load_spacecraft


def launch_history_ui(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.visualization.history_ui import launch_history_ui as _launch_history_ui

    return _launch_history_ui(sc=sc, **kwargs)


def plot_simulation_history(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.visualization.history_ui import plot_simulation_history as _plot_simulation_history

    return _plot_simulation_history(sc=sc, **kwargs)


def run_simulation(
    sc: SpacecraftState | str | PathLike[str] | None = None,
    **kwargs: Any,
):
    from ariss.core.simulation import run_sizing_loop as _run_sizing_loop

    spacecraft = SpacecraftState() if sc is None else load_spacecraft(sc)
    return _run_sizing_loop(spacecraft, **kwargs)


__all__ = [
    "SpacecraftState",
    "launch_history_ui",
    "load_spacecraft",
    "plot_simulation_history",
    "run_simulation",
]
