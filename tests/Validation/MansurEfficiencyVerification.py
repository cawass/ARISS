from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState


def sweep_mansur_efficiencies(
    config_path: str | Path | None = None,
    collection_efficiencies: tuple[float, ...] = (0.35, 0.40, 0.45),
    isp_values: np.ndarray | None = None,
    max_iterations: int = 400,
    mass_tolerance: float = 1.0e-3,
) -> dict[float, dict[str, np.ndarray]]:
    base_path = Path(config_path) if config_path is not None else Path(__file__).with_name("MansurVerification.toml")
    base_state = SpacecraftState.from_toml(base_path)
    isp_grid = np.asarray(isp_values if isp_values is not None else np.linspace(1800.0, 7000.0, 64), dtype=float)
    results: dict[float, dict[str, np.ndarray]] = {}

    previous_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)
    try:
        for efficiency in collection_efficiencies:
            converged_altitudes: list[float] = []
            converged_isp: list[float] = []
            for isp in isp_grid:
                print(f"Running ISP: {isp}")
                sc = deepcopy(base_state)
                sc.refueling.coll_eff = float(efficiency)
                sc.thruster.specific_impulse = float(isp)
                with redirect_stdout(io.StringIO()):
                    final_sc, converged, _history = run_sizing_loop(
                        sc,
                        max_iterations=max_iterations,
                        mass_tolerance=mass_tolerance,
                    )
                if converged:
                    converged_altitudes.append(float(final_sc.orbit.altitude))
                    converged_isp.append(float(isp))
            results[float(efficiency)] = {
                "altitude_km": np.asarray(converged_altitudes, dtype=float),
                "isp_s": np.asarray(converged_isp, dtype=float),
            }
    finally:
        simulation_logger.setLevel(previous_level)

    return results


def plot_mansur_efficiency_verification(
    config_path: str | Path | None = None,
    collection_efficiencies: tuple[float, ...] = (0.35, 0.40, 0.45),
    isp_values: np.ndarray | None = None,
    max_iterations: int = 400,
    mass_tolerance: float = 1.0e-3,
    show: bool = True,
    save_path: str | Path | None = None,
):
    results = sweep_mansur_efficiencies(
        config_path=config_path,
        collection_efficiencies=collection_efficiencies,
        isp_values=isp_values,
        max_iterations=max_iterations,
        mass_tolerance=mass_tolerance,
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    colors = ["#1f77b4", "#d95f02", "#e6ab02"]

    for color, efficiency in zip(colors, collection_efficiencies):
        data = results.get(float(efficiency), {})
        altitude_km = np.asarray(data.get("altitude_km", np.array([])), dtype=float)
        isp_s = np.asarray(data.get("isp_s", np.array([])), dtype=float)
        if altitude_km.size == 0:
            continue
        axis.plot(altitude_km, isp_s, color=color, linewidth=1.6, label=fr"$I_{{sp}}$ for $\eta_c = {efficiency:.2f}$")
        solution_limit = float(np.min(altitude_km))
        axis.axvline(
            solution_limit,
            color=color,
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            label=f"Solution limit at {solution_limit:.1f} km",
        )

    axis.set_xlabel("Converged altitude (km)")
    axis.set_ylabel("Isp (s)")
    axis.set_title("Mansur Verification Sweep")
    axis.set_xlim(140.0, 200.0)
    axis.set_ylim(1500.0, 6000.0)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(Path(save_path), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axis, results


if __name__ == "__main__":
    plot_mansur_efficiency_verification()
