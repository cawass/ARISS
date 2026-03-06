"""Budget-vs-driver plotting for simulation history."""

import os
import sys
from pathlib import Path

import matplotlib

if os.environ.get("SIM_BUDGET_PLOT_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState
from ariss.utils import constants as const


def _prop_mflow_from_state(sc: SpacecraftState) -> float:
    v_exhaust = const.EARTH_GRAVITY * sc.thruster.specific_impulse
    delta_v = v_exhaust - sc.orbit.velocity
    if delta_v <= 0.0:
        return 0.0
    return sc.thruster.thrust / delta_v


def _mass_budget_contributions(sc: SpacecraftState) -> dict[str, float]:
    b = sc.budget
    mflow_prop = _prop_mflow_from_state(sc)
    adcs_mass = b.B_mass_mass_ADCS * sc.total_mass
    return {
        "B_mass_total": b.B_mass_total,
        "B_mass_volume_in": b.B_mass_volume_in * (sc.geometry.A_in * sc.geometry.L_in),
        "B_mass_volume_body": b.B_mass_volume_body * (sc.geometry.A_body * sc.geometry.L_body),
        "B_mass_surface_solar": b.B_mass_surface_solar * sc.geometry.A_solar,
        "B_mass_surface_rad": b.B_mass_surface_rad * sc.geometry.A_rad,
        "B_mass_power_prop": b.B_mass_power_prop * sc.thruster.power_required,
        "B_mass_mass_ADCS": adcs_mass,
        "B_mass_payload": b.B_mass_payload,
        "B_mass_mflow_prop": b.B_mass_mflow_prop * mflow_prop,
        "B_mass_mflow_ref": b.B_mass_mflow_ref * 0.0,
    }


def _power_budget_contributions(sc: SpacecraftState) -> dict[str, float]:
    b = sc.budget
    mflow_prop = _prop_mflow_from_state(sc)
    adcs_mass = b.B_mass_mass_ADCS * sc.total_mass
    return {
        "B_power_power_solar": b.B_power_power_solar * sc.geometry.A_solar,
        "B_power_power_prop": b.B_power_power_prop * sc.thruster.power_required,
        "B_power_mass_ADCS": b.B_power_mass_ADCS * adcs_mass,
        "B_power_payload": b.B_power_payload,
        "B_power_mflow_prop": b.B_power_mflow_prop * mflow_prop,
        "B_power_mflow_ref": b.B_power_mflow_ref * 0.0,
    }


def _driving_amounts(sc: SpacecraftState) -> dict[str, float]:
    mflow_prop = _prop_mflow_from_state(sc)
    adcs_mass = sc.budget.B_mass_mass_ADCS * sc.total_mass
    return {
        "prop_power": sc.thruster.power_required,
        "prop_mflow": mflow_prop,
        "ref_mflow": 0.0,
        "A_in": sc.geometry.A_in,
        "A_solar": sc.geometry.A_solar,
        "A_rad": sc.geometry.A_rad,
        "volume_in": sc.geometry.A_in * sc.geometry.L_in,
        "volume_body": sc.geometry.A_body * sc.geometry.L_body,
        "total_mass": sc.total_mass,
        "adcs_mass": adcs_mass,
        "unit": 1.0,
    }


def _to_series(history: list[SpacecraftState], fn):
    keys = list(fn(history[0]).keys())
    series = {key: [] for key in keys}
    for state in history:
        values = fn(state)
        for key in keys:
            series[key].append(values[key])
    return series


def _positive_for_log(values: list[float], floor: float = 1e-18) -> list[float]:
    return [value if value > floor else floor for value in values]


def _plot_group(ax, it: list[int], pairs: list[tuple[str, list[float], str, list[float], str]], title: str):
    for budget_label, budget_values, driver_label, driver_values, color in pairs:
        ax.plot(it, _positive_for_log(budget_values), linewidth=1.4, marker="o", color=color, linestyle="-", label=f"{budget_label} (budget)")
        ax.plot(it, _positive_for_log(driver_values), linewidth=1.2, marker="o", color=color, linestyle=":", label=f"{driver_label} (driver)")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)


def _budget_driver_groups(mass_series: dict[str, list[float]], power_series: dict[str, list[float]], driver_series: dict[str, list[float]]):
    return [
        (
            "Volume-Driven Mass Budgets",
            [
                ("B_mass_volume_in", mass_series["B_mass_volume_in"], "volume_in", driver_series["volume_in"], "tab:blue"),
                ("B_mass_volume_body", mass_series["B_mass_volume_body"], "volume_body", driver_series["volume_body"], "tab:orange"),
            ],
        ),
        (
            "Area-Driven Budgets",
            [
                ("B_mass_surface_solar", mass_series["B_mass_surface_solar"], "A_solar", driver_series["A_solar"], "tab:green"),
                ("B_power_power_solar", power_series["B_power_power_solar"], "A_solar", driver_series["A_solar"], "tab:red"),
                ("B_mass_surface_rad", mass_series["B_mass_surface_rad"], "A_rad", driver_series["A_rad"], "tab:purple"),
            ],
        ),
        (
            "Propulsion-Power-Driven Budgets",
            [
                ("B_mass_power_prop", mass_series["B_mass_power_prop"], "prop_power", driver_series["prop_power"], "tab:brown"),
                ("B_power_power_prop", power_series["B_power_power_prop"], "prop_power", driver_series["prop_power"], "tab:pink"),
            ],
        ),
        (
            "Massflow-Driven Budgets",
            [
                ("B_mass_mflow_prop", mass_series["B_mass_mflow_prop"], "prop_mflow", driver_series["prop_mflow"], "tab:cyan"),
                ("B_power_mflow_prop", power_series["B_power_mflow_prop"], "prop_mflow", driver_series["prop_mflow"], "tab:gray"),
                ("B_mass_mflow_ref", mass_series["B_mass_mflow_ref"], "ref_mflow", driver_series["ref_mflow"], "tab:olive"),
                ("B_power_mflow_ref", power_series["B_power_mflow_ref"], "ref_mflow", driver_series["ref_mflow"], "tab:orange"),
            ],
        ),
        (
            "Constant / Coupled-Coef Budgets",
            [
                ("B_mass_total", mass_series["B_mass_total"], "unit", driver_series["unit"], "tab:blue"),
                ("B_mass_mass_ADCS", mass_series["B_mass_mass_ADCS"], "total_mass", driver_series["total_mass"], "tab:green"),
                ("B_mass_payload", mass_series["B_mass_payload"], "unit", driver_series["unit"], "tab:red"),
                ("B_power_payload", power_series["B_power_payload"], "unit", driver_series["unit"], "tab:purple"),
                ("B_power_mass_ADCS", power_series["B_power_mass_ADCS"], "adcs_mass", driver_series["adcs_mass"], "tab:brown"),
            ],
        ),
    ]


def plot_simulation_budgets(
    sc: SpacecraftState | None = None,
    max_iterations: int = 10,
    mass_tolerance: float = 1e-9,
    show: bool = True,
):
    """Plot grouped budget terms against their driving quantities."""
    if sc is None:
        sc = SpacecraftState()
    _, _, history = run_sizing_loop(sc, max_iterations=max_iterations, mass_tolerance=mass_tolerance)

    it = list(range(len(history)))
    mass_series = _to_series(history, _mass_budget_contributions)
    power_series = _to_series(history, _power_budget_contributions)
    driver_series = _to_series(history, _driving_amounts)
    groups = _budget_driver_groups(mass_series, power_series, driver_series)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    flat_axes = axes.flatten()
    for idx, (title, pairs) in enumerate(groups):
        _plot_group(flat_axes[idx], it, pairs, title)
    flat_axes[0].set_ylabel("Magnitude (log scale)")
    for idx in range(len(groups), len(flat_axes)):
        flat_axes[idx].axis("off")

    if show:
        plt.show()
    return fig, axes, history


if __name__ == "__main__":
    plot_simulation_budgets()
