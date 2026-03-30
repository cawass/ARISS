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
#      Core sensitivity utilities for ARISS. Provides point-wise parameter
#      sweeps for direct sizing and refueling-time search modes.
#
#  Project:        ARISS
#  Module:         sensitivity.py
#  Author:         Carlos Carrasco Requejo, Lucas Calderon del Rio
# ============================================================================ #
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Sequence
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ariss.core.simulation import load_spacecraft_from_base_config, run_sizing_loop

SECONDS_PER_MONTH = 30.4375 * 24.0 * 3600.0

SensitivityResult = dict[str, Any]


# --------------------------------------------------------------------------------------
# Core path access
# --------------------------------------------------------------------------------------
def get_path(obj: Any, path: str) -> Any:
    # Inputs:
    #   obj: root object to traverse.
    #   path: dotted attribute path (e.g. "orbit.altitude").
    #
    # Output:
    #   Value stored at the target attribute path.
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    # Inputs:
    #   obj: root object to mutate.
    #   path: dotted attribute path to set.
    #   value: new value for the target attribute.
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    object.__setattr__(obj, parts[-1], value)


# --------------------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------------------
def find_min_refuel_time(
    sc: Any,
    *,
    max_iterations: int,
    min_value: float = 1.0,
    shrink: float = 0.9,
    expand: float = 1.25,
    coarse_steps: int = 80,
    refine_steps: int = 20,
) -> tuple[Any | None, bool, str | None]:
    # Inputs:
    #   sc: baseline spacecraft state.
    #   max_iterations: maximum sizing-loop iterations per trial.
    #   min_value: lower bound for refueling time [s].
    #   shrink/expand: multiplicative search factors.
    #   coarse_steps: iterations for expand/shrink bracketing.
    #   refine_steps: bisection refinement iterations.
    #
    # Outputs:
    #   (best_sc, converged, error_message)
    #   best_sc carries the minimum converged refueling time when converged.
    sc = deepcopy(sc)
    set_path(sc, "mission_profile.active_refueling", True)

    first_error = None
    t = max(float(get_path(sc, "refueling.t_refuel")), min_value)
    best_sc = None
    best_t = None

    def solve(value: float) -> tuple[Any | None, bool]:
        nonlocal first_error
        trial = deepcopy(sc)
        set_path(trial, "refueling.t_refuel", value)
        try:
            final_sc, ok, _ = run_sizing_loop(trial, max_iterations=max_iterations)
            return final_sc, ok
        except Exception as exc:
            if first_error is None:
                first_error = str(exc)
            return None, False

    # Expand until success.
    for _ in range(coarse_steps):
        final_sc, ok = solve(t)
        if ok:
            best_sc, best_t = final_sc, t
            break
        t *= expand

    if best_sc is None:
        return None, False, first_error or "No converged refueling point found."

    # Shrink until first failure.
    low = best_t
    high = None
    t = best_t

    for _ in range(coarse_steps):
        next_t = max(t * shrink, min_value)
        if next_t >= t:
            break

        final_sc, ok = solve(next_t)
        if ok:
            low = next_t
            best_t = next_t
            best_sc = final_sc
            t = next_t
            if next_t <= min_value:
                break
        else:
            high = next_t
            break

    # Refine with bisection.
    if high is not None:
        for _ in range(refine_steps):
            if abs(high - low) <= max(1e-6, 1e-4 * max(low, 1.0)):
                break

            mid = 0.5 * (low + high)
            final_sc, ok = solve(mid)
            if ok:
                low = mid
                best_t = mid
                best_sc = final_sc
            else:
                high = mid

    if best_sc is None or best_t is None:
        return None, False, first_error or "Failed to determine minimum refueling time."

    set_path(best_sc, "refueling.t_refuel", best_t)
    return best_sc, True, None


def run_sensitivity_analysis(
    variable_paths: str | Sequence[str],
    variable_values: Sequence[Any] | Sequence[Sequence[Any]],
    output_paths: str | Sequence[str],
    *,
    combine: str = "zip",
    case_path=None,
    base_config_path=None,
    base_sc: Any | None = None,
    max_iterations: int = 200,
    mode: str | None = None,
) -> SensitivityResult:
    # Inputs:
    #   variable_paths:
    #       Single dotted path or list of dotted paths to perturb.
    #   variable_values:
    #       Single list for one variable, or one list per variable.
    #   output_paths:
    #       One or more dotted output paths to record.
    #   combine:
    #       "zip" to pair values by index, "product" for Cartesian product.
    #   case_path / base_config_path / base_sc:
    #       Base spacecraft source. base_sc takes precedence when provided.
    #   max_iterations:
    #       Iteration cap passed to the sizing solver.
    #   mode:
    #       "direct" or "refuel_search". If omitted, inferred from outputs.
    #
    # Output:
    #   Dictionary with sweep points, outputs per path, convergence flags,
    #   and error strings.

    paths = [variable_paths] if isinstance(variable_paths, str) else [str(path) for path in variable_paths]
    if not paths:
        raise ValueError("variable_paths cannot be empty.")

    if isinstance(variable_paths, str):
        values = list(variable_values)  # type: ignore[arg-type]
        if not values:
            raise ValueError("variable_values cannot be empty.")
        value_lists: list[list[Any]] = [values]
    else:
        if len(variable_values) != len(paths):  # type: ignore[arg-type]
            raise ValueError("variable_paths and variable_values must have the same length.")

        value_lists = []
        for values in variable_values:  # type: ignore[assignment]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                current = [values]
            else:
                current = list(values)
            if not current:
                raise ValueError("Each entry in variable_values must contain at least one value.")
            value_lists.append(current)

    output_paths = [] if output_paths is None else ([output_paths] if isinstance(output_paths, str) else list(output_paths))
    if not output_paths:
        raise ValueError("output_paths cannot be empty.")

    mode = mode if mode is not None else ("refuel_search" if "refueling.t_refuel" in output_paths else "direct")

    base_sc = deepcopy(base_sc) if base_sc is not None else load_spacecraft_from_base_config(
        case_path=case_path,
        base_config_path=base_config_path,
    )

    if combine == "zip":
        lengths = {len(v) for v in value_lists if len(v) > 1}
        n = max(lengths, default=1)
        if any(len(v) not in (1, n) for v in value_lists):
            raise ValueError(
                "For combine='zip', each value list must have length 1 or the same common length."
            )
        points = [
            {
                path: values[0] if len(values) == 1 else values[i]
                for path, values in zip(paths, value_lists)
            }
            for i in range(n)
        ]
    elif combine == "product":
        points = [
            {path: value for path, value in zip(paths, combo)}
            for combo in product(*value_lists)
        ]
    else:
        raise ValueError("combine must be 'zip' or 'product'.")

    outputs = {path: [] for path in output_paths}
    converged: list[bool] = []
    errors: list[str | None] = []

    for point in points:
        sc = deepcopy(base_sc)
        for path, value in point.items():
            set_path(sc, path, value)

        run_outputs = {path: None for path in output_paths}
        run_ok = False
        run_error: str | None = None
        try:
            if mode == "refuel_search":
                final_sc, run_ok, run_error = find_min_refuel_time(sc, max_iterations=max_iterations)
            else:
                final_sc, run_ok, _ = run_sizing_loop(sc, max_iterations=max_iterations)
                run_error = None

            if run_ok and final_sc is not None:
                for path in output_paths:
                    value = get_path(final_sc, path)
                    if path == "refueling.t_refuel" and value is not None:
                        run_outputs[path] = float(value) / SECONDS_PER_MONTH
                    else:
                        run_outputs[path] = value
            else:
                run_ok = False
                run_error = run_error or "Sizing loop did not converge."
        except Exception as exc:
            run_ok = False
            run_error = str(exc)

        for path in output_paths:
            outputs[path].append(run_outputs[path])
        converged.append(run_ok)
        errors.append(run_error)

    primary_values = [point[paths[0]] for point in points]
    return {
        "label": "Sensitivity",
        "x_label": paths[0] if len(paths) == 1 else "case",
        "values": primary_values if len(paths) == 1 else points,
        "variable_values": primary_values if len(paths) == 1 else list(range(len(points))),
        "outputs": outputs,
        "converged": converged,
        "errors": errors,
        "variable_paths": paths,
        "points": points,
        "point_labels": [", ".join(f"{k}={v}" for k, v in point.items()) for point in points],
        "output_paths": output_paths,
    }
