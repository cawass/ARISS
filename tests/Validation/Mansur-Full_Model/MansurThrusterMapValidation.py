from pathlib import Path
import sys
from copy import deepcopy
import io
import re
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for p in (SRC, VALIDATION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ariss.core.spacecraft import SpacecraftState
from ariss.core.simulation import run_sizing_loop, logger as simulation_logger
from ariss.utils.ploting import plot_validation_mansur_thruster_map
from csv_helper import load_wide_xy_csv
from ariss.utils.ploting import PALETTE
from validation_metrics import datapoint_relative_and_corr_stats, minimum_finite


BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"
CONFIG_PATH = HERE / "MansurValidation1000W.toml"
OUTPUT = HERE / "mansur_thruster_map_validation.png"
PAGE_FIGSIZE = (15.84, 5.4)
MATPLOTLIB_ONLY = True

ETA_DATASET_PATH = HERE / "Eta Dataset.csv"
MFLOW_DATASET_PATH = HERE / "M_flow Dataset.csv"
AIN_DATASET_PATH = HERE / "A_in dataset.csv"

EFF_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MDOT_LEVELS = [0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
AIN_LEVELS = [0.16, 0.2, 0.3, 0.5, 1.2]

EFF_COLOR = PALETTE["l1_teal"]
MDOT_COLOR = PALETTE["sernn_pink"]
AIN_COLOR = PALETTE["choice_mid"]

MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]

MARKER_SIZE = 6.2

# Marker placement in the plot
EFF_MARKER_SPEC = {"mode": "y_targets", "targets": [3225.0, 5550.0], "fallback_count": 2}
MDOT_MARKER_SPEC = {"mode": "x_targets", "targets": [8.5, 26.0], "fallback_count": 2}
AIN_MARKER_SPEC = {"mode": "arc", "count": 2}


def _backend_is_usable() -> bool:
    """Return True when the current backend can create a figure canvas."""
    try:
        probe = plt.figure()
        plt.close(probe)
        return True
    except Exception:
        return False


def _ensure_gui_backend() -> str:
    """
    Ensure a working interactive backend when possible.
    Returns the backend name finally in use.
    """
    current = str(plt.get_backend())
    current_lower = current.lower()
    if "agg" not in current_lower and _backend_is_usable():
        return current

    for candidate in ("QtAgg", "TkAgg", "Qt5Agg", "WXAgg"):
        try:
            plt.switch_backend(candidate)
            if _backend_is_usable():
                return str(plt.get_backend())
        except Exception:
            continue

    plt.switch_backend("Agg")
    return str(plt.get_backend())


# ------------------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------------------ #

def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _extract_first_number(text):
    if text is None:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    return float(matches[0]) if matches else None


def _find_level_key(mapping, target, tol=1e-10):
    for key in mapping:
        if abs(float(key) - float(target)) <= tol:
            return key
    return None


def _deduplicate_points(x, y):
    x, y = _clean_xy(x, y)
    if len(x) == 0:
        return x, y

    points = np.column_stack([x, y])
    rounded = np.round(points, decimals=12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    points = points[idx]
    return points[:, 0], points[:, 1]


def sort_points_by_tp(x, y):
    """
    Sort digitized Mansur points by T/P (x-axis) as requested.
    """
    x, y = _deduplicate_points(x, y)
    if len(x) <= 1:
        return x, y

    order = np.argsort(x, kind="mergesort")
    return x[order], y[order]


# ------------------------------------------------------------------------------ #
# Dataset loaders
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path):
    contours = load_wide_xy_csv(
        path,
        label_parser=_extract_first_number,
        sort_by="x",
        min_rows=2,
    )

    ordered: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for level, (x_vals, y_vals) in contours.items():
        x_sorted, y_sorted = sort_points_by_tp(x_vals, y_vals)
        ordered[float(level)] = (x_sorted, y_sorted)
    return ordered


# ------------------------------------------------------------------------------ #
# Contour extraction from ARISS sweep
# ------------------------------------------------------------------------------ #

def crossing_tp_for_level(field_row, tp_row, level):
    hits = []

    for i in range(len(field_row) - 1):
        f0 = field_row[i]
        f1 = field_row[i + 1]
        t0 = tp_row[i]
        t1 = tp_row[i + 1]

        if not (np.isfinite(f0) and np.isfinite(f1) and np.isfinite(t0) and np.isfinite(t1)):
            continue

        if (f0 - level) * (f1 - level) <= 0.0:
            if f1 != f0:
                frac = (level - f0) / (f1 - f0)
                hits.append(t0 + frac * (t1 - t0))

    return np.unique(np.asarray(hits, dtype=float)) if hits else np.array([], dtype=float)


def stitch_branches(rows):
    branches = []
    active = []

    for isp_value, hits in rows:
        hits = list(np.sort(np.asarray(hits, dtype=float)))

        if not hits:
            active = []
            continue

        if not active:
            active = [[(x, isp_value)] for x in hits]
            branches.extend(active)
            continue

        new_active = []

        for branch in active:
            if not hits:
                continue

            previous_x = branch[-1][0]
            idx = int(np.argmin(np.abs(np.asarray(hits) - previous_x)))
            x_match = hits.pop(idx)
            branch.append((x_match, isp_value))
            new_active.append(branch)

        for x_remaining in hits:
            new_branch = [(x_remaining, isp_value)]
            branches.append(new_branch)
            new_active.append(new_branch)

        active = new_active

    stitched = []
    for branch in branches:
        if len(branch) <= 1:
            continue
        x = np.asarray([p[0] for p in branch], dtype=float)
        y = np.asarray([p[1] for p in branch], dtype=float)
        stitched.append((x, y))

    return stitched


# ------------------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------------------ #

def run_sweep():
    base = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    base.update_from_toml(CONFIG_PATH)

    efficiencies = np.geomspace(0.05, 1.0, 60)
    isp_grid = np.linspace(1000.0, 6000.0, 60)

    tp_mn_per_kw = np.full((len(isp_grid), len(efficiencies)), np.nan, dtype=float)
    eff_grid = np.full_like(tp_mn_per_kw, np.nan)
    mdot_mg_per_s = np.full_like(tp_mn_per_kw, np.nan)
    ain_m2 = np.full_like(tp_mn_per_kw, np.nan)

    old_level = simulation_logger.level
    simulation_logger.setLevel(50)

    try:
        for i, isp_s in enumerate(isp_grid):
            for j, efficiency in enumerate(efficiencies):
                spacecraft = deepcopy(base)
                spacecraft.thruster.specific_impulse = float(isp_s)
                spacecraft.thruster.eff = float(efficiency)

                with redirect_stdout(io.StringIO()):
                    try:
                        final_sc, converged, _ = run_sizing_loop(spacecraft)
                    except Exception:
                        continue

                if not converged:
                    continue

                tp_mn_per_kw[i, j] = 1.0e6 * float(final_sc.thruster.thrust) / float(final_sc.thruster.power)
                eff_grid[i, j] = float(final_sc.thruster.eff)
                mdot_mg_per_s[i, j] = 1.0e6 * float(final_sc.thruster.m_flow)
                ain_m2[i, j] = float(final_sc.geometry.A_in)
    finally:
        simulation_logger.setLevel(old_level)

    return isp_grid, tp_mn_per_kw, eff_grid, mdot_mg_per_s, ain_m2


def extract_lines(isp_grid, field, tp_grid, levels):
    lines = {}

    for level in levels:
        rows = []
        for i, isp_s in enumerate(isp_grid):
            hits = crossing_tp_for_level(field[i], tp_grid[i], float(level))
            rows.append((float(isp_s), hits))

        branches = stitch_branches(rows)
        if branches:
            lines[float(level)] = branches

    return lines


def _print_family_relative_stats(title, model_lines, dataset, levels) -> float | None:
    print(title)
    pearson_values: list[float] = []
    for level in levels:
        model_key = _find_level_key(model_lines, level)
        data_key = _find_level_key(dataset, level)
        if model_key is None or data_key is None:
            print(f"  level={level:g} n/a")
            continue

        branches = sorted(model_lines[model_key], key=lambda seg: len(seg[0]), reverse=True)
        if not branches:
            print(f"  level={level:g} n/a")
            continue

        model_x, model_y = branches[0]
        ref_x, ref_y = dataset[data_key]
        ref_x, ref_y = sort_points_by_tp(ref_x, ref_y)
        stats = datapoint_relative_and_corr_stats(model_x, model_y, ref_x, ref_y)
        if stats is None:
            print(f"  level={level:g} n/a")
            continue

        max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats
        line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
        print(
            f"  level={level:g} "
            f"max_relative_error={max_rel:10.6f} ({100.0 * max_rel:7.3f}%) (line {line_text}), "
            f"mean_relative_error={mean_rel:10.6f} ({100.0 * mean_rel:7.3f}%), "
            f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
        )
        if np.isfinite(pearson_r):
            pearson_values.append(pearson_r)

    family_min = minimum_finite(pearson_values)
    if family_min is not None:
        print(f"  Minimum Pearson correlation coefficient: {family_min:.6f}")
        return family_min

    print("  Minimum Pearson correlation coefficient: n/a")
    return None


# ------------------------------------------------------------------------------ #
# Main plot
# ------------------------------------------------------------------------------ #

def plot(show: bool = True, use_matplotlib_tab: bool = True):
    if use_matplotlib_tab:
        _ensure_gui_backend()
    backend_is_agg = "agg" in plt.get_backend().lower()
    force_file_output = MATPLOTLIB_ONLY and backend_is_agg
    save_figure = (not MATPLOTLIB_ONLY) or force_file_output
    output_path = OUTPUT if save_figure else None

    isp_grid, tp_mn_per_kw, eff_grid, mdot_mg_per_s, ain_m2 = run_sweep()

    efficiency_lines = extract_lines(isp_grid, eff_grid, tp_mn_per_kw, EFF_LEVELS)
    mass_flow_lines = extract_lines(isp_grid, mdot_mg_per_s, tp_mn_per_kw, MDOT_LEVELS)
    intake_area_lines = extract_lines(isp_grid, ain_m2, tp_mn_per_kw, AIN_LEVELS)

    mansur_eta = load_wide_xy_dataset(ETA_DATASET_PATH)
    mansur_mdot = load_wide_xy_dataset(MFLOW_DATASET_PATH)
    mansur_ain = load_wide_xy_dataset(AIN_DATASET_PATH)

    min_r_values = []
    min_r_eta = _print_family_relative_stats(
        "Datapoint relative-error and correlation (eta family):",
        efficiency_lines,
        mansur_eta,
        EFF_LEVELS,
    )
    if min_r_eta is not None:
        min_r_values.append(min_r_eta)

    min_r_mdot = _print_family_relative_stats(
        "Datapoint relative-error and correlation (m_flow family):",
        mass_flow_lines,
        mansur_mdot,
        MDOT_LEVELS,
    )
    if min_r_mdot is not None:
        min_r_values.append(min_r_mdot)

    min_r_ain = _print_family_relative_stats(
        "Datapoint relative-error and correlation (A_in family):",
        intake_area_lines,
        mansur_ain,
        AIN_LEVELS,
    )
    if min_r_ain is not None:
        min_r_values.append(min_r_ain)

    if min_r_values:
        print(f"Overall minimum Pearson correlation coefficient: {min(min_r_values):.6f}")
    else:
        print("Overall minimum Pearson correlation coefficient: n/a")

    plot_result = plot_validation_mansur_thruster_map(
        efficiency_lines,
        mass_flow_lines,
        intake_area_lines,
        mansur_eta,
        mansur_mdot,
        mansur_ain,
        eff_levels=EFF_LEVELS,
        mdot_levels=MDOT_LEVELS,
        ain_levels=AIN_LEVELS,
        eff_color=EFF_COLOR,
        mdot_color=MDOT_COLOR,
        ain_color=AIN_COLOR,
        marker_cycle=MARKER_CYCLE,
        marker_size=MARKER_SIZE,
        eff_marker_spec=EFF_MARKER_SPEC,
        mdot_marker_spec=MDOT_MARKER_SPEC,
        ain_marker_spec=AIN_MARKER_SPEC,
        output_path=output_path,
        page_figsize=PAGE_FIGSIZE,
        show=False,
        save=save_figure,
        return_figure=True,
    )
    fig, axes = plot_result
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Mansur Thruster Map Validation")

    if show and not backend_is_agg:
        plt.show()

    return fig, axes

if __name__ == "__main__":
    plot(show=True, use_matplotlib_tab=True)
    if MATPLOTLIB_ONLY and "agg" not in plt.get_backend().lower():
        print("Opened editable Matplotlib tab/window (no file saved).")
    else:
        print(f"Saved figure to: {OUTPUT}")


