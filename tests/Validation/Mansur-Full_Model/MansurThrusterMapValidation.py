from pathlib import Path
import sys
from copy import deepcopy
import io
import csv
import re
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

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
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"
CONFIG_PATH = HERE / "MansurValidation.toml"
OUTPUT = HERE / "mansur_thruster_map_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

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


def smooth_path(x, y, n=280):
    """
    Smooth along arc length. Does not assume monotonic y.
    """
    x, y = _clean_xy(x, y)
    if len(x) < 3:
        return x, y

    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    unique_s, idx = np.unique(s, return_index=True)
    x = x[idx]
    y = y[idx]

    if len(unique_s) < 3:
        return x, y

    fx = PchipInterpolator(unique_s, x)
    fy = PchipInterpolator(unique_s, y)

    ss = np.linspace(unique_s.min(), unique_s.max(), n)
    return fx(ss), fy(ss)


def spaced_marker_indices(x, y, n_markers=2, pad_fraction=0.14):
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return []

    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]

    if not np.isfinite(total) or total <= 0:
        return [len(x) // 2]

    if n_markers <= 1:
        targets = np.array([0.5 * total])
    else:
        lo = pad_fraction * total
        hi = (1.0 - pad_fraction) * total
        if hi <= lo:
            targets = np.array([0.5 * total])
        else:
            targets = np.linspace(lo, hi, n_markers)

    idx = []
    for t in targets:
        i = int(np.argmin(np.abs(s - t)))
        if i not in idx:
            idx.append(i)
    return idx


def target_marker_indices(x, y, *, x_targets=None, y_targets=None, fallback_count=2):
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return []

    idx = []

    if x_targets is not None:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        for target in x_targets:
            if xmin <= float(target) <= xmax:
                i = int(np.argmin(np.abs(x - float(target))))
                if i not in idx:
                    idx.append(i)

    elif y_targets is not None:
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        for target in y_targets:
            if ymin <= float(target) <= ymax:
                i = int(np.argmin(np.abs(y - float(target))))
                if i not in idx:
                    idx.append(i)

    if not idx:
        return spaced_marker_indices(x, y, n_markers=fallback_count)

    return idx


def marker_indices_from_spec(x, y, marker_spec):
    mode = marker_spec.get("mode", "arc")

    if mode == "x_targets":
        return target_marker_indices(
            x, y,
            x_targets=marker_spec.get("targets", []),
            fallback_count=marker_spec.get("fallback_count", 2),
        )

    if mode == "y_targets":
        return target_marker_indices(
            x, y,
            y_targets=marker_spec.get("targets", []),
            fallback_count=marker_spec.get("fallback_count", 2),
        )

    return spaced_marker_indices(x, y, n_markers=marker_spec.get("count", 2))


def plot_curve_with_markers(
    axis,
    x,
    y,
    *,
    color,
    marker,
    lw,
    ls,
    alpha,
    zorder,
    filled,
    halo=False,
    marker_spec=None,
):
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return

    if marker_spec is None:
        mark_idx = spaced_marker_indices(x, y, n_markers=2)
    else:
        mark_idx = marker_indices_from_spec(x, y, marker_spec)

    line, = axis.plot(
        x,
        y,
        color=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
        dash_capstyle="round",
        marker=marker,
        markevery=mark_idx if mark_idx else None,
        ms=MARKER_SIZE,
        mec="white" if filled else color,
        mew=1.0,
        mfc=color if filled else "white",
    )

    if halo:
        line.set_path_effects([
            pe.Stroke(linewidth=lw + 1.4, foreground="white"),
            pe.Normal(),
        ])


def build_marker_map(levels):
    if len(levels) > len(MARKER_CYCLE):
        raise ValueError("Not enough markers defined for the requested levels.")
    return {float(level): MARKER_CYCLE[i] for i, level in enumerate(levels)}


# ------------------------------------------------------------------------------ #
# Dataset loaders
# ------------------------------------------------------------------------------ #

def load_wide_xy_dataset(path):
    """
    Expected format:
      row 0: labels in columns 0,2,4,...
      row 1: X,Y,X,Y,...
      rows 2+: values
    """
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        return {}

    header = rows[0]
    contours = {}

    for col in range(0, len(header), 2):
        level = _extract_first_number(header[col])
        if level is None:
            continue

        x_vals = []
        y_vals = []

        for row in rows[2:]:
            if col + 1 >= len(row):
                continue

            x_raw = row[col].strip() if row[col] is not None else ""
            y_raw = row[col + 1].strip() if row[col + 1] is not None else ""

            if not x_raw or not y_raw:
                continue

            try:
                x_vals.append(float(x_raw))
                y_vals.append(float(y_raw))
            except ValueError:
                continue

        if not x_vals:
            continue

        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        x, y = sort_points_by_tp(x, y)

        contours[float(level)] = (x, y)

    return contours


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
                    final_sc, converged, _ = run_sizing_loop(spacecraft)

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


# ------------------------------------------------------------------------------ #
# Plot helpers
# ------------------------------------------------------------------------------ #

def _plot_ariss_family(axis, lines, levels, color, linewidth, markers, marker_spec, zorder):
    for level in levels:
        key = _find_level_key(lines, level)
        if key is None:
            continue

        marker = markers[float(level)]
        branches = sorted(lines[key], key=lambda seg: len(seg[0]), reverse=True)

        for x, y in branches:
            xs, ys = smooth_path(x, y, n=320)
            plot_curve_with_markers(
                axis,
                xs,
                ys,
                color=color,
                marker=marker,
                lw=linewidth,
                ls="-",
                alpha=0.98,
                zorder=zorder,
                filled=True,
                halo=True,
                marker_spec=marker_spec,
            )


def _plot_mansur_family(axis, dataset, levels, color, linewidth, markers, marker_spec, zorder):
    for level in levels:
        key = _find_level_key(dataset, level)
        if key is None:
            continue

        marker = markers[float(level)]
        x_raw, y_raw = dataset[key]
        x_raw, y_raw = sort_points_by_tp(x_raw, y_raw)

        if len(x_raw) < 2:
            continue

        xs, ys = smooth_path(x_raw, y_raw, n=280)

        plot_curve_with_markers(
            axis,
            xs,
            ys,
            color=color,
            marker=marker,
            lw=linewidth,
            ls=(0, (4, 2)),
            alpha=0.62,
            zorder=zorder,
            filled=False,
            halo=False,
            marker_spec=marker_spec,
        )


def _paired_model_reference_samples(
    model_x: np.ndarray,
    model_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model_x = np.asarray(model_x, dtype=float)
    model_y = np.asarray(model_y, dtype=float)
    ref_x = np.asarray(ref_x, dtype=float)
    ref_y = np.asarray(ref_y, dtype=float)

    valid_model = np.isfinite(model_x) & np.isfinite(model_y)
    if np.count_nonzero(valid_model) < 2:
        return None

    model_x = model_x[valid_model]
    model_y = model_y[valid_model]
    order = np.argsort(model_x)
    model_x = model_x[order]
    model_y = model_y[order]

    model_x_unique, unique_idx = np.unique(model_x, return_index=True)
    model_y_unique = model_y[unique_idx]
    if len(model_x_unique) < 2:
        return None

    line_ids = np.arange(1, len(ref_y) + 1, dtype=int)
    valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y)
    if not np.any(valid_ref):
        return None

    ref_x = ref_x[valid_ref]
    ref_y = ref_y[valid_ref]
    line_ids = line_ids[valid_ref]

    x_low = float(np.min(model_x_unique))
    x_high = float(np.max(model_x_unique))
    in_range = (ref_x >= x_low) & (ref_x <= x_high)
    if not np.any(in_range):
        return None

    ref_x = ref_x[in_range]
    ref_y = ref_y[in_range]
    line_ids = line_ids[in_range]

    model_at_ref = np.interp(ref_x, model_x_unique, model_y_unique)
    return model_at_ref, ref_y, line_ids


def _relative_and_corr_stats(
    model_x: np.ndarray,
    model_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> tuple[float, float, int, int, float, int] | None:
    paired = _paired_model_reference_samples(model_x, model_y, ref_x, ref_y)
    if paired is None:
        return None

    model_at_ref, ref_y_used, line_ids = paired

    nonzero_ref = np.abs(ref_y_used) > 1.0e-12
    if np.any(nonzero_ref):
        relative_error = np.abs(model_at_ref[nonzero_ref] - ref_y_used[nonzero_ref]) / np.abs(ref_y_used[nonzero_ref])
        rel_line_ids = line_ids[nonzero_ref]
        i_max = int(np.argmax(relative_error))
        max_relative_error = float(relative_error[i_max])
        max_rel_line = int(rel_line_ids[i_max])
        mean_relative_error = float(np.mean(relative_error))
        n_rel = int(len(relative_error))
    else:
        max_relative_error = float("nan")
        max_rel_line = -1
        mean_relative_error = float("nan")
        n_rel = 0

    if len(model_at_ref) >= 2:
        pearson_r = float(np.corrcoef(model_at_ref, ref_y_used)[0, 1])
    else:
        pearson_r = float("nan")

    return (
        max_relative_error,
        mean_relative_error,
        max_rel_line,
        n_rel,
        pearson_r,
        int(len(model_at_ref)),
    )


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
        stats = _relative_and_corr_stats(model_x, model_y, ref_x, ref_y)
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

    if pearson_values:
        family_min = min(pearson_values)
        print(f"  Minimum Pearson correlation coefficient: {family_min:.6f}")
        return family_min

    print("  Minimum Pearson correlation coefficient: n/a")
    return None


# ------------------------------------------------------------------------------ #
# Main plot
# ------------------------------------------------------------------------------ #

def plot():
    apply_validation_style()
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

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

    eta_markers = build_marker_map(EFF_LEVELS)
    mdot_markers = build_marker_map(MDOT_LEVELS)
    ain_markers = build_marker_map(AIN_LEVELS)

    figure = plt.figure(figsize=PAGE_FIGSIZE)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.07)

    axis = figure.add_subplot(grid[0, 0])
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.tick_params(axis="both", which="both", labelsize=12)
    legend_axis = figure.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)

    # Mansur first
    _plot_mansur_family(
        axis,
        mansur_eta,
        EFF_LEVELS,
        color=EFF_COLOR,
        linewidth=1.0,
        markers=eta_markers,
        marker_spec=EFF_MARKER_SPEC,
        zorder=1,
    )
    _plot_mansur_family(
        axis,
        mansur_mdot,
        MDOT_LEVELS,
        color=MDOT_COLOR,
        linewidth=0.95,
        markers=mdot_markers,
        marker_spec=MDOT_MARKER_SPEC,
        zorder=1,
    )
    _plot_mansur_family(
        axis,
        mansur_ain,
        AIN_LEVELS,
        color=AIN_COLOR,
        linewidth=0.9,
        markers=ain_markers,
        marker_spec=AIN_MARKER_SPEC,
        zorder=1,
    )

    # ARISS on top
    _plot_ariss_family(
        axis,
        efficiency_lines,
        EFF_LEVELS,
        color=EFF_COLOR,
        linewidth=1.35,
        markers=eta_markers,
        marker_spec=EFF_MARKER_SPEC,
        zorder=4,
    )
    _plot_ariss_family(
        axis,
        mass_flow_lines,
        MDOT_LEVELS,
        color=MDOT_COLOR,
        linewidth=1.20,
        markers=mdot_markers,
        marker_spec=MDOT_MARKER_SPEC,
        zorder=3,
    )
    _plot_ariss_family(
        axis,
        intake_area_lines,
        AIN_LEVELS,
        color=AIN_COLOR,
        linewidth=1.05,
        markers=ain_markers,
        marker_spec=AIN_MARKER_SPEC,
        zorder=2,
    )

    axis.set_xlabel("T/P (mN/kW)")
    axis.set_ylabel("Isp (s)")
    axis.set_xlim(5, 60)
    axis.set_ylim(2500, 6000)

    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    style_axis(axis)
    axis.grid(which="major", color="0.88", linewidth=0.7)
    axis.grid(which="minor", color="0.94", linewidth=0.5)

    axis.tick_params(axis="both", which="major", width=0.9, length=5, labelsize=12)
    axis.tick_params(axis="both", which="minor", width=0.7, length=3, labelsize=12)

    # ------------------------------------------------------------------ #
    # Source legend
    # ------------------------------------------------------------------ #
    source_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=1.25,
            ls="-",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="ARISS",
        ),
        Line2D(
            [0], [0],
            color="black",
            lw=1.0,
            ls=(0, (4, 2)),
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            label="Mansur",
        ),
    ]

    leg_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.96),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_source)
    leg_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_source)

    # ------------------------------------------------------------------ #
    # Families legend
    # ------------------------------------------------------------------ #
    family_handles = [
        Line2D([0], [0], color=EFF_COLOR, lw=1.35, label=r"$\eta_T$"),
        Line2D([0], [0], color=MDOT_COLOR, lw=1.20, label=r"$\dot{m}$ (mg/s)"),
        Line2D([0], [0], color=AIN_COLOR, lw=1.05, label=r"$A_i$ (m$^2$)"),
    ]

    leg_family = legend_axis.legend(
        handles=family_handles,
        title="Families",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.76),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.7,
        handlelength=2.6,
        handletextpad=0.6,
    )
    style_legend(leg_family)
    leg_family.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_family)

    # ------------------------------------------------------------------ #
    # Eta legend: two horizontal rows
    # ------------------------------------------------------------------ #
    eta_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=eta_markers[float(level)],
            markersize=MARKER_SIZE,
            markerfacecolor=EFF_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"{level:g}",
        )
        for level in EFF_LEVELS
    ]

    leg_eta = legend_axis.legend(
        handles=eta_handles,
        title=r"$\eta_T$",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.52),
        frameon=False,
        borderaxespad=0.0,
        ncol=4,
        columnspacing=1.0,
        labelspacing=0.70,
        handletextpad=0.45,
    )
    style_legend(leg_eta)
    leg_eta.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_eta)

    # ------------------------------------------------------------------ #
    # Mass-flow legend: rows
    # ------------------------------------------------------------------ #
    mdot_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=mdot_markers[float(level)],
            markersize=MARKER_SIZE,
            markerfacecolor=MDOT_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"{level:g}",
        )
        for level in MDOT_LEVELS
    ]

    leg_mdot = legend_axis.legend(
        handles=mdot_handles,
        title=r"$\dot{m}$ (mg/s)",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.30),
        frameon=False,
        borderaxespad=0.0,
        ncol=4,
        columnspacing=1.0,
        labelspacing=0.70,
        handletextpad=0.45,
    )
    style_legend(leg_mdot)
    leg_mdot.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_mdot)

    # ------------------------------------------------------------------ #
    # Intake-area legend
    # ------------------------------------------------------------------ #
    ain_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=ain_markers[float(level)],
            markersize=MARKER_SIZE,
            markerfacecolor=AIN_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=f"{level:g}",
        )
        for level in AIN_LEVELS
    ]

    leg_ain = legend_axis.legend(
        handles=ain_handles,
        title=r"$A_i$ (m$^2$)",
        loc="upper left",
        bbox_to_anchor=(0.00, 0.08),
        frameon=False,
        borderaxespad=0.0,
        ncol=3,
        columnspacing=1.0,
        labelspacing=0.70,
        handletextpad=0.45,
    )
    style_legend(leg_ain)
    leg_ain.get_title().set_fontweight("bold")

    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    plot()
    print(f"Saved figure to: {OUTPUT}")
