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
#      Efficiency sweep utility for Mansur-style validation curves.
#
#  Project:        ARISS
#  Module:         MansurEfficiencyVerification.py
# ============================================================================== #
import sys
import io
import csv
import logging
import warnings
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import PchipInterpolator


# ------------------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for path in (SRC, VALIDATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ------------------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------------------ #

from ariss.core.simulation import load_spacecraft_from_base_config
from ariss.core.simulation import logger as simulation_logger
from ariss.core.simulation import run_sizing_loop
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

CONFIG_PATH = Path(__file__).with_name("MansurValidation.toml")
DATASET_PATH = Path(__file__).with_name("Isp Altitude Dataset.csv")

OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.png")
VECTOR_OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.svg")
PDF_OUTPUT_PATH = Path(__file__).with_name("mansur_efficiency_validation.pdf")
PAGE_FIGSIZE = (13.2, 5.4)

COLLECTION_EFFICIENCIES = (0.35, 0.40, 0.45)
ISP = np.linspace(2000, 6000, 40)

PLOT_COLORS = [
    PALETTE["l1_teal"],
    PALETTE["sernn_pink"],
    PALETTE["choice_mid"],
]

# ------------------------------------------------------------------------------ #
# Global style
# ------------------------------------------------------------------------------ #

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

logging.getLogger("fontTools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------------------ #

def _sanitize_color(color: str) -> str:
    if str(color).lower() in {"grey", "gray", "#808080", "#7f7f7f"}:
        return "black"
    return color


def _soft_curve_points(x: np.ndarray, y: np.ndarray):
    if x.size < 3:
        return x, y

    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate(([0.0], np.cumsum(seg)))

    if s[-1] == 0:
        return x, y

    t = s / s[-1]
    t, idx = np.unique(t, return_index=True)

    if t.size < 3:
        return x[idx], y[idx]

    fx = PchipInterpolator(t, x[idx])
    fy = PchipInterpolator(t, y[idx])

    tt = np.linspace(0, 1, max(140, 24 * (t.size - 1)))
    return fx(tt), fy(tt)


def _plot_curve(ax, x, y, color, marker, linestyle):
    color = _sanitize_color(color)
    sx, sy = _soft_curve_points(x, y)

    ax.plot(sx, sy, color=color, linestyle=linestyle, linewidth=1.1)
    ax.plot(
        x, y,
        linestyle="None",
        marker=marker,
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
    )


def _plot_sweep(ax, x, y, color):
    if x.size == 0:
        return

    idx = np.argmin(x)

    _plot_curve(ax, x[:idx+1], y[:idx+1], color, "o", "-")

    if idx < len(x) - 1:
        _plot_curve(ax, x[idx:], y[idx:], color, "o", "-")


def _plot_reference(ax, x, y, color):
    if x.size == 0:
        return

    order = np.argsort(y)[::-1]
    _plot_curve(ax, x[order], y[order], color, "s", "--")


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


# ------------------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------------------ #

def sweep_mansur_efficiencies():
    base = load_spacecraft_from_base_config(CONFIG_PATH)
    results = {}

    prev_level = simulation_logger.level
    simulation_logger.setLevel(logging.CRITICAL)

    try:
        for eff in COLLECTION_EFFICIENCIES:
            alt, isp_vals = [], []

            for isp in ISP:
                sc = deepcopy(base)
                sc.refueling.coll_eff = eff
                sc.thruster.specific_impulse = float(isp)

                with redirect_stdout(io.StringIO()):
                    final, ok, _ = run_sizing_loop(sc)

                if ok:
                    alt.append(final.orbit.altitude)
                    isp_vals.append(final.thruster.specific_impulse)

            results[eff] = (np.array(alt), np.array(isp_vals))

    finally:
        simulation_logger.setLevel(prev_level)

    return results


def load_mansur_paper_dataset():
    data = {}

    with DATASET_PATH.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return data

    header = rows[0]

    for i in range(0, len(header), 2):
        try:
            eff = float(header[i].split("=")[1].strip())
        except:
            continue

        x, y = [], []

        for r in rows[2:]:
            if i+1 >= len(r):
                continue
            if r[i] and r[i+1]:
                x.append(float(r[i]))
                y.append(float(r[i+1]))

        data[eff] = (np.array(x), np.array(y))

    return data


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def plot_results(results, paper=None, save_path=OUTPUT_PATH, show=True):

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

    fig = plt.figure(figsize=PAGE_FIGSIZE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    handles, labels = [], []

    print("Datapoint relative-error and correlation against Mansur efficiency curves:")
    pearson_values: list[float] = []

    for color, eff in zip(PLOT_COLORS, COLLECTION_EFFICIENCIES):

        x, y = results.get(eff, (np.array([]), np.array([])))
        px, py = paper.get(eff, (np.array([]), np.array([]))) if paper else (np.array([]), np.array([]))

        if x.size == 0:
            continue

        clean_color = _sanitize_color(color)

        _plot_sweep(ax, x, y, clean_color)
        _plot_reference(ax, px, py, clean_color)

        if px.size > 0 and py.size > 0:
            stats = _relative_and_corr_stats(x, y, px, py)
            if stats is None:
                print(f"  eta_c={eff:.2f} n/a")
            else:
                max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats
                line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
                print(
                    f"  eta_c={eff:.2f} "
                    f"max_relative_error={max_rel:10.6f} ({100.0 * max_rel:7.3f}%) (line {line_text}), "
                    f"mean_relative_error={mean_rel:10.6f} ({100.0 * mean_rel:7.3f}%), "
                    f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
                )
                if np.isfinite(pearson_r):
                    pearson_values.append(pearson_r)

        h_lim = np.min(x)
        handles.append(Line2D([0], [0], color=clean_color, marker="o", linestyle="-"))
        labels.append(f"ARISS ηc={eff:.2f}, h_lim={h_lim:.1f} km")

        if px.size:
            h_lim_ref = np.min(px)
            handles.append(Line2D([0], [0], color=clean_color, marker="s", linestyle="--"))
            labels.append(f"Mansur ηc={eff:.2f}, h_lim={h_lim_ref:.1f} km")

    if pearson_values:
        print(f"  Minimum Pearson correlation coefficient: {min(pearson_values):.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    ax.set_xlabel("Converged altitude (km)")
    ax.set_ylabel("Isp (s)")

    ax.set_xlim(150, 230)
    ax.set_ylim(2000, 6000)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    style_axis(ax)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(which="major", color="0.88", linewidth=0.7)
    ax.grid(which="minor", color="0.94", linewidth=0.5)

    # Force black axis
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")

    if handles:
        legend = legend_axis.legend(
            handles,
            labels,
            title="Curves",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.96),
            frameon=False,
            borderaxespad=0.0,
            labelspacing=0.8,
            handlelength=2.5,
            handletextpad=0.6,
        )
        style_legend(legend)
        legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    fig.savefig(VECTOR_OUTPUT_PATH, bbox_inches="tight")
    fig.savefig(PDF_OUTPUT_PATH, bbox_inches="tight")

    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)

    return save_path


# ------------------------------------------------------------------------------ #
# Entry
# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    results = sweep_mansur_efficiencies()
    paper = load_mansur_paper_dataset()
    path = plot_results(results, paper)
    print(f"Saved: {path}")
