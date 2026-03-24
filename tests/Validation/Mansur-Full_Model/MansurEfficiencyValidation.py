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

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    handles, labels = [], []

    for color, eff in zip(PLOT_COLORS, COLLECTION_EFFICIENCIES):

        x, y = results.get(eff, (np.array([]), np.array([])))
        px, py = paper.get(eff, (np.array([]), np.array([]))) if paper else (np.array([]), np.array([]))

        if x.size == 0:
            continue

        clean_color = _sanitize_color(color)

        _plot_sweep(ax, x, y, clean_color)
        _plot_reference(ax, px, py, clean_color)

        h_lim = np.min(x)
        handles.append(Line2D([0], [0], color=clean_color, marker="o", linestyle="-"))
        labels.append(f"ARISS ηc={eff:.2f}, h_lim={h_lim:.1f} km")

        if px.size:
            h_lim_ref = np.min(px)
            handles.append(Line2D([0], [0], color=clean_color, marker="s", linestyle="--"))
            labels.append(f"Mansur ηc={eff:.2f}, h_lim={h_lim_ref:.1f} km")

    ax.set_xlabel("Converged altitude (km)")
    ax.set_ylabel("Isp (s)")

    ax.set_xlim(150, 230)
    ax.set_ylim(2000, 6000)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    style_axis(ax)

    # Force black axis
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black")

    if handles:
        ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
        style_legend(ax.get_legend())

    fig.tight_layout()
    fig.savefig(save_path, dpi=1200, bbox_inches="tight")
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