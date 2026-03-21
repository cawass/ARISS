from pathlib import Path
import csv
import sys
from copy import deepcopy
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from plot_style import PALETTE, apply_validation_style, style_axis, style_legend

from ariss.core.spacecraft import SpacecraftState
from ariss.core.simulation import run_sizing_loop, logger as simulation_logger

HERE = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "src" / "ariss" / "core" / "base_config.toml"
CONFIG_PATH = HERE / "MansurValidation.toml"
DATASET_PATH = HERE / "TP Dataset.csv"
OUTPUT = HERE / "mansur_envelope_validation.png"

ALT_LEVELS = [150, 155, 160, 165, 170, 180, 190, 200, 220]
EFF_LEVELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
G0 = 9.80665


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _apply_publication_style():
    apply_validation_style()

def smooth_xy(x, y, n=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return x, y

    x, u = np.unique(x, return_index=True)
    y = y[u]

    if len(x) < 3:
        return x, y

    f = PchipInterpolator(x, y)
    xs = np.linspace(x.min(), x.max(), n)
    return xs, f(xs)


def smooth_by_y(x, y, n=200):
    """
    Smooth a curve where y is the natural marching coordinate.
    Returns x_smooth, y_smooth.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return x, y

    order = np.argsort(y)
    y = y[order]
    x = x[order]

    y, u = np.unique(y, return_index=True)
    x = x[u]

    if len(y) < 3:
        return x, y

    f = PchipInterpolator(y, x)
    ys = np.linspace(y.min(), y.max(), n)
    return f(ys), ys


def label_line(ax, x, y, text, color, position="start", x_offset=0.5, y_offset=0.0):
    if len(x) == 0 or len(y) == 0:
        return

    if position == "end":
        idx = -1
        ha = "right"
    elif position == "middle":
        idx = len(x) // 2
        ha = "center"
    else:
        idx = 0
        ha = "left"

    ax.text(
        float(x[idx]) + x_offset,
        float(y[idx]) + y_offset,
        text,
        color=color,
        fontsize=9,
        va="center",
        ha=ha,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.1),
        zorder=5,
    )


def label_line_at_tp(
    ax,
    x,
    y,
    tp_target,
    text,
    color,
    x_offset=0.5,
    y_offset=0.0,
    rotate=False,
    bbox_edgecolor="none",
    bbox_linestyle="-",
    bbox_linewidth=0.8,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x, u = np.unique(x, return_index=True)
    y = y[u]

    if len(x) < 2:
        return

    if not (x.min() <= tp_target <= x.max()):
        return

    y_target = np.interp(tp_target, x, y)

    rotation = 0.0
    if rotate:
        idx = np.searchsorted(x, tp_target)
        idx0 = max(0, min(len(x) - 2, idx - 1))
        idx1 = idx0 + 1
        dx = x[idx1] - x[idx0]
        dy = y[idx1] - y[idx0]
        if abs(dx) > 1e-12:
            rotation = np.degrees(np.arctan2(dy, dx))
            if rotation > 90.0:
                rotation -= 180.0
            elif rotation < -90.0:
                rotation += 180.0

    ax.text(
        float(tp_target) + x_offset,
        float(y_target) + y_offset,
        text,
        color=color,
        fontsize=9,
        va="center",
        ha="left",
        rotation=rotation,
        rotation_mode="anchor",
        bbox=dict(
            facecolor="white",
            edgecolor=bbox_edgecolor,
            linestyle=bbox_linestyle,
            linewidth=bbox_linewidth,
            pad=0.15,
        ),
        zorder=5,
    )


def tp_from_efficiency(eta, isp_s):
    isp_s = np.asarray(isp_s, dtype=float)
    return 1e6 * (2.0 * float(eta) / (G0 * isp_s))


def plot_efficiency_lines(ax):
    y_min, y_max = 2500.0, 6000.0
    y_grid = np.linspace(y_min, y_max, 300)
    label_y_targets = {
        0.2: 3500.0,
        0.3: 3600.0,
        0.4: 3750.0,
        0.5: 3900.0,
        0.6: 4050.0,
        0.7: 4200.0,
    }

    for eta in EFF_LEVELS:
        x_grid = tp_from_efficiency(eta, y_grid)
        mask = (x_grid >= 5.0) & (x_grid <= 60.0)
        if np.count_nonzero(mask) < 2:
            continue

        x_plot = x_grid[mask]
        y_plot = y_grid[mask]
        ax.plot(x_plot, y_plot, color=PALETTE["secondary_text"], lw=0.9, zorder=1)

        y_label = label_y_targets.get(float(eta), 3800.0)
        x_label = float(tp_from_efficiency(eta, y_label))
        if 5.0 <= x_label <= 60.0:
            ax.text(
                x_label + 0.4,
                y_label,
                f"{eta:.1f}",
                color=PALETTE["secondary_text"],
                fontsize=10,
                rotation=-63.0,
                rotation_mode="anchor",
                va="center",
                ha="left",
                bbox=dict(facecolor=PALETTE["panel_bg"], edgecolor="none", pad=0.1),
                zorder=4,
            )


def load_dataset(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header = rows[0]

    contours = {}
    sol = (np.array([]), np.array([]))

    for i in range(0, len(header), 2):
        label = header[i].lower().strip()
        xs, ys = [], []

        for r in rows[2:]:
            if i + 1 < len(r) and r[i] and r[i + 1]:
                xs.append(float(r[i]))
                ys.append(float(r[i + 1]))

        if not xs:
            continue

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        o = np.argsort(xs)

        if label.startswith("h"):
            contours[float(label.split()[1])] = (xs[o], ys[o])
        elif "solution" in label:
            sol = (xs[o], ys[o])

    return contours, sol


# ------------------------------------------------------------------------------
# Simulation sweep
# ------------------------------------------------------------------------------

def run_sweep():
    base = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    base.update_from_toml(CONFIG_PATH)

    eta = np.geomspace(0.05, 1, 60)
    ISP = np.linspace(1000, 6000, 60)

    alt = np.full((len(ISP), len(eta)), np.nan, dtype=float)
    tp = np.full_like(alt, np.nan)

    old = simulation_logger.level
    simulation_logger.setLevel(50)

    try:
        for i, isp in enumerate(ISP):
            for j, p in enumerate(eta):
                sc = deepcopy(base)
                sc.geometry.use_intake_area_ratio = False
                sc.thruster.specific_impulse = isp
                sc.thruster.eff = p

                with redirect_stdout(io.StringIO()):
                    sc, conv, _ = run_sizing_loop(sc)

                if not conv:
                    continue

                alt[i, j] = sc.orbit.altitude
                tp[i, j] = 1e6 * sc.thruster.thrust / sc.thruster.power

    finally:
        simulation_logger.setLevel(old)

    return ISP, alt, tp


# ------------------------------------------------------------------------------
# Altitude contour extraction
# ------------------------------------------------------------------------------

def crossing_tp_for_level(a, t, level, eps=1e-9):
    """
    Find all TP values where the altitude trace a crosses 'level',
    preserving the original sweep order. This is the key fix.

    a : altitude samples along increasing power for a single ISP
    t : TP samples along the same power sweep
    """
    a = np.asarray(a, dtype=float)
    t = np.asarray(t, dtype=float)

    hits = []

    for k in range(len(a) - 1):
        a0, a1 = a[k], a[k + 1]
        t0, t1 = t[k], t[k + 1]

        if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(t0) and np.isfinite(t1)):
            continue

        d0 = level - a0
        d1 = level - a1

        # Entire segment lies exactly on the contour level.
        if abs(d0) < eps and abs(d1) < eps:
            hits.extend([t0, t1])
            continue

        # First endpoint exactly on level.
        if abs(d0) < eps:
            hits.append(t0)
            continue

        # Crossing inside segment, or second endpoint exactly on level.
        if d0 * d1 < 0.0 or abs(d1) < eps:
            if abs(a1 - a0) < eps:
                continue
            frac = (level - a0) / (a1 - a0)
            hits.append(t0 + frac * (t1 - t0))

    if not hits:
        return np.array([], dtype=float)

    hits = np.array(sorted(hits), dtype=float)

    # Deduplicate numerically identical crossings.
    dedup = [hits[0]]
    for v in hits[1:]:
        if abs(v - dedup[-1]) > 1e-8:
            dedup.append(v)

    return np.array(dedup, dtype=float)


def stitch_branches(rows):
    """
    rows: list of (isp, tp_hits_array)

    Each ISP row can have 0, 1, or multiple TP crossings.
    Stitch them into continuous branches by nearest-neighbor continuity in TP.
    """
    branches = []
    active = []

    for isp, hits in rows:
        hits = list(np.sort(np.asarray(hits, dtype=float)))

        # No crossings on this ISP: terminate active branches.
        if len(hits) == 0:
            active = []
            continue

        # Start new branches if none are active yet.
        if not active:
            active = []
            for x in hits:
                branch = [(x, float(isp))]
                branches.append(branch)
                active.append(branch)
            continue

        remaining = hits[:]
        next_active = []

        # Match each active branch to the nearest crossing on the current ISP row.
        for branch in sorted(active, key=lambda b: b[-1][0]):
            if not remaining:
                continue

            prev_x = branch[-1][0]
            idx = int(np.argmin(np.abs(np.asarray(remaining) - prev_x)))
            x = remaining.pop(idx)

            branch.append((x, float(isp)))
            next_active.append(branch)

        # Any unmatched crossings start new branches.
        for x in remaining:
            branch = [(x, float(isp))]
            branches.append(branch)
            next_active.append(branch)

        active = next_active

    out = []
    for branch in branches:
        if len(branch) < 2:
            continue
        x = np.array([p[0] for p in branch], dtype=float)
        y = np.array([p[1] for p in branch], dtype=float)
        out.append((x, y))

    return out



def extract_lines(ISP, alt, tp):
    lines = {}

    for level in ALT_LEVELS:
        rows = []

        for i, isp in enumerate(ISP):
            hits = crossing_tp_for_level(alt[i], tp[i], level)
            rows.append((float(isp), hits))

        branches = stitch_branches(rows)

        if branches:
            lines[level] = branches

    return lines


# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

def plot():
    _apply_publication_style()

    ISP, alt, tp = run_sweep()
    paper, _ = load_dataset(DATASET_PATH)
    ariss = extract_lines(ISP, alt, tp)

    colors = plt.cm.viridis(np.linspace(0, 1, len(ALT_LEVELS)))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    for c, h in zip(colors, ALT_LEVELS):
        if h in paper:
            x, y = smooth_xy(*paper[h])
            ax.plot(x, y, color=c, ls="--", lw=1.0, alpha=0.65, zorder=2)
            label_line_at_tp(
                ax,
                x,
                y,
                30.0,
                f"{h}",
                c,
                x_offset=0.4,
                rotate=False,
                bbox_edgecolor=c,
                bbox_linestyle="--",
            )

        if h in ariss:
            # Plot every extracted branch for this altitude level.
            branches = sorted(ariss[h], key=lambda seg: len(seg[0]), reverse=True)

            for k, (x, y) in enumerate(branches):
                xs, ys = smooth_by_y(x, y)
                ax.plot(xs, ys, color=c, lw=1.8, zorder=3)

                # Label only the longest branch once.
                if k == 0:
                    label_line_at_tp(
                        ax,
                        xs,
                        ys,
                        50.0,
                        f"{h}",
                        c,
                        x_offset=0.6,
                        bbox_edgecolor=c,
                        bbox_linestyle="-",
                    )

    ax.set_xlabel("T/P (mN/kW)")
    ax.set_ylabel("Isp (s)")
    ax.set_xlim(5, 60)
    ax.set_ylim(2500, 6000)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    style_axis(ax)
    ax.tick_params(axis="both", which="major", width=0.9, length=5)
    ax.tick_params(axis="both", which="minor", width=0.7, length=3)

    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["secondary_text"], label="ARISS feasible alt (km)"),
            Line2D([0], [0], color=PALETTE["secondary_text"], ls="--", label="Mansur feasible alt (km)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.8,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    style_legend(ax.get_legend())

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=1200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot()
    print(f"Saved figure to: {OUTPUT}")
