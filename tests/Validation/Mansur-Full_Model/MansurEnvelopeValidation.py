import sys
import io
import csv
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D


# ------------------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
VALIDATION_DIR = ROOT / "tests" / "Validation"

for p in (SRC, VALIDATION_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ariss.core.simulation import run_sizing_loop, logger as simulation_logger
from ariss.core.spacecraft import SpacecraftState
from plot_style import apply_validation_style, style_axis, style_legend


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "src/ariss/core/base_config.toml"
CONFIG_PATH = HERE / "MansurValidation.toml"
DATASET_PATH = HERE / "TP Dataset.csv"
OUTPUT = HERE / "mansur_envelope_validation.png"
PAGE_FIGSIZE = (13.2, 5.4)

ALT_LEVELS = [150, 155, 160, 165, 170, 180, 190, 200, 220]
G0 = 9.80665


# ------------------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------------------ #

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})


def _apply_style():
    apply_validation_style()


# ------------------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------------------ #

def _clean_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def smooth_xy(x, y, n=300):
    x, y = _clean_xy(x, y)
    if len(x) < 3:
        return x, y

    x, idx = np.unique(x, return_index=True)
    y = y[idx]

    f = PchipInterpolator(x, y)
    xs = np.linspace(x.min(), x.max(), n)
    return xs, f(xs)


def smooth_by_y(x, y, n=300):
    x, y = _clean_xy(x, y)
    if len(x) < 3:
        return x, y

    order = np.argsort(y)
    y, x = y[order], x[order]

    y, idx = np.unique(y, return_index=True)
    x = x[idx]

    f = PchipInterpolator(y, x)
    ys = np.linspace(y.min(), y.max(), n)
    return f(ys), ys


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


def spaced_marker_indices(x, y, n_markers=2, pad_fraction=0.14):
    """
    Return indices corresponding to equally spaced positions in arc length.
    This is better than equal index spacing because the curves are not
    uniformly parameterized in x or y.
    """
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


def plot_curve_with_markers(
    ax,
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
    n_markers=2,
):
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return

    mark_idx = spaced_marker_indices(x, y, n_markers=n_markers, pad_fraction=0.14)

    line, = ax.plot(
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
        ms=6.8 if filled else 6.2,
        mec="white" if filled else color,
        mew=0.9 if filled else 1.15,
        mfc=color if filled else "white",
    )

    if halo:
        line.set_path_effects([
            pe.Stroke(linewidth=lw + 1.6, foreground="white"),
            pe.Normal(),
        ])


# ------------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------------ #

def load_dataset(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header = rows[0]

    contours, solution = {}, (np.array([]), np.array([]))

    for i in range(0, len(header), 2):
        label = header[i].lower()
        x, y = [], []

        for r in rows[2:]:
            if i + 1 < len(r) and r[i] and r[i + 1]:
                x.append(float(r[i]))
                y.append(float(r[i + 1]))

        if not x:
            continue

        x, y = np.array(x), np.array(y)
        o = np.argsort(x)

        if label.startswith("h"):
            contours[float(label.split()[1])] = (x[o], y[o])
        elif "solution" in label:
            solution = (x[o], y[o])

    return contours, solution


# ------------------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------------------ #

def run_sweep():
    base = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    base.update_from_toml(CONFIG_PATH)

    eta = np.geomspace(0.05, 1, 60)
    isp_vals = np.linspace(2500, 6000, 60)

    alt = np.full((len(isp_vals), len(eta)), np.nan)
    tp = np.full_like(alt, np.nan)

    old = simulation_logger.level
    simulation_logger.setLevel(50)

    try:
        for i, isp in enumerate(isp_vals):
            for j, eff in enumerate(eta):
                sc = deepcopy(base)
                sc.geometry.use_intake_area_ratio = False
                sc.geometry.fixed_body = True
                sc.thruster.specific_impulse = isp
                sc.thruster.eff = eff

                with redirect_stdout(io.StringIO()):
                    sc, ok, _ = run_sizing_loop(sc)

                if ok:
                    alt[i, j] = sc.orbit.altitude
                    tp[i, j] = 1e6 * sc.thruster.thrust / sc.thruster.power
    finally:
        simulation_logger.setLevel(old)

    return isp_vals, alt, tp


# ------------------------------------------------------------------------------ #
# Contours
# ------------------------------------------------------------------------------ #

def crossing_tp(a, t, level):
    hits = []
    for i in range(len(a) - 1):
        if not np.isfinite(a[i:i + 2]).all() or not np.isfinite(t[i:i + 2]).all():
            continue

        if (a[i] - level) * (a[i + 1] - level) <= 0:
            if a[i + 1] != a[i]:
                f = (level - a[i]) / (a[i + 1] - a[i])
                hits.append(t[i] + f * (t[i + 1] - t[i]))

    return np.unique(np.array(hits))


def stitch(rows):
    branches, active = [], []

    for isp, hits in rows:
        hits = list(np.sort(hits))

        if not hits:
            active = []
            continue

        if not active:
            active = [[(x, isp)] for x in hits]
            branches += active
            continue

        new_active = []
        for b in active:
            if not hits:
                continue
            prev = b[-1][0]
            idx = np.argmin(np.abs(np.array(hits) - prev))
            x = hits.pop(idx)
            b.append((x, isp))
            new_active.append(b)

        for x in hits:
            nb = [(x, isp)]
            branches.append(nb)
            new_active.append(nb)

        active = new_active

    return [
        (np.array([p[0] for p in b]), np.array([p[1] for p in b]))
        for b in branches
        if len(b) > 1
    ]


def extract_lines(ISP, alt, tp):
    out = {}
    for lvl in ALT_LEVELS:
        rows = [(isp, crossing_tp(alt[i], tp[i], lvl)) for i, isp in enumerate(ISP)]
        branches = stitch(rows)
        if branches:
            out[lvl] = branches
    return out


# ------------------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------------------ #

def plot():
    _apply_style()
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

    ISP, alt, tp = run_sweep()
    paper, _ = load_dataset(DATASET_PATH)
    ariss = extract_lines(ISP, alt, tp)

    print("Datapoint relative-error and correlation against Mansur envelope contours:")
    pearson_values: list[float] = []
    for h in ALT_LEVELS:
        if h not in paper or h not in ariss:
            continue
        ref_x, ref_y = paper[h]
        branches = sorted(ariss[h], key=lambda b: len(b[0]), reverse=True)
        if not branches:
            print(f"  h={h:>3} km n/a")
            continue
        model_x, model_y = branches[0]
        stats = _relative_and_corr_stats(model_x, model_y, ref_x, ref_y)
        if stats is None:
            print(f"  h={h:>3} km n/a")
            continue
        max_rel, mean_rel, line_max_rel, n_rel, pearson_r, n_corr = stats
        line_text = str(line_max_rel) if line_max_rel > 0 else "n/a"
        print(
            f"  h={h:>3} km "
            f"max_relative_error={max_rel:10.6f} ({100.0 * max_rel:7.3f}%) (line {line_text}), "
            f"mean_relative_error={mean_rel:10.6f} ({100.0 * mean_rel:7.3f}%), "
            f"pearson_r={pearson_r:9.6f}, n_rel={n_rel}, n_corr={n_corr}"
        )
        if np.isfinite(pearson_r):
            pearson_values.append(pearson_r)

    if pearson_values:
        print(f"  Minimum Pearson correlation coefficient: {min(pearson_values):.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    colors = {
        h: plt.cm.viridis(v)
        for h, v in zip(ALT_LEVELS, np.linspace(0.08, 0.95, len(ALT_LEVELS)))
    }

    # One marker shape per altitude pair
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    markers = {h: m for h, m in zip(ALT_LEVELS, marker_cycle)}

    fig = plt.figure(figsize=PAGE_FIGSIZE)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.07)
    ax = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")

    # Draw Mansur first, lighter, same marker per altitude
    for h in ALT_LEVELS:
        if h not in paper:
            continue

        c = colors[h]
        mk = markers[h]
        x, y = smooth_xy(*paper[h], n=320)

        plot_curve_with_markers(
            ax,
            x,
            y,
            color=c,
            marker=mk,
            lw=1.2,
            ls=(0, (4, 2)),
            alpha=0.38,
            zorder=1,
            filled=False,
            halo=False,
            n_markers=2,
        )

    # Draw ARISS on top, same marker per altitude, filled markers + halo
    for h in ALT_LEVELS:
        if h not in ariss:
            continue

        c = colors[h]
        mk = markers[h]

        branches = sorted(ariss[h], key=lambda b: len(b[0]), reverse=True)
        for x, y in branches:
            xs, ys = smooth_by_y(x, y, n=320)

            plot_curve_with_markers(
                ax,
                xs,
                ys,
                color=c,
                marker=mk,
                lw=2.2,
                ls="-",
                alpha=0.98,
                zorder=3,
                filled=True,
                halo=True,
                n_markers=2,
            )

    ax.set_xlabel("T/P (mN/kW)")
    ax.set_ylabel("Isp (s)")
    ax.set_xlim(5, 60)
    ax.set_ylim(2500, 6000)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    style_axis(ax)

    ax.grid(which="major", color="0.88", linewidth=0.7)
    ax.grid(which="minor", color="0.94", linewidth=0.5)

    for s in ax.spines.values():
        s.set_color("black")
    ax.tick_params(colors="black", axis="both", which="both", labelsize=12)

    # Source legend
    source_handles = [
        Line2D([0], [0], color="black", lw=2.2, ls="-", label="ARISS"),
        Line2D([0], [0], color="black", lw=1.2, ls=(0, (4, 2)), label="Mansur"),
    ]
    leg_source = legend_axis.legend(
        handles=source_handles,
        title="Source",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.8,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_source)
    leg_source.get_title().set_fontweight("bold")
    legend_axis.add_artist(leg_source)

    # Altitude legend: marker + color
    alt_handles = [
        Line2D(
            [0], [0],
            linestyle="None",
            marker=markers[h],
            markersize=7.0,
            markerfacecolor=colors[h],
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"{h} km",
        )
        for h in ALT_LEVELS
    ]

    leg_alt = legend_axis.legend(
        handles=alt_handles,
        title="Altitude",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.72),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        labelspacing=0.7,
        handlelength=2.5,
        handletextpad=0.6,
    )
    style_legend(leg_alt)
    leg_alt.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95, wspace=0.07)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ #

if __name__ == "__main__":
    plot()
    print(f"Saved: {OUTPUT}")
