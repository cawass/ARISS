import sys
import io
from copy import deepcopy
from pathlib import Path
from contextlib import redirect_stdout
import time

import numpy as np
import matplotlib.pyplot as plt


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
from ariss.utils.ploting import plot_validation_mansur_envelope
from csv_helper import load_wide_xy_csv, split_labeled_contours
from validation_metrics import datapoint_relative_and_corr_stats, minimum_finite


# ------------------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------------------ #

HERE = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "src/ariss/core/base_config.toml"
CONFIG_PATH = HERE / "MansurValidation3000W.toml"
DATASET_PATH = HERE / "TP Dataset.csv"
OUTPUT = HERE / "mansur_envelope_validation.png"
PAGE_FIGSIZE = (15.84, 5.4)
MATPLOTLIB_ONLY = True

ALT_LEVELS = [150, 155, 160, 165, 170, 180, 190, 200, 220]
SWEEP_ISP_POINTS = 36
SWEEP_ETA_POINTS = 36
MAX_ITERATIONS = 120
PRINT_PROGRESS_EVERY_ROWS = 4


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
# Dataset
# ------------------------------------------------------------------------------ #

def load_dataset(path):
    series = load_wide_xy_csv(path, sort_by="x", min_rows=2)
    return split_labeled_contours(series, contour_prefix="h", solution_token="solution")


# ------------------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------------------ #

def run_sweep():
    base = SpacecraftState.from_toml(BASE_CONFIG_PATH)
    base.update_from_toml(CONFIG_PATH)

    eta = np.geomspace(0.05, 1, SWEEP_ETA_POINTS)
    isp_vals = np.linspace(2500, 6000, SWEEP_ISP_POINTS)

    alt = np.full((len(isp_vals), len(eta)), np.nan)
    tp = np.full_like(alt, np.nan)

    old = simulation_logger.level
    simulation_logger.setLevel(50)
    t0 = time.perf_counter()
    print(
        f"Running Mansur envelope sweep: {len(isp_vals)} x {len(eta)} points "
        f"(max_iterations={MAX_ITERATIONS})"
    )

    try:
        for i, isp in enumerate(isp_vals):
            print(f"  ISP {isp:.0f} s:")
            if (i % PRINT_PROGRESS_EVERY_ROWS) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  row {i + 1:>3}/{len(isp_vals)} | elapsed {elapsed:6.1f}s")
            for j, eff in enumerate(eta):
                sc = deepcopy(base)
                sc.geometry.use_intake_area_ratio = False
                sc.geometry.fixed_body = True
                sc.thruster.specific_impulse = isp
                sc.thruster.eff = eff

                with redirect_stdout(io.StringIO()):
                    try:
                        sc, ok, _ = run_sizing_loop(
                            sc,
                            max_iterations=MAX_ITERATIONS,
                        )
                    except Exception:
                        ok = False

                if ok:
                    alt[i, j] = sc.orbit.altitude
                    tp[i, j] = 1e6 * sc.thruster.thrust / sc.thruster.power
    finally:
        simulation_logger.setLevel(old)
        elapsed = time.perf_counter() - t0
        print(f"Envelope sweep complete in {elapsed:.1f}s")

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

def plot(show: bool = True, use_matplotlib_tab: bool = True):
    if use_matplotlib_tab:
        _ensure_gui_backend()
    backend_is_agg = "agg" in plt.get_backend().lower()
    force_file_output = MATPLOTLIB_ONLY and backend_is_agg
    save_figure = (not MATPLOTLIB_ONLY) or force_file_output
    output_path = OUTPUT if save_figure else None

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
        stats = datapoint_relative_and_corr_stats(model_x, model_y, ref_x, ref_y)
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

    min_pearson = minimum_finite(pearson_values)
    if min_pearson is not None:
        print(f"  Minimum Pearson correlation coefficient: {min_pearson:.6f}")
    else:
        print("  Minimum Pearson correlation coefficient: n/a")

    plot_result = plot_validation_mansur_envelope(
        ariss,
        paper,
        alt_levels=ALT_LEVELS,
        output_path=output_path,
        page_figsize=PAGE_FIGSIZE,
        show=False,
        save=save_figure,
        return_figure=True,
    )
    fig, axes = plot_result
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("Mansur Envelope Validation")

    if show and not backend_is_agg:
        plt.show()

    return fig, axes


# ------------------------------------------------------------------------------ #
if __name__ == "__main__":
    plot(show=True, use_matplotlib_tab=True)
    if MATPLOTLIB_ONLY and "agg" not in plt.get_backend().lower():
        print("Opened editable Matplotlib tab/window (no file saved).")
    else:
        print(f"Saved: {OUTPUT}")

