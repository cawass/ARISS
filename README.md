# ARISS

Atmospheric Refueling Iterative System Solver.

ARISS is a Python codebase for sizing and evaluating very-low-Earth-orbit spacecraft with a coupled atmosphere, drag, propulsion, power, thermal, refueling, and budget model.

This README is written around the code that is currently in this repository. The commands below are based on the actual entry points in `src/ariss/__main__.py`, the base spacecraft definition in `src/ariss/core/base_config.toml`, and the test and validation layout under `tests/`.

## What ARISS does

ARISS solves a spacecraft state iteratively. The core loop updates:

- atmosphere and orbit state
- drag
- propulsion
- refueling
- power
- thermal
- mass and power budgets

The main solver entry point is:

- `src/ariss/core/simulation.py`
  - `load_spacecraft_from_base_config(...)`
  - `run_sizing_loop(...)`

The main subsystem modules live in:

- `src/ariss/modules/Drag.py`
- `src/ariss/modules/Propulsion.py`
- `src/ariss/modules/Power.py`
- `src/ariss/modules/Thermal.py`
- `src/ariss/modules/Refueling.py`
- `src/ariss/modules/Budgets.py`

## Repository layout

```text
ARISS/
  src/ariss/
    __main__.py              CLI entry point
    __init__.py              public Python API helpers
    core/
      base_config.toml       full default spacecraft definition
      simulation.py          main iterative solver
      simulation_ui.py       history and diagnostic UI
      spacecraft.py          state dataclasses
    modules/                 drag, propulsion, power, thermal, etc.
    utils/                   atmosphere and constants helpers
  tests/
    Verification/            automated pytest verification suite
    Validation/              script-style literature and case validations
```

## Environment setup

### 1. Work from the repository root

All commands below assume the current working directory is the repo root:

```powershell
cd C:\Users\carlo\OneDrive\Escritorio\ARISS
```

### 2. Activate the virtual environment

If you are using the repo-local virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can still use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3. Put `src` on `PYTHONPATH`

This repository uses a `src` layout. The safest way to run the CLI and ad hoc scripts is:

```powershell
$env:PYTHONPATH = "src"
```

If you do not set this, direct imports such as `import ariss` may fail with:

- `ModuleNotFoundError: No module named 'ariss'`

This is the single most common setup issue.

### 4. Required Python packages

This repository does not currently ship a full installable package definition. In practice, the code expects a Python environment with at least these packages available:

- `numpy`
- `scipy`
- `matplotlib`
- `pymsis`
- `openpyxl`
- `pytest`

If `pymsis` is missing, atmosphere-dependent verification and validation runs will skip or fail.

### 5. Optional: use a non-GUI backend for plot scripts

If you only want files written to disk and do not want windows opening:

```powershell
$env:MPLBACKEND = "Agg"
```

Do not set `MPLBACKEND=Agg` if you want to use the interactive Tk UI.

## Quick start

### Run the default simulation

```powershell
$env:PYTHONPATH = "src"
python -m ariss sim
```

This loads the default spacecraft from:

- `src/ariss/core/base_config.toml`

### Run a specific spacecraft case override

```powershell
$env:PYTHONPATH = "src"
python -m ariss sim tests\Verification\configs\case_cc_equal_area_ar2_fixed_body_ratio.toml
```

### Get JSON output instead of the text summary

```powershell
$env:PYTHONPATH = "src"
python -m ariss sim tests\Verification\configs\case_cc_equal_area_ar2_fixed_body_ratio.toml --json
```

### Launch the history UI

```powershell
$env:PYTHONPATH = "src"
python -m ariss ui
```

Or with a specific case:

```powershell
$env:PYTHONPATH = "src"
python -m ariss ui tests\Validation\Mansur-Full_Model\MansurVerification.toml
```

### Inspect a spacecraft file

```powershell
$env:PYTHONPATH = "src"
python -m ariss spacecraft tests\Verification\configs\case_cc_equal_area_ar2_fixed_body_ratio.toml
```

## CLI reference

The project currently exposes three CLI commands:

```powershell
python -m ariss ui
python -m ariss sim
python -m ariss spacecraft
```

### `python -m ariss ui`

Launches the simulation history UI.

Arguments:

- `spacecraft`: optional TOML or JSON path
- `--max-iterations`
- `--mass-tolerance`

Example:

```powershell
$env:PYTHONPATH = "src"
python -m ariss ui tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022_6U.toml --max-iterations 100
```

### `python -m ariss sim`

Runs the solver without the UI and prints a compact summary.

Arguments:

- `spacecraft`: optional TOML or JSON path
- `--max-iterations`
- `--mass-tolerance`
- `--json`

Example:

```powershell
$env:PYTHONPATH = "src"
python -m ariss sim tests\Validation\Thermodiver-Full_Model\IEPC2025Validation.toml --max-iterations 200 --json
```

### `python -m ariss spacecraft`

Loads a spacecraft file and prints the resolved dataclass as JSON.

Example:

```powershell
$env:PYTHONPATH = "src"
python -m ariss spacecraft tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022_6U.toml
```

## Spacecraft configuration model

ARISS uses a base-plus-override model.

### Base config

The complete spacecraft definition lives in:

- `src/ariss/core/base_config.toml`

This file defines every field in the `SpacecraftState` hierarchy.

### Case override files

A case file usually overrides only the fields that differ from the base config.

Example location:

- `tests/Verification/configs/case_cc_equal_area_ar2_fixed_body_ratio.toml`

Case files are loaded like this:

1. load `src/ariss/core/base_config.toml`
2. apply the override file on top of it

This is done by:

- `load_spacecraft_from_base_config(...)` in `src/ariss/core/simulation.py`

### Common sections in TOML files

Typical sections are:

- `[orbit]`
- `[geometry]`
- `[thruster]`
- `[mass]`
- `[power]`
- `[solar]`
- `[thermal]`
- `[drag]`
- `[refueling]`
- `[mission_profile]`

## Using ARISS from Python

The top-level API is in:

- `src/ariss/__init__.py`

Main helpers:

- `load_spacecraft(...)`
- `run_simulation(...)`
- `plot_simulation_history(...)`
- `launch_history_ui(...)`

Example:

```python
from ariss import run_simulation

spacecraft, converged, history = run_simulation(
    sc="tests/Verification/configs/case_cc_equal_area_ar2_fixed_body_ratio.toml",
    max_iterations=200,
    mass_tolerance=1e-3,
)

print(converged)
print(spacecraft.orbit.altitude)
print(spacecraft.mass.Mass_total)
```

If you are running this outside `pytest`, make sure `src` is on `PYTHONPATH`.

## Running the UI

The interactive UI is implemented in:

- `src/ariss/core/simulation_ui.py`

Safest way to launch it:

```powershell
$env:PYTHONPATH = "src"
Remove-Item Env:MPLBACKEND -ErrorAction SilentlyContinue
python -m ariss ui tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022_6U.toml
```

Notes:

- the UI expects an interactive Matplotlib backend
- `tkinter` must be available in the Python installation
- if you previously set `MPLBACKEND=Agg`, unset it before launching the UI

## Running verification tests

The automated verification suite lives under:

- `tests/Verification`

Main groups:

- `Drag`
- `Power`
- `Propulsion`
- `Refueling`
- `Sizing`
- `Thermal`
- `full simulation loop`

### Run the full verification suite

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests\Verification -q
```

### Run one verification module

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests\Verification\Propulsion -q
```

### Run one specific file

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests\Verification\Power\test_power.py -q
```

### About `pytest` path handling

`pyproject.toml` already configures:

- `testpaths = ["tests"]`
- `pythonpath = ["src"]`

So `pytest` usually works even when `PYTHONPATH` is not set. Setting it explicitly is still the safest option for consistency with direct script execution.

## Running validation scripts

The validation area is different from verification.

- `tests/Verification`: automated pass/fail tests
- `tests/Validation`: script-style reproductions, sweeps, and literature comparisons

Most validation scripts are not pytest tests. They are regular Python scripts that generate figures or tables.

Validation folders currently include:

- `tests/Validation/CrandallWirz2022-Drag_Simplified_Trust`
- `tests/Validation/EULO-Full_Model`
- `tests/Validation/GOCEE-Drag`
- `tests/Validation/Mansur-Full_Model`
- `tests/Validation/Spacecraft-Sweep`
- `tests/Validation/Thermodiver-Full_Model`

### Example: Crandall and Wirz reduced validation

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022Validation.py
```

### Example: individual Crandall and Wirz figure scripts

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022_Fig26_SolarEfficiency.py
python tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022_Fig27_Accommodation.py
```

### Example: EULO full-model validation

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python tests\Validation\EULO-Full_Model\EULO.py
```

### Example: Mansur full-model validation

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python tests\Validation\Mansur-Full_Model\MansurValidation.py
```

### Example: validation sweep to Excel

```powershell
$env:PYTHONPATH = "src"
python tests\Validation\Spacecraft-Sweep\create_sweep_excel.py
```

This writes:

- `tests/Validation/Spacecraft-Sweep/sweep_results.xlsx`

## Typical workflows

### Workflow 1: inspect a case and run it

```powershell
$env:PYTHONPATH = "src"
python -m ariss spacecraft tests\Verification\configs\case_cc_equal_area_ar2_fixed_body_ratio.toml
python -m ariss sim tests\Verification\configs\case_cc_equal_area_ar2_fixed_body_ratio.toml --json
```

### Workflow 2: iterate on the UI

```powershell
$env:PYTHONPATH = "src"
Remove-Item Env:MPLBACKEND -ErrorAction SilentlyContinue
python -m ariss ui tests\Validation\Thermodiver-Full_Model\IEPC2025Validation.toml
```

### Workflow 3: run verification after changing a module

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests\Verification\Propulsion -q
python -m pytest "tests\Verification\full simulation loop" -q
```

### Workflow 4: regenerate validation plots headlessly

```powershell
$env:PYTHONPATH = "src"
$env:MPLBACKEND = "Agg"
python tests\Validation\CrandallWirz2022-Drag_Simplified_Trust\CrandallWirz2022Validation.py
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'ariss'`

Cause:

- `src` is not on `PYTHONPATH`
- or the script is being run from the wrong working directory

Fix:

```powershell
cd C:\Users\carlo\OneDrive\Escritorio\ARISS
$env:PYTHONPATH = "src"
```

### The UI does not open

Check:

- `MPLBACKEND` is not set to `Agg`
- `tkinter` is available
- you are using `python -m ariss ui ...` from the repo root

### Validation scripts print lots of atmosphere debug lines

That comes from the pymsis debug print currently inside the atmosphere helper. For headless batch runs, redirect stdout or use wrapper scripts that already suppress it where needed.

### `pytest tests/Validation` says no tests ran

That is expected for this repository. The validation folder contains plot and analysis scripts, not pytest test functions.

## Recommended files to know first

If you are new to the codebase, read these in order:

1. `src/ariss/core/spacecraft.py`
2. `src/ariss/core/base_config.toml`
3. `src/ariss/core/simulation.py`
4. `src/ariss/modules/Drag.py`
5. `src/ariss/modules/Propulsion.py`
6. `src/ariss/core/simulation_ui.py`

## Current status of the README

This README documents how to run the code as it exists now. It does not attempt to describe every modeling assumption in the literature-validation scripts. Those assumptions belong in the validation folders next to the scripts that use them.
