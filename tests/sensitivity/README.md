# Sensitivity Runs

This folder contains a dedicated runner that uses `ariss.core.sensitivity` with a `+/-10%` perturbation.

## Run

```powershell
python tests/sensitivity/run_10pct_sensitivity.py
```

Core baseline run (no case override, same behavior as `sensitivity.py` defaults):

```powershell
python tests/sensitivity/run_10pct_sensitivity.py --core-base
```

## Outputs

Generated files are written to:

- `tests/sensitivity/results/*_10pct_sensitivity.csv`
- `tests/sensitivity/results/*_10pct_sensitivity.tex`
