# Sensitivity Runs

This folder contains a dedicated runner that:

- runs core-base sensitivity by default,
- can run additional sensitivity cases when `--cases` are provided,
- sends plot calls through `ariss.utils.ploting` using data payloads,
- computes and prints ranking in this folder (no direct call to `run_sensitivity_ranking`).

## Run

```powershell
python tests/sensitivity/run_10pct_sensitivity.py
```

Skip core base and run only listed cases:

```powershell
python tests/sensitivity/run_10pct_sensitivity.py --skip-core-base
```

Run specific validation cases:

```powershell
python tests/sensitivity/run_10pct_sensitivity.py --cases tests/Validation/CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022_3U.toml tests/Validation/CrandallWirz2022-Drag_Simplified_Trust/CrandallWirz2022_6U.toml
```

## Outputs

Generated files are written to:

- `tests/sensitivity/results/core_base_sensitivity.csv`
- `tests/sensitivity/results/*_sensitivity.csv`
- `tests/sensitivity/results/*_sensitivity_curves.png`
- `tests/sensitivity/results/*_geometry_sensitivity.png`
