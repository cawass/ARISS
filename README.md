# ARISS

Atmospheric Refueling Iterative System Solver.

ARISS is a Python codebase for sizing and evaluating very-low-Earth-orbit spacecraft with a coupled atmosphere, drag, propulsion, power, thermal, refueling, and budget model.

This README is written for the way this repository is actually used: inside VS Code, with the repository opened as a workspace, a selected Python interpreter, and the main workflows run from the editor, the Test Explorer, or the integrated terminal.

## What ARISS does

ARISS solves a spacecraft state iteratively. The main subsystem modules are the following:

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

### Required Python packages

At minimum, the environment should provide:

- `numpy`, `scipy`, `matplotlib`, `pymsis`, `openpyxl`, `pytest`, `tkinter`

## Core concepts

### First run
For intial testing and characterization with the program, it is recomended to run 
- `src/ariss/core/simulation_ui.py`
- `src/ariss/core/simulation.py`

You can desing your own spacecraft using the 
- `src/ariss/core/base_config.py`

## Spacecraft types
There are two main clasification of ABEP spacecraft for ARISS, Refueling and Geometry:
### Refueling
#### Refueling Spacecraft
This is any case with:

- `[mission_profile].active_refueling = true`

In this mode, the propulsion model allocates part of the intake to refueling:

- `A_ref` becomes non-zero
- `refueling.m_flow` is solved
- the propulsion branch also includes the refueling ram load in the required force balance

You normally configure these cases through:

- `[mission_profile]`
  - `active_refueling`
  - `delta_v`
- `[refueling]`
  - `coll_eff`
  - `t_refuel`
  - `eta_refuel`
  - `p_tank`

Use this mode when the spacecraft is meant to collect atmospheric mass for storage and later mission use, not just immediate drag compensation.

#### Non-Refuelign Spacecraft
This is any case with:

- `[mission_profile].active_refueling = false`

In this mode:

- `A_ref = 0`
- `refueling.m_flow = 0`
- the intake is only split between useful propulsion capture and drag-only intake area

This is the simpler and more common mode for drag-compensation studies, verification cases, and literature reproductions that do not include tank refill.

### Geometry
#### Geometry Model 1 - Fixed Body and Free Inlet AR
This corresponds to:

- `[geometry].use_intake_area_ratio = false`

This is the `free intake` propulsion branch in `src/ariss/modules/Propulsion.py`.

Behavior:

- `A_body` stays as the user-defined body frontal area
- `A_in` is not imposed by a fixed intake/body ratio
- the solver finds the useful propulsion area first
- then it reconstructs the total intake from collection efficiency

Use this when:

- you want the body geometry fixed
- but you do not want to force the intake to follow a prescribed area ratio

#### Geometry Model 2 - Fixed Body Fixed Inlet AR
This corresponds to:

- `[geometry].use_intake_area_ratio = true`
- `[geometry].fixed_body = true`

This is the `fixed body ratio mode` branch in `src/ariss/modules/Propulsion.py`.

Behavior:

- `A_body` is fixed by the case file
- `A_in` is imposed by:
  - `A_in = intake_area_ratio * A_body`
- body and inlet frontal dimensions are therefore fixed by the input case
- the solver uses collection efficiency to split the fixed intake into:
  - `A_prop`
  - `A_ref`
  - `A_in_drag`

Use this when:

- you want the cleanest fixed-geometry study
- the intake size should stay tied to the body through a prescribed ratio

This mode has a caveat, an is that as the geometry is over constrained the solver also has to solve for ISP, so in this mode your ISP will change

#### Geometry Model 3 - Free Body Fixed Inlet AR
This corresponds to:

- `[geometry].use_intake_area_ratio = true`
- `[geometry].fixed_body = false`

This is the `variable body ratio mode` branch in `src/ariss/modules/Propulsion.py`.

Behavior:

- the intake/body area ratio is fixed
- but `A_body` is allowed to move
- the solver reconstructs `A_in`
- then updates:
  - `A_body = A_in / intake_area_ratio`

So this mode preserves the ratio, not the absolute body area.

Use this when:

- you want to enforce a geometric scaling rule between body and intake
- but you want the body size itself to be solved rather than prescribed

#### Circular or Rectangular Intakes
Cross-section type is controlled by:

- `[geometry].S_in`
- `[geometry].S_body`

Codes accepted by the geometry utilities:

- round / elliptic:
  - `"c"`
  - `"e"`
- rectangular:
  - `"s"`
  - `"r"`

## The spacecraft class
Every spacecraft in ARISS is represented by `SpacecraftState` in `src/ariss/core/spacecraft.py`.

In practice, you do not build this class by hand in Python. You define the spacecraft in a TOML file, and ARISS loads that TOML into the nested dataclasses automatically.

Important point:

- `src/ariss/core/base_config.toml` already contains every field
- a normal case file only overrides the values you want to change
- many fields are solver outputs, so you usually do not need to edit them directly

The top-level TOML tables map directly to the top-level `SpacecraftState` fields:

- `[orbit]`
- `[geometry]`
- `[thruster]`
- `[rate]`
- `[mass]`
- `[power]`
- `[solar]`
- `[thermal]`
- `[drag]`
- `[refueling]`
- `[mission_profile]`

Below is what each section means and the units expected by the code.

### `[orbit]`
Orbital environment and atmosphere settings.

Main inputs:

- `altitude` `[km]`: mission altitude used as the starting orbit altitude
- `alpha` `[rad]`: incidence angle used by drag and thermal models
- `p_orb` `[Pa]`: ambient pressure used by the refueling compression model
- `gamma` `[-]`: gas specific-heat ratio
- `R_spec` `[J/kg/K]`: specific gas constant
- `msis_date` `[ISO-8601 UTC string]`: date used for the MSIS atmosphere call
- `msis_f107` `[sfu]`: F10.7 solar flux
- `msis_ap` `[-]`: geomagnetic activity index
- `latitude` `[deg]`: latitude for point-sampled atmosphere queries
- `longitude` `[deg]`: longitude for point-sampled atmosphere queries
- `use_average` `[bool]`: if `true`, use a latitude/longitude-averaged atmosphere instead of a single point

Mostly solver-derived outputs:

- `velocity` `[m/s]`
- `density` `[kg/m^3]`
- `temperature` `[K]`
- `molar_mass` `[kg/mol]`

### `[geometry]`
Body, inlet, solar array, and radiator geometry.

Main inputs:

- `S_in`, `S_body` `[shape code]`: cross-section type, typically circular/elliptic or square/rectangular
- `use_intake_area_ratio` `[bool]`: whether to enforce `A_in = intake_area_ratio * A_body`
- `fixed_body` `[bool]`: whether body area is fixed in ratio mode
- `intake_area_ratio` `[-]`: intake-to-body frontal area ratio
- `AR_in`, `AR_body`, `AR_solar`, `AR_rad` `[-]`: aspect ratios
- `epsilon_in`, `epsilon_body`, `epsilon_solar`, `epsilon_rad`, `epsilon_in_norm` `[-]`: drag accommodation-related coefficients used by the drag model
- `wake_in`, `wake_body`, `wake_solar`, `wake_radiator` `[-]`: exposed fractions or wake factors
- `A_in`, `A_body`, `A_solar`, `A_rad` `[m^2]`: frontal or planform areas
- `t_solar`, `t_rad` `[m]`: panel thicknesses used for frontal-edge drag
- `L_in`, `L_body` `[m]`: inlet and body lengths
- `X_solar`, `X_rad` `[m]`: panel/radiator longitudinal placement

Usually solver-updated:

- `A_ref` `[m^2]`: intake area allocated to refueling
- `A_prop` `[m^2]`: intake area allocated to propulsion
- `A_in_drag` `[m^2]`: intake area contributing drag only

### `[thruster]`
Propulsion operating point.

Main inputs:

- `specific_impulse` `[s]`
- `eff` `[-]`: thruster efficiency
- `thermal_eff` `[-]`: thruster thermal efficiency
- `power` `[W]`: electrical power delivered to the thruster

Often solver-updated or checked after the solve:

- `thrust` `[N]`
- `m_flow` `[kg/s]`
- `propellant_mass` `[kg/s]`
- `propulsive_ram_load` `[N]`
- `refueling_ram_load` `[N]`
- `required_load` `[N]`
- `force_residual` `[N]`

### `[rate]`
Sizing-law coefficients used by the budgets model.

Inputs:

- `R_mass_volume_in` `[kg/m^3]`
- `R_mass_volume_body` `[kg/m^3]`
- `R_mass_surface_solar` `[kg/m^2]`
- `R_mass_surface_rad` `[kg/m^2]`

### `[mass]`
Subsystem mass bookkeeping.

Typical direct inputs:

- `Mass_prop` `[kg]`
- `Mass_ADCS` `[kg]`
- `Mass_payload` `[kg]`
- `Mass_refprop` `[kg]`

Typically solver-derived:

- `Mass_in`, `Mass_body`, `Mass_solar`, `Mass_rad`, `Mass_total` `[kg]`

### `[power]`
Subsystem power bookkeeping.

Typical direct inputs:

- `Power_ADCS` `[W]`
- `Power_payload` `[W]`

Typically solver-derived:

- `Power_in`, `Power_body`, `Power_solar`, `Power_rad`, `Power_prop`, `Power_refprop`, `Power_total` `[W]`

### `[solar]`
Solar conversion and pointing assumptions.

Inputs:

- `av_aligment` `[deg]`: average sun-pointing alignment angle
- `eta_solar` `[-]`: solar-cell efficiency
- `eta_power` `[-]`: power-chain efficiency

### `[thermal]`
Thermal design and optical properties.

Inputs:

- `T_des` `[K]`: design temperature
- `alpha_body`, `alpha_solar` `[-]`: absorptivity values
- `epsilon_therm_in`, `epsilon_therm_body`, `epsilon_therm_solar`, `epsilon_therm_rad` `[-]`: emissivity values

### `[drag]`
Drag coefficients and drag force outputs.

These are usually diagnostic outputs, not primary case inputs.

Coefficients:

- `cd_solar`, `cd_solar_front`, `cd_rad`, `cd_rad_front`, `cd_body_side`, `cd_inlet_side`, `cd_inlet_front` `[-]`

Forces:

- `drag_total`, `drag_solar`, `drag_solar_front`, `drag_rad`, `drag_rad_front`, `drag_body_side`, `drag_inlet_side`, `drag_inlet_front` `[N]`

### `[refueling]`
Atmospheric collection and tanking settings.

Inputs:

- `coll_eff` `[-]`: useful collection efficiency
- `t_refuel` `[s]`: time allowed for refueling
- `eta_refuel` `[-]`: compression efficiency
- `p_tank` `[Pa]`: tank pressure
- `V_prop` `[m^3]`: propellant tank volume

Usually solver-updated:

- `m_flow` `[kg/s]`: refueling mass flow

### `[mission_profile]`
Mission-level settings.

Inputs:

- `active_refueling` `[bool]`
- `delta_v` `[m/s]`

Usually solver-updated:

- `required_fuel` `[kg]`

### Practical rule
When creating a new spacecraft case, most of the time you only need to edit:

- `[orbit]`
- `[geometry]`
- `[thruster]`
- `[solar]`
- `[thermal]`
- `[refueling]`
- `[mission_profile]`

and sometimes a few direct subsystem masses or powers.

You usually do not need to hand-edit the drag outputs, total mass, total power, or other quantities that ARISS recomputes during the solve.

## Validation folders and what they are for

### `CrandallWirz2022-Drag_Simplified_Trust`

Use this for reduced drag and power-limited validation against Crandall and Wirz style figures.

### `Mansur-Full_Model`

Use this for Mansur-style full-model validation and sweeps.

### `EULO-Full_Model`

Use this for the integrated EULO validation runs and parameter sweeps.

### `GOCEE-Drag`

Use this for GOCE-like drag validation and drag-coefficient comparisons.

### `Thermodiver-Full_Model`

Use this for the IEPC 2025 style integrated validation case.
