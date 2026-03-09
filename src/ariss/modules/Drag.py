# ============================================================================== #
#       ___    ____  ____  _____ _____
#      /   |  / __ \/  _// ___// ___/
#     / /| | / /_/ // / \__ \ \__ \
#    / ___ |/ _, _// / ___/ /___/ /
#   /_/  |_/_/ |_/___//____//____/
#
#        ARISS — Atmospheric Refueling Iterative System Solver
# ============================================================================== #
#  Description:
#      Aerodynamic drag model for the ARISS spacecraft geometry.
#
#  Project:        ARISS
#  Module:         Drag.py
#  Author:         Carlos Carrasco Requejo
# ============================================================================

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

from ariss.utils import constants as const


# Stores sampled capture fractions and integrated drag terms for post-processing.
@dataclass(frozen=True)
class DragDiagnostics:
    x_array: list[float]
    fz_array: list[float]
    fy_array: list[float]
    d_body_array: list[float]
    d_in_array: list[float]
    D_body_cumulative: list[float]
    D_in_cumulative: list[float]

def _drag_coefficient(S: float, epsilon: float, alpha: float, T_orb: float, T_r: float, N) -> float:
    # Inputs:
    #   S: speed ratio.
    #   epsilon: surface accommodation parameter.
    #   alpha: incidence angle [rad].
    #   T_orb: freestream temperature [K].
    #   T_r: wall/reference temperature [K].
    #   N: geometric multiplier.
    #
    # Output:
    #   Drag coefficient for the selected surface.
    #
    # Equations used:
    #   C_D = friction + pressure + thermal
    #   pressure term uses erf(S sin(alpha))

    # Friction term from the tangential momentum exchange at the surface.
    friction = (1.0 - epsilon * np.cos(2.0 * alpha)) / (np.sqrt(np.pi) * S) *np.exp(-(S ** 2) * (np.sin(alpha) ** 2))

    # Pressure term from the normal momentum exchange of the impinging flow.
    pressure = np.sin(alpha) / (S ** 2)* (1.0 + 2.0 * (S ** 2) + epsilon * (1.0 - 2.0 * (S ** 2) * np.cos(2.0 * alpha)))* erf(S * np.sin(alpha)) * N

    # Thermal re-emission term from particles leaving the heated wall.
    thermal = (1.0 - epsilon) / S  * np.sqrt(np.pi) * (np.sin(alpha) ** 2) * np.sqrt(T_r / T_orb)

    return float(friction + pressure + thermal)

def _half_gap(x: float, inlet_dim: float, body_dim: float, L_body: float, L_in: float) -> float:
    # Inputs:
    #   x: axial position [m].
    #   inlet_dim: inlet height or width [m].
    #   body_dim: body height or width [m].
    #   L_body, L_in: body and inlet lengths [m].
    #
    # Output:
    #   Half-gap between body and inlet dimensions [m].
    #
    # Equation used:
    #   gap = |inlet_dim - body_dim| / 2
    _ = (x, L_body, L_in)

    # Current implementation assumes a constant linear gap set only by the
    # difference between inlet and body dimensions.
    gap = abs(inlet_dim - body_dim) / 2.0

    return gap

def _capture_fraction(offset: float, x: float, V_orb: float, T_orb: float, L_body: float, L_in: float, molar_mass: float) -> float:
    # Inputs:
    #   offset: half-gap distance [m].
    #   x: axial position [m].
    #   V_orb: orbital velocity [m/s].
    #   T_orb: freestream temperature [K].
    #   L_body, L_in: body and inlet lengths [m].
    #   molar_mass: atmospheric molar mass [kg/mol].
    #
    # Output:
    #   Capture fraction at the selected axial position.
    #
    # Equations used:
    #   t_rem = (L_body + L_in - x) / V_orb
    #   m = molar_mass / N_A
    #   sigma = sqrt(k_B T_orb / m)
    #   z = offset / (sqrt(2) sigma t_rem)
    #   f = 1 - erf(z)

    # Remaining travel time available for the thermal plume to spread before
    # reaching the inlet exit plane.
    remaining_time = (L_body + L_in - x) / V_orb
    if remaining_time <= 0.0:
        return 0.0

    # Zero or negative offset means the whole local stream tube is captured.
    if offset <= 0.0:
        return 1.0

    # Convert molar mass into single-particle mass to estimate the thermal
    # velocity dispersion normal to the main orbital flow.
    molecule_mass = molar_mass / const.AVOGADRO_NUMBER
    sigma = np.sqrt(const.BOLTZMANN_CONSTANT * T_orb / molecule_mass)
    denominator = np.sqrt(2.0) * sigma * remaining_time
    if denominator <= 0.0:
        return 0.0

    # The error function gives the fraction of the broadened distribution that
    # spills outside the capture half-gap.
    z = offset / denominator
    return float(1.0 - erf(z))


def _capture_fraction_array(offset: float, x_array: np.ndarray, V_orb: float, T_orb: float, total_length: float, molar_mass: float) -> np.ndarray:
    # Inputs:
    #   offset: half-gap distance [m].
    #   x_array: axial sample positions [m].
    #   V_orb: orbital velocity [m/s].
    #   T_orb: freestream temperature [K].
    #   total_length: L_body + L_in [m].
    #   molar_mass: atmospheric molar mass [kg/mol].
    #
    # Output:
    #   Capture fraction profile for all sampled axial positions.
    #
    # Equations used:
    #   t_rem = (L_body + L_in - x) / V_orb
    #   m = molar_mass / N_A
    #   sigma = sqrt(k_B T_orb / m)
    #   z = offset / (sqrt(2) sigma t_rem)
    #   f = 1 - erf(z)

    # Zero or negative offset means the whole sampled stream is captured.
    if offset <= 0.0:
        return np.ones_like(x_array, dtype=float)

    # Compute the remaining downstream travel time for every sampled axial station.
    remaining_time = (total_length - x_array) / V_orb

    # Convert molar mass into particle mass and then into the thermal velocity spread.
    molecule_mass = molar_mass / const.AVOGADRO_NUMBER
    sigma = np.sqrt(const.BOLTZMANN_CONSTANT * T_orb / molecule_mass)
    denominator = np.sqrt(2.0) * sigma * remaining_time

    # Evaluate the capture fraction only where the plume still has a finite
    # downstream travel time; the final station naturally collapses to zero.
    capture_fraction = np.zeros_like(x_array, dtype=float)
    valid = denominator > 0.0
    capture_fraction[valid] = 1.0 - erf(offset / denominator[valid])
    return capture_fraction



def drag_model(sc, n_points: int = 200):
    # Inputs:
    #   sc: spacecraft state with orbit, geometry, thermal, and drag data.
    #   n_points: number of axial integration points.
    #
    # Outputs:
    #   diagnostics: sampled and cumulative drag terms.
    #   sc.drag is updated in place with the drag coefficients.
    #
    # Equations used:
    #   S = V_orb * sqrt(M / (2 R T_orb))
    #   H = sqrt(A / AR)
    #   W = A / H
    #   D_i = 0.5 (d_i + d_(i-1)) dx

    # Compute the free-molecular speed ratio that controls the drag coefficient model.
    S = sc.orbit.velocity * np.sqrt(sc.orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * sc.orbit.temperature))

    # Recover intake and body dimensions from area and aspect ratio.
    H_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    W_in = sc.geometry.A_in / H_in
    H_body = np.sqrt(sc.geometry.A_body / sc.geometry.AR_body)
    W_body = sc.geometry.A_body / H_body

    # Evaluate drag coefficients for the exposed spacecraft surfaces.
    CD_solar = _drag_coefficient(S, sc.geometry.epsilon_solar, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2)
    CD_rad = _drag_coefficient(S, sc.geometry.epsilon_rad, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 2)
    CD_body = _drag_coefficient(S, sc.geometry.epsilon_body, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 1)
    CD_in_norm = _drag_coefficient(S, sc.geometry.epsilon_in_norm, sc.orbit.alpha + np.pi / 2, sc.orbit.temperature, sc.thermal.T_des, 1)

    # Use a corrected inlet incidence angle when the inlet narrows relative to the body.
    if W_in < W_body:
        alpha_in = np.arctan(np.abs(W_body - W_in) / sc.geometry.L_in)
        CD_in = _drag_coefficient(S, sc.geometry.epsilon_in, alpha_in, sc.orbit.temperature, sc.thermal.T_des, 1)
    else:
        CD_in = _drag_coefficient(S, sc.geometry.epsilon_in, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des, 1)

    total_length = sc.geometry.L_body + sc.geometry.L_in
    x_array = np.linspace(0.0, total_length, n_points, dtype=float)

    # The body-to-inlet gap is constant in the current geometry model, so the
    # capture fractions can be evaluated for the full axial array in one vectorized pass.
    hz = _half_gap(0.0, H_in, H_body, sc.geometry.L_body, sc.geometry.L_in)
    hy = _half_gap(0.0, W_in, W_body, sc.geometry.L_body, sc.geometry.L_in)
    fz_array = _capture_fraction_array(hz, x_array, sc.orbit.velocity, sc.orbit.temperature, total_length, sc.orbit.molar_mass)
    fy_array = _capture_fraction_array(hy, x_array, sc.orbit.velocity, sc.orbit.temperature, total_length, sc.orbit.molar_mass)

    # If the inlet is already larger than the body shadow, all sampled flow is captured.
    if sc.geometry.A_in < sc.geometry.A_body:
        fz_array = np.ones_like(x_array, dtype=float)
        fy_array = np.ones_like(x_array, dtype=float)

    # Build the local body and inlet drag-density profiles with boolean masks
    # instead of a Python loop over every sampled station.
    body_mask = x_array < sc.geometry.L_body
    d_body_array = np.where(body_mask, CD_body * H_body * fz_array + CD_body * W_body * fy_array, 0.0)
    d_in_array = np.where(~body_mask, CD_in * H_in * fz_array + CD_in * W_in * fy_array, 0.0)

    # Integrate the local drag-density arrays with a vectorized trapezoidal rule.
    dx = np.diff(x_array)
    body_increment = 0.5 * (d_body_array[1:] + d_body_array[:-1]) * dx
    inlet_increment = 0.5 * (d_in_array[1:] + d_in_array[:-1]) * dx
    D_body_cumulative = np.concatenate(([0.0], np.cumsum(body_increment)))
    D_in_cumulative = np.concatenate(([0.0], np.cumsum(inlet_increment)))

    # Convert the integrated side drag back into effective side drag coefficients.
    sc.drag.cd_solar = CD_solar
    sc.drag.cd_rad = CD_rad
    sc.drag.cd_body_side = D_body_cumulative[-1] / ((2 * W_body + 2 * H_body) * sc.geometry.L_body)
    sc.drag.cd_inlet_side = D_in_cumulative[-1] / ((2 * W_in + 2 * H_in + 2 * W_body + 2 * H_body) / 2 * sc.geometry.L_in)
    sc.drag.cd_inlet_front = CD_in_norm

    # Return the full sampled profiles for plotting and post-processing.
    diagnostics = DragDiagnostics(
        x_array=x_array.tolist(),
        fz_array=fz_array.tolist(),
        fy_array=fy_array.tolist(),
        d_body_array=d_body_array.tolist(),
        d_in_array=d_in_array.tolist(),
        D_body_cumulative=D_body_cumulative.tolist(),
        D_in_cumulative=D_in_cumulative.tolist(),
    )

    return diagnostics
