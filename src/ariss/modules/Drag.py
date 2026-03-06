"""Aerodynamic drag model for ARISS.

This module computes drag contributions from solar panels, radiators, body,
and inlet using a free-molecular style coefficient model plus a simple
capture-fraction integration along body/inlet length.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import erf

from ariss.utils import constants as const


@dataclass(frozen=True)
class DragDiagnostics:
    """Detailed drag outputs for post-processing.

    Attributes
    ----------
    x_array : list of float
        Sampled axial coordinates along ``L_body + L_in`` [m].
    fz_array : list of float
        Capture fraction in z-direction at each ``x``.
    fy_array : list of float
        Capture fraction in y-direction at each ``x``.
    d_body_array : list of float
        Local body drag density contribution at each ``x``.
    d_in_array : list of float
        Local inlet drag density contribution at each ``x``.
    D_body_cumulative : list of float
        Cumulative integral of ``d_body_array`` along ``x``.
    D_in_cumulative : list of float
        Cumulative integral of ``d_in_array`` along ``x``.
    totals : dict of str to float
        Aggregate drag values with keys:
        ``"solar"``, ``"rad"``, ``"body"``, ``"inlet"``, ``"total"``.
    """
    x_array: list[float]
    fz_array: list[float]
    fy_array: list[float]
    d_body_array: list[float]
    d_in_array: list[float]
    D_body_cumulative: list[float]
    D_in_cumulative: list[float]




def _drag_coefficient(S: float, epsilon: float, alpha: float, T_orb: float, T_r: float) -> float:

    friction = (1.0 - epsilon * np.cos(2.0 * alpha)) / (np.sqrt(np.pi) * S)
    friction *= np.exp(-(S ** 2) * (np.sin(alpha) ** 2))

    pressure = np.sin(alpha) / (S ** 2)
    pressure *= 1.0 + 2.0 * (S ** 2) + epsilon * (1.0 - 2.0 * (S ** 2) * np.cos(2.0 * alpha))
    pressure *= erf(S * np.sin(alpha))

    thermal = (1.0 - epsilon) / S
    thermal *= np.sqrt(np.pi) * (np.sin(alpha) ** 2) * np.sqrt(T_r / T_orb)

    return float(friction + pressure + thermal)


def _half_gap(x: float, inlet_dim: float, body_dim: float, L_body: float, L_in: float) -> float:
    _ = (x, L_body, L_in)
    gap = abs(inlet_dim - body_dim) / 2.0
    return gap


def _capture_fraction(offset: float, x: float, V_orb: float, T_orb: float, L_body: float, L_in: float, molar_mass: float) -> float:
    
    remaining_time = (L_body + L_in - x) / V_orb
    if remaining_time <= 0.0:
        return 0.0
    if offset <= 0.0:
        return 1.0
    molecule_mass = molar_mass / const.AVOGADRO_NUMBER
    sigma = np.sqrt(const.BOLTZMANN_CONSTANT * T_orb / molecule_mass)
    denominator = np.sqrt(2.0) * sigma * remaining_time
    if denominator <= 0.0:
        return 0.0
    z = offset / denominator
    return float(1.0 - erf(z))


def drag_model(sc, n_points: int = 200):

    S = sc.orbit.velocity * np.sqrt(sc.orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * sc.orbit.temperature))

    CD_solar = _drag_coefficient(S, sc.geometry.epsilon_solar, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des)
    CD_rad = _drag_coefficient(S, sc.geometry.epsilon_rad, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des)
    CD_body = _drag_coefficient(S, sc.geometry.epsilon_body, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des)
    CD_in = _drag_coefficient(S, sc.geometry.epsilon_in, sc.orbit.alpha, sc.orbit.temperature, sc.thermal.T_des)

    H_in = np.sqrt( sc.geometry.A_in /  sc.geometry.AR_in)
    W_in =  sc.geometry.A_in / H_in
    H_body = np.sqrt( sc.geometry.A_body /  sc.geometry.AR_body)
    W_body =  sc.geometry.A_body / H_body

    x_array: list[float] = []
    fz_array: list[float] = []
    fy_array: list[float] = []
    d_body_array: list[float] = []
    d_in_array: list[float] = []

    total_length =  sc.geometry.L_body +  sc.geometry.L_in
    x_step = total_length / (n_points - 1)

    for i in range(n_points):
        x = i * x_step
        
        hz = _half_gap(x, H_in, H_body,  sc.geometry.L_body,  sc.geometry.L_in)
        hy = _half_gap(x, W_in, W_body,  sc.geometry.L_body,  sc.geometry.L_in)
        fz = _capture_fraction(hz, x, sc.orbit.velocity, sc.orbit.temperature,  sc.geometry.L_body,  sc.geometry.L_in, sc.orbit.molar_mass)
        fy = _capture_fraction(hy, x, sc.orbit.velocity, sc.orbit.temperature,  sc.geometry.L_body,  sc.geometry.L_in, sc.orbit.molar_mass)

        x_array.append(float(x))
        fz_array.append(float(fz))
        fy_array.append(float(fy))

       
        if x >=  sc.geometry.L_body:
            d_in_array.append(float(CD_in * H_in * fz + CD_in * W_in * fy))
            d_body_array.append(0.0)
        else:
            d_in_array.append(0.0)
            d_body_array.append(float(CD_body * H_body * fz + CD_body * W_body * fy))

    D_body_cumulative: list[float] = [0.0]
    D_in_cumulative: list[float] = [0.0]

    for i in range(1, n_points):
        dx = x_array[i] - x_array[i - 1]
        body_increment = 0.5 * (d_body_array[i] + d_body_array[i - 1]) * dx
        inlet_increment = 0.5 * (d_in_array[i] + d_in_array[i - 1]) * dx
        D_body_cumulative.append(D_body_cumulative[-1] + body_increment)
        D_in_cumulative.append(D_in_cumulative[-1] + inlet_increment)


    dynamic_pressure = 0.5 * (sc.orbit.velocity ** 2)
    sc.drag.drag_solar = float(dynamic_pressure * CD_solar * sc.geometry.A_solar)
    sc.drag.drag_rad = float(dynamic_pressure * CD_rad * sc.geometry.A_rad)
    sc.drag.drag_body = float(dynamic_pressure * D_body_cumulative[-1])
    sc.drag.drag_inlet = float(dynamic_pressure * D_in_cumulative[-1])

    sc.drag.drag_total = sc.drag.drag_solar + sc.drag.drag_rad + sc.drag.drag_body + sc.drag.drag_inlet
    print(f"Total Drag: {sc.drag.drag_total:.6e} N")
    diagnostics = DragDiagnostics(
        x_array=x_array,
        fz_array=fz_array,
        fy_array=fy_array,
        d_body_array=d_body_array,
        d_in_array=d_in_array,
        D_body_cumulative=D_body_cumulative,
        D_in_cumulative=D_in_cumulative
    )

    return diagnostics
