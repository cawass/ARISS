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
    totals: Dict[str, float]


def _extract_states(sc):
    """Extract orbit/geometry/thermal states from a spacecraft object.

    Parameters
    ----------
    sc : object
        Spacecraft object exposing either modern attributes
        (``orbit``, ``geometry``, ``temperature_sc``) or legacy
        alternatives (``OrbitState``, ``Geometry``, ``SpacecraftState``).

    Returns
    -------
    tuple
        ``(orbit, geometry, temperature_sc, alpha)``.

    Raises
    ------
    AttributeError
        If required orbit/geometry/temperature attributes are missing.
    """
    orbit = getattr(sc, "orbit", getattr(sc, "OrbitState", None))
    geometry = getattr(sc, "geometry", getattr(sc, "Geometry", None))
    if orbit is None or geometry is None:
        raise AttributeError("Spacecraft object must expose orbit/geometry state.")
    temperature_sc = getattr(sc, "temperature_sc", None)
    if temperature_sc is None:
        legacy_state = getattr(sc, "SpacecraftState", None)
        temperature_sc = getattr(legacy_state, "temperature_sc", None)
    if temperature_sc is None:
        raise AttributeError("Spacecraft object must expose temperature_sc.")
    alpha = getattr(orbit, "alpha", 0.0)
    return orbit, geometry, temperature_sc, alpha


def _drag_coefficient(S: float, epsilon: float, alpha: float, T_orb: float, T_r: float) -> float:
    """Compute combined drag coefficient.

    Parameters
    ----------
    S : float
        Molecular speed ratio.
    epsilon : float
        Surface accommodation/specularity parameter.
    alpha : float
        Angle of attack [rad].
    T_orb : float
        Orbital gas temperature [K].
    T_r : float
        Reference/surface temperature [K].

    Returns
    -------
    float
        Total drag coefficient from friction + pressure + thermal terms.
    """
    friction = (1.0 - epsilon * np.cos(2.0 * alpha)) / (np.sqrt(np.pi) * S)
    friction *= np.exp(-(S ** 2) * (np.sin(alpha) ** 2))

    pressure = np.sin(alpha) / (S ** 2)
    pressure *= 1.0 + 2.0 * (S ** 2) + epsilon * (1.0 - 2.0 * (S ** 2) * np.cos(2.0 * alpha))
    pressure *= erf(S * np.sin(alpha))

    thermal = (1.0 - epsilon) / S
    thermal *= np.sqrt(np.pi) * (np.sin(alpha) ** 2) * np.sqrt(T_r / T_orb)

    return float(friction + pressure + thermal)


def _half_gap(x: float, inlet_dim: float, body_dim: float, L_body: float, L_in: float) -> float:
    """Return local half-gap between inlet and body dimensions.

    Parameters
    ----------
    x : float
        Axial location along body+inlet [m].
    inlet_dim : float
        Inlet characteristic dimension (height or width) [m].
    body_dim : float
        Body characteristic dimension (height or width) [m].
    L_body : float
        Body length [m].
    L_in : float
        Inlet length [m].

    Returns
    -------
    float
        Half-gap used by capture-fraction calculation.
    """
    # Constant half-gap assumption across body/inlet sections.
    _ = (x, L_body, L_in)
    gap = abs(inlet_dim - body_dim) / 2.0
    return gap


def _capture_fraction(offset: float, x: float, V_orb: float, T_orb: float, L_body: float, L_in: float, molar_mass: float) -> float:
    """Compute thermal capture fraction at one axial location.

    The model uses one velocity component from a Maxwell-Boltzmann
    distribution and evaluates
    ``P(|v| >= offset / remaining_time)``.

    Parameters
    ----------
    offset : float
        Transverse distance to a side wall [m].
    x : float
        Axial location [m].
    V_orb : float
        Orbital velocity [m/s].
    T_orb : float
        Orbital gas temperature [K].
    L_body : float
        Body length [m].
    L_in : float
        Inlet length [m].
    molar_mass : float
        Gas molar mass [kg/mol].

    Returns
    -------
    float
        Capture fraction in ``[0, 1]``.
    """
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


def drag_model(sc, n_points: int = 200, return_diagnostics: bool = False):
    """Compute aggregate drag and optional diagnostics.

    Parameters
    ----------
    sc : object
        Spacecraft state object accepted by :func:`_extract_states`.
    n_points : int, default=200
        Number of axial samples used for integration.
    return_diagnostics : bool, default=False
        If ``True``, also return a :class:`DragDiagnostics` instance.

    Returns
    -------
    tuple
        If ``return_diagnostics`` is ``False``:
        ``(D_solar, D_rad, D_body, D_inlet)``.
        If ``return_diagnostics`` is ``True``:
        ``((D_solar, D_rad, D_body, D_inlet), diagnostics)``.

    Raises
    ------
    ValueError
        If ``n_points < 2`` or if orbit conditions are non-positive.
    AttributeError
        If required spacecraft attributes are missing.
    """
    orbit, geometry, temperature_sc, alpha = _extract_states(sc)

    S = orbit.velocity * np.sqrt(orbit.molar_mass / (2.0 * const.UNIVERSAL_GAS * orbit.temperature))

    CD_solar = _drag_coefficient(S, geometry.epsilon_solar, alpha, orbit.temperature, temperature_sc)
    CD_rad = _drag_coefficient(S, geometry.epsilon_rad, alpha, orbit.temperature, temperature_sc)
    CD_body = _drag_coefficient(S, geometry.epsilon_body, alpha, orbit.temperature, temperature_sc)
    CD_in = _drag_coefficient(S, geometry.epsilon_in, alpha, orbit.temperature, temperature_sc)

    H_in = np.sqrt(geometry.A_in / geometry.AR_in)
    W_in = geometry.A_in / H_in
    H_body = np.sqrt(geometry.A_body / geometry.AR_body)
    W_body = geometry.A_body / H_body

    x_array: list[float] = []
    fz_array: list[float] = []
    fy_array: list[float] = []
    d_body_array: list[float] = []
    d_in_array: list[float] = []

    total_length = geometry.L_body + geometry.L_in
    x_step = total_length / (n_points - 1)

    for i in range(n_points):
        x = i * x_step
        hz = _half_gap(x, H_in, H_body, geometry.L_body, geometry.L_in)
        hy = _half_gap(x, W_in, W_body, geometry.L_body, geometry.L_in)
        fz = _capture_fraction(hz, x, orbit.velocity, orbit.temperature, geometry.L_body, geometry.L_in, orbit.molar_mass)
        fy = _capture_fraction(hy, x, orbit.velocity, orbit.temperature, geometry.L_body, geometry.L_in, orbit.molar_mass)

        x_array.append(float(x))
        fz_array.append(float(fz))
        fy_array.append(float(fy))

       
        if x >= geometry.L_body:
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

    totals = {
        "solar": float(CD_solar * geometry.A_solar),
        "rad": float(CD_rad * geometry.A_rad),
        "body": float(D_body_cumulative[-1]),
        "inlet": float(D_in_cumulative[-1]),
    }
    totals["total"] = totals["solar"] + totals["rad"] + totals["body"] + totals["inlet"]

    diagnostics = DragDiagnostics(
        x_array=x_array,
        fz_array=fz_array,
        fy_array=fy_array,
        d_body_array=d_body_array,
        d_in_array=d_in_array,
        D_body_cumulative=D_body_cumulative,
        D_in_cumulative=D_in_cumulative,
        totals=totals,
    )

    drag = (
        diagnostics.totals["solar"],
        diagnostics.totals["rad"],
        diagnostics.totals["body"],
        diagnostics.totals["inlet"],
    )
    if return_diagnostics:
        return drag, diagnostics
    return drag
