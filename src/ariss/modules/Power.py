"""Minimal solar power sizing model."""

from dataclasses import dataclass

import numpy as np

from ariss.utils import constants as const


@dataclass(frozen=True)
class PowerDiagnostics:
    power_required: float
    efficiency: float
    alignment_deg: float
    projected_flux: float
    required_area: float
    fixed_top_area: float
    deployable_area: float


def power_model(sc, power_required: float, efficiency: float = 0.2, alignment_deg: float = 0.0, return_diagnostics: bool = False):
    """Return required deployable solar area [m^2]."""
    projected_flux = efficiency * const.SOLAR_CONSTANT * np.cos(np.radians(alignment_deg))
    required_area = power_required / projected_flux

    h_in = np.sqrt(sc.geometry.A_in / sc.geometry.AR_in)
    w_in = sc.geometry.A_in / h_in
    h_body = np.sqrt(sc.geometry.A_body / sc.geometry.AR_body)
    w_body = sc.geometry.A_body / h_body
    fixed_top_area = (w_in + w_body) * 0.5 * sc.geometry.L_in + w_body * sc.geometry.L_body
    deployable_area = max(0.0, required_area - fixed_top_area)
    
    if return_diagnostics:
        return deployable_area, PowerDiagnostics(
            power_required=power_required,
            efficiency=efficiency,
            alignment_deg=alignment_deg,
            projected_flux=projected_flux,
            required_area=required_area,
            fixed_top_area=fixed_top_area,
            deployable_area=deployable_area,
        )
    return deployable_area
