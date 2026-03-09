import pytest
import numpy as np
from src.ariss.modules.Thermal import thermal_model
from src.ariss.core.spacecraft import SpacecraftState
from src.ariss.utils import constants as const
from dataclasses import replace

def test_drag_heating():
    """Test whether drag heating is computed accurately based on hand calculations"""
    sc = SpacecraftState()
    sc.orbit.velocity = 7000
    sc.orbit.density = 10**-7
    sc.geometry.A_in = 4.0387
    diagnostics = thermal_model(sc)
    assert np.isclose(diagnostics.Q_drag, 69263.705)


def test_sun_heating():
    """Test sun heating calculation"""
    sc = SpacecraftState()
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 4.0387
    sc.geometry.A_body = 1.21
    sc.geometry.L_in = 2.26
    sc.geometry.L_body = 2.80
    sc.geometry.A_solar = 5
    # sc.thermal.alpha_solar = 0.9
    # sc.solar.eta_solar = 0.3

    diagnostics = thermal_model(sc)
    assert np.isclose(diagnostics.Q_sun, 9940.96319)


def test_albedo_heating():
    """Test earth albedo heating calculation"""
    sc = SpacecraftState()
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 4.0387
    sc.geometry.A_body = 1.21
    sc.geometry.L_in = 2.26
    sc.geometry.L_body = 2.80
    sc.geometry.A_solar = 5

    diagnostics = thermal_model(sc)
    assert np.isclose(diagnostics.Q_albedo, 269.2292)


def test_ir_heating():
    """Test earth infrared heating calculation"""
    sc = SpacecraftState()
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 4.0387
    sc.geometry.A_body = 1.21
    sc.geometry.L_in = 2.26
    sc.geometry.L_body = 2.80
    sc.geometry.A_solar = 5

    sc.orbit.altitude = 1000
    
    diagnostics = thermal_model(sc)

    assert np.isclose(diagnostics.Q_ir,1063.89, rtol=0.001)


def test_internal_heating():
    """Test internal heating calculation"""
    sc = SpacecraftState()
    # Set power
    sc.power.Power_total = 300.0
    sc.power.Power_prop = 100.0
    sc.power.Power_refprop = 100.0

    sc.thruster.thruster_thermal_eff = 0.8 
    sc.refueling.eta_refuel = 0.1

    diagnostics = thermal_model(sc)
    assert np.isclose(diagnostics.Q_internal, 210)


def test_radiated_heating():
    """Test radiated heat calculation"""
    sc = SpacecraftState()
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 4.0387
    sc.geometry.A_body = 1.21
    sc.geometry.L_in = 2.26
    sc.geometry.L_body = 2.80
    sc.geometry.A_solar = 5

    # sc.thermal.epsilon_therm_in: float = 0.5
    # sc.thermal.epsilon_therm_body: float = 0.9
    # sc.thermal.epsilon_therm_solar: float = 0.85
    # sc.thermal.epsilon_therm_rad: float = 0.9
    # sc.thermal.T_des = 300

    diagnostics = thermal_model(sc)

    assert np.isclose(diagnostics.Q_radiated, 14577.84676)


def test_temperature_limits():
    """Test temperature limits (currently not implemented, set to 0.0)"""
    sc = SpacecraftState()
    diagnostics = thermal_model(sc)
    assert diagnostics.T_max == 0.0
    assert diagnostics.T_min == 0.0


def test_radiator_area_zero_cold_case():
    """Test that radiator area is zero when heat inputs are minimal (cold case)"""
    sc = SpacecraftState()
    # Set minimal heat inputs
    sc.orbit.velocity = 0.0
    sc.orbit.density = 0.0
    sc.power.Power_total = 0.0
    sc.geometry.A_solar = 0.0
    sc.orbit.altitude = 10000.0  # High altitude to minimize IR heating
    
    diagnostics = thermal_model(sc)
    assert sc.geometry.A_rad == 0.0


def test_radiator_area_positive_hot_case():
    """Test that radiator area is positive when heat inputs exceed radiated heat"""
    sc = SpacecraftState()
    # Increase drag heating to exceed radiated heat
    sc.orbit.velocity = 8000.0
    sc.orbit.density = 1e-7
    diagnostics = thermal_model(sc)
    assert sc.geometry.A_rad > 0.0


@pytest.mark.parametrize("velocity", [7000, 8000, 9000])
@pytest.mark.parametrize("density", [1e-7, 2e-7])
def test_thermal_balance_at_design_temperature(velocity, density):
    """Test that at design temperature, heat in equals heat out with radiators for various inputs"""
    sc = SpacecraftState()
    # Set parameters to require radiators
    sc.orbit.velocity = velocity
    sc.orbit.density = density
    diagnostics = thermal_model(sc)
    Q_in_total = diagnostics.Q_drag + diagnostics.Q_sun + diagnostics.Q_albedo + diagnostics.Q_ir + diagnostics.Q_internal
    # Total radiator area includes back of solar panels + 2 * additional radiators (double-sided)
    total_radiator_area = sc.geometry.A_solar + 2 * sc.geometry.A_rad
    Q_out_total = diagnostics.Q_radiated + total_radiator_area * const.STEFAN_BOLTZMANN * sc.thermal.T_des**4 * sc.thermal.epsilon_therm_body
    # Should be approximately equal
    assert np.isclose(Q_in_total, Q_out_total, rtol=1e-6)



