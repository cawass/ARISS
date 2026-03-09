import pytest
import numpy as np
from src.ariss.modules.Thermal import thermal_model
from src.ariss.core.spacecraft import SpacecraftState
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
    sc.power.Power_refprop = 20.0

    diagnostics = thermal_model(sc)
    # Expected Q_internal = Power_total - Power_prop * thruster_eff - Power_refprop * 0.5
    # thruster_eff = 0.53
    expected_Q_internal = 100.0 - 10.0 * 0.53 - 20.0 * 0.5
    assert np.isclose(diagnostics.Q_internal, expected_Q_internal)


def test_radiated_heating():
    """Test radiated heat calculation"""
    sc = SpacecraftState()
    # Set simple geometry
    sc.geometry.AR_in = 1.0
    sc.geometry.AR_body = 1.0
    sc.geometry.A_in = 1.0
    sc.geometry.A_body = 1.0
    sc.geometry.L_in = 1.0
    sc.geometry.L_body = 1.0
    sc.geometry.A_solar = 0.0
    # Zero drag
    sc.orbit.velocity = 0.0
    sc.orbit.density = 0.0
    # Zero internal power
    sc.power.Power_total = 0.0
    sc.power.Power_prop = 0.0
    sc.power.Power_refprop = 0.0
    # High altitude
    sc.orbit.altitude = 1e7
    diagnostics = thermal_model(sc)
    # Ae_total calculation as above ≈8.5
    # Q_radiated = Ae_total * T_des^4 * STEFAN_BOLTZMANN
    # T_des = 300, STEFAN = 5.670374419e-8
    Ae_total = 8.5
    expected_Q_radiated = Ae_total * (300**4) * 5.670374419e-8
    assert np.isclose(diagnostics.Q_radiated, expected_Q_radiated)


def test_temperature_limits():
    """Test temperature limits (currently not implemented, set to 0.0)"""
    sc = SpacecraftState()
    diagnostics = thermal_model(sc)
    assert diagnostics.T_max == 0.0
    assert diagnostics.T_min == 0.0