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