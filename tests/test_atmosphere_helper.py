import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.core.simulation import run_sizing_loop
from ariss.core.spacecraft import SpacecraftState
from ariss.utils.atmosphere import orbit_updates_from_height, sample_atmosphere_at_height

HAS_PYMSIS = find_spec("pymsis") is not None


@unittest.skipUnless(HAS_PYMSIS, "pymsis is required for atmosphere helper tests")
class AtmosphereHelperTest(unittest.TestCase):
    def test_sample_atmosphere_outputs_finite_state(self):
        sample = sample_atmosphere_at_height(200.0)
        self.assertGreater(sample.density, 0.0)
        self.assertGreater(sample.temperature, 0.0)
        self.assertGreater(sample.molar_mass, 0.0)
        self.assertGreater(sample.orbital_velocity, 0.0)
        self.assertGreaterEqual(sample.dynamic_pressure, 0.0)

    def test_orbit_updates_from_height(self):
        updates = orbit_updates_from_height(250.0)
        self.assertAlmostEqual(updates["altitude"], 250000.0, places=6)
        self.assertGreater(updates["density"], 0.0)
        self.assertGreater(updates["velocity"], 0.0)

    def test_simulation_uses_mission_profile_height(self):
        sc = SpacecraftState().update(mission_profile={"mission_height": 300.0})
        final_sc, _, _ = run_sizing_loop(sc, max_iterations=2, mass_tolerance=1e-6)
        self.assertAlmostEqual(final_sc.orbit.altitude, 300000.0, places=6)
        self.assertGreater(final_sc.orbit.density, 0.0)
        self.assertGreater(final_sc.orbit.velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
