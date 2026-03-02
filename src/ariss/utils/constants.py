"""
ARISS Physical Constants
---------------------------------

This module centrally stores all fundamental physical, planetary, 
and mathematical constants used throughout the ARISS simulation modules.

"""

from typing import Final


# ==========================================
# Universal / Mathematical Constants
# ==========================================
STEFAN_BOLTZMANN: Final[float] = 5.670374419e-8    # [W/(m^2 * K^4)]
UNIVERSAL_GAS: Final[float] = 8.31446261815324     # [J/(mol * K)]

# ==========================================
# Air / Atmospheric Properties (Default assumed values)
# ==========================================
MOLAR_MASS_AIR: Final[float] = 21.88e-3            # [kg/mol]
SPECIFIC_GAS_AIR: Final[float] = UNIVERSAL_GAS / MOLAR_MASS_AIR # [J/(kg * K)]
HEAT_CAPACITY_AIR: Final[float] = 1026.32          # [J/(kg * K)]
SPECIFIC_HEAT_RATIO: Final[float] = 1.5404         # []

# ==========================================
# Earth Constants
# ==========================================
EARTH_RADIUS: Final[float] = 6371e3                # [m]
EARTH_GRAVITY: Final[float] = 9.80665              # [m/s^2]
EARTH_MU: Final[float] = 3.986004418e14            # [m^3/s^2] Standard gravitational parameter
EARTH_ALBEDO: Final[float] = 0.3                   # []
EARTH_IR_EMISSION: Final[float] = STEFAN_BOLTZMANN * (255 ** 4) # [W/m^2] Approximate Earth IR out

# ==========================================
# Solar System / Environment Variations
# ==========================================
SOLAR_CONSTANT: Final[float] = 1361.0              # [W/m^2]
