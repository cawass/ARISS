import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iteration.SpacecraftClass import SpacecraftClass


def power_model(sc: SpacecraftClass) -> None:

    sc.S_dict['pow'] = sc.P_tot / (sc._Sc * sc._eta_solar)*1.5
