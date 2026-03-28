from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.utils import atmosphere as atmosphere_module
from ariss.utils.atmosphere import MSIS_AP, MSIS_F107


class _FakeVariable:
    MASS_DENSITY = 0
    N2 = 1
    O2 = 2
    O = 3
    TEMPERATURE = 4


class _FakeMsis:
    Variable = _FakeVariable

    def __init__(self) -> None:
        self.last_kwargs: dict[str, np.ndarray] | None = None

    def calculate(self, et, lons, lats, heights, **kwargs):
        self.last_kwargs = kwargs
        n = len(np.asarray(heights, dtype=float))
        return np.ones((n, 5), dtype=float)


def test_atmospheric_properties_from_height_omits_activity_kwargs_for_zero_pair(monkeypatch) -> None:
    fake_msis = _FakeMsis()
    monkeypatch.setattr(atmosphere_module, "msis", fake_msis)

    atmosphere_module.atmospheric_properties_from_height(np.array([200.0]), msis_f107=0.0, msis_ap=0.0)
    assert fake_msis.last_kwargs == {}


def test_atmospheric_properties_from_height_uses_defaults_for_missing_activity_inputs(monkeypatch) -> None:
    fake_msis = _FakeMsis()
    monkeypatch.setattr(atmosphere_module, "msis", fake_msis)

    atmosphere_module.atmospheric_properties_from_height(np.array([200.0]), msis_f107=None, msis_ap=None)
    assert fake_msis.last_kwargs is not None
    assert set(fake_msis.last_kwargs.keys()) == {"f107s", "f107as", "aps"}
    assert float(fake_msis.last_kwargs["f107s"][0]) == MSIS_F107
    assert float(fake_msis.last_kwargs["f107as"][0]) == MSIS_F107
    assert float(fake_msis.last_kwargs["aps"][0][0]) == MSIS_AP


def test_atmospheric_properties_from_height_passes_activity_kwargs_for_explicit_values(monkeypatch) -> None:
    f107 = 125.0
    ap = 11.0

    fake_msis = _FakeMsis()
    monkeypatch.setattr(atmosphere_module, "msis", fake_msis)

    atmosphere_module.atmospheric_properties_from_height(np.array([200.0]), msis_f107=f107, msis_ap=ap)
    assert fake_msis.last_kwargs is not None
    assert set(fake_msis.last_kwargs.keys()) == {"f107s", "f107as", "aps"}
    assert float(fake_msis.last_kwargs["f107s"][0]) == f107
    assert float(fake_msis.last_kwargs["f107as"][0]) == f107
    assert float(fake_msis.last_kwargs["aps"][0][0]) == ap
