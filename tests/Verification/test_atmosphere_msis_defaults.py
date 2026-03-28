from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ariss.utils import atmosphere as atmosphere_module
from ariss.utils.atmosphere import MSIS_AP, MSIS_F107, _resolved_msis_inputs, _use_pymsis_default_activity


def test_resolved_msis_inputs_preserves_zero_pair_sentinel() -> None:
    _, f107, ap, _, _, _ = _resolved_msis_inputs(
        msis_date=None,
        msis_f107=0.0,
        msis_ap=0.0,
        latitude=None,
        longitude=None,
        use_average=None,
    )
    assert f107 == 0.0
    assert ap == 0.0
    assert _use_pymsis_default_activity(f107, ap)


def test_resolved_msis_inputs_uses_defaults_for_missing_values() -> None:
    _, f107, ap, _, _, _ = _resolved_msis_inputs(
        msis_date=None,
        msis_f107=None,
        msis_ap=None,
        latitude=None,
        longitude=None,
        use_average=None,
    )
    assert f107 == MSIS_F107
    assert ap == MSIS_AP
    assert not _use_pymsis_default_activity(f107, ap)


def test_resolved_msis_inputs_keeps_explicit_non_default_values() -> None:
    _, f107, ap, _, _, _ = _resolved_msis_inputs(
        msis_date=None,
        msis_f107=110.0,
        msis_ap=22.0,
        latitude=None,
        longitude=None,
        use_average=None,
    )
    assert f107 == 110.0
    assert ap == 22.0
    assert not _use_pymsis_default_activity(f107, ap)


def test_atmos_omits_activity_kwargs_for_zero_pair(monkeypatch) -> None:
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
            data = np.ones((n, 5), dtype=float)
            return data

    fake_msis = _FakeMsis()
    monkeypatch.setattr(atmosphere_module, "msis", fake_msis)

    atmosphere_module.atmos(np.array([200.0]), msis_f107=0.0, msis_ap=0.0)
    assert fake_msis.last_kwargs == {}


def test_atmos_passes_activity_kwargs_for_explicit_values(monkeypatch) -> None:
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
            data = np.ones((n, 5), dtype=float)
            return data

    fake_msis = _FakeMsis()
    monkeypatch.setattr(atmosphere_module, "msis", fake_msis)

    atmosphere_module.atmos(np.array([200.0]), msis_f107=125.0, msis_ap=11.0)
    assert fake_msis.last_kwargs is not None
    assert set(fake_msis.last_kwargs.keys()) == {"f107s", "f107as", "aps"}
