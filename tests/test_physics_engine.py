# -*- coding: utf-8 -*-
import numpy as np
import pytest
from src import physics_engine


def test_mackenzie_sound_velocity_standard_values():
    # At T=0C, S=35 PSU, D=0m, Mackenzie c = 1448.96 m/s
    c = physics_engine.calculate_mackenzie_sound_velocity(np.array([0.0]), np.array([35.0]), np.array([0.0]))
    assert np.isclose(c[0], 1448.96, atol=0.01)

    # Typical warm tropical surface water: T=25C, S=35, D=0m -> ~1534 m/s
    c_warm = physics_engine.calculate_mackenzie_sound_velocity(np.array([25.0]), np.array([35.0]), np.array([0.0]))
    assert 1530.0 < c_warm[0] < 1540.0


def test_unesco_density_approx():
    # Cold standard seawater: T=4C, S=35 PSU, P=0 dbar -> ~1028 kg/m3
    rho = physics_engine.calculate_unesco_density_approx(np.array([4.0]), np.array([35.0]), np.array([0.0]))
    assert 1025.0 < rho[0] < 1030.0


def test_hydrostatic_pressure():
    # At D=1000m -> P ~ 1019.7 dbar
    p = physics_engine.compute_hydrostatic_pressure(np.array([1000.0]))
    assert np.isclose(p[0], 1019.716, atol=0.1)
