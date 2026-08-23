import numpy as np
from src.core import physics_engine

def test_mackenzie_sound_velocity():
    c = physics_engine.calculate_mackenzie_sound_velocity(np.array([0.0]), np.array([35.0]), np.array([0.0]))
    assert np.isclose(c[0], 1448.96, atol=0.01)

def test_unesco_density():
    rho = physics_engine.calculate_unesco_density_approx(np.array([4.0]), np.array([35.0]), np.array([0.0]))
    assert 1025.0 < rho[0] < 1030.0

def test_hydrostatic_pressure():
    p = physics_engine.compute_hydrostatic_pressure(np.array([1000.0]))
    assert np.isclose(p[0], 1019.716, atol=0.1)
