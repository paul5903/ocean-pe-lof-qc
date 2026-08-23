# -*- coding: utf-8 -*-
"""
Physical Oceanography Engine:
Implements standardized physical equations for acoustic velocity, seawater density,
and hydrostatic pressure consistency in ocean observations.
"""
import numpy as np
from . import config


def calculate_mackenzie_sound_velocity(
    temperature: np.ndarray,
    salinity: np.ndarray,
    depth: np.ndarray
) -> np.ndarray:
    """
    Mackenzie (1981) 9-term equation for sound velocity in seawater (m/s):
    c(D, S, T) = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
                 + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D^2
                 - 1.025e-2*T*(S - 35) - 7.139e-13*T*D^3
    """
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)
    D = np.asarray(depth, dtype=float)

    # Use nominal salinity 34.5 PSU if missing
    S_eff = np.where(np.isnan(S), 34.5, S)

    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * (T**2)
        + 2.374e-4 * (T**3)
        + 1.340 * (S_eff - 35.0)
        + 1.630e-2 * D
        + 1.675e-7 * (D**2)
        - 1.025e-2 * T * (S_eff - 35.0)
        - 7.139e-13 * T * (D**3)
    )
    return c


def calculate_unesco_density_approx(
    temperature: np.ndarray,
    salinity: np.ndarray,
    pressure: np.ndarray
) -> np.ndarray:
    """
    UNESCO Equation of State (EOS-80) approximation for in-situ density (kg/m^3).
    """
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)
    P = np.asarray(pressure, dtype=float)

    S_eff = np.where(np.isnan(S), 34.5, S)
    P_eff = np.where(np.isnan(P), 0.0, P)

    # Pure water density SMOW
    rho_0 = (
        999.842594 + 6.793952e-2 * T - 9.095290e-3 * (T**2)
        + 1.001685e-4 * (T**3) - 1.120083e-6 * (T**4) + 6.536332e-9 * (T**5)
    )
    # Salinity terms
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * (T**2) - 8.2467e-7 * (T**3) + 5.3875e-9 * (T**4)
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * (T**2)
    C = 4.8314e-4

    rho_sec = rho_0 + A * S_eff + B * (S_eff**1.5) + C * (S_eff**2)
    # Hydrostatic compressibility correction
    rho = rho_sec + (P_eff * 0.0045)
    return rho


def compute_hydrostatic_pressure(depth: np.ndarray) -> np.ndarray:
    """Calculates theoretical hydrostatic pressure (dbar) from depth (m)."""
    return np.asarray(depth, dtype=float) * config.PRESSURE_RATIO_APPROX


def compute_temperature_gradient_log(temperature: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Calculates logarithmic vertical temperature decay gradient: T / (ln(1 + D) + 1)."""
    T = np.asarray(temperature, dtype=float)
    D = np.asarray(depth, dtype=float)
    return T / (np.log1p(np.maximum(0, D)) + 1.0)
