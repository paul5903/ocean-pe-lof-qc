# -*- coding: utf-8 -*-
"""
Synthetic Oceanographic CTD Profile Generator:
Generates physically realistic CTD vertical profiles with thermocline, halocline,
and acoustic channels, along with controlled sensor anomaly injections for reproducible benchmarking.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from . import physics_engine, config


def generate_synthetic_profile(
    profile_id: str,
    lat: float = 15.0,
    lon: float = 112.0,
    n_levels: int = 50,
    max_depth: float = 2000.0,
    inject_anomaly: bool = False,
    anomaly_type: str = "spike",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a single synthetic ocean vertical profile obeying oceanographic dynamics.
    """
    rng = np.random.RandomState(random_seed)

    depths = np.linspace(5.0, max_depth, n_levels)
    pressures = depths * config.PRESSURE_RATIO_APPROX + rng.normal(0, 0.05, n_levels)

    # 1. Physical Thermocline Profile: Surface mixed layer + thermocline decay + deep cold layer
    t_surf = 28.0 + rng.normal(0, 0.5)
    t_deep = 2.5 + rng.normal(0, 0.2)
    z_therm = 150.0  # Thermocline depth (m)
    scale_therm = 100.0
    temperatures = t_deep + (t_surf - t_deep) / (1.0 + np.exp((depths - z_therm) / scale_therm))
    temperatures += rng.normal(0, 0.05, n_levels)

    # 2. Physical Halocline Profile: Surface fresher + subsurface salinity maximum + deep stable
    s_surf = 33.8 + rng.normal(0, 0.2)
    s_max = 34.6
    s_deep = 34.7
    salinities = s_surf + (s_max - s_surf) * np.exp(-((depths - 120.0) / 80.0)**2) + (s_deep - s_surf) * (depths / max_depth) * 0.5
    salinities += rng.normal(0, 0.02, n_levels)

    # 3. Direct Mackenzie Acoustic Speed
    sound_speeds = physics_engine.calculate_mackenzie_sound_velocity(temperatures, salinities, depths)
    sound_speeds += rng.normal(0, 0.1, n_levels)

    # Ground truth flags
    ground_truth = np.zeros(n_levels, dtype=int)

    # 4. Inject Controlled Anomalies if requested
    if inject_anomaly:
        candidate_pool = np.arange(5, n_levels - 5) if n_levels > 12 else np.arange(n_levels)
        inject_size = max(1, min(len(candidate_pool), int(n_levels * 0.1)))
        inject_indices = rng.choice(candidate_pool, size=inject_size, replace=False)
        ground_truth[inject_indices] = 1

        if anomaly_type == "spike":
            temperatures[inject_indices] += rng.choice([-8.0, 10.0], size=len(inject_indices))
        elif anomaly_type == "drift":
            salinities[inject_indices] += np.linspace(1.5, 4.0, len(inject_indices))
        elif anomaly_type == "stuck":
            temperatures[inject_indices] = temperatures[inject_indices[0]]
            sound_speeds[inject_indices] = sound_speeds[inject_indices[0]]
        elif anomaly_type == "acoustic_inconsistency":
            sound_speeds[inject_indices] += 45.0  # Violates Mackenzie relationship

    df_prof = pd.DataFrame({
        "source": "Synthetic_Benchmark",
        "profile_id": profile_id,
        "station_id": profile_id,
        "timestamp": "2026-01-01 12:00:00",
        "lat": lat,
        "lon": lon,
        "depth": depths,
        "pressure": pressures,
        "temperature": temperatures,
        "salinity": salinities,
        "sound_vel_direct": sound_speeds,
        "qc_raw_flag": np.where(ground_truth == 1, 4, 1),
        "is_ground_truth_anomaly": ground_truth
    })

    return df_prof


def generate_synthetic_benchmark_dataset(
    n_profiles: int = 50,
    levels_per_profile: int = 40,
    anomaly_profile_ratio: float = 0.2,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generates a complete multi-profile synthetic CTD dataset."""
    rng = np.random.RandomState(random_seed)
    profiles = []

    for i in range(n_profiles):
        p_id = f"synth_prof_{i:04d}"
        lat = float(rng.uniform(10.0, 20.0))
        lon = float(rng.uniform(108.0, 118.0))
        is_anomaly = (rng.rand() < anomaly_profile_ratio)
        a_type = rng.choice(["spike", "drift", "stuck", "acoustic_inconsistency"])

        p_df = generate_synthetic_profile(
            profile_id=p_id,
            lat=lat,
            lon=lon,
            n_levels=levels_per_profile,
            inject_anomaly=is_anomaly,
            anomaly_type=a_type,
            random_seed=random_seed + i
        )
        profiles.append(p_df)

    return pd.concat(profiles, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Ocean Profile Generator")
    parser.add_argument("--profiles", type=int, default=30, help="Number of vertical profiles")
    parser.add_argument("--levels", type=int, default=30, help="Number of depth levels per profile")
    parser.add_argument("--output", type=str, default="data/synthetic_benchmark.csv", help="Output file path")
    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_synth = generate_synthetic_benchmark_dataset(n_profiles=args.profiles, levels_per_profile=args.levels)
    df_synth.to_csv(out_p, index=False)
    print(f"[SUCCESS] Generated {len(df_synth)} synthetic observations across {args.profiles} profiles -> {out_p}")
