from sklearn.preprocessing import StandardScaler

import pandas as pd

NUMERIC = [
    "speed",
    "course",
    "distance_from_shore",
    "distance_from_port",
    "distance_to_port_km",
    "distance_to_lane_km",
    "distance_to_protected_km",
    "bathymetry_m",
    "speed_change",
    "acceleration",
    "acceleration_abs",
    "turning_rate",
    "log_distance_port",
    "log_distance_shore",
    "depth_abs",
    "historical_points",
    "trip_length",
    "gps_gap",
]

def normalize(df):

    scaler = StandardScaler()

    df[NUMERIC] = scaler.fit_transform(
        df[NUMERIC]
    )

    return df