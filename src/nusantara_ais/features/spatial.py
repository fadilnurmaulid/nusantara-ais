import numpy as np


def add_spatial_features(df):

    df["log_distance_port"] = np.log1p(
        df["distance_to_port_km"]
    )

    df["log_distance_shore"] = np.log1p(
        df["distance_from_shore"]
    )

    df["depth_abs"] = (
        df["bathymetry_m"].abs()
    )

    return df