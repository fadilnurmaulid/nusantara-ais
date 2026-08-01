import pandas as pd


AIS_FEATURES = [
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
    "heading_sin",
    "heading_cos",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "movement_state",
    "log_distance_port",
    "log_distance_shore",
    "depth_abs",
    "historical_points",
    "trip_length",
    "gps_gap",
    "missing_course",
    "is_weekend",
]


def build_ais_nodes(df: pd.DataFrame):

    nodes = df.copy()

    nodes["node_id"] = range(len(nodes))

    x = nodes[AIS_FEATURES].fillna(0)

    return nodes, x


def build_port_nodes(df):

    ports = (
        df[
            [
                "nearest_port",
                "distance_to_port_km",
            ]
        ]
        .drop_duplicates("nearest_port")
        .reset_index(drop=True)
    )

    ports["node_id"] = range(len(ports))

    return ports


def build_trip_nodes(df):

    trips = (
        df.groupby("trip_id")
        .agg(
            duration=("time_gap_second", "sum"),
            points=("trip_id", "size"),
            mean_speed=("speed", "mean"),
        )
        .reset_index()
    )

    trips["node_id"] = range(len(trips))

    return trips


def build_protected_nodes(df):

    protected = (
        df[
            [
                "protected_name",
                "protected_category",
            ]
        ]
        .drop_duplicates("protected_name")
        .reset_index(drop=True)
    )

    protected["node_id"] = range(len(protected))

    return protected