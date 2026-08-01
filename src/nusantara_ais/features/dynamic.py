import numpy as np


def add_dynamic_features(df):

    # ==========================================
    # Speed
    # ==========================================

    df["speed_change"] = (
        df["speed"] -
        df["previous_speed"].fillna(df["speed"])
    )

    df["acceleration_abs"] = (
        df["acceleration"].abs()
    )

    # ==========================================
    # Turning
    # ==========================================

    df["turning_rate"] = (
        df["course_change"] /
        (df["time_gap_second"] + 1)
    )

    # ==========================================
    # Heading
    # ==========================================

    rad = np.deg2rad(df["course"].fillna(0))

    df["heading_sin"] = np.sin(rad)
    df["heading_cos"] = np.cos(rad)

    # ==========================================
    # Time Encoding
    # ==========================================

    df["hour_sin"] = np.sin(
        2*np.pi*df["hour"]/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*df["hour"]/24
    )

    df["month_sin"] = np.sin(
        2*np.pi*df["month"]/12
    )

    df["month_cos"] = np.cos(
        2*np.pi*df["month"]/12
    )

    # ==========================================
    # Stop Moving
    # ==========================================

    df["movement_state"] = (
        df["speed"] > 1
    ).astype(int)

    return df