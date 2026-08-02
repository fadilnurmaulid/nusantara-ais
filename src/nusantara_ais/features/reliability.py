import numpy as np
import pandas as pd

def compute_ari(df: pd.DataFrame):

    score = np.ones(len(df))

    # -------------------------------------------------
    # Missing heading
    # -------------------------------------------------

    score -= df["missing_course"] * 0.20

    # -------------------------------------------------
    # GPS gap
    # -------------------------------------------------

    score -= np.clip(
        df["gps_gap"] / 5,
        0,
        0.20,
    )

    # -------------------------------------------------
    # Sudden acceleration
    # -------------------------------------------------

    score -= np.clip(
        df["acceleration_abs"] / 8,
        0,
        0.20,
    )

    # -------------------------------------------------
    # Extreme turning
    # -------------------------------------------------

    score -= np.clip(
        df["turning_rate"] / 180,
        0,
        0.20,
    )

    # -------------------------------------------------

    score = np.clip(score, 0, 1)

    df["ari"] = score

    return df

def add_reliability_features(df):

    df["gps_gap"] = (
        df["time_gap_second"] > 3600
    ).astype(int)

    df["missing_course"] = (
        df["course"].isna()
    ).astype(int)

    return df