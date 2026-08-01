def add_reliability_features(df):

    df["gps_gap"] = (
        df["time_gap_second"] > 3600
    ).astype(int)

    df["missing_course"] = (
        df["course"].isna()
    ).astype(int)

    return df