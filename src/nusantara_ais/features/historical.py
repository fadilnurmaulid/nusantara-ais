def add_historical_features(df):

    vessel_size = (
        df.groupby("mmsi")
          .size()
    )

    df["historical_points"] = (
        df["mmsi"]
        .map(vessel_size)
    )

    trip_size = (
        df.groupby("trip_id")
          .size()
    )

    df["trip_length"] = (
        df["trip_id"]
        .map(trip_size)
    )

    return df