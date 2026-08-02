import numpy as np
import pandas as pd


def anomaly_summary(df):

    score = df["anomaly_score"]

    summary = {

        "count": len(score),
        "mean": score.mean(),
        "median": score.median(),
        "std": score.std(),
        "min": score.min(),
        "max": score.max(),

        "p90": score.quantile(0.90),
        "p95": score.quantile(0.95),
        "p99": score.quantile(0.99),

    }

    return pd.Series(summary)


def top_percent(df, percent=1):

    threshold = np.percentile(
        df["anomaly_score"],
        100-percent
    )

    return df[
        df["anomaly_score"] >= threshold
    ].copy()


def zscore(df):

    s = df["anomaly_score"]

    return (
        s-s.mean()
    )/s.std()


def categorize(df):

    df = df.copy()

    z = zscore(df)

    df["risk_level"] = "Normal"

    df.loc[
        z > 1.5,
        "risk_level"
    ] = "Moderate"

    df.loc[
        z > 2.5,
        "risk_level"
    ] = "High"

    df.loc[
        z > 4,
        "risk_level"
    ] = "Extreme"

    return df