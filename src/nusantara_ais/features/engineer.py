from .dynamic import add_dynamic_features
from .spatial import add_spatial_features
from .historical import add_historical_features
from .reliability import add_reliability_features


def build_features(df):

    df = add_dynamic_features(df)
    df = add_spatial_features(df)
    df = add_historical_features(df)
    df = add_reliability_features(df)

    return df