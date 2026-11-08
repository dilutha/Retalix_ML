"""Inference: the single entry point shared by evaluation, the app and monitoring."""
import numpy as np
import pandas as pd

from . import config
from .features import CATEGORICAL_FEATURES, build_features

MEDIUM_RISK_MARGIN = 0.15


def risk_category(probability, threshold):
    """High at/above the decision threshold, Medium within a margin below it, else Low."""
    probability = np.asarray(probability, dtype=float)
    medium_floor = max(threshold - MEDIUM_RISK_MARGIN, 0.0)
    return np.select([probability >= threshold, probability >= medium_floor],
                     ["High", "Medium"], default="Low")


def predict_features(bundle, features):
    columns = bundle["feature_columns"]
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(f"feature table is missing model columns: {missing}")
    probability = bundle["pipeline"].predict_proba(features[columns])[:, 1]
    out = features[[c for c in config.GROUP_KEYS if c in features.columns]].copy()
    out["probability"] = probability
    out["predicted_discontinued"] = (probability >= bundle["threshold"]).astype(int)
    out["risk_category"] = risk_category(probability, bundle["threshold"])
    return out


def predict_weekly(bundle, weekly, history=None):
    """Build point-in-time features from raw weekly rows and score them."""
    features = build_features(weekly, bundle["horizon"], history=history)
    return features, predict_features(bundle, features)


def reference_values(features, columns):
    """Typical values (median / mode) used as the 'neutral' input for local explanations."""
    ref = {}
    for col in columns:
        series = features[col].dropna()
        if col in CATEGORICAL_FEATURES:
            ref[col] = series.mode().iloc[0]
        else:
            ref[col] = float(series.median())
    return ref


def explain_prediction(bundle, feature_row):
    """Feature-reset sensitivity analysis (not SHAP; the app's explanations are in explain.py).

    Change in probability when one feature at a time is reset to its typical training
    value; positive values mean the product's actual value pushes the risk up. It ignores
    interactions and is retained only as an additional sensitivity check.
    """
    if len(feature_row) != 1:
        raise ValueError("explain_prediction expects exactly one row")
    columns = bundle["feature_columns"]
    reference = bundle["reference_values"]
    row = feature_row[columns]
    variants = pd.concat([row] * (len(columns) + 1), ignore_index=True)
    for i, col in enumerate(columns, start=1):
        variants.loc[i, col] = reference[col]
    probs = bundle["pipeline"].predict_proba(variants)[:, 1]
    effects = pd.DataFrame({
        "feature": columns,
        "value": [row.iloc[0][c] for c in columns],
        "typical_value": [reference[c] for c in columns],
        "effect": probs[0] - probs[1:],
    })
    return effects.reindex(effects["effect"].abs().sort_values(ascending=False).index)
