"""Inference: the single entry point shared by evaluation, the app and monitoring."""
import numpy as np
import pandas as pd

from . import config
from .calibration import apply_calibration
from .features import CATEGORICAL_FEATURES, build_features


SCENARIO_INPUT_TOLERANCE = 0.006


def risk_category(probability, threshold, medium_floor):
    """High at/above the decision threshold, Medium from the validated floor, else Low."""
    probability = np.asarray(probability, dtype=float)
    return np.select([probability >= threshold, probability >= medium_floor],
                     ["High", "Medium"], default="Low")


def positive_probability(bundle, features):
    """Calibrated probability of DiscontinuedTF = 1 (raw if the bundle has no calibrator)."""
    pipeline = bundle["pipeline"]
    positive = np.flatnonzero(np.asarray(pipeline.classes_) == 1)
    if positive.size != 1:
        raise ValueError(f"model has no positive class 1: classes_={list(pipeline.classes_)}")
    raw = pipeline.predict_proba(features[bundle["feature_columns"]])[:, positive[0]]
    return apply_calibration(raw, bundle.get("calibrator"))


def predict_features(bundle, features):
    missing = [c for c in bundle["feature_columns"] if c not in features.columns]
    if missing:
        raise ValueError(f"feature table is missing model columns: {missing}")
    probability = positive_probability(bundle, features)
    out = features[[c for c in config.GROUP_KEYS if c in features.columns]].copy()
    out["probability"] = probability
    out["predicted_discontinued"] = (probability >= bundle["threshold"]).astype(int)
    out["risk_category"] = risk_category(probability, bundle["threshold"], bundle["risk_medium_floor"])
    return out


def predict_weekly(bundle, weekly, history=None):
    """Build point-in-time features from raw weekly rows and score them."""
    features = build_features(weekly, bundle["horizon"], history=history)
    return features, predict_features(bundle, features)


def _validated(name, value, allow_zero=True):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a finite number at or above zero, got {value}")
    return value


def _same(first, second):
    """Equal to the two decimals shown by the app's number inputs."""
    return bool(np.isclose(first, second, rtol=0, atol=SCENARIO_INPUT_TOLERANCE))


def _reshape(series, cutoff, lag, current, earlier):
    """Move a weekly series to new values at the prediction week and four weeks before it."""
    out = series.astype(float).copy()
    if current is None and earlier is None:
        return out
    old_current = out.loc[cutoff]
    current = old_current if current is None else current
    unchanged_earlier = earlier is None or lag not in out.index or _same(earlier, out.loc[lag])
    if _same(current, old_current) and unchanged_earlier:
        return out
    if earlier is None or lag not in out.index:
        return out * (current / old_current) if old_current > 0 else out * 0 + current
    old_earlier = out.loc[lag]
    before = out.index < lag
    out.loc[before] = out.loc[before] * (earlier / old_earlier) if old_earlier > 0 else earlier
    window = (out.index >= lag) & (out.index <= cutoff)
    out.loc[window] = np.interp(out.index[window], [lag, cutoff], [earlier, current])
    return out


def apply_scenario(product_weekly, horizon, history, actuals=None, forecast=None, actuals_4_weeks_earlier=None,
                   forecast_4_weeks_earlier=None, price=None, weeks_listed=None, new_to_range=False):
    """Weekly rows and history for a what-if version of one product-edition.

    Scenarios are expressed as raw weekly data, so they reach the model through `build_features`,
    the same code as training. Weeks between the two entered points are interpolated and earlier
    weeks rescaled, so a level change does not appear as an artificial one-week spike.
    """
    if product_weekly[config.GROUP_KEYS].drop_duplicates().shape[0] != 1:
        raise ValueError("a scenario needs the rows of exactly one product-edition")
    cutoff, lag = -horizon, -horizon - config.TREND_LOOKBACK_WEEKS
    rows = product_weekly[product_weekly["WeeksOut"] <= cutoff].sort_values("WeeksOut").set_index("WeeksOut", drop=False)
    if cutoff not in rows.index:
        raise ValueError(f"the product has no data at WeeksOut = {cutoff}")
    rows = rows.drop(columns=[config.TARGET], errors="ignore")

    if weeks_listed is not None:
        if not 1 <= int(weeks_listed) <= config.MAX_WEEKS_OUT - horizon + 1:
            raise ValueError(f"Weeks listed must be between 1 and {config.MAX_WEEKS_OUT - horizon + 1}")
        rows = rows[rows["WeeksOut"] > cutoff - int(weeks_listed)]
    rows["ActualsPerWeek"] = _reshape(rows["ActualsPerWeek"], cutoff, lag, _validated("Average weekly sales", actuals),
                                      _validated("Average weekly sales 4 weeks earlier", actuals_4_weeks_earlier))
    rows["ForecastPerWeek"] = _reshape(rows["ForecastPerWeek"], cutoff, lag, _validated("Expected weekly sales", forecast),
                                       _validated("Expected weekly sales 4 weeks earlier", forecast_4_weeks_earlier))
    if price is not None:
        rows["SalePriceIncVAT"] = _validated("Selling price", price)
    if new_to_range:
        history = history[history["ProductKey"] != rows["ProductKey"].iloc[0]]
    return rows.reset_index(drop=True), history


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
    probs = positive_probability(bundle, variants)
    effects = pd.DataFrame({
        "feature": columns,
        "value": [row.iloc[0][c] for c in columns],
        "typical_value": [reference[c] for c in columns],
        "effect": probs[0] - probs[1:],
    })
    return effects.reindex(effects["effect"].abs().sort_values(ascending=False).index)
