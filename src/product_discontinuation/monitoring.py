"""Post-deployment monitoring: input checks, prediction logging, drift and outcome tracking.

Without labels only inputs and prediction distributions can be monitored. Model
performance is computed only once actual DiscontinuedTF outcomes become available
after the range refresh.
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import check_columns, validate_catalogue
from .evaluation import classification_metrics
from .features import CATEGORICAL_FEATURES, REQUIRED_COLUMNS

PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
MAX_DISCRETE_LEVELS = 10
LOG_COLUMNS = ["logged_at", "model_version", "horizon", "source", "CatEdition", "ProductKey",
               "probability", "predicted_discontinued", "risk_category"]
_EPS = 1e-4


def validate_scoring_input(weekly):
    """Schema and integrity checks for unlabelled weekly data before scoring."""
    check_columns(weekly, REQUIRED_COLUMNS, "scoring data")
    return validate_catalogue(weekly, require_target=False)


def _numeric_reference(values, n_bins):
    values = pd.Series(values, dtype=float)
    missing_rate = float(values.isna().mean())
    values = values.dropna()
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([values.min() - 0.5, values.max() + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(values, bins=edges)[0]
    return {"type": "numeric", "edges": edges.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
            "missing_rate": missing_rate}


def build_drift_reference(features, columns, n_bins=10):
    """Training distribution of each input feature, stored with the model.

    Categorical and low-cardinality features (e.g. binary flags) use level
    frequencies; continuous features use training-quantile bins.
    """
    reference = {}
    for col in columns:
        is_discrete = features[col].nunique(dropna=False) <= MAX_DISCRETE_LEVELS
        if col in CATEGORICAL_FEATURES or is_discrete:
            freq = features[col].astype(str).value_counts(normalize=True)
            reference[col] = {"type": "categorical", "proportions": freq.to_dict()}
        else:
            reference[col] = _numeric_reference(features[col], n_bins)
    return reference


def population_stability_index(expected, actual):
    expected = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _psi_for(ref, values):
    if ref["type"] == "categorical":
        expected = pd.Series(ref["proportions"])
        actual = values.astype(str).value_counts(normalize=True)
        categories = expected.index.union(actual.index)
        return population_stability_index(expected.reindex(categories, fill_value=0),
                                          actual.reindex(categories, fill_value=0))
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return np.nan
    counts = np.histogram(values, bins=np.asarray(ref["edges"]))[0]
    return population_stability_index(ref["proportions"], counts / counts.sum())


def drift_status(psi):
    if np.isnan(psi):
        return "no data"
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def feature_drift_report(reference, current_features):
    rows = []
    for col, ref in reference.items():
        if col not in current_features:
            rows.append({"feature": col, "psi": np.nan, "status": "missing column"})
            continue
        psi = _psi_for(ref, current_features[col])
        rows.append({"feature": col, "psi": psi, "status": drift_status(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def prediction_summary(predictions):
    probability = predictions["probability"]
    summary = {
        "n_predictions": int(len(predictions)),
        "mean_probability": float(probability.mean()),
        "predicted_positive_rate": float(predictions["predicted_discontinued"].mean()),
    }
    counts = predictions["risk_category"].value_counts(normalize=True)
    for level in ("High", "Medium", "Low"):
        summary[f"share_{level.lower()}_risk"] = float(counts.get(level, 0.0))
    return summary


def prediction_drift(reference_probabilities, current_probabilities, n_bins=10):
    ref = _numeric_reference(reference_probabilities, n_bins)
    return _psi_for(ref, pd.Series(current_probabilities))


def log_predictions(predictions, bundle, source, path=None):
    """Append scored products with a timestamp and model version to a CSV log."""
    check_columns(predictions, config.GROUP_KEYS + ["probability", "predicted_discontinued",
                                                    "risk_category"], "predictions")
    path = Path(path or config.PREDICTION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = predictions.copy()
    entry["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry["model_version"] = bundle["model_version"]
    entry["horizon"] = bundle["horizon"]
    entry["source"] = source
    entry[LOG_COLUMNS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def load_prediction_log(path=None):
    path = Path(path or config.PREDICTION_LOG_PATH)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)


def evaluate_against_outcomes(log, outcomes, threshold):
    """Join logged predictions with actual outcomes once they are known.

    `outcomes` holds CatEdition, ProductKey and DiscontinuedTF. Only the latest
    logged prediction per product-edition and model version is evaluated.
    """
    check_columns(outcomes, config.GROUP_KEYS + [config.TARGET], "outcomes")
    latest = (log.sort_values("logged_at")
                 .drop_duplicates(config.GROUP_KEYS + ["model_version"], keep="last"))
    labels = outcomes[config.GROUP_KEYS + [config.TARGET]].drop_duplicates(config.GROUP_KEYS)
    joined = latest.merge(labels, on=config.GROUP_KEYS, how="left")
    labelled = joined[joined[config.TARGET].notna()]
    result = {"logged": int(len(latest)), "with_outcome": int(len(labelled)),
              "label_coverage": float(len(labelled) / max(len(latest), 1))}
    if labelled.empty or labelled[config.TARGET].nunique() < 2:
        result["metrics"] = None
        return result
    result["metrics"] = classification_metrics(labelled[config.TARGET].astype(int),
                                               labelled["probability"], threshold)
    return result
