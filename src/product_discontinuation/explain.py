"""Local SHAP explanations for the deployed discontinuation model.

Method: ``shap.PermutationExplainer`` applied to the complete fitted pipeline (preprocessing
plus classifier) over its original input columns, with an independent masker built from a
stratified sample of development editions. The explained output is the calibrated log-odds of
the positive class (DiscontinuedTF = 1). Platt calibration is linear on this scale, so the
values describe exactly the probability shown to the user.

Permutation SHAP is a Monte Carlo estimate of the Shapley values, but it satisfies local
accuracy exactly: contributions always sum to f(x) - E[f(background)].

TreeExplainer is kept only as a cross-check. Its path-dependent mode is exact but takes no
explicit background; its interventional mode fails SHAP's additivity check for this model
with shap 0.49.1 (see shap_analysis.py and reports/results/shap_validation_h13.json).
"""
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.special import expit, logit

from . import config
from .calibration import calibrated_log_odds
from .features import CATEGORICAL_FEATURES
from .predict import positive_probability

ADDITIVITY_TOLERANCE = 1e-6
BACKGROUND_SIZE = 200
MAX_EVALS = 1000
_PROBABILITY_CLIP = 1e-12
_SHAP_LOCK = threading.Lock()  # shap's permutations and matplotlib both use global state


class _ExactMasker(shap.maskers.Independent):
    """Independent masker that treats a value as unchanged only when exactly equal.

    SHAP's default uses np.isclose, skipping re-evaluation for values within 1e-5 relative
    tolerance; a tree split can fall between such values, which breaks local accuracy.
    """

    def invariants(self, x):
        return np.equal(x, self.data)


def background_paths(horizon, models_dir=None, metadata_dir=None):
    name = f"shap_background_h{horizon}"
    return (Path(models_dir or config.MODELS_DIR) / f"{name}.parquet",
            Path(metadata_dir or config.METADATA_DIR) / f"{name}.json")


def _check_development_only(background):
    if "CatEdition" not in background.columns:
        return
    held_out = {config.TEST_EDITION, *config.CENSORED_EDITIONS}
    found = held_out.intersection(background["CatEdition"].unique().tolist())
    if found:
        raise ValueError(f"SHAP background contains held-out editions: {sorted(found)}")


def build_background(dev, columns, size=BACKGROUND_SIZE, seed=config.RANDOM_STATE):
    """Sample development rows, stratified by edition and outcome in proportion to their size."""
    _check_development_only(dev)
    size = min(size, len(dev))
    strata = ["CatEdition", config.TARGET]
    ordered = dev.sample(frac=1, random_state=seed).sort_values(strata, kind="stable")
    positions = ((np.arange(size) + 0.5) * len(ordered) / size).astype(int)
    return ordered.iloc[positions][strata + list(columns)].reset_index(drop=True)


def save_background(background, dev, bundle_info, horizon, models_dir=None, metadata_dir=None,
                    seed=config.RANDOM_STATE):
    data_path, info_path = background_paths(horizon, models_dir, metadata_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    background.to_parquet(data_path, index=False)
    info = {**bundle_info, "rows": len(background), "seed": seed, "source_rows": len(dev),
            "editions": sorted(int(e) for e in background["CatEdition"].unique()),
            "strategy": "systematic sample of shuffled rows ordered by edition and outcome",
            "positive_rate_background": float(background[config.TARGET].mean()),
            "positive_rate_source": float(dev[config.TARGET].mean())}
    info_path.write_text(json.dumps(info, indent=2))
    return data_path, info_path


def load_background(horizon, models_dir=None, metadata_dir=None):
    path = background_paths(horizon, models_dir, metadata_dir)[0]
    if not path.exists():
        raise FileNotFoundError(f"SHAP background not found: {path}. "
                                "Run: python -m product_discontinuation.shap_analysis")
    return pd.read_parquet(path)


def output_feature_groups(preprocessor, columns):
    """Original model input that each preprocessed column was derived from."""
    groups = []
    for name in preprocessor.get_feature_names_out():
        if name in columns:
            groups.append(name)
            continue
        owners = [c for c in columns if name.startswith(f"{c}_")]
        if not owners:
            raise ValueError(f"cannot map preprocessed column {name!r} to a model input")
        groups.append(max(owners, key=len))
    uncovered = set(columns) - set(groups)
    if uncovered:
        raise ValueError(f"model inputs dropped by preprocessing: {sorted(uncovered)}")
    return np.array(groups)


def positive_class(values, expected_value):
    """Select the DiscontinuedTF = 1 output from any SHAP binary-output layout."""
    if isinstance(values, list):
        values, expected_value = values[1], np.ravel(expected_value)[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values, expected_value = values[..., 1], np.ravel(expected_value)[1]
    return values, float(np.ravel(expected_value)[-1])


def tree_path_dependent_values(bundle, rows):
    """Exact path-dependent TreeSHAP per original input; a cross-check, not used by the app."""
    columns = list(bundle["feature_columns"])
    preprocess, model = bundle["pipeline"][:-1], bundle["pipeline"][-1]
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    values, base = positive_class(explainer.shap_values(preprocess.transform(rows[columns])),
                                  explainer.expected_value)
    groups = output_feature_groups(preprocess, columns)
    table = pd.DataFrame(values).T.groupby(groups).sum().T[columns]
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        table, base = table * calibrator["a"], float(calibrated_log_odds(base, calibrator))
    return table, base


@dataclass
class Explanation:
    probability: float
    base_value: float
    contributions: pd.DataFrame

    @property
    def log_odds(self):
        return self.base_value + self.contributions["contribution"].sum()


class LocalExplainer:
    """Permutation SHAP for single feature rows, against a fixed development-data background."""

    method = "shap.PermutationExplainer on the complete pipeline"

    def __init__(self, bundle, background, max_evals=MAX_EVALS, seed=config.RANDOM_STATE):
        self.bundle, self.pipeline = bundle, bundle["pipeline"]
        self.columns = list(bundle["feature_columns"])
        self.max_evals, self.seed = max_evals, seed
        if list(self.pipeline.classes_) != [0, 1]:
            raise ValueError(f"expected binary classes [0, 1], got {list(self.pipeline.classes_)}")
        missing = [c for c in self.columns if c not in background.columns]
        if missing:
            raise ValueError(f"SHAP background is missing model columns: {missing}")
        _check_development_only(background)
        self.background = background[self.columns].reset_index(drop=True)
        self.categorical = [c for c in self.columns if c in CATEGORICAL_FEATURES]
        self.levels = {c: sorted(self.background[c].unique().tolist()) for c in self.categorical}
        self.masker = _ExactMasker(self._encode(self.background, self.levels), max_samples=len(self.background))
        self.base_value = float(self._log_odds(self.background).mean())

    def _log_odds(self, frame):
        probability = self.pipeline.predict_proba(frame)[:, 1]
        return calibrated_log_odds(logit(np.clip(probability, _PROBABILITY_CLIP, 1 - _PROBABILITY_CLIP)),
                                   self.bundle.get("calibrator"))

    def _encode(self, frame, levels):
        """Reversible numeric form of the inputs, required by SHAP's tabular masker."""
        encoded = frame.copy()
        for col in self.categorical:
            lookup = {v: i for i, v in enumerate(levels[col])}
            encoded[col] = [lookup.get(v, len(levels[col]) - 1) for v in frame[col]]
        return encoded.to_numpy(dtype=float)

    def _decode(self, values, levels):
        frame = pd.DataFrame(values, columns=self.columns)
        for col in self.categorical:
            codes = values[:, self.columns.index(col)].astype(int)
            frame[col] = np.asarray(levels[col], dtype=object)[codes]
        return frame

    def _validated(self, feature_row):
        if len(feature_row) != 1:
            raise ValueError("explain expects exactly one feature row")
        missing = [c for c in self.columns if c not in feature_row.columns]
        if missing:
            raise ValueError(f"feature row is missing model columns: {missing}")
        row = feature_row[self.columns].copy()
        for col in row.columns.difference(self.categorical):
            row[col] = pd.to_numeric(row[col], errors="raise").astype(float)
            if np.isinf(row[col].iloc[0]):
                raise ValueError(f"feature {col!r} is infinite")
        return row

    def explain(self, feature_row):
        row = self._validated(feature_row)
        levels = {c: self.levels[c] + ([] if row[c].iloc[0] in self.levels[c] else [row[c].iloc[0]])
                  for c in self.categorical}
        predict = lambda values: self._log_odds(self._decode(values, levels))  # noqa: E731
        with _SHAP_LOCK:
            state = np.random.get_state()
            try:
                explainer = shap.PermutationExplainer(predict, self.masker, seed=self.seed)
                result = explainer(self._encode(row, levels), max_evals=self.max_evals, silent=True)
            finally:
                np.random.set_state(state)
        values, base = positive_class(result.values, result.base_values)
        if values.shape != (1, len(self.columns)):
            raise RuntimeError(f"unexpected SHAP output shape {values.shape}")
        probability = float(positive_probability(self.bundle, row)[0])
        total = base + values.sum()
        if abs(float(expit(total)) - probability) > ADDITIVITY_TOLERANCE:
            raise RuntimeError(f"SHAP values do not reproduce the prediction "
                               f"({float(expit(total)):.6f} vs {probability:.6f})")
        table = pd.DataFrame({"feature": self.columns, "value": feature_row[self.columns].iloc[0].to_numpy(),
                              "contribution": values[0]})
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
        return Explanation(probability, base, table.reset_index(drop=True))


def waterfall_figure(explanation, max_display=12):
    """SHAP waterfall plot for one explanation, on the log-odds scale."""
    import matplotlib.pyplot as plt

    table = explanation.contributions
    data = shap.Explanation(values=table["contribution"].to_numpy(dtype=float),
                            base_values=explanation.base_value,
                            data=table["value"].to_numpy(dtype=object),
                            feature_names=table["feature"].tolist())
    with _SHAP_LOCK:
        plt.figure()
        shap.plots.waterfall(data, max_display=max_display, show=False)
        figure = plt.gcf()
        figure.axes[0].set_xlabel("Log-odds of DiscontinuedTF = 1 (calibrated)", labelpad=26)
    return figure
