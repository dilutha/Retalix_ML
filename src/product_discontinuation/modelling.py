"""Preprocessing, model definitions, edition-based CV and hyperparameter search."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import feature_columns

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}
REFIT_METRIC = "average_precision"

PARAM_SPACES = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [12, 20, None],
        "model__min_samples_leaf": [5, 20, 50],
        "model__max_features": ["sqrt", 0.2],
        "model__class_weight": [None, "balanced_subsample"],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 100],
        "model__l2_regularization": [0.0, 1.0],
    },
}
MODEL_NAMES = tuple(PARAM_SPACES)


def make_preprocessor(horizon, include_status=False, exclude=()):
    numeric, categorical, binary = feature_columns(horizon, include_status, exclude)
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
    ], verbose_feature_names_out=False)


def make_estimator(name, random_state=config.RANDOM_STATE):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=random_state)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def make_pipeline(name, horizon, include_status=False, params=None, exclude=()):
    pipe = Pipeline([
        ("preprocess", make_preprocessor(horizon, include_status, exclude)),
        ("model", make_estimator(name)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def edition_splits(editions, test_editions=config.CV_TEST_EDITIONS):
    """Expanding-window folds: train on all earlier editions, test on the next one."""
    editions = np.asarray(editions)
    splits = []
    for test_edition in test_editions:
        train_idx = np.flatnonzero(editions < test_edition)
        test_idx = np.flatnonzero(editions == test_edition)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"empty fold for test edition {test_edition}")
        splits.append((train_idx, test_idx))
    return splits


def tune(name, X, y, editions, horizon, include_status=False, n_iter=8, n_jobs=1, verbose=1):
    """Search hyperparameters with edition-based CV; refit on all supplied data."""
    pipe = make_pipeline(name, horizon, include_status)
    space = PARAM_SPACES[name]
    common = dict(scoring=SCORING, refit=REFIT_METRIC, cv=edition_splits(editions),
                  n_jobs=n_jobs, verbose=verbose, return_train_score=False)
    if len(ParameterGrid(space)) <= n_iter:
        search = GridSearchCV(pipe, space, **common)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter,
                                    random_state=config.RANDOM_STATE, **common)
    return search.fit(X, y)


def cv_results_table(search, name):
    results = pd.DataFrame(search.cv_results_)
    cols = ["params"] + [f"{s}_{m}" for m in SCORING for s in ("mean_test", "std_test")]
    folds = [c for c in results if c.startswith("split") and c.endswith(f"_test_{REFIT_METRIC}")]
    table = results[cols + folds + ["mean_fit_time", f"rank_test_{REFIT_METRIC}"]].copy()
    table.insert(0, "model", name)
    table["params"] = table["params"].astype(str)
    return table.sort_values(f"rank_test_{REFIT_METRIC}")


def out_of_fold_predictions(pipe, X, y, editions):
    """Probabilities for each CV test edition from a model trained on earlier editions."""
    oof = pd.Series(np.nan, index=X.index)
    for train_idx, test_idx in edition_splits(editions):
        model = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof.iloc[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return oof.dropna()
