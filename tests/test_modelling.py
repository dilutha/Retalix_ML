import numpy as np
import pytest

from product_discontinuation import config
from product_discontinuation.evaluation import best_f1_threshold, classification_metrics
from product_discontinuation.features import all_feature_columns, build_features
from product_discontinuation.modelling import edition_splits, make_pipeline, make_preprocessor, tune
from product_discontinuation.train import check_no_contamination, modelling_sets

HORIZON = 13


@pytest.fixture(scope="module")
def sets(weekly):
    return modelling_sets(build_features(weekly, HORIZON))


def test_modelling_sets_are_chronological(sets):
    dev, test = sets
    assert dev["CatEdition"].min() == config.MODEL_FIRST_EDITION
    assert set(test["CatEdition"]) == {config.TEST_EDITION}
    assert not set(config.CENSORED_EDITIONS) & (set(dev["CatEdition"]) | set(test["CatEdition"]))
    check_no_contamination(dev, test)


def test_contamination_check_catches_overlap(sets):
    dev, test = sets
    with pytest.raises(AssertionError):
        check_no_contamination(test, test)


def test_edition_splits_expanding_window():
    editions = np.repeat([89, 90, 91, 92], 5)
    splits = edition_splits(editions)
    assert len(splits) == len(config.CV_TEST_EDITIONS)
    for (train_idx, test_idx), test_edition in zip(splits, config.CV_TEST_EDITIONS):
        assert (editions[train_idx] < test_edition).all()
        assert (editions[test_idx] == test_edition).all()
        assert not set(train_idx) & set(test_idx)


def test_preprocessor_handles_unseen_categories(sets):
    dev, test = sets
    cols = all_feature_columns(HORIZON)
    pre = make_preprocessor(HORIZON).fit(dev[cols])
    unseen = test[cols].copy()
    unseen["Supplier"] = "999999"
    transformed = pre.transform(unseen)
    assert transformed.shape[0] == len(unseen)
    assert not np.isnan(transformed).any()


def test_preprocessor_fitted_on_training_only(sets):
    dev, test = sets
    cols = all_feature_columns(HORIZON)
    pre = make_preprocessor(HORIZON).fit(dev[cols])
    scaler = pre.named_transformers_["numeric"].named_steps["scale"]
    imputed = pre.named_transformers_["numeric"].named_steps["impute"].transform(dev[pre.transformers_[0][2]])
    assert np.allclose(scaler.mean_, imputed.mean(axis=0))


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_pipelines_learn_signal(sets, name):
    dev, test = sets
    cols = all_feature_columns(HORIZON)
    params = {"model__n_estimators": 50} if name == "random_forest" else None
    pipe = make_pipeline(name, HORIZON, params=params).fit(dev[cols], dev[config.TARGET])
    prob = pipe.predict_proba(test[cols])[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()
    assert classification_metrics(test[config.TARGET], prob)["roc_auc"] > 0.8


def test_tune_runs_with_edition_cv(sets):
    dev, _ = sets
    cols = all_feature_columns(HORIZON)
    search = tune("logistic_regression", dev[cols], dev[config.TARGET], dev["CatEdition"],
                  HORIZON, n_iter=2, verbose=0)
    assert 0 <= search.best_score_ <= 1
    assert len(search.cv_results_["params"]) == 2


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        make_pipeline("svm", HORIZON)


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p, 0.5)
    assert m["tp"] == 1 and m["fn"] == 1 and m["tn"] == 2
    assert 0 < best_f1_threshold(y, p) <= 0.8
    with pytest.raises(ValueError):
        classification_metrics([1, 1], [0.2, 0.9])
