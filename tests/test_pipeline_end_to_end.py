"""Training, persistence, inference, monitoring and app tests on synthetic artefacts."""
import json

import numpy as np
import pandas as pd
import pytest

from product_discontinuation import config, monitoring
from product_discontinuation.features import build_features, edition_history
from product_discontinuation.persistence import load_bundle, load_metadata, model_paths, save_bundle
from product_discontinuation.predict import explain_prediction, predict_weekly, risk_category
from product_discontinuation.train import OutputDirs, horizon_study, save_app_data, train_primary

HORIZON = config.PRIMARY_HORIZON


@pytest.fixture(scope="module")
def trained(weekly, tmp_path_factory):
    root = tmp_path_factory.mktemp("artefacts")
    dirs = OutputDirs.from_root(root)
    dirs.create()
    history = edition_history(weekly)
    searches, selected, threshold, test_table = train_primary(
        weekly, history, HORIZON, ["logistic_regression", "hist_gradient_boosting"], 2, dirs)
    save_app_data(weekly, history, dirs)
    model_path, meta_path = model_paths(HORIZON, dirs.models, dirs.metadata)
    return dict(root=root, dirs=dirs, searches=searches, selected=selected, threshold=threshold,
                test_table=test_table, bundle=load_bundle(model_path), metadata=load_metadata(meta_path),
                history=history)


def test_training_outputs_created(trained):
    dirs = trained["dirs"]
    for path in [dirs.results / f"test_metrics_h{HORIZON}.csv", dirs.experiments / f"cv_results_h{HORIZON}.csv",
                 dirs.figures / f"roc_pr_curves_h{HORIZON}.png", dirs.figures / f"confusion_matrix_h{HORIZON}.png",
                 dirs.figures / f"calibration_h{HORIZON}.png", dirs.processed / "app_weekly.parquet"]:
        assert path.exists(), path
    assert 0 < trained["threshold"] < 1


def test_metadata_contents(trained):
    meta = trained["metadata"]
    assert meta["model_version"] == config.MODEL_VERSION
    assert meta["test_edition"] == config.TEST_EDITION
    assert config.TEST_EDITION not in meta["train_editions"]
    assert "status_ro" not in meta["feature_columns"]
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy"]:
        assert 0 <= meta["test_metrics"][key] <= 1


def test_saved_model_reproduces_predictions(trained, weekly):
    bundle = trained["bundle"]
    test_rows = weekly[weekly["CatEdition"] == config.TEST_EDITION].drop(columns=[config.TARGET])
    features, preds = predict_weekly(bundle, test_rows, history=trained["history"])
    in_memory = trained["searches"][trained["selected"]].best_estimator_
    expected = in_memory.predict_proba(features[bundle["feature_columns"]])[:, 1]
    np.testing.assert_allclose(preds["probability"], expected)
    assert set(preds["risk_category"]) <= {"High", "Medium", "Low"}


def test_inference_features_match_training_columns(trained, weekly):
    bundle = trained["bundle"]
    live = weekly[weekly["CatEdition"].isin(config.CENSORED_EDITIONS)].drop(columns=[config.TARGET])
    features, preds = predict_weekly(bundle, live, history=trained["history"])
    fitted_columns = list(bundle["pipeline"].named_steps["preprocess"].feature_names_in_)
    assert fitted_columns == bundle["feature_columns"]
    assert set(bundle["feature_columns"]) <= set(features.columns)
    assert len(preds) == live.loc[live["WeeksOut"] == -HORIZON].groupby(config.GROUP_KEYS).ngroups


def test_bundle_round_trip_and_validation(trained, tmp_path):
    bundle = {k: v for k, v in trained["bundle"].items() if k not in ("trained_at", "sklearn_version")}
    path, _ = save_bundle(bundle, {"note": "test"}, HORIZON, tmp_path, tmp_path)
    assert load_bundle(path)["threshold"] == bundle["threshold"]
    with pytest.raises(ValueError):
        save_bundle({"pipeline": None}, {}, HORIZON, tmp_path, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


def test_explanation_for_single_prediction(trained, weekly):
    bundle = trained["bundle"]
    features = build_features(weekly, HORIZON)
    effects = explain_prediction(bundle, features.iloc[[0]])
    assert len(effects) == len(bundle["feature_columns"])
    assert effects["effect"].abs().is_monotonic_decreasing
    with pytest.raises(ValueError):
        explain_prediction(bundle, features.iloc[:2])


def test_risk_category_bands():
    assert list(risk_category([0.9, 0.45, 0.1], threshold=0.5)) == ["High", "Medium", "Low"]


def test_horizon_study_runs(trained, weekly):
    table = horizon_study(weekly, trained["history"], [24, 1], trained["searches"], trained["selected"],
                          trained["dirs"])
    assert set(table["horizon"]) == {24, 1}
    assert "status_rule" in set(table["model"])


def test_monitoring_drift_and_logging(trained, weekly, tmp_path):
    bundle = trained["bundle"]
    live = weekly[weekly["CatEdition"].isin(config.CENSORED_EDITIONS)].drop(columns=[config.TARGET])
    monitoring.validate_scoring_input(live)
    features, preds = predict_weekly(bundle, live, history=trained["history"])
    report = monitoring.feature_drift_report(bundle["drift_reference"], features)
    assert set(report["feature"]) == set(bundle["feature_columns"])
    assert report.loc[report.feature == "SpringSummer", "status"].iloc[0] == "significant"

    log_path = tmp_path / "log.csv"
    monitoring.log_predictions(preds, bundle, "test", log_path)
    monitoring.log_predictions(preds, bundle, "test", log_path)
    log = monitoring.load_prediction_log(log_path)
    assert len(log) == 2 * len(preds)
    assert {"logged_at", "model_version"} <= set(log.columns)

    summary = monitoring.prediction_summary(preds)
    assert summary["n_predictions"] == len(preds)


def test_outcome_evaluation_requires_labels(trained, weekly, tmp_path):
    bundle = trained["bundle"]
    test = weekly[weekly["CatEdition"] == config.TEST_EDITION]
    _, preds = predict_weekly(bundle, test.drop(columns=[config.TARGET]), history=trained["history"])
    log_path = tmp_path / "log.csv"
    monitoring.log_predictions(preds, bundle, "backtest", log_path)
    log = monitoring.load_prediction_log(log_path)

    no_labels = pd.DataFrame(columns=config.GROUP_KEYS + [config.TARGET])
    assert monitoring.evaluate_against_outcomes(log, no_labels, bundle["threshold"])["metrics"] is None

    outcomes = test[config.GROUP_KEYS + [config.TARGET]].drop_duplicates()
    result = monitoring.evaluate_against_outcomes(log, outcomes, bundle["threshold"])
    assert result["label_coverage"] == 1.0
    assert result["metrics"]["roc_auc"] == pytest.approx(trained["metadata"]["test_metrics"]["roc_auc"])


def test_psi_properties():
    assert monitoring.population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert monitoring.population_stability_index([0.9, 0.1], [0.1, 0.9]) > monitoring.PSI_SIGNIFICANT
    rng = np.random.default_rng(0)
    ref = monitoring.build_drift_reference(pd.DataFrame({"x": rng.normal(size=2000)}), ["x"])
    same = monitoring.feature_drift_report(ref, pd.DataFrame({"x": rng.normal(size=2000)}))
    shifted = monitoring.feature_drift_report(ref, pd.DataFrame({"x": rng.normal(2, 1, size=2000)}))
    assert same["status"].iloc[0] == "stable" and shifted["status"].iloc[0] == "significant"


def test_monitor_cli(trained):
    from product_discontinuation.monitor import run

    summary, drift = run(trained["dirs"], HORIZON)
    assert summary["backtest_outcome_monitoring"]["metrics"] is not None
    assert json.loads((trained["dirs"].results / "monitoring_summary.json").read_text())


def test_streamlit_app_runs(trained, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PD_ARTIFACT_ROOT", str(trained["root"]))
    app = AppTest.from_file(str(config.PROJECT_ROOT / "deployment" / "app.py"), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Discontinuation probability" for m in app.metric)
    app.selectbox[0].select_index(1).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Actual outcome" in i.value for i in app.info)
