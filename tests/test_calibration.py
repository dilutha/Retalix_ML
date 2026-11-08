"""Calibration, risk bands, input checks, what-if scenarios and the prediction-selection fix."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from product_discontinuation import config
from product_discontinuation.calibration import (LOW_BAND_MISS_BUDGET, apply_calibration, band_table, calibrated_log_odds,
                                                 fit_platt, low_band_floor, rolling_origin_comparison)
from product_discontinuation.explain import LocalExplainer, background_paths, load_background
from product_discontinuation.features import build_features
from product_discontinuation.monitoring import assess_inputs
from product_discontinuation.persistence import load_bundle, load_metadata, model_paths
from product_discontinuation.predict import apply_scenario, positive_probability, predict_weekly, risk_category

HORIZON = config.PRIMARY_HORIZON


def under_confident_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(-0.5, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-latent))).astype(int)
    return y, 1 / (1 + np.exp(-(latent - 1.0))), rng.choice([90, 91, 92], n)


def test_platt_corrects_bias_and_preserves_ranking():
    y, raw, _ = under_confident_sample()
    calibrator = fit_platt(raw, y)
    calibrated = apply_calibration(raw, calibrator)
    assert raw.mean() < y.mean() - 0.1
    assert calibrated.mean() == pytest.approx(y.mean(), abs=0.01)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, raw), abs=1e-12)
    assert calibrator["a"] > 0


def test_platt_is_linear_in_log_odds_so_shap_rescales_exactly():
    calibrator = {"method": "platt", "a": 0.9, "b": 1.0}
    raw = np.array([0.02, 0.3, 0.8])
    np.testing.assert_allclose(logit(apply_calibration(raw, calibrator)), calibrated_log_odds(logit(raw), calibrator))


def test_calibration_none_is_identity_and_negative_slope_rejected():
    np.testing.assert_array_equal(apply_calibration([0.2, 0.7], None), [0.2, 0.7])
    with pytest.raises(ValueError, match="slope"):
        fit_platt(np.array([0.1, 0.2, 0.8, 0.9] * 20), np.array([1, 1, 0, 0] * 20))


def test_rolling_origin_only_uses_earlier_editions():
    y, raw, editions = under_confident_sample()
    table = rolling_origin_comparison(raw, y, editions)
    assert set(table["evaluated_edition"]) == {91, 92}
    assert set(table["method"]) == {"uncalibrated", "platt", "isotonic"}
    pivot = table.pivot(index="evaluated_edition", columns="method", values="brier")
    assert (pivot["platt"] < pivot["uncalibrated"]).all()


def test_band_floor_respects_miss_budget_and_threshold():
    y, raw, _ = under_confident_sample()
    calibrated = apply_calibration(raw, fit_platt(raw, y))
    floor = low_band_floor(y, calibrated, threshold=0.9)
    assert np.mean(calibrated[y == 1] < floor) == pytest.approx(LOW_BAND_MISS_BUDGET, abs=0.002)
    assert low_band_floor(y, calibrated, threshold=0.01) == 0.01


def test_all_three_bands_reachable_when_floor_below_threshold():
    assert set(risk_category([0.9, 0.2, 0.05], threshold=0.28, medium_floor=0.14)) == {"High", "Medium", "Low"}
    table = band_table([1, 0, 1, 0], [0.9, 0.2, 0.05, 0.03], 0.28, 0.14, "demo")
    assert table["share_of_products"].sum() == pytest.approx(1)
    assert table["share_of_discontinued"].sum() == pytest.approx(1)


def test_positive_class_column_is_found_by_label():
    class Reversed:
        classes_ = np.array([1, 0])

        @staticmethod
        def predict_proba(frame):
            return np.column_stack([np.full(len(frame), 0.8), np.full(len(frame), 0.2)])

    frame = pd.DataFrame({"x": [1.0, 2.0]})
    bundle = {"pipeline": Reversed(), "feature_columns": ["x"]}
    np.testing.assert_allclose(positive_probability(bundle, frame), [0.8, 0.8])
    Reversed.classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match="positive class"):
        positive_probability(bundle, frame)


@pytest.fixture(scope="module")
def product(weekly):
    edition = weekly[weekly["CatEdition"] == config.TEST_EDITION]
    key = edition[edition["WeeksOut"] == -HORIZON]["ProductKey"].iloc[0]
    rows = edition[edition["ProductKey"] == key]
    history = weekly[["CatEdition", "ProductKey", config.TARGET]].drop_duplicates(config.GROUP_KEYS)
    history[config.TARGET] = history[config.TARGET].astype(float)
    return rows, history


def test_scenario_without_changes_matches_training_features(product):
    rows, history = product
    same, same_history = apply_scenario(rows, HORIZON, history)
    pd.testing.assert_frame_equal(build_features(same, HORIZON, history=same_history).drop(columns=[config.TARGET], errors="ignore"),
                                  build_features(rows, HORIZON, history=history).drop(columns=[config.TARGET]),
                                  check_like=True)


def test_scenario_changes_flow_through_feature_engineering(product):
    rows, history = product
    changed, _ = apply_scenario(rows, HORIZON, history, actuals=200, actuals_4_weeks_earlier=400, forecast=500, price=10)
    row = build_features(changed, HORIZON, history=history).iloc[0]
    assert row["log_actuals"] == pytest.approx(np.log1p(200))
    assert row["actuals_trend"] == pytest.approx(np.log1p(200) - np.log1p(400))
    assert row["forecast_to_actual"] == pytest.approx(np.log1p(500) - np.log1p(200))
    assert row["log_price"] == pytest.approx(np.log(10))


def test_scenario_limited_history(product):
    rows, history = product
    changed, reduced = apply_scenario(rows, HORIZON, history, weeks_listed=2, new_to_range=True)
    row = build_features(changed, HORIZON, history=reduced).iloc[0]
    assert row["weeks_in_edition"] == 2 and np.isnan(row["actuals_trend"])
    assert row["in_previous_edition"] == 0 and row["prior_editions"] == 0


@pytest.mark.parametrize("bad", [dict(actuals=-1), dict(forecast=np.inf), dict(price=np.nan), dict(weeks_listed=0),
                                 dict(weeks_listed=99)])
def test_scenario_rejects_invalid_values(product, bad):
    rows, history = product
    with pytest.raises(ValueError):
        apply_scenario(rows, HORIZON, history, **bad)


real_model = pytest.mark.skipif(not model_paths(HORIZON)[0].exists() or not background_paths(HORIZON)[0].exists()
                                or not (config.PROCESSED_DIR / "app_weekly.parquet").exists(),
                                reason="trained project artefacts not available")


@pytest.fixture(scope="module")
def real():
    bundle = load_bundle(model_paths(HORIZON)[0])
    weekly = pd.read_parquet(config.PROCESSED_DIR / "app_weekly.parquet")
    history = pd.read_parquet(config.PROCESSED_DIR / "edition_history.parquet")
    test = weekly[weekly.CatEdition == config.TEST_EDITION]
    features, preds = predict_weekly(bundle, test.drop(columns=[config.TARGET]), history=history)
    y = test[test.WeeksOut == -HORIZON].set_index("ProductKey")[config.TARGET].reindex(features["ProductKey"]).astype(int).to_numpy()
    return dict(bundle=bundle, weekly=weekly, history=history, features=features, preds=preds, y=y,
                metadata=load_metadata(model_paths(HORIZON)[1]))


@real_model
def test_real_probabilities_valid_and_positive_class(real):
    assert list(real["bundle"]["pipeline"].classes_) == [0, 1]
    assert real["preds"]["probability"].between(0, 1).all()


@real_model
def test_real_calibration_preserves_flagged_products_and_auc(real):
    bundle, preds = real["bundle"], real["preds"]
    raw = bundle["pipeline"].predict_proba(real["features"][bundle["feature_columns"]])[:, 1]
    np.testing.assert_array_equal(raw >= bundle["raw_threshold"], preds["predicted_discontinued"].astype(bool))
    assert roc_auc_score(real["y"], preds["probability"]) == pytest.approx(roc_auc_score(real["y"], raw), abs=1e-12)
    assert real["metadata"]["test_metrics"]["roc_auc"] == pytest.approx(roc_auc_score(real["y"], raw))


@real_model
def test_real_bands_are_reachable_ordered_and_match_thresholds(real):
    bundle, preds = real["bundle"], real["preds"]
    assert 0 < bundle["risk_medium_floor"] < bundle["threshold"] < 1
    assert set(preds["risk_category"]) == {"High", "Medium", "Low"}
    np.testing.assert_array_equal(risk_category(preds["probability"], bundle["threshold"], bundle["risk_medium_floor"]),
                                  preds["risk_category"])
    observed = pd.Series(real["y"]).groupby(preds["risk_category"].to_numpy()).mean()
    assert observed["Low"] < observed["Medium"] < observed["High"]


@real_model
def test_real_calibrated_explanation_matches_shown_probability(real):
    explainer = LocalExplainer(real["bundle"], load_background(HORIZON))
    features, preds = real["features"], real["preds"]
    for i in np.random.default_rng(1).choice(len(features), 15, replace=False):
        exp = explainer.explain(features.iloc[[i]])
        assert exp.probability == pytest.approx(preds["probability"].iloc[i], abs=1e-12)
        assert exp.base_value + exp.contributions["contribution"].sum() == pytest.approx(logit(preds["probability"].iloc[i]), abs=1e-8)


def _median_product(real):
    """A returning product with full history, near the median score of its edition."""
    features, preds = real["features"], real["preds"]
    full = (features["weeks_in_edition"] == 12) & (features["in_previous_edition"] == 1)
    gap = (preds["probability"] - preds.loc[full, "probability"].median()).abs()[full]
    key = int(preds.loc[gap.idxmin(), "ProductKey"])
    weekly = real["weekly"]
    return weekly[(weekly.CatEdition == config.TEST_EDITION) & (weekly.ProductKey == key)], key


@real_model
@pytest.mark.parametrize("name,changes", [
    ("strong sales", dict(actuals=5000, actuals_4_weeks_earlier=4500, forecast=4500, forecast_4_weeks_earlier=4500)),
    ("declining sales", dict(actuals=20, actuals_4_weeks_earlier=80)),
    ("sales far below forecast", dict(actuals=5, forecast=120)),
    ("strong sales, recently listed", dict(actuals=5000, forecast=4500, weeks_listed=4)),
    ("limited history", dict(weeks_listed=1, new_to_range=True)),
    ("extreme values", dict(actuals=9_000_000, forecast=9_000_000, price=250_000)),
])
def test_real_scenarios_score_consistently(real, name, changes):
    bundle = real["bundle"]
    rows, _ = _median_product(real)
    scenario, history = apply_scenario(rows, HORIZON, real["history"], **changes)
    features, preds = predict_weekly(bundle, scenario, history=history)
    result = preds.iloc[0]
    assert 0 <= result["probability"] <= 1
    assert result["risk_category"] == risk_category([result["probability"]], bundle["threshold"], bundle["risk_medium_floor"])[0]
    messages = assess_inputs(bundle, features)
    if name == "extreme values":
        assert any("higher than almost all" in m for m in messages)
    if name == "limited history":
        assert any("not enough weekly history" in m for m in messages)


@real_model
def test_real_unedited_scenario_reproduces_actual_prediction(real):
    """What the what-if form submits when the user changes nothing, including products with week gaps."""
    bundle, weekly, preds = real["bundle"], real["weekly"], real["preds"].set_index("ProductKey")
    test = weekly[weekly.CatEdition == config.TEST_EDITION]
    span = test[test.WeeksOut <= -HORIZON].groupby("ProductKey")["WeeksOut"].agg(["min", "size"])
    gaps = span.index[span["size"] != -HORIZON - span["min"] + 1].intersection(preds.index)[:25]
    rng = np.random.default_rng(3)
    for key in list(gaps) + list(rng.choice(preds.index.to_numpy(), 75, replace=False)):
        rows = test[test.ProductKey == key]
        now, earlier = rows[rows.WeeksOut == -HORIZON].iloc[0], rows[rows.WeeksOut == -HORIZON - 4]
        scenario, history = apply_scenario(
            rows, HORIZON, real["history"], actuals=round(now.ActualsPerWeek, 2), forecast=round(now.ForecastPerWeek, 2),
            price=now.SalePriceIncVAT, weeks_listed=int(-HORIZON - rows.WeeksOut.min() + 1),
            actuals_4_weeks_earlier=round(earlier.ActualsPerWeek.iloc[0], 2) if len(earlier) else None,
            forecast_4_weeks_earlier=round(earlier.ForecastPerWeek.iloc[0], 2) if len(earlier) else None)
        _, scored = predict_weekly(bundle, scenario, history=history)
        assert scored["probability"].iloc[0] == pytest.approx(preds.loc[key, "probability"], abs=1e-12)


@real_model
def test_real_app_does_not_open_on_highest_risk_product_and_runs_what_if():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "deployment" / "app.py"), default_timeout=180).run()
    assert not app.exception, [e.value for e in app.exception]
    keys = [int(o.split()[1]) for o in app.selectbox[1].options]
    assert keys == sorted(keys)
    assert any("High ·" in c.value and "Low" in c.value for c in app.caption)

    app.radio[0].set_value("Low").run()
    assert not app.exception
    assert all("Low risk" in o for o in app.selectbox[1].options)

    app.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Discontinuation probability" for m in app.metric)
    assert any("calibrated estimate" in c.value for c in app.caption)
    submit = next(b for b in app.button if b.label == "Run scenario")
    next(n for n in app.number_input if n.label == "Average weekly sales").set_value(9_000_000.0)
    submit.click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Scenario probability" for m in app.metric)
    assert any("higher than almost all" in w.value for w in app.warning)

    next(n for n in app.number_input if n.label == "Average weekly sales").set_value(-5.0)
    next(b for b in app.button if b.label == "Run scenario").click().run()
    assert not app.exception
    assert any("Average weekly sales must be" in e.value for e in app.error)
