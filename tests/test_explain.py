"""SHAP explanations and plain-English wording (synthetic data + real artefacts if present)."""
import re
import time

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from product_discontinuation import config, narrative
from product_discontinuation.explain import (LocalExplainer, background_paths, build_background, load_background,
                                             output_feature_groups, positive_class, tree_path_dependent_values,
                                             waterfall_figure)
from product_discontinuation.features import all_feature_columns, build_features, edition_history
from product_discontinuation.modelling import make_pipeline
from product_discontinuation.persistence import load_bundle, model_paths
from product_discontinuation.predict import predict_weekly, reference_values

HORIZON = config.PRIMARY_HORIZON
TECHNICAL_TERMS = re.compile(r"shap|log[-_ ]?odds|\blog\b|feature|attribution|coefficient|_", re.I)


@pytest.fixture(scope="module")
def setup(weekly):
    history = edition_history(weekly)
    features = build_features(weekly, HORIZON, history=history)
    columns = all_feature_columns(HORIZON)
    dev = features[(features.CatEdition >= config.MODEL_FIRST_EDITION) & (features.CatEdition < config.TEST_EDITION)]
    pipe = make_pipeline("hist_gradient_boosting", HORIZON).fit(dev[columns], dev[config.TARGET])
    bundle = {"pipeline": pipe, "feature_columns": columns, "threshold": 0.5, "horizon": HORIZON,
              "reference_values": reference_values(dev[columns], columns)}
    test = features[features.CatEdition == config.TEST_EDITION].reset_index(drop=True)
    background = build_background(dev, columns, size=40)
    explainer = LocalExplainer(bundle, background, max_evals=300)
    return dict(bundle=bundle, explainer=explainer, test=test, history=history, columns=columns, dev=dev,
                background=background)


def test_contributions_reproduce_positive_class_prediction(setup):
    explainer, test, columns = setup["explainer"], setup["test"], setup["columns"]
    for i in range(0, len(test), max(len(test) // 25, 1)):
        exp = explainer.explain(test.iloc[[i]])
        p1 = setup["bundle"]["pipeline"].predict_proba(test.iloc[[i]][columns])[0, 1]
        assert exp.probability == pytest.approx(p1)
        assert exp.log_odds == pytest.approx(logit(p1), abs=1e-8)
        assert sorted(exp.contributions["feature"]) == sorted(columns)


def test_contribution_direction_matches_signal(setup):
    """Synthetic discontinued products have low sales: low sales must raise the risk."""
    explainer, test = setup["explainer"], setup["test"]
    order = test["log_actuals"].sort_values().index
    lows, highs = order[:15], order[-15:]
    effect = lambda idx: np.mean([explainer.explain(test.loc[[i]]).contributions.set_index("feature")
                                  .loc["log_actuals", "contribution"] for i in idx])
    assert effect(lows) > 0 > effect(highs)


def test_explanation_handles_missing_and_unseen_values(setup):
    row = setup["test"].iloc[[0]].copy()
    row[["actuals_trend", "previous_discontinued", "log_price"]] = np.nan
    row["Supplier"] = "unseen-supplier"
    exp = setup["explainer"].explain(row)
    assert np.isfinite(exp.contributions["contribution"]).all()
    p1 = setup["bundle"]["pipeline"].predict_proba(row[setup["columns"]])[0, 1]
    assert exp.log_odds == pytest.approx(logit(p1), abs=1e-8)


def test_explanation_rejects_invalid_input(setup):
    with pytest.raises(ValueError, match="exactly one"):
        setup["explainer"].explain(setup["test"].iloc[:2])
    with pytest.raises(ValueError, match="missing model columns"):
        setup["explainer"].explain(setup["test"].iloc[[0]].drop(columns=["log_actuals"]))


def test_explanation_ignores_future_rows(setup, weekly):
    """Changing rows after the prediction week must not change the explanation."""
    test_weekly = weekly[weekly.CatEdition == config.TEST_EDITION].drop(columns=[config.TARGET])
    altered = test_weekly.copy()
    future = altered["WeeksOut"] > -HORIZON
    altered.loc[future, ["ActualsPerWeek", "ForecastPerWeek"]] *= 50
    altered.loc[future, "SalePriceIncVAT"] = 0
    key = test_weekly["ProductKey"].iloc[0]
    explain = lambda w: setup["explainer"].explain(
        (f := predict_weekly(setup["bundle"], w, history=setup["history"])[0])[f.ProductKey == key])
    pd.testing.assert_frame_equal(explain(test_weekly).contributions, explain(altered).contributions)


def test_explainer_initialises_with_development_background(setup):
    explainer = setup["explainer"]
    assert explainer.columns == setup["columns"]
    assert len(explainer.background) == 40
    expected = logit(setup["bundle"]["pipeline"].predict_proba(setup["background"][setup["columns"]])[:, 1]).mean()
    assert explainer.base_value == pytest.approx(expected)


def test_background_is_reproducible_stratified_and_development_only(setup):
    dev, columns = setup["dev"], setup["columns"]
    first, second = build_background(dev, columns, size=40), build_background(dev, columns, size=40)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 40 and config.TEST_EDITION not in set(first["CatEdition"])
    assert abs(first[config.TARGET].mean() - dev[config.TARGET].mean()) < 0.05
    assert not first.equals(build_background(dev, columns, size=40, seed=1))


def test_held_out_background_is_rejected(setup):
    leaked = setup["test"]
    with pytest.raises(ValueError, match="held-out"):
        build_background(leaked, setup["columns"], size=10)
    with pytest.raises(ValueError, match="held-out"):
        LocalExplainer(setup["bundle"], leaked)


def test_output_shape_and_feature_names(setup):
    table = setup["explainer"].explain(setup["test"].iloc[[0]]).contributions
    assert list(table.columns) == ["feature", "value", "contribution"]
    assert table.shape[0] == len(setup["columns"]) and set(table["feature"]) == set(setup["columns"])
    assert table["contribution"].abs().is_monotonic_decreasing


def test_masker_only_ignores_exactly_equal_values(setup):
    data = setup["explainer"].masker.data
    near = data[0].copy()
    near[np.flatnonzero(near)] *= 1 + 1e-7
    assert not setup["explainer"].masker.invariants(near)[0][np.flatnonzero(data[0])].any()
    assert setup["explainer"].masker.invariants(data[0])[0][~np.isnan(data[0])].all()


def test_explanation_is_deterministic(setup):
    row = setup["test"].iloc[[1]]
    state = np.random.get_state()[1].copy()
    first, second = setup["explainer"].explain(row), setup["explainer"].explain(row)
    pd.testing.assert_frame_equal(first.contributions, second.contributions)
    np.testing.assert_array_equal(state, np.random.get_state()[1])


@pytest.mark.parametrize("bad", ["not-a-number", np.inf])
def test_invalid_numeric_values_rejected(setup, bad):
    row = setup["test"].iloc[[0]].copy()
    row["log_actuals"] = row["log_actuals"].astype(object)
    row.iloc[0, row.columns.get_loc("log_actuals")] = bad
    with pytest.raises((ValueError, TypeError)):
        setup["explainer"].explain(row)


def test_missing_categorical_value_is_explained(setup):
    row = setup["test"].iloc[[0]].copy()
    row["Supplier"] = np.nan
    exp = setup["explainer"].explain(row)
    p1 = setup["bundle"]["pipeline"].predict_proba(row[setup["columns"]])[0, 1]
    assert exp.log_odds == pytest.approx(logit(p1), abs=1e-8)


def test_prediction_not_altered_by_explaining(setup):
    columns, row = setup["columns"], setup["test"].iloc[[2]]
    before = setup["bundle"]["pipeline"].predict_proba(setup["test"][columns])
    setup["explainer"].explain(row)
    np.testing.assert_array_equal(before, setup["bundle"]["pipeline"].predict_proba(setup["test"][columns]))


def test_agrees_with_tree_shap_on_strongest_input(setup):
    rows = setup["test"].iloc[:10]
    tree, _ = tree_path_dependent_values(setup["bundle"], rows)
    permutation = pd.DataFrame([setup["explainer"].explain(rows.iloc[[i]]).contributions.set_index("feature")
                               ["contribution"] for i in range(len(rows))])[setup["columns"]]
    assert np.corrcoef(tree["log_actuals"], permutation["log_actuals"])[0, 1] > 0.7


def test_waterfall_figure_shows_explanation(setup):
    figure = waterfall_figure(setup["explainer"].explain(setup["test"].iloc[[0]]))
    assert "Log-odds" in figure.axes[0].get_xlabel()
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_positive_class_layouts():
    two_d = np.array([[0.1, -0.2]])
    assert positive_class(two_d, -1.0)[1] == -1.0
    values, base = positive_class([-two_d, two_d], [1.0, -1.0])
    np.testing.assert_array_equal(values, two_d)
    assert base == -1.0
    values, base = positive_class(np.stack([-two_d, two_d], axis=-1), np.array([0.7, 0.3]))
    np.testing.assert_array_equal(values, two_d)
    assert base == 0.3


def test_feature_groups_cover_every_input(setup):
    pre = setup["bundle"]["pipeline"][:-1]
    groups = output_feature_groups(pre, setup["columns"])
    assert set(groups) == set(setup["columns"])
    assert len(groups) == len(pre.get_feature_names_out())


def test_negligible_factors_not_described():
    table = pd.DataFrame({"feature": ["actuals_trend", "log_price", "Seasonal"],
                          "value": [-0.3, 3.0, 1], "contribution": [0.6, -0.01, 0.04]})
    raising, lowering = narrative.key_factors(table, {"log_price": 3.4})
    assert [f["feature"] for f in raising] == ["actuals_trend"]
    assert lowering == []
    assert "fallen by about 26%" in raising[0]["sentence"] and "raised" in raising[0]["sentence"]


@pytest.mark.parametrize("feature,value,expected", [
    ("actuals_trend", -0.3, "fallen"), ("actuals_trend", 0.01, "stable"), ("actuals_trend", np.nan, "not enough"),
    ("forecast_to_actual", 0.7, "below the forecast"), ("forecast_to_actual", -0.7, "above the forecast"),
    ("forecast_to_actual", 0.02, "close to"), ("previous_discontinued", np.nan, "no previous"),
    ("previous_discontinued", 1.0, "marked for discontinuation"), ("zero_actuals", 1, "No sales"),
    ("log_price", np.nan, "No valid selling price"), ("in_previous_edition", 0.0, "new to the range"),
])
def test_value_descriptions(feature, value, expected):
    assert expected.lower() in narrative.describe(feature, value, 0.1).lower()


def test_wording_direction_and_language(setup):
    columns, ref = setup["columns"], setup["bundle"]["reference_values"]
    row = setup["test"].iloc[0]
    for feature in columns:
        for sign in (1, -1):
            text = narrative.factor_sentence(feature, row[feature], sign * 0.5, ref.get(feature))
            assert ("raised" in text) == (sign > 0) and ("lowered" in text) == (sign < 0)
            assert not TECHNICAL_TERMS.search(text), text
            assert "cause" not in text.lower()
        assert not TECHNICAL_TERMS.search(narrative.label(feature))


def test_summary_is_product_specific(setup):
    texts = set()
    bundle, test = setup["bundle"], setup["test"]
    probs = bundle["pipeline"].predict_proba(test[setup["columns"]])[:, 1]
    for i in np.argsort(probs)[[0, len(probs) // 2, -1]]:
        exp = setup["explainer"].explain(test.iloc[[i]])
        risk = "High" if exp.probability >= 0.5 else "Medium"
        raising, lowering = narrative.key_factors(exp.contributions, bundle["reference_values"])
        text = narrative.summary(exp.probability, risk, 0.5, raising, lowering)
        assert not TECHNICAL_TERMS.search(text), text
        texts.add(text)
    assert len(texts) == 3


def test_recommendations():
    raising = [{"feature": "forecast_to_actual", "contribution": 0.8}, {"feature": "actuals_trend", "contribution": 0.3}]
    high = narrative.recommendations("High", raising)
    assert len(high) == 3 and "catalogue planning team" in high[-1]
    low = narrative.recommendations("Medium", raising)
    assert low[0].startswith("No immediate action") and len(low) == 2
    assert narrative.recommendations("Medium", [{"feature": "Seasonal", "contribution": 0.06}]) == \
        ["No immediate action is suggested; continue routine range monitoring."]


def test_explanation_is_fast(setup):
    row = setup["test"].iloc[[0]]
    start = time.perf_counter()
    for _ in range(3):
        setup["explainer"].explain(row)
    assert (time.perf_counter() - start) / 3 < 5


real_model = pytest.mark.skipif(not model_paths(HORIZON)[0].exists()
                                or not background_paths(HORIZON)[0].exists()
                                or not (config.PROCESSED_DIR / "app_weekly.parquet").exists(),
                                reason="trained project artefacts not available")


@pytest.fixture(scope="module")
def real():
    bundle = load_bundle(model_paths(HORIZON)[0])
    weekly = pd.read_parquet(config.PROCESSED_DIR / "app_weekly.parquet")
    history = pd.read_parquet(config.PROCESSED_DIR / "edition_history.parquet")
    test = weekly[weekly.CatEdition == config.TEST_EDITION].drop(columns=[config.TARGET])
    features, preds = predict_weekly(bundle, test, history=history)
    return dict(bundle=bundle, features=features, preds=preds)


@real_model
def test_real_predictions_unchanged(real):
    """App scoring reproduces the test-edition probabilities saved at training time."""
    np.testing.assert_allclose(real["preds"]["probability"].to_numpy(),
                               np.asarray(real["bundle"]["reference_probabilities"]), rtol=0, atol=1e-12)


@real_model
def test_real_model_explanations_exact(real):
    explainer = LocalExplainer(real["bundle"], load_background(HORIZON))
    features = real["features"]
    for i in np.random.default_rng(0).choice(len(features), 10, replace=False):
        exp = explainer.explain(features.iloc[[i]])
        assert exp.probability == pytest.approx(real["preds"]["probability"].iloc[i], abs=1e-12)
        assert exp.log_odds == pytest.approx(logit(exp.probability), abs=1e-8)


@real_model
def test_real_app_renders_explanation_quickly():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.PROJECT_ROOT / "deployment" / "app.py"), default_timeout=180).run()
    assert not app.exception, [e.value for e in app.exception]
    start = time.perf_counter()
    app.button[0].click().run()
    first_check = time.perf_counter() - start
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Discontinuation probability" for m in app.metric)
    assert any(s.value == "Why was this product flagged?" for s in app.subheader)
    assert any("Suggested next steps" == s.value for s in app.subheader)
    assert any("technical SHAP view" in e.label for e in app.expander)
    assert any("How to read this chart" in m.value for m in app.markdown)
    factors = [m.value for m in app.markdown if 'class="factor"' in m.value]
    assert factors and not any(re.search(r"shap|log-odds", text, re.I) for text in factors)
    app.selectbox[1].select_index(5).run()
    start = time.perf_counter()
    app.button[0].click().run()
    assert not app.exception
    assert first_check < 10 and time.perf_counter() - start < 10
