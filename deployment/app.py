"""Retalix Product Health Check - decision support for catalogue range refresh planning.

Run from the project root:  streamlit run deployment/app.py
"""
import os
import sys
from io import BytesIO
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from product_discontinuation import config, narrative  # noqa: E402
from product_discontinuation.explain import LocalExplainer, load_background, waterfall_figure  # noqa: E402
from product_discontinuation.monitoring import (assess_inputs, feature_drift_report, input_notes,  # noqa: E402
                                                log_predictions, prediction_summary)
from product_discontinuation.persistence import load_bundle, load_metadata, model_paths  # noqa: E402
from product_discontinuation.predict import apply_scenario, predict_weekly  # noqa: E402
from product_discontinuation.train import OutputDirs  # noqa: E402

RAISES, LOWERS = "#d03b3b", "#2a78d6"
ACTUAL, FORECAST = "#2a78d6", "#eb6834"
RISK_STYLE = {  # status colour, tint, icon
    "High": ("#d03b3b", "#fdf0ef", "▲"),
    "Medium": ("#b7791f", "#fef7e8", "●"),
    "Low": ("#0a8a0a", "#eef8ee", "▼"),
}
# The cut-off is a review flag at about 28%, so these must not read as "more likely than not".
OUTCOME = {1: "Flagged for discontinuation review", 0: "Not flagged for review"}
MAX_CHART_FACTORS = 8
RISK_FILTERS = ["All", "High", "Medium", "Low"]

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
h1, h2, h3 {letter-spacing: -0.01em;}
.app-sub {color: #52514e; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.2rem;}
.result-card {border-radius: 14px; padding: 1.2rem 1.4rem; border: 1px solid; height: 100%;}
.badge {display: inline-block; padding: 0.2rem 0.8rem; border-radius: 999px; font-weight: 600;
        font-size: 0.95rem; color: #fff;}
.outcome {font-size: 1.45rem; font-weight: 650; margin: 0.6rem 0 0.2rem; color: #0b0b0b;}
.summary {font-size: 1.04rem; line-height: 1.6; color: #1f1f1d;}
.factor {border-left: 4px solid; background: #fafaf9; border-radius: 6px; padding: 0.55rem 0.8rem;
         margin-bottom: 0.5rem; line-height: 1.45;}
.factor b {display: block; font-size: 0.82rem; color: #52514e; font-weight: 600; text-transform: uppercase;
           letter-spacing: 0.03em;}
.muted {color: #6b6a66; font-size: 0.9rem;}
</style>
"""


def artefact_dirs():
    return OutputDirs.from_root(os.environ.get("PD_ARTIFACT_ROOT"))


@st.cache_resource
def load_model(horizon):
    dirs = artefact_dirs()
    model_path, meta_path = model_paths(horizon, dirs.models, dirs.metadata)
    return load_bundle(model_path), load_metadata(meta_path)


@st.cache_resource
def load_explainer(_bundle, model_version, trained_at):
    dirs = artefact_dirs()
    return LocalExplainer(_bundle, load_background(_bundle["horizon"], dirs.models, dirs.metadata))


@st.cache_data
def load_app_data():
    dirs = artefact_dirs()
    weekly = pd.read_parquet(dirs.processed / "app_weekly.parquet")
    history = pd.read_parquet(dirs.processed / "edition_history.parquet")
    return weekly, history


@st.cache_data
def score_edition(_bundle, model_version, edition, _weekly, _history):
    rows = _weekly[_weekly["CatEdition"] == edition].drop(columns=[config.TARGET], errors="ignore")
    return predict_weekly(_bundle, rows, history=_history)


@st.cache_data(max_entries=256)
def explain_product(_explainer, _features, trained_at, edition, product):
    return _explainer.explain(_features[_features["ProductKey"] == product])


@st.cache_data(max_entries=256)
def waterfall_png(_explanation, trained_at, edition, product):
    figure = waterfall_figure(_explanation)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def edition_options():
    options = {f"Edition {e} · current catalogue (outcome not yet known)": e for e in config.CENSORED_EDITIONS}
    options[f"Edition {config.TEST_EDITION} · past catalogue (outcome known)"] = config.TEST_EDITION
    return options


def band_evidence(metadata, risk):
    """Observed discontinuation rate of the band on out-of-fold development predictions."""
    row = next(b for b in metadata["risk_bands"] if b["sample"] == "development out-of-fold" and b["band"] == risk)
    return (f"In past catalogues (validation on editions {min(metadata['cv_test_editions'])}-"
            f"{max(metadata['cv_test_editions'])}), {row['observed_rate']:.0%} of products in the {risk} band "
            "were discontinued.")


def show_input_checks(bundle, feature_row):
    messages = assess_inputs(bundle, feature_row)
    if messages:
        st.warning("**Check these inputs before relying on this estimate**\n\n" + "\n".join(f"- {m}" for m in messages))
    for note in input_notes(bundle, feature_row):
        st.caption(f"Note: {note}")


def result_card(probability, risk, predicted, summary_text, rank_share, evidence):
    colour, tint, icon = RISK_STYLE[risk]
    left, right = st.columns([1, 1.9], gap="large")
    with left:
        st.markdown(f'<div class="result-card" style="border-color:{colour}; background:{tint};">'
                    f'<span class="badge" style="background:{colour};">{icon} {risk} risk</span>'
                    f'<div class="outcome">{OUTCOME[predicted]}</div></div>', unsafe_allow_html=True)
        st.metric("Discontinuation probability", f"{probability:.0%}",
                  help="The model's estimate, calibrated against later catalogue editions, that this product is "
                       "discontinued at the next range refresh.")
        st.caption(f"Higher than {rank_share:.0%} of products in this catalogue. {evidence}")
    with right:
        st.markdown("#### What this means")
        st.markdown(f'<div class="summary">{summary_text}</div>', unsafe_allow_html=True)
        st.caption("Treat this as a calibrated estimate for prioritising review, not an exact chance: calibration "
                   "is approximate and can shift between catalogue editions.")


def product_snapshot(past, meta):
    latest = past.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Average weekly sales", f"{latest.ActualsPerWeek:,.0f} units")
    cols[1].metric("Expected weekly sales", f"{latest.ForecastPerWeek:,.0f} units")
    price = latest.SalePriceIncVAT
    cols[2].metric("Selling price", f"£{price:,.2f}" if price > 0 else "Not recorded")
    cols[3].metric("Weeks listed so far", f"{int(past['WeeksOut'].max() - past['WeeksOut'].min() + 1)}")
    st.markdown(f'<p class="muted">Supplier {meta.Supplier} · Product type {meta.HierarchyLevel1} · '
                f'Department {meta.HierarchyLevel2} · {"Imported" if meta.DIorDOM == "DI" else "Domestic"} · '
                f'{"Seasonal" if meta.Seasonal else "Non-seasonal"}</p>', unsafe_allow_html=True)


def sales_chart(past):
    data = past.melt(id_vars="WeeksOut", value_vars=["ActualsPerWeek", "ForecastPerWeek"],
                     var_name="series", value_name="units")
    data["series"] = data["series"].map({"ActualsPerWeek": "Average weekly sales",
                                         "ForecastPerWeek": "Expected weekly sales"})
    data["weeks_before"] = -data["WeeksOut"]
    return alt.Chart(data).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36)).encode(
        x=alt.X("weeks_before:Q", title="Weeks before range refresh", scale=alt.Scale(reverse=True),
                axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("units:Q", title="Units per week", axis=alt.Axis(gridOpacity=0.4)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Average weekly sales", "Expected weekly sales"],
                                                    range=[ACTUAL, FORECAST]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("weeks_before:Q", title="Weeks before refresh"),
                 alt.Tooltip("units:Q", title="Units", format=",.0f")],
    ).properties(height=260)


def factor_chart(contributions):
    data = contributions.loc[contributions["contribution"].abs() >= narrative.NEGLIGIBLE,
                             ["feature", "contribution"]].head(MAX_CHART_FACTORS).copy()
    if data.empty:
        return None
    data["factor"] = data["feature"].map(narrative.label)
    data["effect"] = data["contribution"].map(lambda c: "Raises risk" if c > 0 else "Lowers risk")
    data["order"] = data["contribution"].abs()
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("contribution:Q", title="◀ lowers risk          raises risk ▶",
                axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
        y=alt.Y("factor:N", sort=alt.EncodingSortField("order", order="descending"), title=None,
                axis=alt.Axis(labelLimit=260, domain=False, ticks=False)),
        color=alt.Color("effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES, LOWERS]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("effect:N", title="Effect")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8984").encode(x="x:Q")
    return (bars + zero).properties(height=36 * len(data) + 40)


def factor_list(factors, colour, empty_text):
    if not factors:
        st.markdown(f'<p class="muted">{empty_text}</p>', unsafe_allow_html=True)
    for f in factors:
        st.markdown(f'<div class="factor" style="border-color:{colour};"><b>{f["label"]}</b>{f["sentence"]}</div>',
                    unsafe_allow_html=True)


def contributor_table(rows):
    table = rows[["feature", "value", "contribution"]].assign(value=lambda d: d["value"].astype(str))
    st.dataframe(table.rename(columns={"contribution": "SHAP (log-odds)"}).round(4), hide_index=True)


def technical_contribution_chart(contributions, top_n=12):
    data = contributions[["feature", "contribution"]].head(top_n).copy()
    data["direction"] = data["contribution"].map(lambda c: "Raises log-odds" if c > 0 else "Lowers log-odds")
    return alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("contribution:Q", title="SHAP value (calibrated log-odds of DiscontinuedTF = 1)", axis=alt.Axis(gridOpacity=0.4)),
        y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, ticks=False)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises log-odds", "Lowers log-odds"],
                                                       range=[RAISES, LOWERS]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["feature", alt.Tooltip("contribution:Q", format=".4f")],
    ).properties(height=26 * len(data) + 40)


def shap_details(explainer, explanation, trained_at, edition, product):
    with st.expander("How this explanation was calculated (technical SHAP view)"):
        contributions = explanation.contributions
        st.markdown(
            f"**Method:** {explainer.method}, explaining `DiscontinuedTF = True` in calibrated log-odds over the model's "
            f"{len(explainer.columns)} original inputs. **Background:** {len(explainer.background)} rows sampled from "
            f"development editions only, stratified by edition and outcome (seed {explainer.seed}). "
            f"**Evaluations:** {explainer.max_evals}. Permutation SHAP is a seeded Monte Carlo estimate; "
            f"contributions with |log-odds| < {narrative.NEGLIGIBLE} are not described in the plain-English view.")
        st.image(waterfall_png(explanation, trained_at, edition, product))
        st.markdown(
            f"**How to read this chart:** the plot starts at E[f(X)] = {explanation.base_value:.3f}, the average "
            f"log-odds over the background sample, and each bar adds one input's contribution to reach f(x) = "
            f"{explanation.log_odds:.3f}, which equals the model probability {explanation.probability:.4f}. Red bars "
            "push towards discontinuation and blue bars away from it. Values describe this model's behaviour "
            "relative to the background; they are not causal effects.")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Top positive contributors**")
            contributor_table(contributions[contributions["contribution"] > 0].head(5))
        with right:
            st.markdown("**Top negative contributors**")
            contributor_table(contributions[contributions["contribution"] < 0].head(5))
        st.markdown("**Feature contributions**")
        st.altair_chart(technical_contribution_chart(contributions), use_container_width=True)
        st.markdown(f"Local accuracy check: {explanation.base_value:.4f} + "
                    f"{contributions['contribution'].sum():.4f} = {explanation.log_odds:.4f} log-odds.")


def technical_details(bundle, metadata):
    with st.expander("Model information (technical)"):
        threshold, floor, calibration = bundle["threshold"], bundle["risk_medium_floor"], metadata["calibration"]
        st.markdown(
            f"**Model:** {metadata['model_name'].replace('_', ' ').title()} · **Version:** {bundle['model_version']} · "
            f"**Trained:** {bundle.get('trained_at', 'unknown')}  \n"
            f"**Prediction point:** {bundle['horizon']} weeks before the range refresh, using only weekly data up to "
            f"WeeksOut = -{bundle['horizon']} and outcomes of earlier editions. `Status` is not a model input.  \n"
            f"**Training editions:** {metadata['train_editions']} · **Held-out test edition:** {metadata['test_edition']}")
        st.markdown(
            f"**What the probability means:** the estimated chance that the product is discontinued "
            f"(`DiscontinuedTF = True`) at the next refresh, after Platt calibration fitted on out-of-fold predictions "
            f"from development editions. Risk bands: **High** ≥ {threshold:.3f} (the max-F1 decision threshold), "
            f"**Medium** ≥ {floor:.3f}, otherwise **Low**. {metadata['risk_band_definition']['Low'].capitalize()}.")
        st.markdown(
            f"**Calibration on held-out edition {metadata['test_edition']}:** Brier score "
            f"{calibration['test_raw']['brier']:.4f} → {calibration['test_calibrated']['brier']:.4f}; mean prediction "
            f"{calibration['test_raw']['mean_predicted']:.3f} → {calibration['test_calibrated']['mean_predicted']:.3f} "
            f"against an observed rate of {calibration['test_calibrated']['observed_rate']:.3f}; "
            f"expected calibration error {calibration['test_raw']['ece_10_quantile_bins']:.3f} → "
            f"{calibration['test_calibrated']['ece_10_quantile_bins']:.3f}.")
        bands = pd.DataFrame(metadata["risk_bands"])[["sample", "band", "n", "share_of_products", "mean_predicted",
                                                      "observed_rate", "share_of_discontinued"]]
        st.dataframe(bands.round(3), hide_index=True)
        test = metadata["test_metrics"]
        st.markdown(f"**Held-out test performance (edition {metadata['test_edition']})**")
        cols = st.columns(6)
        for col, (key, name) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision", "Precision"),
                                           ("recall", "Recall"), ("f1", "F1"),
                                           ("balanced_accuracy", "Balanced acc.")]):
            col.metric(name, f"{test[key]:.3f}")


def health_check_tab(bundle, metadata, explainer, weekly, history):
    options = edition_options()
    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    edition = options[c1.selectbox("Catalogue edition", list(options))]
    features, preds = score_edition(bundle, bundle["model_version"], edition, weekly, history)
    risk_filter = c3.radio("Show products", RISK_FILTERS, horizontal=True,
                           help="Filter the product list by risk band.")
    shown = preds if risk_filter == "All" else preds[preds["risk_category"] == risk_filter]
    if shown.empty:
        st.info(f"No products in this catalogue are in the {risk_filter} risk band.")
        return
    ordered = shown.sort_values("ProductKey")
    labels = {int(k): f"Product {k} · {r} risk" for k, r in zip(ordered["ProductKey"], ordered["risk_category"])}
    product = c2.selectbox("Product", list(labels), format_func=lambda k: labels.get(k, k),
                           help="Products are listed by product number. Type to search, or filter by risk band.")
    counts = preds["risk_category"].value_counts(normalize=True)
    st.caption(f"This catalogue: {counts.get('High', 0):.0%} High · {counts.get('Medium', 0):.0%} Medium · "
               f"{counts.get('Low', 0):.0%} Low · average estimate {preds['probability'].mean():.0%} · predictions "
               f"are made {bundle['horizon']} weeks before the refresh.")

    if st.button("Check product health", type="primary"):
        st.session_state["checked"] = (edition, product)
    if st.session_state.get("checked") != (edition, product):
        st.info("Choose a catalogue edition and product, then press **Check product health**.")
        return

    row = preds[preds["ProductKey"] == product].iloc[0]
    probability, risk, predicted = float(row["probability"]), str(row["risk_category"]), int(row["predicted_discontinued"])
    trained_at = bundle.get("trained_at")
    explanation = None
    if explainer:
        with st.spinner("Working out what influenced this estimate..."):
            explanation = explain_product(explainer, features, trained_at, edition, product)
    raising, lowering = narrative.key_factors(explanation.contributions, bundle["reference_values"]) \
        if explanation else ([], [])

    st.divider()
    summary_text = narrative.summary(probability, risk, bundle["threshold"], raising, lowering) if explanation \
        else narrative.outcome_statement(probability, risk, bundle["threshold"])
    rank_share = float((preds["probability"] < probability).mean())
    result_card(probability, risk, predicted, summary_text, rank_share, band_evidence(metadata, risk))
    feature_row = features[features["ProductKey"] == product]
    show_input_checks(bundle, feature_row)
    raw = weekly[(weekly["CatEdition"] == edition) & (weekly["ProductKey"] == product)]
    if edition == config.TEST_EDITION:
        actual = bool(raw[config.TARGET].iloc[0])
        st.info(f"Actual outcome for this past catalogue: **{'Discontinued' if actual else 'Stayed in the range'}**")

    st.divider()
    if explanation is None:
        st.warning("Explanations are not available yet. Run `python -m product_discontinuation.shap_analysis` "
                   "once to prepare them; the prediction above is unaffected.")
    else:
        explanation_section(explanation, raising, lowering, predicted, risk)

    st.divider()
    st.subheader("Product snapshot")
    past = raw[raw["WeeksOut"] <= -bundle["horizon"]].sort_values("WeeksOut")
    product_snapshot(past, past.iloc[-1])
    st.altair_chart(sales_chart(past), use_container_width=True)
    st.caption(f"Only data up to {bundle['horizon']} weeks before the refresh is used for the prediction.")
    what_if_section(bundle, explainer, raw, past.iloc[-1], history, probability, edition, product)
    if explanation is not None:
        shap_details(explainer, explanation, trained_at, edition, product)
    technical_details(bundle, metadata)


def explanation_section(explanation, raising, lowering, predicted, risk):
    st.subheader("Why was this product flagged?" if predicted else "What influenced this result?")
    st.caption("The factors that moved this product's estimate most, compared with an average product. "
               "They show what the model took into account, not proven reasons.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Raising the risk")
        factor_list(raising, RAISES, "No factor noticeably raised the risk.")
    with right:
        st.markdown("##### Lowering the risk")
        factor_list(lowering, LOWERS, "No factor noticeably lowered the risk.")
    chart = factor_chart(explanation.contributions)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Suggested next steps")
    for action in narrative.recommendations(risk, raising):
        st.markdown(f"- {action}")
    st.caption("These are prompts for review, not decisions. Combine them with your own knowledge of the range.")


def what_if_section(bundle, explainer, raw, latest, history, probability, edition, product):
    """Re-score an edited copy of the product through the same feature code as training."""
    horizon = bundle["horizon"]
    earlier = raw[raw["WeeksOut"] == -horizon - config.TREND_LOOKBACK_WEEKS]
    has_earlier = not earlier.empty
    st.divider()
    st.subheader("Try a what-if scenario")
    with st.container():  # not an expander: Streamlit collapses it on submit, which would hide the result
        st.caption(f"Change this product's figures at the prediction point, {horizon} weeks before the refresh, and "
                   "re-run the same model. This shows how the model responds; it does not forecast what would happen.")
        # No min/max on these inputs: Streamlit would keep the last valid value while showing the typed one.
        with st.form(f"what_if_{edition}_{product}"):
            a, b, c = st.columns(3)
            actuals = a.number_input("Average weekly sales", value=float(latest.ActualsPerWeek))
            forecast = b.number_input("Expected weekly sales", value=float(latest.ForecastPerWeek))
            price = c.number_input("Selling price (£)", value=float(latest.SalePriceIncVAT),
                                   help="A price of 0 is treated as not recorded.")
            a2, b2, c2 = st.columns(3)
            earlier_actuals = a2.number_input("Average weekly sales 4 weeks earlier", disabled=not has_earlier,
                                              value=float(earlier["ActualsPerWeek"].iloc[0]) if has_earlier else 0.0)
            earlier_forecast = b2.number_input("Expected weekly sales 4 weeks earlier", disabled=not has_earlier,
                                               value=float(earlier["ForecastPerWeek"].iloc[0]) if has_earlier else 0.0)
            weeks = c2.number_input("Weeks listed so far", step=1, value=int(-horizon - raw["WeeksOut"].min() + 1))
            new_to_range = st.checkbox("Treat as new to the range (no earlier catalogues)")
            submitted = st.form_submit_button("Run scenario")
        if not submitted:
            return
        try:
            rows, scenario_history = apply_scenario(
                raw, horizon, history, actuals=actuals, forecast=forecast, price=price, weeks_listed=weeks,
                actuals_4_weeks_earlier=earlier_actuals if has_earlier else None,
                forecast_4_weeks_earlier=earlier_forecast if has_earlier else None, new_to_range=new_to_range)
            features, scored = predict_weekly(bundle, rows, history=scenario_history)
        except ValueError as err:
            st.error(f"This scenario could not be scored: {err}")
            return
        result = scored.iloc[0]
        left, right = st.columns(2)
        left.metric("Scenario probability", f"{result['probability']:.0%}",
                    delta=f"{(result['probability'] - probability) * 100:+.0f} points vs. actual", delta_color="inverse")
        right.metric("Scenario risk band", str(result["risk_category"]))
        show_input_checks(bundle, features)
        if explainer is not None:
            raising, lowering = narrative.key_factors(explainer.explain(features).contributions, bundle["reference_values"])
            first, second = st.columns(2, gap="large")
            with first:
                st.markdown("##### Raising the risk")
                factor_list(raising, RAISES, "No factor noticeably raised the risk.")
            with second:
                st.markdown("##### Lowering the risk")
                factor_list(lowering, LOWERS, "No factor noticeably lowered the risk.")


def performance_section(metadata):
    st.subheader(f"Selected model: {metadata['model_name'].replace('_', ' ').title()}")
    st.write(f"Trained on editions {metadata['train_editions']}, evaluated once on held-out edition "
             f"{metadata['test_edition']}. Threshold chosen by maximising F1 on out-of-fold development predictions.")
    test = metadata["test_metrics"]
    cols = st.columns(6)
    for col, key in zip(cols, ["roc_auc", "pr_auc", "precision", "recall", "f1", "balanced_accuracy"]):
        col.metric(key.replace("_", " ").upper(), f"{test[key]:.3f}")
    context = metadata["calibration"]["test_edition_context"]
    st.info(f"Edition {metadata['test_edition']} included {context['departments_fully_discontinued']} departments "
            f"({context['share_of_test_rows']:.0%} of products) that were discontinued in full. Excluding them, test "
            f"ROC-AUC is {context['roc_auc_excluding']:.3f} and PR-AUC is {context['pr_auc_excluding']:.3f}. The "
            "cross-validation results below are a better guide to a typical edition.")
    st.write("Cross-validation (expanding window over editions):")
    st.dataframe(pd.DataFrame(metadata["cv_summary"]), hide_index=True)
    figures = artefact_dirs().figures
    horizon = metadata["horizon_weeks"]
    names = [f"roc_pr_curves_h{horizon}.png", f"calibration_before_after_h{horizon}.png",
             f"confusion_matrix_h{horizon}.png", f"feature_importance_h{horizon}.png"]
    existing = [figures / n for n in names if (figures / n).exists()]
    for col, path in zip(st.columns(2) * 2, existing):
        col.image(str(path))


def monitoring_section(bundle, weekly, history):
    st.subheader("Live catalogue monitoring")
    st.write("Live edition outcomes are unknown until the refresh, so only input and prediction "
             "distributions can be monitored now. Accuracy is tracked once labels arrive.")
    edition = config.CENSORED_EDITIONS[0]
    features, preds = score_edition(bundle, bundle["model_version"], edition, weekly, history)
    summary = prediction_summary(preds)
    c1, c2, c3 = st.columns(3)
    c1.metric("Products scored", summary["n_predictions"])
    c2.metric("Predicted discontinued", f"{summary['predicted_positive_rate']:.1%}")
    c3.metric("Mean probability", f"{summary['mean_probability']:.3f}")
    st.bar_chart(preds["risk_category"].value_counts())
    st.markdown("**Input drift vs training data (PSI)**")
    st.dataframe(feature_drift_report(bundle["drift_reference"], features).round(3), hide_index=True)
    st.caption("PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant.")
    if st.button("Log these predictions"):
        path = log_predictions(preds, bundle, source="streamlit",
                               path=artefact_dirs().logs / config.PREDICTION_LOG_PATH.name)
        st.success(f"Logged {len(preds)} predictions to {path.name}")


def main():
    st.set_page_config(page_title="Retalix Product Health Check", page_icon="🛒", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Retalix Product Health Check")
    st.markdown('<p class="app-sub">Estimate how likely a catalogue product is to be discontinued at the next '
                'range refresh, see which factors moved the estimate, and decide what to review next.</p>',
                unsafe_allow_html=True)
    try:
        bundle, metadata = load_model(config.PRIMARY_HORIZON)
        weekly, history = load_app_data()
    except FileNotFoundError as err:
        st.error(f"{err}\n\nTrain the model first: `python -m product_discontinuation.train`")
        st.stop()
    try:
        explainer = load_explainer(bundle, bundle["model_version"], bundle.get("trained_at"))
    except FileNotFoundError:
        explainer = None

    check, technical = st.tabs(["Product health check", "Model performance & monitoring"])
    with check:
        health_check_tab(bundle, metadata, explainer, weekly, history)
    with technical:
        performance_section(metadata)
        st.divider()
        monitoring_section(bundle, weekly, history)


main()
