"""Pre-submission validation evidence for the deployed model.

Read-only for the model: refits below are diagnostic copies and are never saved over the deployed
bundle. Results go to reports/results/final_*_h13.csv and final_validation_h13.json.

Usage: python -m product_discontinuation.final_validation [--n-explain 150]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config, narrative
from .calibration import apply_calibration, fit_platt, low_band_floor, probability_metrics
from .data import load_dataset
from .evaluation import best_f1_threshold
from .explain import LocalExplainer, load_background
from .features import all_feature_columns, build_features, edition_history
from .modelling import make_pipeline, out_of_fold_predictions
from .monitoring import assess_inputs, input_notes
from .persistence import load_bundle, load_metadata, model_paths, to_jsonable
from .predict import positive_probability, predict_features
from .train import OutputDirs, log, modelling_sets

BUDGETS = (0.01, 0.05, 0.10)
TARGET = config.TARGET
MODEL = "hist_gradient_boosting"
SEED = config.RANDOM_STATE


def ranking(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    return {"n": int(len(y)), "positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p))}


def full_exit_mask(frame):
    """Departments discontinued in full in the row's own edition (a hindsight description, not a feature)."""
    rate = frame.groupby(["CatEdition", "HierarchyLevel2"])[TARGET].transform("mean")
    return (rate == 1).to_numpy()


def within_department_auc(frame, p):
    """Row-weighted ROC-AUC inside departments: ignores differences between departments."""
    aucs, weights = [], []
    for _, idx in frame.reset_index(drop=True).groupby("HierarchyLevel2").indices.items():
        y = frame[TARGET].to_numpy()[idx]
        if len(idx) >= 30 and 0 < y.mean() < 1:
            aucs.append(roc_auc_score(y, np.asarray(p)[idx]))
            weights.append(len(idx))
    return float(np.average(aucs, weights=weights)) if aucs else None


def department_cluster_interval(frame, p, n_boot=1000):
    """95% interval when whole departments are resampled, because outcomes cluster by department."""
    rng = np.random.default_rng(SEED)
    y, p = frame[TARGET].astype(int).to_numpy(), np.asarray(p)
    groups = list(frame.reset_index(drop=True).groupby("HierarchyLevel2").indices.values())
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if 0 < y[idx].mean() < 1:
            rows.append((roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx])))
    low, high = np.percentile(rows, [2.5, 97.5], axis=0)
    return {"roc_auc_95": [float(low[0]), float(high[0])], "pr_auc_95": [float(low[1]), float(high[1])],
            "departments": len(groups)}


def calibration_provenance(weekly, features, dev, test, live, bundle, oof, y_oof):
    log("Calibration provenance: rebuilding without editions 93 and 94")
    columns = bundle["feature_columns"]
    restricted = weekly[~weekly["CatEdition"].isin([config.TEST_EDITION, *config.CENSORED_EDITIONS])]
    restricted_dev, _ = _dev_only(restricted)
    keys = config.GROUP_KEYS
    # Natural row order is kept: HistGradientBoosting's seeded early-stopping split depends on row position.
    right = restricted_dev.reset_index(drop=True)
    pd.testing.assert_frame_equal(dev[keys + columns + [TARGET]], right[keys + columns + [TARGET]], check_exact=True)
    restricted_oof = out_of_fold_predictions(bundle["pipeline"], right[columns], right[TARGET], right["CatEdition"])
    refit = fit_platt(restricted_oof.to_numpy(), right.loc[restricted_oof.index, TARGET])
    reordered = right.sort_values(keys).reset_index(drop=True)
    reordered_oof = out_of_fold_predictions(bundle["pipeline"], reordered[columns], reordered[TARGET], reordered["CatEdition"])
    order_sensitivity = fit_platt(reordered_oof.to_numpy(), reordered.loc[reordered_oof.index, TARGET])
    saved = bundle["calibrator"]

    raw = {name: bundle["pipeline"].predict_proba(frame[columns])[:, 1] for name, frame in
           [("development", dev), ("test", test), ("live", live)]}
    order = {}
    for name, values in raw.items():
        calibrated = apply_calibration(values, saved)
        order[name] = bool((pd.Series(values).rank(method="dense") == pd.Series(calibrated).rank(method="dense")).all())

    y = test[TARGET].astype(int).to_numpy()
    calibrated_test = positive_probability(bundle, test)
    hand = expit(saved["a"] * logit(raw["test"]) + saved["b"])
    return {
        "oof_editions_used": sorted(int(e) for e in dev.loc[oof.index, "CatEdition"].unique()),
        "oof_rows": int(len(oof)),
        "development_features_identical_without_93_94": True,
        "saved_a_b": [saved["a"], saved["b"]],
        "refit_without_93_94_a_b": [refit["a"], refit["b"]],
        "max_abs_difference_a_b": float(max(abs(saved["a"] - refit["a"]), abs(saved["b"] - refit["b"]))),
        "row_order_sensitivity_a_b_when_rows_are_sorted": [order_sensitivity["a"], order_sensitivity["b"]],
        "dense_rank_identical_raw_vs_calibrated": order,
        "app_path_vs_hand_formula_max_difference": float(np.abs(calibrated_test - hand).max()),
        "held_out_recomputed_by_hand": {
            "n": int(len(y)), "editions": sorted(int(e) for e in test["CatEdition"].unique()),
            "brier_raw": float(np.mean((raw["test"] - y) ** 2)), "brier_calibrated": float(np.mean((calibrated_test - y) ** 2)),
            "mean_calibrated": float(calibrated_test.mean()), "observed_rate": float(y.mean())},
    }


def _dev_only(weekly):
    features = build_features(weekly, config.PRIMARY_HORIZON, history=edition_history(weekly))
    labelled = features[features["CatEdition"] >= config.MODEL_FIRST_EDITION]
    return labelled.reset_index(drop=True), None


def _band_row(sample, budget, floor, threshold, p, y):
    bands = np.select([p >= threshold, p >= floor], ["High", "Medium"], default="Low")
    row = {"sample": sample, "miss_budget": budget, "floor": floor, "n": int(len(p))}
    for band in ("Low", "Medium", "High"):
        mask = bands == band
        row[f"{band}_share"] = float(mask.mean())
        row[f"{band}_observed_rate"] = float(y[mask].mean()) if y is not None and mask.any() else None
    row["realised_miss_rate"] = float(y[bands == "Low"].sum() / y.sum()) if y is not None else None
    return row


def risk_band_study(oof, y_oof, editions, raw_test, y_test, raw_live, bundle):
    log("Risk-band budgets and stability")
    calibrator, threshold = bundle["calibrator"], bundle["threshold"]
    calibrated = apply_calibration(oof, calibrator)
    test_p, live_p = apply_calibration(raw_test, calibrator), apply_calibration(raw_live, calibrator)
    budget_rows, stability = [], []
    for budget in BUDGETS:
        floor = low_band_floor(y_oof, calibrated, threshold, budget)
        budget_rows.append(_band_row("development out-of-fold (pooled)", budget, floor, threshold, calibrated, y_oof))
        for edition in np.unique(editions):
            mask = editions == edition
            budget_rows.append(_band_row(f"edition {edition} (out-of-fold)", budget, floor, threshold, calibrated[mask], y_oof[mask]))
            stability.append({"kind": "within-edition floor, deployed calibrator", "edition": int(edition), "miss_budget": budget,
                              "floor": low_band_floor(y_oof[mask], calibrated[mask], 1.0, budget)})
        budget_rows.append(_band_row("edition 93 (held out)", budget, floor, threshold, test_p, y_test))
        budget_rows.append(_band_row("edition 94 (live, no labels)", budget, floor, threshold, live_p, None))

    for budget in BUDGETS:   # out of time: calibrator and floor from earlier editions only
        for edition in [e for e in np.unique(editions)[1:]] + [config.TEST_EDITION]:
            earlier = editions < edition
            platt = fit_platt(oof[earlier], y_oof[earlier])
            raw_t = best_f1_threshold(y_oof[earlier], oof[earlier])
            t = float(apply_calibration([raw_t], platt)[0])
            floor = low_band_floor(y_oof[earlier], apply_calibration(oof[earlier], platt), t, budget)
            raw_now, y_now = (raw_test, y_test) if edition == config.TEST_EDITION else (oof[editions == edition], y_oof[editions == edition])
            now = apply_calibration(raw_now, platt)
            stability.append({"kind": "out-of-time: fitted on earlier editions only", "edition": int(edition), "miss_budget": budget,
                              "floor": floor, "threshold": t, "realised_miss_rate": float(y_now[now < floor].sum() / y_now.sum()),
                              "low_share": float((now < floor).mean()),
                              "low_observed_rate": float(y_now[now < floor].mean()) if (now < floor).any() else None,
                              "high_share": float((now >= t).mean()), "high_observed_rate": float(y_now[now >= t].mean()),
                              "high_recall": float(y_now[now >= t].sum() / y_now.sum())})
    return pd.DataFrame(budget_rows), pd.DataFrame(stability)


def generalisation_table(dev, test, oof, raw_test):
    log("Generalisation segments")
    pool = pd.concat([dev.loc[oof.index].assign(p=oof.to_numpy()), test.assign(p=raw_test)], ignore_index=True)
    pool["full_exit"] = full_exit_mask(pool)
    pool[TARGET] = pool[TARGET].astype(int)
    rows = []
    def add(name, mask, group):
        part = pool[mask]
        if part[TARGET].nunique() == 2:
            rows.append({"group": group, "segment": name, **ranking(part[TARGET], part["p"])})

    for edition in (*config.CV_TEST_EDITIONS, config.TEST_EDITION):
        at = pool["CatEdition"] == edition
        group = "held-out test" if edition == config.TEST_EDITION else "chronological CV"
        add(f"edition {edition}: all", at, group)
        add(f"edition {edition}: excluding fully discontinued departments", at & ~pool["full_exit"], group)
        add(f"edition {edition}: new to range", at & (pool["in_previous_edition"] == 0), group)
        add(f"edition {edition}: returning", at & (pool["in_previous_edition"] == 1), group)
    cv = pool["CatEdition"] < config.TEST_EDITION
    add("CV pooled: all", cv, "chronological CV")
    add("CV pooled: new to range", cv & (pool["in_previous_edition"] == 0), "chronological CV")

    seen = set(dev["ProductKey"])
    add("edition 93: ProductKey never in editions 89-92", (pool["CatEdition"] == 93) & ~pool["ProductKey"].isin(seen), "product level")
    first = pool.sort_values("CatEdition").drop_duplicates("ProductKey")
    add_part = pool.index.isin(first.index)
    add("one row per product: earliest evaluated edition", add_part, "product level")
    shuffled = pool.sample(frac=1, random_state=SEED).drop_duplicates("ProductKey")
    add("one row per product: random edition", pool.index.isin(shuffled.index), "product level")

    table = pd.DataFrame(rows)
    per_edition = table[table["segment"].str.fullmatch(r"edition 9[0-2]: all")]
    table = pd.concat([table, pd.DataFrame([{"group": "chronological CV", "segment": "CV mean of editions 90-92 (macro)",
                                             "n": int(per_edition["n"].sum()), "positive_rate": per_edition["positive_rate"].mean(),
                                             "roc_auc": per_edition["roc_auc"].mean(), "pr_auc": per_edition["pr_auc"].mean()}])], ignore_index=True)

    level = []
    labelled = pd.concat([dev, test], ignore_index=True)
    for edition, part in pool.groupby("CatEdition"):
        earlier = labelled[labelled["CatEdition"] < edition].assign(**{TARGET: lambda d: d[TARGET].astype(int)})
        department = part["HierarchyLevel2"].map(earlier.groupby("HierarchyLevel2")[TARGET].mean()).fillna(earlier[TARGET].mean())
        dept = part.groupby("HierarchyLevel2").agg(pred=("p", "mean"), actual=(TARGET, "mean"), n=(TARGET, "size"))
        dept = dept[dept["n"] >= 30]
        level.append({"edition": int(edition), "n": int(len(part)), "positive_rate": float(part[TARGET].mean()),
                      "fully_discontinued_departments": int(part.loc[part["full_exit"], "HierarchyLevel2"].nunique()),
                      "share_in_those_departments": float(part["full_exit"].mean()),
                      "model_roc_auc": float(roc_auc_score(part[TARGET], part["p"])),
                      "department_history_baseline_roc_auc": float(roc_auc_score(part[TARGET], department)),
                      "within_department_roc_auc": within_department_auc(part, part["p"].to_numpy()),
                      "department_level_spearman_pred_vs_actual": float(dept["pred"].corr(dept["actual"], method="spearman"))})
    return table, pd.DataFrame(level), pool


def refit_score(dev, test, edition, params, exclude=(), window=None, horizon=config.PRIMARY_HORIZON):
    frame = pd.concat([dev, test], ignore_index=True)
    columns = all_feature_columns(horizon, exclude=exclude)
    train = frame[frame["CatEdition"] < edition]
    if window is not None:
        train = train[train["CatEdition"] >= edition - window]
    at = frame[frame["CatEdition"] == edition]
    pipe = make_pipeline(MODEL, horizon, params=params, exclude=exclude).fit(train[columns], train[TARGET])
    return at, pipe.predict_proba(at[columns])[:, 1]


def diagnostic_refits(dev, test, params):
    log("Diagnostic refits (never saved)")
    rows = []
    editions = (*config.CV_TEST_EDITIONS, config.TEST_EDITION)
    variants = {"all 20 features": (), "without HierarchyLevel2": ("HierarchyLevel2",),
                "without all product category inputs": ("HierarchyLevel1", "HierarchyLevel2", "Supplier")}
    for label, exclude in variants.items():
        for edition in editions:
            at, p = refit_score(dev, test, edition, params, exclude)
            rows.append({"study": "feature ablation", "variant": label, "edition": edition, **ranking(at[TARGET], p)})
    for window in (1, 2, 3, 4):
        at, p = refit_score(dev, test, config.TEST_EDITION, params, window=window)
        rows.append({"study": "training window for edition 93", "variant": f"last {window} edition(s)", "edition": 93, **ranking(at[TARGET], p)})

    columns = all_feature_columns(config.PRIMARY_HORIZON)
    frame = pd.concat([dev, test], ignore_index=True)
    for edition in editions:   # every prediction comes from a model that never saw that department
        train_all, at = frame[frame["CatEdition"] < edition], frame[frame["CatEdition"] == edition].reset_index(drop=True)
        p = np.full(len(at), np.nan)
        for _, held in GroupKFold(5).split(at, groups=at["HierarchyLevel2"]):
            departments = at.iloc[held]["HierarchyLevel2"].unique()
            train = train_all[~train_all["HierarchyLevel2"].isin(departments)]
            p[held] = make_pipeline(MODEL, config.PRIMARY_HORIZON, params=params).fit(train[columns], train[TARGET]).predict_proba(at.iloc[held][columns])[:, 1]
        rows.append({"study": "unseen departments", "variant": "department never in training", "edition": edition, **ranking(at[TARGET], p),
                     "within_department_roc_auc": within_department_auc(at, p)})
    return pd.DataFrame(rows)


def exit_department_context(dev, test, raw_test):
    y = test[TARGET].astype(int)
    rates = y.groupby(test["HierarchyLevel2"]).mean()
    full = rates[rates == 1].index
    in_full = test["HierarchyLevel2"].isin(full).to_numpy()
    by_edition = dev[dev["HierarchyLevel2"].isin(full)].groupby("CatEdition")[TARGET].mean()
    others = dev[~dev["HierarchyLevel2"].isin(full)].groupby("CatEdition")[TARGET].mean()
    return {"departments": len(full),
            "test_mean_raw_score_in_those_departments": float(raw_test[in_full].mean()),
            "test_mean_raw_score_elsewhere": float(raw_test[~in_full].mean()),
            "training_rate_by_edition_those_departments": {int(k): float(v) for k, v in by_edition.items()},
            "training_rate_by_edition_other_departments": {int(k): float(v) for k, v in others.items()}}


CAUSAL = re.compile(r"\b(because|caused?|causes|causing|due to|leads? to|results? in|drives?|drove|driven by|will be discontinued|guarantee)\b", re.I)


def _sentence_is_truthful(feature, value, sentence, typical):
    """Independent re-derivation of the numbers quoted in one plain-English sentence."""
    number = lambda pattern: float(re.search(pattern, sentence).group(1).replace(",", ""))
    if pd.isna(value):
        return True
    if feature in ("log_actuals", "log_forecast"):
        return number(r"about ([\d,]+) units") == round(np.expm1(value))
    if feature == "log_price":
        return number(r"£([\d,.]+\d)") == float(f"{np.exp(value):.2f}")
    if feature in ("actuals_trend", "forecast_trend"):
        if "stable" in sentence:
            return abs(np.expm1(value)) < narrative.STABLE_CHANGE
        up = ("risen" in sentence) or ("revised up" in sentence)
        return up == (value > 0) and number(r"about ([\d]+)%") == float(f"{abs(np.expm1(value)) * 100:.0f}")
    if feature == "forecast_to_actual":
        ratio = np.exp(value)
        if "close to" in sentence:
            return abs(ratio - 1) < 0.10
        return ("below" in sentence) == (ratio > 1)
    if feature == "weeks_in_edition":
        return number(r"listed for (\d+) weeks") == int(value)
    if feature == "prior_editions":
        return (int(value) == 0 and "first catalogue" in sentence) or (int(value) > 0 and number(r"appeared in (\d+)") == int(value))
    if feature == "previous_discontinued":
        return ("marked for discontinuation" in sentence) == bool(value)
    if feature == "in_previous_edition":
        return ("listed in the previous" in sentence) == bool(value)
    if feature == "zero_actuals":
        return ("No sales" in sentence) == bool(value)
    if feature == "price_missing":
        return ("No valid" in sentence) == bool(value)
    if feature == "Seasonal":
        return ("not seasonal" in sentence) == (not value)
    if feature == "SpringSummer":
        return ("Spring/Summer" in sentence) == bool(value)
    if feature == "DIorDOM":
        return ("imported" in sentence) == (value == "DI")
    if feature in ("Supplier", "HierarchyLevel1", "HierarchyLevel2"):
        return str(value) in sentence
    return True


def shap_language_audit(bundle, test, live, n_explain):
    log(f"SHAP and wording audit on {n_explain} products")
    explainer = LocalExplainer(bundle, load_background(bundle["horizon"]))
    rng = np.random.default_rng(SEED)
    rows = pd.concat([test.iloc[rng.choice(len(test), n_explain // 2, replace=False)],
                      live.iloc[rng.choice(len(live), n_explain - n_explain // 2, replace=False)]], ignore_index=True)
    shown = predict_features(bundle, rows)
    background = explainer.background
    counts = dict(sentences=0, wrong_direction=0, wrong_list=0, wrong_fact=0, causal=0, negligible_mentioned=0)
    examples, additivity, probability_gap, faith_total, faith_ok = [], [], [], 0, 0
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        exp = explainer.explain(row)
        p = float(shown["probability"].iloc[i])
        additivity.append(abs(exp.base_value + exp.contributions["contribution"].sum() - logit(p)))
        probability_gap.append(abs(exp.probability - p))
        raising, lowering = narrative.key_factors(exp.contributions, bundle["reference_values"])
        counts["wrong_list"] += sum(f["contribution"] <= 0 for f in raising) + sum(f["contribution"] >= 0 for f in lowering)
        lookup = exp.contributions.set_index("feature")
        for f in raising + lowering:
            counts["sentences"] += 1
            value, contribution = lookup.loc[f["feature"], "value"], float(lookup.loc[f["feature"], "contribution"])
            says_up, says_down = " raised the estimated risk" in f["sentence"], " lowered the estimated risk" in f["sentence"]
            if says_up != (contribution > 0) or says_down != (contribution < 0):
                counts["wrong_direction"] += 1
            if abs(contribution) < narrative.NEGLIGIBLE:
                counts["negligible_mentioned"] += 1
            if not _sentence_is_truthful(f["feature"], value, f["sentence"], bundle["reference_values"].get(f["feature"])):
                counts["wrong_fact"] += 1
                examples.append({"feature": f["feature"], "value": str(value), "sentence": f["sentence"]})
        text = narrative.summary(p, str(shown["risk_category"].iloc[i]), bundle["threshold"], raising, lowering)
        counts["causal"] += int(bool(CAUSAL.search(text + " ".join(f["sentence"] for f in raising + lowering))))
        top = exp.contributions.iloc[0]   # faithfulness: resetting the top input to background values should move f(x) against its sign
        if abs(top["contribution"]) >= narrative.NEGLIGIBLE:
            variants = pd.concat([row[explainer.columns]] * len(background), ignore_index=True)
            variants[top["feature"]] = background[top["feature"]].to_numpy()
            faith_total += 1
            change = logit(p) - float(np.mean(logit(positive_probability(bundle, variants))))
            faith_ok += int(np.sign(change) == np.sign(top["contribution"]))
    source = (config.PROJECT_ROOT / "deployment" / "app.py").read_text() + Path(narrative.__file__).read_text()
    return {"n_products": len(rows), "max_additivity_error_vs_displayed_calibrated_log_odds": float(max(additivity)),
            "max_probability_difference_shap_vs_displayed": float(max(probability_gap)), "sentence_checks": counts,
            "wrong_fact_examples": examples[:10],
            "top_factor_faithfulness": {"checked": faith_total, "same_sign": faith_ok},
            "causal_terms_in_source": sorted(set(m.group(0).lower() for m in CAUSAL.finditer(source)))}


def input_warning_rates(bundle, test, live):
    def share(frame):
        n = np.array([len(assess_inputs(bundle, frame.iloc[[i]])) for i in range(len(frame))])
        notes = np.array([len(input_notes(bundle, frame.iloc[[i]])) for i in range(len(frame))])
        return {"products": len(frame), "share_with_any_warning": float((n > 0).mean()), "mean_warnings": float(n.mean()),
                "share_with_rare_group_note": float((notes > 0).mean())}
    rng = np.random.default_rng(SEED)
    return {"edition_93_sample": share(test.iloc[rng.choice(len(test), 3000, replace=False)]),
            "edition_94_sample": share(live.iloc[rng.choice(len(live), 3000, replace=False)])}


def run(dirs, horizon=config.PRIMARY_HORIZON, n_explain=150):
    bundle = load_bundle(model_paths(horizon, dirs.models, dirs.metadata)[0])
    metadata = load_metadata(model_paths(horizon, dirs.models, dirs.metadata)[1])
    columns, pipeline, params = bundle["feature_columns"], bundle["pipeline"], metadata["best_params"]
    weekly = load_dataset()
    features = build_features(weekly, horizon, history=edition_history(weekly))
    dev, test = modelling_sets(features)
    live = features[features["CatEdition"].isin(config.CENSORED_EDITIONS)].reset_index(drop=True)
    oof = out_of_fold_predictions(pipeline, dev[columns], dev[TARGET], dev["CatEdition"])
    y_oof = dev.loc[oof.index, TARGET].astype(int).to_numpy()
    editions = dev.loc[oof.index, "CatEdition"].to_numpy()
    raw_test, raw_live = pipeline.predict_proba(test[columns])[:, 1], pipeline.predict_proba(live[columns])[:, 1]
    y_test = test[TARGET].astype(int).to_numpy()

    report = {"calibration_provenance": calibration_provenance(weekly, features, dev, test, live, bundle, oof, y_oof)}
    report["probability_metrics"] = {"test_raw": probability_metrics(y_test, raw_test),
                                     "test_calibrated": probability_metrics(y_test, apply_calibration(raw_test, bundle["calibrator"]))}
    budgets, stability = risk_band_study(oof.to_numpy(), y_oof, editions, raw_test, y_test, raw_live, bundle)
    budgets.to_csv(dirs.results / f"final_risk_band_budgets_h{horizon}.csv", index=False)
    stability.to_csv(dirs.results / f"final_risk_band_stability_h{horizon}.csv", index=False)

    table, level, pool = generalisation_table(dev, test, oof, raw_test)
    table.to_csv(dirs.results / f"final_generalisation_h{horizon}.csv", index=False)
    level.to_csv(dirs.results / f"final_edition_level_h{horizon}.csv", index=False)
    report["edition_93_department_cluster_bootstrap"] = department_cluster_interval(test, raw_test)
    report["cv_department_cluster_bootstrap"] = {str(e): department_cluster_interval(part, part["p"].to_numpy(), 500)
                                                 for e, part in pool[pool["CatEdition"] < config.TEST_EDITION].groupby("CatEdition")}
    refits = diagnostic_refits(dev, test, params)
    refits.to_csv(dirs.results / f"final_diagnostic_refits_h{horizon}.csv", index=False)
    report["edition_93_exit_departments"] = exit_department_context(dev, test, raw_test)
    report["input_warning_rates"] = input_warning_rates(bundle, test, live)
    report["shap_and_wording"] = shap_language_audit(bundle, test, live, n_explain)
    (dirs.results / f"final_validation_h{horizon}.json").write_text(json.dumps(to_jsonable(report), indent=2))
    log("Done")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the pre-submission validation evidence.")
    parser.add_argument("--n-explain", type=int, default=150)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    run(OutputDirs.from_root(args.output_root), n_explain=args.n_explain)


if __name__ == "__main__":
    main()
