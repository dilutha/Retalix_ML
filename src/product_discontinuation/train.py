"""End-to-end training: validation, tuning, model selection, evaluation and persistence.

Usage (from the project root):
    python -m product_discontinuation.train                  # full run
    python -m product_discontinuation.train --sample-frac 0.1 --n-iter 2 --output-root /tmp/dev
"""
import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import evaluation as ev
from .calibration import attach_calibration
from .data import (find_incomplete_editions, load_catalogue, load_product_details,
                   merge_product_details, split_labelled, validate_catalogue, validate_product_details)
from .explain import build_background, save_background
from .features import (FORECAST_FEATURES, HISTORY_FEATURES, all_feature_columns, build_features,
                       edition_history)
from .modelling import MODEL_NAMES, cv_results_table, make_pipeline, out_of_fold_predictions, tune
from .monitoring import build_drift_reference
from .persistence import to_jsonable, save_bundle
from .predict import reference_values

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}


@dataclass
class OutputDirs:
    models: Path
    metadata: Path
    figures: Path
    results: Path
    experiments: Path
    processed: Path
    logs: Path

    @classmethod
    def from_root(cls, root=None):
        if root is None:
            return cls(config.MODELS_DIR, config.METADATA_DIR, config.FIGURES_DIR,
                       config.RESULTS_DIR, config.EXPERIMENTS_DIR, config.PROCESSED_DIR,
                       config.PREDICTION_LOG_PATH.parent)
        root = Path(root)
        return cls(root / "models", root / "metadata", root / "figures", root / "results",
                   root / "experiments", root / "processed", root / "logs")

    def create(self):
        for path in vars(self).values():
            Path(path).mkdir(parents=True, exist_ok=True)


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sample_products(weekly, frac, seed=config.RANDOM_STATE):
    """Keep a random subset of products (all their editions and weeks)."""
    if frac >= 1:
        return weekly
    if not 0 < frac < 1:
        raise ValueError("sample fraction must be in (0, 1]")
    products = weekly["ProductKey"].drop_duplicates().sample(frac=frac, random_state=seed)
    return weekly[weekly["ProductKey"].isin(products)].reset_index(drop=True)


def modelling_sets(features):
    """Development editions (CV + threshold) and the held-out test edition."""
    labelled = features[
        (features["CatEdition"] >= config.MODEL_FIRST_EDITION)
        & ~features["CatEdition"].isin(config.CENSORED_EDITIONS)
    ]
    dev = labelled[labelled["CatEdition"] < config.TEST_EDITION].reset_index(drop=True)
    test = labelled[labelled["CatEdition"] == config.TEST_EDITION].reset_index(drop=True)
    if dev.empty or test.empty:
        raise ValueError("development or test set is empty")
    return dev, test


def check_no_contamination(dev, test):
    """Train/test editions must be disjoint and strictly earlier than the test edition."""
    if dev["CatEdition"].max() >= test["CatEdition"].min():
        raise AssertionError("training data overlaps or follows the test edition")
    keys = set(zip(dev["CatEdition"], dev["ProductKey"]))
    if any(k in keys for k in zip(test["CatEdition"], test["ProductKey"])):
        raise AssertionError("product-edition rows shared between train and test")


def validate_and_report(catalogue, products, dirs):
    """Validate the raw data and save a data-quality report."""
    report = validate_catalogue(catalogue)
    validate_product_details(products)
    report["product_details_rows"] = len(products)
    report["incomplete_editions"] = find_incomplete_editions(catalogue)
    report["censored_editions_config"] = list(config.CENSORED_EDITIONS)
    if set(report["incomplete_editions"]) - set(config.CENSORED_EDITIONS):
        log(f"WARNING: unexpected incomplete editions {report['incomplete_editions']}")
    labelled, _ = split_labelled(catalogue)
    group_target = labelled.groupby(config.GROUP_KEYS)[config.TARGET].first()
    report["labelled_product_editions"] = int(len(group_target))
    report["labelled_group_positive_rate"] = float(group_target.mean())
    (dirs.results / "data_validation.json").write_text(json.dumps(to_jsonable(report), indent=2))
    return report


def train_primary(weekly, history, horizon, model_names, n_iter, dirs):
    log(f"Building features for primary horizon {horizon}")
    features = build_features(weekly, horizon, history=history)
    dev, test = modelling_sets(features)
    check_no_contamination(dev, test)
    columns = all_feature_columns(horizon)
    X_dev, y_dev, X_test, y_test = dev[columns], dev[config.TARGET], test[columns], test[config.TARGET]
    log(f"dev rows={len(dev)} (editions {sorted(dev.CatEdition.unique())}), test rows={len(test)}")

    searches, cv_tables = {}, []
    for name in model_names:
        log(f"Tuning {name}")
        start = time.time()
        searches[name] = tune(name, X_dev, y_dev, dev["CatEdition"], horizon, n_iter=n_iter)
        cv_tables.append(cv_results_table(searches[name], name))
        log(f"  best CV PR-AUC={searches[name].best_score_:.4f} in {time.time() - start:.0f}s "
            f"params={searches[name].best_params_}")
    cv_table = pd.concat(cv_tables, ignore_index=True)
    cv_table.to_csv(dirs.experiments / f"cv_results_h{horizon}.csv", index=False)

    best_cv = {n: s.cv_results_ for n, s in searches.items()}
    summary = pd.DataFrame([{
        "model": n,
        "cv_pr_auc": s.best_score_,
        "cv_pr_auc_std": best_cv[n]["std_test_average_precision"][s.best_index_],
        "cv_roc_auc": best_cv[n]["mean_test_roc_auc"][s.best_index_],
        "cv_balanced_accuracy": best_cv[n]["mean_test_balanced_accuracy"][s.best_index_],
        "best_params": str(s.best_params_),
    } for n, s in searches.items()]).sort_values("cv_pr_auc", ascending=False)
    summary.to_csv(dirs.results / f"model_selection_h{horizon}.csv", index=False)
    selected = summary.iloc[0]["model"]
    log(f"Selected model by CV PR-AUC: {selected}")

    log("Out-of-fold predictions for threshold selection")
    oof = out_of_fold_predictions(searches[selected].best_estimator_, X_dev, y_dev, dev["CatEdition"])
    threshold = ev.best_f1_threshold(y_dev.loc[oof.index], oof)
    ev.plot_threshold_tradeoff(y_dev.loc[oof.index], oof, threshold,
                               dirs.figures / f"threshold_tradeoff_h{horizon}.png")
    ev.threshold_table(y_dev.loc[oof.index], oof).to_csv(
        dirs.results / f"threshold_table_validation_h{horizon}.csv", index=False)
    log(f"Chosen threshold (max F1 on out-of-fold dev predictions) = {threshold:.3f}")

    log("Final evaluation on the held-out test edition")
    test_probs = {n: s.best_estimator_.predict_proba(X_test)[:, 1] for n, s in searches.items()}
    test_rows = [{"model": n, "threshold_type": "selected", **ev.classification_metrics(y_test, p, threshold)}
                 for n, p in test_probs.items()]
    test_rows += [{"model": n, "threshold_type": "default_0.5", **ev.classification_metrics(y_test, p, 0.5)}
                  for n, p in test_probs.items()]
    status_rule = test["status_ro"].to_numpy()
    test_rows.append({"model": "baseline_status_rule (leakage benchmark)", "threshold_type": "rule",
                      **ev.classification_metrics(y_test, status_rule, 0.5)})
    test_rows.append({"model": "baseline_majority_class", "threshold_type": "rule",
                      **ev.classification_metrics(y_test, np.zeros(len(y_test)), 0.5)})

    log("Leakage investigation: selected model family with Status features")
    status_cols = all_feature_columns(horizon, include_status=True)
    with_status = make_pipeline(selected, horizon, include_status=True,
                                params=searches[selected].best_params_)
    with_status.fit(dev[status_cols], y_dev)
    status_probs = with_status.predict_proba(test[status_cols])[:, 1]
    test_rows.append({"model": f"{selected}_with_status (leakage investigation)",
                      "threshold_type": "default_0.5",
                      **ev.classification_metrics(y_test, status_probs, 0.5)})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(dirs.results / f"test_metrics_h{horizon}.csv", index=False)

    final = searches[selected].best_estimator_
    final_probs = test_probs[selected]
    ev.plot_confusion_matrix(y_test, final_probs, threshold, f"{DISPLAY_NAMES[selected]} - test edition",
                             dirs.figures / f"confusion_matrix_h{horizon}.png")
    ev.plot_roc_pr(y_test, {DISPLAY_NAMES[n]: p for n, p in test_probs.items()},
                   dirs.figures / f"roc_pr_curves_h{horizon}.png",
                   baselines={"Status rule (leakage benchmark)": status_rule})
    ev.plot_calibration(y_test, {DISPLAY_NAMES[n]: p for n, p in test_probs.items()},
                        dirs.figures / f"calibration_h{horizon}.png")
    ev.threshold_table(y_test, final_probs).to_csv(
        dirs.results / f"threshold_table_test_h{horizon}.csv", index=False)

    segments = np.where(test["in_previous_edition"] == 1, "returning product", "new to range")
    ev.segment_metrics(y_test, final_probs, segments, threshold).to_csv(
        dirs.results / f"segment_metrics_h{horizon}.csv", index=False)

    log("Permutation importance (test sample)")
    importance = ev.grouped_permutation_importance(final, X_test, y_test)
    importance.to_csv(dirs.results / f"feature_importance_h{horizon}.csv", index=False)
    ev.plot_feature_importance(importance, f"Permutation importance - {DISPLAY_NAMES[selected]}",
                               dirs.figures / f"feature_importance_h{horizon}.png")
    if "logistic_regression" in searches:
        lr = searches["logistic_regression"].best_estimator_
        coefs = pd.DataFrame({"feature": lr[:-1].get_feature_names_out(),
                              "coefficient": lr[-1].coef_[0]})
        coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index).to_csv(
            dirs.results / f"logistic_coefficients_h{horizon}.csv", index=False)

    selected_test = test_table[(test_table.model == selected) & (test_table.threshold_type == "selected")]
    bundle = {
        "pipeline": final,
        "model_name": selected,
        "horizon": horizon,
        "include_status": False,
        "threshold": threshold,
        "feature_columns": columns,
        "model_version": config.MODEL_VERSION,
        "reference_values": reference_values(X_dev, columns),
        "drift_reference": build_drift_reference(X_dev, columns),
        "reference_probabilities": final_probs.tolist(),
    }
    metadata = {
        "model_name": selected,
        "horizon_weeks": horizon,
        "prediction_unit": "product within a catalogue edition, observed at WeeksOut = -horizon",
        "decision_threshold": threshold,
        "threshold_method": "max F1 on out-of-fold predictions of development editions",
        "feature_columns": columns,
        "excluded_features": ["Status (planner listing decision; see leakage investigation)",
                              "ProductKey (identifier)", "observations after the horizon"],
        "train_editions": sorted(int(e) for e in dev["CatEdition"].unique()),
        "test_edition": config.TEST_EDITION,
        "cv_test_editions": list(config.CV_TEST_EDITIONS),
        "best_params": searches[selected].best_params_,
        "cv_summary": summary.to_dict(orient="records"),
        "test_metrics": selected_test.drop(columns=["model", "threshold_type"]).iloc[0].to_dict(),
        "n_train_rows": len(dev),
        "n_test_rows": len(test),
        "random_state": config.RANDOM_STATE,
    }
    bundle, metadata = attach_calibration(bundle, metadata, oof, y_dev.loc[oof.index], dev.loc[oof.index, "CatEdition"],
                                          dev, test, final_probs, horizon, dirs)
    model_path, meta_path = save_bundle(bundle, metadata, horizon, dirs.models, dirs.metadata)
    save_background(build_background(dev, columns), dev,
                    {"model_version": config.MODEL_VERSION, "trained_at": json.loads(meta_path.read_text())["trained_at"]},
                    horizon, dirs.models, dirs.metadata)
    log(f"Saved model to {model_path}")
    return searches, selected, threshold, test_table


def ablation_study(weekly, history, horizon, name, params, dirs):
    """Feature-group ablation for the selected model family, on CV folds and the test edition."""
    features = build_features(weekly, horizon, history=history)
    dev, test = modelling_sets(features)
    variants = {
        "all features (primary)": dict(),
        "without forecast features": dict(exclude=FORECAST_FEATURES),
        "without prior-edition history": dict(exclude=HISTORY_FEATURES),
        "with Status (leakage investigation)": dict(include_status=True),
    }
    rows = []
    for label, kwargs in variants.items():
        log(f"Ablation: {label}")
        columns = all_feature_columns(horizon, kwargs.get("include_status", False),
                                      kwargs.get("exclude", ()))
        pipe = make_pipeline(name, horizon, params=params, **kwargs)
        oof = out_of_fold_predictions(pipe, dev[columns], dev[config.TARGET], dev["CatEdition"])
        y_oof = dev.loc[oof.index, config.TARGET]
        fold_ap = [ev.classification_metrics(y_oof[dev.loc[oof.index, "CatEdition"] == e],
                                             oof[dev.loc[oof.index, "CatEdition"] == e])["pr_auc"]
                   for e in config.CV_TEST_EDITIONS]
        prob = pipe.fit(dev[columns], dev[config.TARGET]).predict_proba(test[columns])[:, 1]
        test_m = ev.classification_metrics(test[config.TARGET], prob)
        rows.append({"variant": label, "model": name, "n_features": len(columns),
                     "cv_pr_auc_mean": np.mean(fold_ap), "cv_pr_auc_std": np.std(fold_ap),
                     **{f"cv_pr_auc_ed{e}": ap for e, ap in zip(config.CV_TEST_EDITIONS, fold_ap)},
                     "test_pr_auc": test_m["pr_auc"], "test_roc_auc": test_m["roc_auc"]})
    table = pd.DataFrame(rows)
    table.to_csv(dirs.results / f"ablation_h{horizon}.csv", index=False)
    return table


def horizon_study(weekly, history, horizons, searches, primary_selected, dirs):
    """Refit each model family (tuned at the primary horizon) at every horizon."""
    rows = []
    for horizon in horizons:
        log(f"Horizon study: {horizon} weeks")
        features = build_features(weekly, horizon, history=history)
        dev, test = modelling_sets(features)
        check_no_contamination(dev, test)
        columns = all_feature_columns(horizon)
        for name, search in searches.items():
            pipe = make_pipeline(name, horizon, params=search.best_params_).fit(dev[columns], dev[config.TARGET])
            prob = pipe.predict_proba(test[columns])[:, 1]
            rows.append({"horizon": horizon, "model": name, **ev.classification_metrics(test[config.TARGET], prob)})
        status_cols = all_feature_columns(horizon, include_status=True)
        pipe = make_pipeline(primary_selected, horizon, include_status=True,
                             params=searches[primary_selected].best_params_)
        prob = pipe.fit(dev[status_cols], dev[config.TARGET]).predict_proba(test[status_cols])[:, 1]
        rows.append({"horizon": horizon, "model": f"{primary_selected}_with_status",
                     **ev.classification_metrics(test[config.TARGET], prob)})
        rows.append({"horizon": horizon, "model": "status_rule",
                     **ev.classification_metrics(test[config.TARGET], test["status_ro"])})
    table = pd.DataFrame(rows)
    table.to_csv(dirs.results / "horizon_comparison.csv", index=False)
    ev.plot_horizon_comparison(table, dirs.figures / "horizon_comparison_pr_auc.png", "pr_auc")
    ev.plot_horizon_comparison(table, dirs.figures / "horizon_comparison_roc_auc.png", "roc_auc")
    return table


def save_app_data(weekly, history, dirs):
    """Raw weekly rows for the test and live editions plus edition history for the app."""
    app_editions = [config.TEST_EDITION, *config.CENSORED_EDITIONS]
    weekly[weekly["CatEdition"].isin(app_editions)].to_parquet(dirs.processed / "app_weekly.parquet", index=False)
    history.to_parquet(dirs.processed / "edition_history.parquet", index=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the product discontinuation model.")
    parser.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON)
    parser.add_argument("--study-horizons", type=int, nargs="*", default=list(config.HORIZONS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--n-iter", type=int, default=8, help="random-search iterations per model")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="fraction of products (dev runs)")
    parser.add_argument("--output-root", default=None, help="write outputs here instead of the project")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(config.RANDOM_STATE)
    dirs = OutputDirs.from_root(args.output_root)
    dirs.create()
    start = time.time()

    log("Loading and validating data")
    catalogue, products = load_catalogue(), load_product_details()
    report = validate_and_report(catalogue, products, dirs)
    weekly = merge_product_details(catalogue, products)
    weekly = sample_products(weekly, args.sample_frac)
    del catalogue
    log(f"rows={report['rows']}, product-editions={report['product_editions']}, "
        f"incomplete editions={report['incomplete_editions']}")
    history = edition_history(weekly)

    searches, selected, threshold, test_table = train_primary(
        weekly, history, args.horizon, args.models, args.n_iter, dirs)
    if not args.skip_ablation:
        ablation_study(weekly, history, args.horizon, selected, searches[selected].best_params_, dirs)
    horizon_table = horizon_study(weekly, history, args.study_horizons, searches, selected, dirs) \
        if args.study_horizons else None
    save_app_data(weekly, history, dirs)

    entry = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": config.MODEL_VERSION,
        "args": vars(args),
        "selected_model": selected,
        "threshold": threshold,
        "test_metrics": test_table.to_dict(orient="records"),
        "horizon_study": None if horizon_table is None else horizon_table.to_dict(orient="records"),
        "duration_seconds": round(time.time() - start),
    }
    with open(dirs.experiments / "experiment_log.jsonl", "a") as fh:
        fh.write(json.dumps(to_jsonable(entry)) + "\n")
    log(f"Done in {time.time() - start:.0f}s")
    print(test_table[["model", "threshold_type", "roc_auc", "pr_auc", "precision", "recall", "f1",
                      "balanced_accuracy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
