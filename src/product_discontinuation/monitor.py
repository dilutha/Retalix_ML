"""Batch scoring and monitoring run.

1. Scores the live (unlabelled) edition, logs predictions and checks input and
   prediction drift against the training reference.
2. Back-test of outcome monitoring: the held-out test edition is scored and
   compared with its known outcomes, illustrating what happens once real labels
   arrive after a refresh.

Usage: python -m product_discontinuation.monitor [--output-root DIR]
"""
import argparse
import json

import pandas as pd

from . import config
from .monitoring import (evaluate_against_outcomes, feature_drift_report, log_predictions,
                         prediction_drift, prediction_summary, validate_scoring_input)
from .persistence import load_bundle, model_paths, to_jsonable
from .predict import predict_weekly
from .train import OutputDirs


def run(dirs, horizon=config.PRIMARY_HORIZON):
    model_path, _ = model_paths(horizon, dirs.models, dirs.metadata)
    bundle = load_bundle(model_path)
    weekly = pd.read_parquet(dirs.processed / "app_weekly.parquet")
    history = pd.read_parquet(dirs.processed / "edition_history.parquet")
    log_path = dirs.logs / config.PREDICTION_LOG_PATH.name

    live_editions = list(config.CENSORED_EDITIONS)
    live = weekly[weekly["CatEdition"].isin(live_editions)].drop(columns=[config.TARGET])
    validation = validate_scoring_input(live)
    live_features, live_preds = predict_weekly(bundle, live, history=history)
    log_predictions(live_preds, bundle, source="live", path=log_path)

    drift = feature_drift_report(bundle["drift_reference"], live_features)
    drift.to_csv(dirs.results / "drift_report_live.csv", index=False)
    summary = {
        "live_editions": live_editions,
        "input_validation": validation,
        "prediction_summary": prediction_summary(live_preds),
        "prediction_psi_vs_test": prediction_drift(bundle["reference_probabilities"],
                                                   live_preds["probability"]),
        "features_with_significant_drift": drift.loc[drift.status == "significant", "feature"].tolist(),
    }

    test = weekly[weekly["CatEdition"] == config.TEST_EDITION]
    _, test_preds = predict_weekly(bundle, test.drop(columns=[config.TARGET]), history=history)
    log_predictions(test_preds, bundle, source="backtest", path=log_path)
    outcomes = test[config.GROUP_KEYS + [config.TARGET]].drop_duplicates(config.GROUP_KEYS)
    logged = pd.read_csv(log_path)
    backtest_log = logged[logged["source"] == "backtest"]
    summary["backtest_outcome_monitoring"] = evaluate_against_outcomes(
        backtest_log, outcomes, bundle["threshold"])

    (dirs.results / "monitoring_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    return summary, drift


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the live edition and run monitoring checks.")
    parser.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    summary, drift = run(OutputDirs.from_root(args.output_root), args.horizon)
    print(json.dumps(to_jsonable({k: v for k, v in summary.items() if k != "input_validation"}), indent=2))
    print(drift.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
