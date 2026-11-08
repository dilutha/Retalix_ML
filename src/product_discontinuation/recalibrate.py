"""Add calibration, risk bands and the input profile to an already trained model.

The fitted pipeline is not changed or refitted. Out-of-fold predictions are regenerated with the
same hyperparameters and the same edition folds that training used.

Usage: python -m product_discontinuation.recalibrate [--output-root DIR]
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np

from . import config
from . import evaluation as ev
from .calibration import attach_calibration
from .data import load_dataset
from .features import build_features, edition_history
from .modelling import out_of_fold_predictions
from .persistence import load_metadata, model_paths, to_jsonable
from .train import OutputDirs, log, modelling_sets


def run(dirs, horizon=config.PRIMARY_HORIZON):
    model_path, meta_path = model_paths(horizon, dirs.models, dirs.metadata)
    bundle, metadata = joblib.load(model_path), load_metadata(meta_path)
    log("Rebuilding development and test features from raw data")
    weekly = load_dataset()
    features = build_features(weekly, horizon, history=edition_history(weekly))
    dev, test = modelling_sets(features)
    if (len(dev), len(test)) != (metadata["n_train_rows"], metadata["n_test_rows"]):
        raise AssertionError("rebuilt development/test rows do not match the saved model's metadata")

    columns, pipeline = bundle["feature_columns"], bundle["pipeline"]
    log("Regenerating out-of-fold predictions for the development editions")
    oof = out_of_fold_predictions(pipeline, dev[columns], dev[config.TARGET], dev["CatEdition"])
    y_oof = dev.loc[oof.index, config.TARGET]

    bundle["threshold"] = bundle.get("raw_threshold", bundle["threshold"])
    regenerated = ev.best_f1_threshold(y_oof, oof)
    if not np.isclose(regenerated, bundle["threshold"]):
        raise AssertionError(f"regenerated threshold {regenerated} differs from the trained {bundle['threshold']}")

    raw_test = pipeline.predict_proba(test[columns])[:, 1]
    bundle, metadata = attach_calibration(bundle, metadata, oof, y_oof, dev.loc[oof.index, "CatEdition"], dev, test,
                                          raw_test, horizon, dirs)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle["calibrated_at"], metadata["calibrated_at"] = stamp, stamp
    joblib.dump(bundle, model_path, compress=3)
    meta_path.write_text(json.dumps(to_jsonable(metadata), indent=2))
    log(f"Saved calibrated bundle to {model_path}")
    return bundle, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate an already trained model.")
    parser.add_argument("--horizon", type=int, default=config.PRIMARY_HORIZON)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    run(OutputDirs.from_root(args.output_root), args.horizon)


if __name__ == "__main__":
    main()
