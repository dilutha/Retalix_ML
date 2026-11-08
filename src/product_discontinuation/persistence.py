"""Saving and loading the model bundle (fitted pipeline + inference settings)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from . import config

REQUIRED_BUNDLE_KEYS = {"pipeline", "model_name", "horizon", "include_status", "threshold",
                        "feature_columns", "model_version", "calibrator", "risk_medium_floor",
                        "input_profile"}


def model_paths(horizon, models_dir=None, metadata_dir=None):
    name = f"discontinuation_model_h{horizon}"
    models_dir = Path(models_dir or config.MODELS_DIR)
    metadata_dir = Path(metadata_dir or config.METADATA_DIR)
    return models_dir / f"{name}.joblib", metadata_dir / f"{name}.json"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def save_bundle(bundle, metadata, horizon, models_dir=None, metadata_dir=None):
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"bundle is missing keys: {sorted(missing)}")
    model_path, meta_path = model_paths(horizon, models_dir, metadata_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle = {**bundle, "trained_at": stamp, "sklearn_version": sklearn.__version__}
    metadata = {**metadata, "model_version": bundle["model_version"], "trained_at": stamp,
                "sklearn_version": sklearn.__version__, "model_file": model_path.name}
    joblib.dump(bundle, model_path, compress=3)
    meta_path.write_text(json.dumps(to_jsonable(metadata), indent=2))
    return model_path, meta_path


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}. Run the training script first.")
    bundle = joblib.load(path)
    missing = REQUIRED_BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"{path.name} is not a valid model bundle; missing {sorted(missing)}. "
                         "Run: python -m product_discontinuation.recalibrate")
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"Warning: model trained with scikit-learn {bundle.get('sklearn_version')}, "
              f"running {sklearn.__version__}")
    return bundle


def load_metadata(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    return json.loads(path.read_text())
