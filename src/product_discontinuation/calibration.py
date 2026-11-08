"""Probability calibration and validation-based risk bands.

The boosted model ranks products well but its probabilities are too low on future editions
(edition 93: mean prediction 0.225 against an observed rate of 0.364). A Platt map is fitted on
the expanding-window out-of-fold predictions of the development editions, so neither the test
edition nor any in-sample prediction is used.

Platt scaling is used because it is linear on the log-odds scale: it preserves the ranking (ROC-AUC,
PR-AUC and the set of flagged products) and rescales SHAP values exactly. Isotonic regression is
fitted alongside it only as a comparison, in `rolling_origin_comparison`.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score  # noqa: E402

from . import config  # noqa: E402
from . import evaluation as ev  # noqa: E402
from .monitoring import build_input_profile  # noqa: E402
from .persistence import to_jsonable  # noqa: E402

PROBABILITY_CLIP = 1e-12
LOW_BAND_MISS_BUDGET = 0.05  # at most 5% of discontinued development products may fall in the Low band
BANDS = ("Low", "Medium", "High")


def _clipped_logit(probability):
    return logit(np.clip(np.asarray(probability, dtype=float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))


def fit_platt(probability, y):
    """Fit log-odds(calibrated) = a * log-odds(raw) + b, with effectively no regularisation."""
    model = LogisticRegression(C=1e6, max_iter=1000).fit(_clipped_logit(probability).reshape(-1, 1),
                                                         np.asarray(y).astype(int))
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if a <= 0:
        raise ValueError(f"Platt slope must be positive to preserve the ranking, got {a}")
    return {"method": "platt", "a": a, "b": b}


def calibrated_log_odds(raw_log_odds, calibrator):
    """Map raw log-odds to the calibrated scale; `calibrator=None` leaves them unchanged."""
    if calibrator is None:
        return raw_log_odds
    return calibrator["a"] * np.asarray(raw_log_odds, dtype=float) + calibrator["b"]


def apply_calibration(probability, calibrator):
    if calibrator is None:
        return np.asarray(probability, dtype=float)
    return expit(calibrated_log_odds(_clipped_logit(probability), calibrator))


def expected_calibration_error(y, probability, n_bins=10):
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(probability, dtype=float)})
    groups = frame.groupby(pd.qcut(frame["p"], n_bins, duplicates="drop"), observed=True)
    return float((groups.size() / len(frame) * (groups["y"].mean() - groups["p"].mean()).abs()).sum())


def probability_metrics(y, probability):
    y, probability = np.asarray(y).astype(int), np.asarray(probability, dtype=float)
    return {"n": int(len(y)), "observed_rate": float(y.mean()), "mean_predicted": float(probability.mean()),
            "brier": float(brier_score_loss(y, probability)),
            "brier_of_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean()))),
            "log_loss": float(log_loss(y, np.clip(probability, 1e-15, 1 - 1e-15))),
            "ece_10_quantile_bins": expected_calibration_error(y, probability),
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability))}


def reliability_table(y, probability, n_bins=10):
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(probability, dtype=float)})
    groups = frame.groupby(pd.qcut(frame["p"], n_bins, duplicates="drop"), observed=True)
    return groups.agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index(drop=True)


def _fit_method(name, probability, y):
    if name == "uncalibrated":
        return lambda p: np.asarray(p, dtype=float)
    if name == "platt":
        calibrator = fit_platt(probability, y)
        return lambda p: apply_calibration(p, calibrator)
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(probability, np.asarray(y).astype(int))
    return model.predict


def rolling_origin_comparison(probability, y, editions):
    """Out-of-time comparison: each method is fitted on earlier out-of-fold editions only."""
    probability, y, editions = np.asarray(probability, dtype=float), np.asarray(y).astype(int), np.asarray(editions)
    rows = []
    for edition in np.unique(editions)[1:]:
        earlier, current = editions < edition, editions == edition
        for method in ("uncalibrated", "platt", "isotonic"):
            calibrated = _fit_method(method, probability[earlier], y[earlier])(probability[current])
            rows.append({"evaluated_edition": int(edition), "method": method,
                         **probability_metrics(y[current], calibrated)})
    return pd.DataFrame(rows)


def low_band_floor(y, calibrated, threshold, miss_budget=LOW_BAND_MISS_BUDGET):
    """Probability below which at most `miss_budget` of the discontinued products fall."""
    positives = np.asarray(calibrated, dtype=float)[np.asarray(y).astype(int) == 1]
    return min(float(np.quantile(positives, miss_budget)), threshold)


def band_table(y, calibrated, threshold, medium_floor, sample):
    y, calibrated = np.asarray(y).astype(int), np.asarray(calibrated, dtype=float)
    bands = np.select([calibrated >= threshold, calibrated >= medium_floor], ["High", "Medium"], default="Low")
    rows = []
    for band in BANDS:
        mask = bands == band
        rows.append({"sample": sample, "band": band, "n": int(mask.sum()), "share_of_products": float(mask.mean()),
                     "mean_predicted": float(calibrated[mask].mean()) if mask.any() else None,
                     "observed_rate": float(y[mask].mean()) if mask.any() else None,
                     "share_of_discontinued": float(y[mask].sum() / y.sum())})
    return pd.DataFrame(rows)


def department_exit_context(test, calibrated):
    """Test-edition metrics with and without departments where every product was discontinued."""
    target, department = test[config.TARGET].astype(int), test["HierarchyLevel2"]
    rates = target.groupby(department).mean()
    keep = ~department.isin(rates[rates == 1].index).to_numpy()
    return {"departments_fully_discontinued": int((rates == 1).sum()),
            "share_of_test_rows": float(1 - keep.mean()),
            "roc_auc_excluding": float(roc_auc_score(target[keep], calibrated[keep])),
            "pr_auc_excluding": float(average_precision_score(target[keep], calibrated[keep])),
            "positive_rate_excluding": float(target[keep].mean())}


def plot_before_after(y, raw, calibrated, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, probability, colour in [("Raw model", raw, "#eb6834"), ("Platt calibrated", calibrated, "#2a78d6")]:
        table = reliability_table(y, probability)
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", color=colour, label=name)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed discontinuation rate")
    ax.set_title("Reliability on held-out edition 93")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def attach_calibration(bundle, metadata, raw_oof, y_oof, editions_oof, dev, test, raw_test, horizon, dirs):
    """Fit the calibrator and risk bands on development data; update the bundle, metadata and reports.

    `bundle["threshold"]` is the max-F1 threshold on the raw scale before this call and its calibrated
    image afterwards, so exactly the same products are flagged.
    """
    raw_oof, y_oof = np.asarray(raw_oof, dtype=float), np.asarray(y_oof).astype(int)
    raw_threshold = float(bundle["threshold"])
    calibrator = fit_platt(raw_oof, y_oof)
    calibrated_oof, calibrated_test = apply_calibration(raw_oof, calibrator), apply_calibration(raw_test, calibrator)
    threshold = float(apply_calibration([raw_threshold], calibrator)[0])
    medium_floor = low_band_floor(y_oof, calibrated_oof, threshold)
    y_test = test[config.TARGET].astype(int).to_numpy()

    rolling = rolling_origin_comparison(raw_oof, y_oof, editions_oof)
    bands = pd.concat([band_table(y_oof, calibrated_oof, threshold, medium_floor, "development out-of-fold"),
                       band_table(y_test, calibrated_test, threshold, medium_floor, "held-out test")], ignore_index=True)
    reliability = pd.concat([reliability_table(y_test, raw_test).assign(probabilities="raw"),
                             reliability_table(y_test, calibrated_test).assign(probabilities="calibrated")],
                            ignore_index=True)
    rolling.to_csv(dirs.results / f"calibration_rolling_origin_h{horizon}.csv", index=False)
    bands.to_csv(dirs.results / f"risk_bands_h{horizon}.csv", index=False)
    reliability.to_csv(dirs.results / f"reliability_test_h{horizon}.csv", index=False)
    plot_before_after(y_test, raw_test, calibrated_test, dirs.figures / f"calibration_before_after_h{horizon}.png")

    context = department_exit_context(test, calibrated_test)
    report = {
        "calibrator": calibrator,
        "fitted_on": "expanding-window out-of-fold predictions, development editions only",
        "raw_decision_threshold": raw_threshold,
        "decision_threshold": threshold,
        "risk_medium_floor": medium_floor,
        "low_band_miss_budget": LOW_BAND_MISS_BUDGET,
        "development_oof_raw": probability_metrics(y_oof, raw_oof),
        "development_oof_calibrated_in_sample": probability_metrics(y_oof, calibrated_oof),
        "test_raw": probability_metrics(y_test, raw_test),
        "test_calibrated": probability_metrics(y_test, calibrated_test),
        "test_edition_context": context,
    }
    (dirs.results / f"calibration_h{horizon}.json").write_text(json.dumps(to_jsonable(report), indent=2))

    columns = bundle["feature_columns"]
    bundle = {**bundle, "calibrator": calibrator, "raw_threshold": raw_threshold, "threshold": threshold,
              "risk_medium_floor": medium_floor, "input_profile": build_input_profile(dev[columns], columns),
              "reference_probabilities": calibrated_test.tolist(), "model_version": config.MODEL_VERSION}
    metadata = {**metadata, "decision_threshold": threshold, "raw_decision_threshold": raw_threshold,
                "threshold_method": "max F1 on out-of-fold development predictions, mapped through the Platt calibrator",
                "calibration": report, "risk_bands": bands.to_dict(orient="records"),
                "risk_band_definition": {"High": "at or above the decision threshold",
                                         "Medium": "from the Low-band floor up to the decision threshold",
                                         "Low": f"below the floor that leaves at most {LOW_BAND_MISS_BUDGET:.0%} of "
                                                "discontinued development products in this band"},
                "test_metrics": ev.classification_metrics(y_test, calibrated_test, threshold),
                "model_version": config.MODEL_VERSION}
    return bundle, metadata
