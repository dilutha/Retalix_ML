"""Build the SHAP background and generate the explainability evidence for the report.

Reads the raw data once, trains nothing and does not touch the saved model.
Usage: python -m product_discontinuation.shap_analysis [--n-global 500] [--output-root DIR]
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from . import config, narrative  # noqa: E402
from .data import load_dataset  # noqa: E402
from .explain import (BACKGROUND_SIZE, MAX_EVALS, LocalExplainer, background_paths,  # noqa: E402
                      build_background, positive_class, save_background,
                      tree_path_dependent_values, waterfall_figure)
from .features import CATEGORICAL_FEATURES, build_features, edition_history  # noqa: E402
from .persistence import load_bundle, load_metadata, model_paths, to_jsonable  # noqa: E402
from .predict import predict_features  # noqa: E402
from .train import OutputDirs, log, modelling_sets  # noqa: E402

CHECK_ROWS = 20
REFERENCE_EVALS = 8000
ALTERNATIVE_SEED = 43


def load_sets(bundle, metadata):
    weekly = load_dataset()
    features = build_features(weekly, bundle["horizon"], history=edition_history(weekly))
    dev, test = modelling_sets(features)
    if (len(dev), len(test)) != (metadata["n_train_rows"], metadata["n_test_rows"]):
        raise AssertionError("rebuilt development/test rows do not match the saved model's metadata")
    return dev, test


def contribution_matrix(explainer, rows):
    start = time.perf_counter()
    records, bases = [], []
    for i in range(len(rows)):
        exp = explainer.explain(rows.iloc[[i]])
        records.append(exp.contributions.set_index("feature")["contribution"])
        bases.append(exp.base_value)
    seconds = (time.perf_counter() - start) / len(rows)
    return pd.DataFrame(records, index=rows.index)[explainer.columns], np.array(bases), seconds


def agreement(reference, other, columns, negligible=narrative.NEGLIGIBLE):
    diff = (reference[columns] - other[columns]).abs().to_numpy()
    def top(frame, n=3):
        return [set(np.array(columns)[np.argsort(-np.abs(r))[:n]]) for r in frame[columns].to_numpy()]
    meaningful = reference[columns].abs().to_numpy() >= negligible
    signs = np.sign(reference[columns].to_numpy()) == np.sign(other[columns].to_numpy())
    return {"mean_abs_difference": diff.mean(), "p95_abs_difference": np.quantile(diff, 0.95),
            "max_abs_difference": diff.max(),
            "top3_overlap": float(np.mean([len(a & b) / 3 for a, b in zip(top(reference), top(other))])),
            "sign_agreement_on_meaningful": float(signs[meaningful].mean())}


def interventional_check(bundle, background, rows):
    """Measure whether interventional TreeSHAP is usable for this fitted model."""
    columns = bundle["feature_columns"]
    preprocess, model = bundle["pipeline"][:-1], bundle["pipeline"][-1]
    data, X = preprocess.transform(background[columns]), preprocess.transform(rows[columns])
    explainer = shap.TreeExplainer(model, data=data, feature_perturbation="interventional", model_output="raw")
    values, base = positive_class(explainer.shap_values(X, check_additivity=False), explainer.expected_value)
    error = np.abs(base + values.sum(1) - model.decision_function(X))
    return {"max_additivity_error": error.max(), "share_of_rows_above_1e-4": float((error > 1e-4).mean()),
            "reported_base_value": base, "true_mean_background_log_odds": model.decision_function(data).mean()}


def department_context(dev, test, path):
    """Outcome rates per department: context for how strongly the model relies on HierarchyLevel2."""
    target, key = config.TARGET, "HierarchyLevel2"
    table = pd.DataFrame({"dev_rows": dev.groupby(key)[target].size(), "dev_positive_rate": dev.groupby(key)[target].mean(),
                          "test_rows": test.groupby(key)[target].size(),
                          "test_positive_rate": test.groupby(key)[target].mean(),
                          "test_positives": test.groupby(key)[target].sum()})
    table = table.sort_values("test_positives", ascending=False).reset_index()
    table.to_csv(path, index=False)
    positives = table["test_positives"].fillna(0)
    return {"departments_in_test": int(table["test_rows"].notna().sum()),
            "share_of_test_positives_top3_departments": float(positives.head(3).sum() / positives.sum()),
            "share_of_test_rows_top3_by_positives": float(table["test_rows"].head(3).sum() / table["test_rows"].sum()),
            "top3": table.head(3).to_dict(orient="records")}


def department_season_context(dev, test, path):
    """Discontinuation rate per department, edition and season."""
    data = pd.concat([dev, test])
    table = data.groupby(["HierarchyLevel2", "CatEdition", "SpringSummer"])[config.TARGET].agg(["size", "mean"])
    table = table.rename(columns={"size": "rows", "mean": "positive_rate"}).reset_index()
    table.to_csv(path, index=False)
    return table


def fully_discontinued_departments(test, probability):
    """Test metrics with and without departments in which every product was discontinued."""
    target, key = test[config.TARGET], test["HierarchyLevel2"]
    rates = target.groupby(key).mean()
    full = rates[rates == 1].index
    rest = ~key.isin(full)
    return {"departments": sorted(full.tolist()), "n_departments": len(full),
            "share_of_test_rows": float((~rest).mean()), "share_of_test_positives": float(target[~rest].sum() / target.sum()),
            "pr_auc_all": average_precision_score(target, probability),
            "roc_auc_all": roc_auc_score(target, probability),
            "pr_auc_excluding": average_precision_score(target[rest], probability[rest]),
            "roc_auc_excluding": roc_auc_score(target[rest], probability[rest]),
            "positive_rate_excluding": float(target[rest].mean()), "rows_excluding": int(rest.sum())}


def plot_global(table, path):
    data = table.sort_values("mean_abs_shap").tail(15)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.barh(data["feature"], data["mean_abs_shap"], color="#2a78d6", height=0.65)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (log-odds of DiscontinuedTF = 1)")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_numeric_beeswarm(matrix, rows, columns, base, path):
    numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]
    data = shap.Explanation(matrix[numeric].to_numpy(), base_values=np.full(len(matrix), base),
                            data=rows[numeric].to_numpy(dtype=float), feature_names=numeric)
    plt.figure()
    shap.plots.beeswarm(data, max_display=len(numeric), show=False)
    figure = plt.gcf()
    figure.axes[0].set_xlabel("SHAP value (log-odds of DiscontinuedTF = 1)")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def local_example(rows, preds, mask):
    pool = preds[mask]
    return rows.iloc[[(pool["probability"] - pool["probability"].median()).abs().idxmin()]]


def describe_example(name, explainer, row, preds, bundle, figures, horizon):
    exp = explainer.explain(row)
    pred = preds.loc[row.index[0]]
    figure = waterfall_figure(exp)
    figure.savefig(figures / f"shap_waterfall_{name}_h{horizon}.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    raising, lowering = narrative.key_factors(exp.contributions, bundle["reference_values"])
    table = exp.contributions.assign(example=name, ProductKey=int(pred["ProductKey"]), value=lambda d: d["value"].astype(str))
    info = {"example": name, "ProductKey": int(pred["ProductKey"]), "probability": exp.probability,
            "risk_category": pred["risk_category"], "actual_discontinued": int(row[config.TARGET].iloc[0]),
            "base_value": exp.base_value, "log_odds": exp.log_odds,
            "summary": narrative.summary(exp.probability, pred["risk_category"], bundle["threshold"], raising, lowering),
            "raising": [f["sentence"] for f in raising], "lowering": [f["sentence"] for f in lowering]}
    return table, info


def run(dirs, horizon=config.PRIMARY_HORIZON, n_global=500):
    model_path, meta_path = model_paths(horizon, dirs.models, dirs.metadata)
    bundle, metadata = load_bundle(model_path), load_metadata(meta_path)
    columns = list(bundle["feature_columns"])
    log("Rebuilding development and test features from raw data")
    dev, test = load_sets(bundle, metadata)

    log("Building the SHAP background from development editions")
    background = build_background(dev, columns)
    save_background(background, dev, {"model_version": bundle["model_version"], "trained_at": bundle["trained_at"]},
                    horizon, dirs.models, dirs.metadata)
    explainer = LocalExplainer(bundle, background)

    preds = predict_features(bundle, test)
    saved = np.asarray(bundle["reference_probabilities"])
    prediction_difference = float(np.abs(preds["probability"].to_numpy() - saved).max())

    sample = test.sample(n_global, random_state=config.RANDOM_STATE)
    log(f"Explaining {n_global} test-edition products")
    matrix, bases, seconds = contribution_matrix(explainer, sample)
    log_odds = explainer._log_odds(sample[columns])
    additivity = float(np.abs(bases + matrix.sum(1).to_numpy() - log_odds).max())

    importance = pd.read_csv(dirs.results / f"feature_importance_h{horizon}.csv").set_index("feature")
    tree_values, tree_base = tree_path_dependent_values(bundle, sample)
    tree_values.index = matrix.index
    table = pd.DataFrame({"mean_abs_shap": matrix.abs().mean(), "mean_shap": matrix.mean(),
                          "tree_path_dependent_mean_abs": tree_values.abs().mean(),
                          "permutation_importance_pr_auc": importance["importance_mean"].reindex(columns)})
    table = table.sort_values("mean_abs_shap", ascending=False).rename_axis("feature").reset_index()
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    table.to_csv(dirs.results / f"shap_global_importance_h{horizon}.csv", index=False)
    plot_global(table, dirs.figures / f"shap_global_importance_h{horizon}.png")
    plot_numeric_beeswarm(matrix, sample, columns, explainer.base_value,
                          dirs.figures / f"shap_beeswarm_numeric_h{horizon}.png")

    log("Stability checks")
    rows = sample.iloc[:CHECK_ROWS]
    subset = matrix.iloc[:CHECK_ROWS]
    reference, _, reference_seconds = contribution_matrix(LocalExplainer(bundle, background, REFERENCE_EVALS), rows)
    alternative = build_background(dev, columns, seed=ALTERNATIVE_SEED)
    other, other_bases, _ = contribution_matrix(LocalExplainer(bundle, alternative), rows)
    repeat = explainer.explain(rows.iloc[[0]]).contributions["contribution"].to_numpy()
    first = explainer.explain(rows.iloc[[0]]).contributions["contribution"].to_numpy()

    log("Local examples")
    actual = test[config.TARGET] == 1
    flagged = preds["predicted_discontinued"] == 1
    tables, examples = [], []
    for name, mask in [("flagged", flagged & actual), ("not_flagged", ~flagged & ~actual)]:
        one, info = describe_example(name, explainer, local_example(test, preds, mask), preds, bundle,
                                     dirs.figures, horizon)
        tables.append(one)
        examples.append(info)
    pd.concat(tables).to_csv(dirs.results / f"shap_local_examples_h{horizon}.csv", index=False)

    report = {
        "versions": {"shap": shap.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "explainer": {"method": explainer.method, "max_evals": MAX_EVALS, "seed": explainer.seed,
                      "output": "log-odds of DiscontinuedTF = 1"},
        "background": json.loads(background_paths(horizon, dirs.models, dirs.metadata)[1].read_text()),
        "base_value_log_odds": explainer.base_value,
        "base_value_probability": float(1 / (1 + np.exp(-explainer.base_value))),
        "n_explained": n_global, "seconds_per_explanation": seconds,
        "max_local_accuracy_error_log_odds": additivity,
        "max_difference_vs_saved_test_probabilities": prediction_difference,
        "determinism_max_difference": float(np.abs(first - repeat).max()),
        "convergence_vs_8000_evals": {**agreement(reference, subset, columns), "rows": CHECK_ROWS,
                                      "seconds_per_explanation_at_8000": reference_seconds},
        "alternative_background_seed_43": {**agreement(subset, other, columns), "rows": CHECK_ROWS,
                                           "base_value": float(other_bases[0])},
        "tree_path_dependent_cross_check": {
            "base_value": tree_base,
            "max_additivity_error": float(np.abs(tree_base + tree_values.sum(1).to_numpy() - log_odds).max()),
            "spearman_global_mean_abs": float(spearmanr(table["mean_abs_shap"],
                                                        table["tree_path_dependent_mean_abs"]).statistic),
            **agreement(matrix, tree_values, columns)},
        "spearman_shap_vs_permutation_importance": float(spearmanr(table["mean_abs_shap"],
                                                                   table["permutation_importance_pr_auc"]).statistic),
        "interventional_tree_shap_check": interventional_check(bundle, background, sample),
        "department_context": department_context(dev, test, dirs.results / f"shap_department_context_h{horizon}.csv"),
        "fully_discontinued_departments": fully_discontinued_departments(test, preds["probability"]),
        "department_season_rows": len(department_season_context(
            dev, test, dirs.results / f"shap_department_season_h{horizon}.csv")),
        "share_of_total_mean_abs_shap_hierarchy_level2": float(
            table.loc[table.feature == "HierarchyLevel2", "mean_abs_shap"].iloc[0] / table["mean_abs_shap"].sum()),
        "local_examples": examples,
    }
    (dirs.results / f"shap_validation_h{horizon}.json").write_text(json.dumps(to_jsonable(report), indent=2))
    log("Done")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the SHAP background and report evidence.")
    parser.add_argument("--n-global", type=int, default=500)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    run(OutputDirs.from_root(args.output_root), n_global=args.n_global)


if __name__ == "__main__":
    main()
