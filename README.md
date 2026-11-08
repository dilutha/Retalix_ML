# Intelligent Product Discontinuation Prediction (Retalix)

Coursework project for 7DATA001W Machine Learning in Practice. The system predicts, for every product listed in a catalogue edition, whether it will be discontinued at the next range refresh (`DiscontinuedTF`).

## Prediction design

| Decision | Choice | Reason |
|---|---|---|
| Prediction unit | Product within a catalogue edition | Target is constant within each (CatEdition, ProductKey); one decision per product per refresh |
| Prediction point | `WeeksOut = -13` (primary), also 24, 6, 1 | Live edition 94 was extracted at week -13; leaves time to act |
| Information used | Weekly rows with `WeeksOut <= -h`, outcomes of earlier editions, product metadata | Point-in-time correctness |
| Excluded | `Status` (planner listing decision), identifiers, future rows, edition-wide observation counts | Leakage / circularity |
| Edition 94 | Unlabelled live edition (100% "True", ends at week -13) | Outcome not yet known |
| Edition 88 | History only, not a training row | No observable prior-edition history |
| Validation | Expanding-window CV over editions 90-92, test on edition 93 | Chronological, no product-edition shared between train and test |

## Project structure

```
src/product_discontinuation/
  config.py        paths, constants, horizons, split editions
  data.py          loading, schema/integrity validation, product join
  features.py      point-in-time feature engineering (shared by training, app, monitoring)
  modelling.py     preprocessing (ColumnTransformer), models, edition CV, hyperparameter search
  evaluation.py    metrics, threshold selection, plots
  persistence.py   joblib model bundle + JSON metadata
  predict.py       inference, risk categories, feature-reset sensitivity analysis (not SHAP)
  explain.py       SHAP local explanations for the deployed pipeline (PermutationExplainer)
  shap_analysis.py builds the SHAP background and the explainability evidence
  narrative.py     plain-English wording of predictions, explanations and next steps
  monitoring.py    input validation, prediction logging, PSI drift, outcome evaluation
  train.py         training CLI
  monitor.py       batch scoring / monitoring CLI
deployment/app.py  Retalix Product Health Check (Streamlit)
notebooks/         01 data validation & EDA, 02 features, 03 evaluation, 04 deployment & monitoring
tests/             pytest suite (synthetic data + real-data integrity check)
models/            trained/ (joblib) and metadata/ (JSON)
reports/           figures/ and results/ (CSV/JSON produced by training)
experiments/       CV results and experiment log
```

## Setup

```bash
conda activate ml_env                         # Python 3.10 environment
python -m venv --system-site-packages .venv   # already created in this repo
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

In VS Code select `.venv/bin/python` as the interpreter / notebook kernel.

## Run order

```bash
pytest                                          # ~40 s
python -m product_discontinuation.train         # ~30-60 min on the full data
python -m product_discontinuation.monitor       # ~30 s
python -m product_discontinuation.shap_analysis # ~10 min, SHAP background and evidence
python -m streamlit run deployment/app.py
```

Then run notebooks 01-04 in order (01-02 do not depend on training; 03-04 read training outputs).

Quick development run on a product sample (outputs go to a separate folder):

```bash
python -m product_discontinuation.train --sample-frac 0.1 --n-iter 2 --output-root /tmp/pd_dev
```

## Product Health Check app and explainability

`python -m streamlit run deployment/app.py` opens the **Retalix Product Health Check**. Choose a catalogue edition
and product and press *Check product health* to see the predicted outcome, probability and risk band, a plain-English
summary, the factors raising and lowering the estimate, and suggested review steps. A collapsed technical section
shows the SHAP waterfall plot, top contributors and interpretation notes. Predictions come from the unchanged
`predict_weekly` path; explanations are a separate layer and never alter a probability, threshold or risk band.

Prepare the explanations once after training (a training run also writes the background):

```bash
python -m product_discontinuation.shap_analysis    # ~10 min; trains nothing
```

### SHAP design

| Item | Implementation |
|---|---|
| Explainer | `shap.PermutationExplainer` on the complete fitted pipeline, 1,000 maximum evaluations, seed 42 |
| Why | Takes an explicit background, explains the exact pipeline, and needs no mapping of one-hot columns back to inputs |
| Explained output | Log-odds of the positive class `DiscontinuedTF = 1`, the scale on which the boosted model is additive |
| Inputs | The model's 20 original feature columns; categorical values are reversibly encoded only for SHAP's masker |
| Background | 200 rows of development editions 89-92, never edition 93 or 94, in `models/trained/shap_background_h13.parquet` |
| Sampling | Systematic sample of shuffled rows ordered by edition and outcome (seed 42), so strata are proportional |
| Local accuracy | Contributions must reproduce the model's probability on every call, otherwise an error is raised |
| Caching | Streamlit caches the explainer and each product's explanation; a first explanation takes about 1 s |

TreeExplainer loads this model but was not selected. Its path-dependent mode is exact but has no explicit background
and is kept as a cross-check; its interventional mode fails the additivity check for this model in shap 0.49.1 (the
root cause inside SHAP was not identified). Results are in `reports/results/shap_validation_h13.json`.

Limitations: permutation SHAP is a seeded Monte Carlo estimate, and independent masking can create unrealistic
combinations of correlated inputs (for example `forecast_to_actual` is `log_forecast - log_actuals`). Values depend
on the background, describe model behaviour rather than causes, and contributions below 0.05 log-odds are not
described to users. `predict.explain_prediction` is a separate one-feature-at-a-time sensitivity check, not SHAP.
