"""Point-in-time feature engineering at the product-edition level.

A prediction is made for every product listed in an edition at a fixed horizon
(weeks before the range refresh). Only weekly rows with WeeksOut <= -horizon are
used, plus outcomes of earlier editions, which are known by then.
"""
import numpy as np
import pandas as pd

from . import config
from .data import check_columns

CATEGORICAL_FEATURES = ["Supplier", "HierarchyLevel1", "HierarchyLevel2", "DIorDOM"]
BINARY_FEATURES = ["Seasonal", "SpringSummer", "zero_actuals", "price_missing"]
HISTORY_FEATURES = ["in_previous_edition", "previous_discontinued", "prior_editions"]
STATUS_FEATURES = ["status_ro", "status_ro_share", "status_changes"]
BASE_NUMERIC_FEATURES = ["log_price", "log_actuals", "log_forecast", "forecast_to_actual", "weeks_in_edition"]
VOLATILITY_FEATURES = ["forecast_volatility", "actuals_volatility"]
TREND_FEATURES = ["actuals_trend", "forecast_trend"]

REQUIRED_COLUMNS = [
    "CatEdition", "ProductKey", "WeeksOut", "SpringSummer", "Status",
    "SalePriceIncVAT", "ForecastPerWeek", "ActualsPerWeek",
    "Supplier", "HierarchyLevel1", "HierarchyLevel2", "DIorDOM", "Seasonal",
]


def validate_horizon(horizon):
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= config.MAX_WEEKS_OUT:
        raise ValueError(f"horizon must be an integer in [1, {config.MAX_WEEKS_OUT}], got {horizon!r}")


def has_trend_window(horizon):
    return horizon + config.TREND_LOOKBACK_WEEKS <= config.MAX_WEEKS_OUT


FORECAST_FEATURES = ["log_forecast", "forecast_to_actual", "forecast_volatility", "forecast_trend"]


def feature_columns(horizon, include_status=False, exclude=()):
    """Return (numeric, categorical, binary) feature lists for a horizon."""
    validate_horizon(horizon)
    numeric = BASE_NUMERIC_FEATURES + HISTORY_FEATURES
    if horizon < config.MAX_WEEKS_OUT:
        numeric = numeric + VOLATILITY_FEATURES
    if has_trend_window(horizon):
        numeric = numeric + TREND_FEATURES
    binary = list(BINARY_FEATURES)
    if include_status:
        numeric = numeric + ["status_ro_share", "status_changes"]
        binary = binary + ["status_ro"]
    keep = lambda cols: [c for c in cols if c not in set(exclude)]
    return keep(numeric), keep(CATEGORICAL_FEATURES), keep(binary)


def all_feature_columns(horizon, include_status=False, exclude=()):
    numeric, categorical, binary = feature_columns(horizon, include_status, exclude)
    return numeric + categorical + binary


def _value_at(history, week, column):
    rows = history[history["WeeksOut"] == week]
    return rows.set_index(config.GROUP_KEYS)[column]


def edition_history(weekly):
    """Product presence and outcome per edition, used for prior-edition features."""
    cols = config.GROUP_KEYS + ([config.TARGET] if config.TARGET in weekly else [])
    groups = weekly[cols].drop_duplicates(config.GROUP_KEYS)
    groups[config.TARGET] = groups[config.TARGET].astype(float) if config.TARGET in groups else np.nan
    live = groups["CatEdition"].isin(config.CENSORED_EDITIONS)
    groups.loc[live, config.TARGET] = np.nan
    return groups


def _history_features(index, history):
    keys = index.to_frame(index=False)
    previous = history.rename(columns={config.TARGET: "previous_discontinued"}).assign(
        CatEdition=lambda d: d["CatEdition"] + 1, in_previous_edition=1.0
    )
    out = keys.merge(previous, on=config.GROUP_KEYS, how="left")
    out["in_previous_edition"] = out["in_previous_edition"].fillna(0.0)
    out["previous_discontinued"] = out["previous_discontinued"].astype(float)

    counts = history[["CatEdition", "ProductKey"]].merge(
        keys[["ProductKey"]].drop_duplicates(), on="ProductKey"
    )
    prior = keys.merge(counts.rename(columns={"CatEdition": "seen"}), on="ProductKey", how="left")
    prior = prior[prior["seen"] < prior["CatEdition"]].groupby(config.GROUP_KEYS).size()
    out = out.set_index(config.GROUP_KEYS)
    out["prior_editions"] = prior.reindex(out.index).fillna(0).astype(float)

    # The first edition in the data has no observable history.
    first = out.index.get_level_values("CatEdition") <= config.DATA_FIRST_EDITION
    out.loc[first, HISTORY_FEATURES] = np.nan
    return out[HISTORY_FEATURES]


def build_features(weekly, horizon, history=None):
    """Build one feature row per product-edition observed at WeeksOut == -horizon.

    `history` is the edition-level presence/outcome table; by default it is derived
    from `weekly`, which must then include the preceding edition.
    """
    validate_horizon(horizon)
    check_columns(weekly, REQUIRED_COLUMNS, "weekly data")
    cutoff = -horizon

    past = weekly[weekly["WeeksOut"] <= cutoff]
    current = past[past["WeeksOut"] == cutoff].set_index(config.GROUP_KEYS)
    if current.empty:
        raise ValueError(f"no products observed at WeeksOut = {cutoff}")
    if current.index.duplicated().any():
        raise ValueError("duplicate product-edition rows at the prediction week")
    past = past.merge(current.index.to_frame(index=False), on=config.GROUP_KEYS)

    # A price of 0 is not a valid retail price (682 groups are 0 throughout): treat as missing.
    price = current["SalePriceIncVAT"].where(current["SalePriceIncVAT"] > 0)
    feats = pd.DataFrame(index=current.index)
    feats["log_price"] = np.log(price)
    feats["price_missing"] = price.isna().astype(int)
    feats["log_actuals"] = np.log1p(current["ActualsPerWeek"])
    feats["log_forecast"] = np.log1p(current["ForecastPerWeek"])
    feats["forecast_to_actual"] = feats["log_forecast"] - feats["log_actuals"]
    feats["zero_actuals"] = (current["ActualsPerWeek"] == 0).astype(int)

    logs = past.assign(
        log_f=np.log1p(past["ForecastPerWeek"]), log_a=np.log1p(past["ActualsPerWeek"])
    ).groupby(config.GROUP_KEYS)
    agg = logs.agg(first_week=("WeeksOut", "min"), forecast_volatility=("log_f", "std"),
                   actuals_volatility=("log_a", "std"))
    feats["weeks_in_edition"] = cutoff - agg["first_week"] + 1
    feats["forecast_volatility"] = agg["forecast_volatility"]
    feats["actuals_volatility"] = agg["actuals_volatility"]

    if has_trend_window(horizon):
        lag_week = cutoff - config.TREND_LOOKBACK_WEEKS
        feats["actuals_trend"] = feats["log_actuals"] - np.log1p(_value_at(past, lag_week, "ActualsPerWeek"))
        feats["forecast_trend"] = feats["log_forecast"] - np.log1p(_value_at(past, lag_week, "ForecastPerWeek"))

    status = past[config.GROUP_KEYS + ["WeeksOut"]].assign(ro=past["Status"].eq("RO").astype(int))
    status = status.sort_values(config.GROUP_KEYS + ["WeeksOut"])
    status["change"] = status.groupby(config.GROUP_KEYS)["ro"].diff().abs()
    by_group = status.groupby(config.GROUP_KEYS)
    feats["status_ro"] = current["Status"].eq("RO").astype(int)
    feats["status_ro_share"] = by_group["ro"].mean()
    feats["status_changes"] = by_group["change"].sum()

    if history is None:
        history = edition_history(weekly)
    feats = feats.join(_history_features(feats.index, history))

    for col in CATEGORICAL_FEATURES:
        feats[col] = current[col].astype(str)
    feats["Seasonal"] = current["Seasonal"].astype(int)
    feats["SpringSummer"] = current["SpringSummer"].astype(int)
    if config.TARGET in current:
        feats[config.TARGET] = current[config.TARGET].astype(int)
    return feats.reset_index()
