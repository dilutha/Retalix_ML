import numpy as np
import pandas as pd
import pytest

from product_discontinuation import config
from product_discontinuation.features import (
    HISTORY_FEATURES, STATUS_FEATURES, all_feature_columns, build_features, edition_history,
)

HORIZON = 13


@pytest.fixture(scope="module")
def features(weekly):
    return build_features(weekly, HORIZON)


def test_one_row_per_product_edition_at_cutoff(weekly, features):
    expected = weekly[weekly["WeeksOut"] == -HORIZON].groupby(config.GROUP_KEYS).ngroups
    assert len(features) == expected
    assert not features.duplicated(config.GROUP_KEYS).any()


def test_future_rows_do_not_change_features(weekly, features):
    """Core leakage test: altering any row after the cutoff must not affect features."""
    tampered = weekly.copy()
    future = tampered["WeeksOut"] > -HORIZON
    tampered.loc[future, ["ActualsPerWeek", "ForecastPerWeek", "SalePriceIncVAT"]] *= 50
    tampered.loc[future, "Status"] = "RO"
    rebuilt = build_features(tampered, HORIZON)
    pd.testing.assert_frame_equal(features, rebuilt)


def test_removing_future_rows_gives_identical_features(weekly, features):
    truncated = weekly[weekly["WeeksOut"] <= -HORIZON]
    history = edition_history(weekly)
    pd.testing.assert_frame_equal(features, build_features(truncated, HORIZON, history=history))


def test_current_edition_target_not_used_as_feature(weekly, features):
    flipped = weekly.assign(DiscontinuedTF=~weekly["DiscontinuedTF"])
    history = edition_history(weekly)
    rebuilt = build_features(flipped, HORIZON, history=history)
    cols = all_feature_columns(HORIZON, include_status=True)
    pd.testing.assert_frame_equal(features[cols], rebuilt[cols])


def test_history_uses_previous_edition_only(weekly, features):
    row = features[(features["CatEdition"] == 90) & (features["in_previous_edition"] == 1)].iloc[0]
    prev = weekly[(weekly["CatEdition"] == 89) & (weekly["ProductKey"] == row["ProductKey"])]
    assert row["previous_discontinued"] == float(prev["DiscontinuedTF"].iloc[0])


def test_first_edition_history_is_missing(features):
    first = features[features["CatEdition"] == config.DATA_FIRST_EDITION]
    assert first[HISTORY_FEATURES].isna().all().all()


def test_live_edition_label_masked_in_history(weekly):
    history = edition_history(weekly)
    assert history.loc[history["CatEdition"].isin(config.CENSORED_EDITIONS), config.TARGET].isna().all()


def test_default_features_exclude_status_and_identifiers():
    cols = all_feature_columns(HORIZON)
    assert not set(STATUS_FEATURES) & set(cols)
    assert "ProductKey" not in cols and "CatEdition" not in cols and config.TARGET not in cols
    assert set(STATUS_FEATURES) <= set(all_feature_columns(HORIZON, include_status=True))


def test_horizon_specific_columns():
    assert "actuals_trend" not in all_feature_columns(24)
    assert "forecast_volatility" not in all_feature_columns(24)
    assert "actuals_trend" in all_feature_columns(1)


@pytest.mark.parametrize("horizon", [0, 25, 1.5, -3])
def test_invalid_horizon_rejected(weekly, horizon):
    with pytest.raises(ValueError):
        build_features(weekly, horizon)


def test_zero_price_treated_as_missing(weekly):
    zeroed = weekly.copy()
    zeroed.loc[zeroed["ProductKey"] == 1000, "SalePriceIncVAT"] = 0.0
    feats = build_features(zeroed, HORIZON)
    target = feats[feats["ProductKey"] == 1000]
    assert target["log_price"].isna().all() and (target["price_missing"] == 1).all()


def test_trend_matches_manual_calculation(weekly, features):
    row = features.iloc[0]
    rows = weekly[(weekly["CatEdition"] == row["CatEdition"]) & (weekly["ProductKey"] == row["ProductKey"])]
    now = rows.loc[rows["WeeksOut"] == -HORIZON, "ActualsPerWeek"].iloc[0]
    lag = rows.loc[rows["WeeksOut"] == -HORIZON - config.TREND_LOOKBACK_WEEKS, "ActualsPerWeek"].iloc[0]
    assert row["actuals_trend"] == pytest.approx(np.log1p(now) - np.log1p(lag))


def test_missing_input_column_raises(weekly):
    with pytest.raises(ValueError, match="Supplier"):
        build_features(weekly.drop(columns=["Supplier"]), HORIZON)
