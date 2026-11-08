"""Loading, validation and cleaning of the raw Retalix datasets."""
from pathlib import Path

import pandas as pd

from . import config

CATALOGUE_DTYPES = {
    "CatEdition": "int64",
    "SpringSummer": "bool",
    "ProductKey": "int64",
    "WeeksOut": "int64",
    "Status": "object",
    "SalePriceIncVAT": "float64",
    "ForecastPerWeek": "float64",
    "ActualsPerWeek": "float64",
    "DiscontinuedTF": "bool",
}
PRODUCT_DTYPES = {
    "ProductKey": "int64",
    "Supplier": "int64",
    "HierarchyLevel1": "int64",
    "HierarchyLevel2": "int64",
    "DIorDOM": "object",
    "Seasonal": "bool",
}
VALID_STATUS = {"RI", "RO"}
VALID_SOURCING = {"DI", "DOM"}
NUMERIC_MEASURES = ["SalePriceIncVAT", "ForecastPerWeek", "ActualsPerWeek"]
WEEKLY_KEYS = config.GROUP_KEYS + ["WeeksOut"]


class DataValidationError(ValueError):
    """Raised when a dataset violates the expected schema or integrity rules."""


def _read_csv(path, dtypes, nrows=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = set(dtypes) - set(header)
    if missing:
        raise DataValidationError(f"{path.name} is missing columns: {sorted(missing)}")
    return pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, nrows=nrows)


def load_catalogue(path=None, nrows=None):
    return _read_csv(path or config.CATALOGUE_PATH, CATALOGUE_DTYPES, nrows)


def load_product_details(path=None):
    return _read_csv(path or config.PRODUCT_DETAILS_PATH, PRODUCT_DTYPES)


def check_columns(df, required, name="data"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing columns: {missing}")


def validate_catalogue(df, require_target=True):
    """Raise on hard integrity errors; return a dict of soft data-quality findings."""
    required = [c for c in CATALOGUE_DTYPES if require_target or c != config.TARGET]
    check_columns(df, required, "catalogue")
    if df.empty:
        raise DataValidationError("catalogue is empty")

    nulls = df[required].isna().sum()
    if nulls.any():
        raise DataValidationError(f"catalogue has missing values: {nulls[nulls > 0].to_dict()}")

    duplicates = int(df.duplicated(WEEKLY_KEYS).sum())
    if duplicates:
        raise DataValidationError(f"{duplicates} duplicate (CatEdition, ProductKey, WeeksOut) rows")

    bad_status = set(df["Status"].unique()) - VALID_STATUS
    if bad_status:
        raise DataValidationError(f"unexpected Status values: {sorted(bad_status)}")

    weeks = df["WeeksOut"]
    if (weeks >= 0).any() or (weeks < -config.MAX_WEEKS_OUT).any():
        raise DataValidationError(f"WeeksOut must lie in [-{config.MAX_WEEKS_OUT}, -1]")

    negatives = (df[NUMERIC_MEASURES] < 0).sum()
    if negatives.any():
        raise DataValidationError(f"negative values found: {negatives[negatives > 0].to_dict()}")

    report = {
        "rows": len(df),
        "products": int(df["ProductKey"].nunique()),
        "product_editions": int(df.groupby(config.GROUP_KEYS).ngroups),
        "editions": sorted(int(e) for e in df["CatEdition"].unique()),
        "zero_price_rows": int((df["SalePriceIncVAT"] == 0).sum()),
        "zero_actuals_rows": int((df["ActualsPerWeek"] == 0).sum()),
        "groups_with_week_gaps": count_groups_with_gaps(df),
    }
    if require_target:
        report["inconsistent_target_groups"] = check_target_consistency(df)
        if report["inconsistent_target_groups"]:
            raise DataValidationError("target varies within a product-edition group")
    return report


def validate_product_details(df):
    check_columns(df, list(PRODUCT_DTYPES), "product details")
    if df[list(PRODUCT_DTYPES)].isna().any().any():
        raise DataValidationError("product details contain missing values")
    if df["ProductKey"].duplicated().any():
        raise DataValidationError("ProductKey is not unique in product details")
    bad = set(df["DIorDOM"].unique()) - VALID_SOURCING
    if bad:
        raise DataValidationError(f"unexpected DIorDOM values: {sorted(bad)}")


def check_target_consistency(df):
    """Number of product-edition groups whose target is not constant."""
    return int((df.groupby(config.GROUP_KEYS)[config.TARGET].nunique() > 1).sum())


def count_groups_with_gaps(df):
    g = df.groupby(config.GROUP_KEYS)["WeeksOut"].agg(["min", "max", "size"])
    return int(((g["max"] - g["min"] + 1) != g["size"]).sum())


def find_incomplete_editions(df):
    """Editions that never reach WeeksOut = -1 or whose labels are all one class."""
    summary = df.groupby("CatEdition").agg(
        last_week=("WeeksOut", "max"), target_rate=(config.TARGET, "mean")
    )
    flagged = (summary["last_week"] < -1) | summary["target_rate"].isin([0.0, 1.0])
    return sorted(int(e) for e in summary.index[flagged])


def merge_product_details(catalogue, products):
    """Many-to-one join; fails if any catalogue product has no metadata."""
    merged = catalogue.merge(products, on="ProductKey", how="left", validate="many_to_one")
    if len(merged) != len(catalogue):
        raise DataValidationError("join changed the number of rows")
    unmatched = merged["Supplier"].isna().sum()
    if unmatched:
        raise DataValidationError(f"{unmatched} catalogue rows have no product details")
    return merged


def split_labelled(df):
    """Separate labelled editions from censored (live) editions."""
    live = df["CatEdition"].isin(config.CENSORED_EDITIONS)
    return df[~live].reset_index(drop=True), df[live].reset_index(drop=True)


def load_dataset(nrows=None, validate=True):
    """Load, validate and join the weekly dataset.

    Zero prices are kept here and treated as missing during feature engineering.
    """
    catalogue = load_catalogue(nrows=nrows)
    products = load_product_details()
    if validate:
        validate_catalogue(catalogue)
        validate_product_details(products)
    return merge_product_details(catalogue, products)
