import pandas as pd
import pytest

from product_discontinuation import config, data
from product_discontinuation.data import DataValidationError


def test_load_catalogue_from_csv(tmp_path, synthetic):
    catalogue, _ = synthetic
    path = tmp_path / "cat.csv"
    catalogue.head(50).to_csv(path, index=False)
    loaded = data.load_catalogue(path)
    assert len(loaded) == 50
    assert loaded["DiscontinuedTF"].dtype == bool
    assert loaded["WeeksOut"].dtype == "int64"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_catalogue(tmp_path / "missing.csv")


def test_load_missing_column_raises(tmp_path, synthetic):
    catalogue, _ = synthetic
    path = tmp_path / "cat.csv"
    catalogue.drop(columns=["Status"]).head(5).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Status"):
        data.load_catalogue(path)


def test_validate_catalogue_passes_and_reports(synthetic):
    catalogue, _ = synthetic
    report = data.validate_catalogue(catalogue)
    assert report["inconsistent_target_groups"] == 0
    assert report["product_editions"] == catalogue.groupby(config.GROUP_KEYS).ngroups


@pytest.mark.parametrize("corrupt, message", [
    (lambda d: pd.concat([d, d.head(1)]), "duplicate"),
    (lambda d: d.assign(Status=d["Status"].where(d.index > 0, "XX")), "Status"),
    (lambda d: d.assign(WeeksOut=d["WeeksOut"].where(d.index > 0, 0)), "WeeksOut"),
    (lambda d: d.assign(ActualsPerWeek=d["ActualsPerWeek"].where(d.index > 0, -1.0)), "negative"),
    (lambda d: d.assign(SalePriceIncVAT=d["SalePriceIncVAT"].where(d.index > 0)), "missing"),
])
def test_validate_catalogue_rejects_corrupt_data(synthetic, corrupt, message):
    catalogue, _ = synthetic
    with pytest.raises(DataValidationError, match=message):
        data.validate_catalogue(corrupt(catalogue.copy()))


def test_inconsistent_target_detected(synthetic):
    catalogue, _ = synthetic
    bad = catalogue.copy()
    bad.loc[bad.index[0], "DiscontinuedTF"] = not bad.loc[bad.index[0], "DiscontinuedTF"]
    assert data.check_target_consistency(bad) == 1
    with pytest.raises(DataValidationError, match="target"):
        data.validate_catalogue(bad)


def test_scoring_validation_without_target(synthetic):
    catalogue, _ = synthetic
    report = data.validate_catalogue(catalogue.drop(columns=["DiscontinuedTF"]), require_target=False)
    assert "inconsistent_target_groups" not in report


def test_merge_is_many_to_one(synthetic):
    catalogue, products = synthetic
    merged = data.merge_product_details(catalogue, products)
    assert len(merged) == len(catalogue)
    assert merged["Supplier"].notna().all()


def test_merge_rejects_duplicate_product_details(synthetic):
    catalogue, products = synthetic
    with pytest.raises(pd.errors.MergeError):
        data.merge_product_details(catalogue, pd.concat([products, products.head(1)]))


def test_merge_rejects_unmatched_products(synthetic):
    catalogue, products = synthetic
    with pytest.raises(DataValidationError, match="no product details"):
        data.merge_product_details(catalogue, products.iloc[1:])


def test_validate_product_details_rejects_duplicates(synthetic):
    _, products = synthetic
    with pytest.raises(DataValidationError):
        data.validate_product_details(pd.concat([products, products.head(1)]))


def test_incomplete_edition_detected_and_split(synthetic):
    catalogue, _ = synthetic
    assert data.find_incomplete_editions(catalogue) == [94]
    labelled, live = data.split_labelled(catalogue)
    assert set(live["CatEdition"]) == {94}
    assert 94 not in set(labelled["CatEdition"])


@pytest.mark.full_data
def test_real_dataset_integrity():
    if not config.CATALOGUE_PATH.exists():
        pytest.skip("raw data not available")
    catalogue = data.load_catalogue()
    products = data.load_product_details()
    report = data.validate_catalogue(catalogue)
    data.validate_product_details(products)
    assert report["product_editions"] == 178905
    assert report["inconsistent_target_groups"] == 0
    assert data.find_incomplete_editions(catalogue) == list(config.CENSORED_EDITIONS)
    merged = data.merge_product_details(catalogue, products)
    assert len(merged) == len(catalogue)
