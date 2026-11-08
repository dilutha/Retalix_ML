import numpy as np
import pandas as pd
import pytest

from product_discontinuation import config


def make_weekly(n_products=120, editions=range(88, 95), seed=0):
    """Synthetic weekly data with the real schema and a learnable signal.

    Discontinued products have lower sales; edition 94 is censored like the real data.
    """
    rng = np.random.default_rng(seed)
    products = pd.DataFrame({
        "ProductKey": np.arange(1000, 1000 + n_products),
        "Supplier": rng.choice([10, 11, 12, 13], n_products),
        "HierarchyLevel1": rng.choice([100, 101, 102], n_products),
        "HierarchyLevel2": rng.choice([200, 201], n_products),
        "DIorDOM": rng.choice(["DI", "DOM"], n_products),
        "Seasonal": rng.random(n_products) < 0.1,
    })
    rows = []
    for edition in editions:
        last_week = -13 if edition in config.CENSORED_EDITIONS else -1
        for key in products["ProductKey"]:
            if rng.random() < 0.15:
                continue
            discontinued = edition in config.CENSORED_EDITIONS or rng.random() < 0.35
            base = rng.uniform(5, 40) if discontinued else rng.uniform(30, 120)
            price = round(rng.uniform(5, 200), 2)
            for week in range(-24, last_week + 1):
                rows.append({
                    "CatEdition": edition,
                    "SpringSummer": edition % 2 == 1,
                    "ProductKey": key,
                    "WeeksOut": week,
                    "Status": "RO" if (discontinued and rng.random() < 0.8) else "RI",
                    "SalePriceIncVAT": price,
                    "ForecastPerWeek": max(base * rng.normal(1, 0.1), 0),
                    "ActualsPerWeek": max(base * rng.normal(1, 0.1), 0),
                    "DiscontinuedTF": discontinued,
                })
    catalogue = pd.DataFrame(rows)
    return catalogue, products


@pytest.fixture(scope="session")
def synthetic():
    return make_weekly()


@pytest.fixture(scope="session")
def weekly(synthetic):
    catalogue, products = synthetic
    return catalogue.merge(products, on="ProductKey", how="left")
