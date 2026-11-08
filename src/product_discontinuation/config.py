"""Project-wide paths and constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "product_discontinuation"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CATALOGUE_PATH = RAW_DIR / "CatalogueDiscontinuation.csv"
PRODUCT_DETAILS_PATH = RAW_DIR / "ProductDetails.csv"
DATA_DICTIONARY_PATH = RAW_DIR / "DataDictionary.xlsx"

MODELS_DIR = PROJECT_ROOT / "models" / "trained"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
PREDICTION_LOG_PATH = DEPLOYMENT_DIR / "logs" / "prediction_log.csv"

RANDOM_STATE = 42

TARGET = "DiscontinuedTF"
GROUP_KEYS = ["CatEdition", "ProductKey"]

# Edition 94 only covers WeeksOut -24..-13 and every row is labelled True:
# its outcome is not yet known, so it is treated as an unlabelled live edition.
CENSORED_EDITIONS = (94,)

MAX_WEEKS_OUT = 24
HORIZONS = (24, 13, 6, 1)
PRIMARY_HORIZON = 13
TREND_LOOKBACK_WEEKS = 4

# Chronological split by catalogue edition (labelled editions 88-93). Edition 88 is
# the first edition in the data, so it has no prior-edition history; it is used only
# as history for edition 89 onwards, not as a training row.
DATA_FIRST_EDITION = 88
MODEL_FIRST_EDITION = 89
TEST_EDITION = 93
CV_TEST_EDITIONS = (90, 91, 92)

MODEL_VERSION = "1.0.0"
