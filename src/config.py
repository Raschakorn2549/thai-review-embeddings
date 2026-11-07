from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"

RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS_FILE = RESULTS_DIR / "metrics.csv"

REPORT_DIR = ROOT / "report"
REPORT_FIGURE_DIR = REPORT_DIR / "figures"

DATASET_URL = "https://raw.githubusercontent.com/wongnai/wongnai-corpus/master/review/review_dataset.zip"
RAW_FILE = RAW_DIR / "w_review_train.csv"

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15

STAR_TO_LABEL = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}
LABELS = ["negative", "neutral", "positive"]

# Fixed colors per class so every figure uses the same mapping
LABEL_COLORS = {
    "negative": "#2a78d6",
    "neutral": "#eb6834",
    "positive": "#1baf7a",
}
BAR_COLOR = "#2a78d6"

# Thai-capable font shipped with Windows, needed for word labels in figures
PLOT_FONT = "Leelawadee UI"
FIGURE_DPI = 200

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MAX_LENGTH = 512
EMBEDDING_BATCH_SIZE = 16

PCA_DIMS = [1024, 512, 256, 128, 64, 32, 16]
MODELS = ["logreg", "linear_svm", "knn", "lightgbm"]
