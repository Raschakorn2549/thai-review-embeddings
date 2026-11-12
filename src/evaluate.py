"""Model factory, metrics and the results CSV."""
import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src import config


def make_model(name, params):
    if name == "logreg":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            random_state=config.SEED,
        )
    if name == "linear_svm":
        return LinearSVC(
            C=params["C"],
            class_weight="balanced",
            random_state=config.SEED,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            metric="cosine",
            weights="distance",
            n_jobs=-1,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            num_leaves=params["num_leaves"],
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def grid_points(name):
    """Expand a grid dict into a list of param dicts. Every grid has one key."""
    grid = config.MODEL_GRIDS[name]
    key = list(grid)[0]
    points = []
    for value in grid[key]:
        points.append({key: value})
    return points


def score(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, labels=config.LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_neg": per_class[0],
        "f1_neu": per_class[1],
        "f1_pos": per_class[2],
    }


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=config.LABELS)


def vector_mb(n_rows, dims):
    return n_rows * dims * 4 / 1e6


def append_metrics(row):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["best_params"] = json.dumps(row["best_params"], sort_keys=True)
    frame = pd.DataFrame([row], columns=config.METRICS_COLUMNS)
    write_header = not config.METRICS_FILE.exists()
    frame.to_csv(config.METRICS_FILE, mode="a", header=write_header, index=False)


def done_runs():
    """(feature, dims, model) tuples already in the CSV, so a run can resume."""
    if not config.METRICS_FILE.exists():
        return set()
    frame = pd.read_csv(config.METRICS_FILE)
    done = set()
    for _, row in frame.iterrows():
        done.add((row["feature"], int(row["dims"]), row["model"]))
    return done
