"""Train every (feature, dims, model) combination and append the results to metrics.csv.

Run from the repo root:
    python -m src.train                      # everything
    python -m src.train --part embeddings --models knn
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import config, data, evaluate, features


def load_labels():
    labels = {}
    for split in config.SPLITS:
        labels[split] = data.load_split(split)["label"].to_numpy()
    return labels


def fit_and_select(model_name, x_train, y_train, x_val, y_val):
    """Fit every grid point, keep the one with the best macro-F1 on val.

    The winning estimator is returned as fitted, so the test score comes from a
    model that never saw val or test during training.
    """
    best = None
    for params in evaluate.grid_points(model_name):
        model = evaluate.make_model(model_name, params)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_sec = time.perf_counter() - started

        val_f1 = evaluate.score(y_val, model.predict(x_val))["macro_f1"]
        print(f"    {params} val macro-F1 {val_f1:.4f} ({train_sec:.1f}s)")
        if best is None or val_f1 > best["val_f1"]:
            best = {"model": model, "params": params, "train_sec": train_sec, "val_f1": val_f1}
    return best


def run_one(feature, dims, model_name, x_train, y_train, x_val, y_val, x_test, y_test,
            explained_var, best_so_far):
    print(f"  {model_name}")
    best = fit_and_select(model_name, x_train, y_train, x_val, y_val)

    started = time.perf_counter()
    y_pred = best["model"].predict(x_test)
    predict_sec = time.perf_counter() - started

    metrics = evaluate.score(y_test, y_pred)
    row = {
        "feature": feature,
        "dims": dims,
        "model": model_name,
        "best_params": best["params"],
        "train_sec": best["train_sec"],
        "predict_sec": predict_sec,
        "vector_mb": evaluate.vector_mb(len(y_test), dims),
        "explained_var": explained_var,
    }
    row.update(metrics)
    evaluate.append_metrics(row)
    print(f"    test macro-F1 {metrics['macro_f1']:.4f}  acc {metrics['accuracy']:.4f}")

    if metrics["macro_f1"] > best_so_far:
        save_predictions(y_test, y_pred)
        return metrics["macro_f1"]
    return best_so_far


def save_predictions(y_true, y_pred):
    """Keep the test predictions of the best config for the error analysis."""
    test = data.load_split("test")
    frame = pd.DataFrame({"id": test["id"], "y_true": y_true, "y_pred": y_pred})
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(config.PREDICTIONS_FILE, index=False)


def best_f1_so_far():
    if not config.METRICS_FILE.exists():
        return -1.0
    return float(pd.read_csv(config.METRICS_FILE)["macro_f1"].max())


def run_tfidf(labels, model_names, done):
    model_names = [name for name in model_names if name in ("logreg", "linear_svm")]
    if not model_names:
        return

    print("TF-IDF baseline")
    train_tokens = features.load_tokens("train")
    vectorizer = features.build_tfidf(train_tokens)
    x_train = vectorizer.transform(train_tokens)
    x_val = vectorizer.transform(features.load_tokens("val"))
    x_test = vectorizer.transform(features.load_tokens("test"))
    dims = x_train.shape[1]
    print(f"  {dims:,} features")

    best_so_far = best_f1_so_far()
    for model_name in model_names:
        if ("tfidf", dims, model_name) in done:
            print(f"  {model_name} already done, skipping")
            continue
        best_so_far = run_one(
            "tfidf", dims, model_name,
            x_train, labels["train"], x_val, labels["val"], x_test, labels["test"],
            explained_var="", best_so_far=best_so_far,
        )


def fit_pca(x_train):
    """One full PCA on train. Every smaller dims is a slice of these components."""
    print(f"Fitting PCA with {config.EMBEDDING_DIMS} components on train...")
    started = time.perf_counter()
    pca = PCA(n_components=config.EMBEDDING_DIMS, random_state=config.SEED)
    pca.fit(x_train)
    print(f"  done in {time.perf_counter() - started:.0f}s")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EXPLAINED_VARIANCE_FILE, pca.explained_variance_ratio_)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for threshold in (0.90, 0.95):
        needed = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  {threshold:.0%} of variance needs {needed} components")
    return pca


def project(pca, x, dims):
    return (x - pca.mean_) @ pca.components_[:dims].T


def run_embeddings(labels, model_names, dims_list, done):
    raw = {}
    for split in config.SPLITS:
        raw[split] = features.load_embeddings(split)

    pca = fit_pca(raw["train"])
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    best_so_far = best_f1_so_far()

    for dims in dims_list:
        pending = [name for name in model_names if (config.EMBEDDING_NAME, dims, name) not in done]
        if not pending:
            print(f"\ndims {dims}: already done, skipping")
            continue

        print(f"\ndims {dims}")
        if dims == config.EMBEDDING_DIMS:
            # Full size means no reduction at all, so the raw vectors are used as is
            parts = raw
            explained_var = 1.0
        else:
            parts = {split: project(pca, raw[split], dims) for split in config.SPLITS}
            explained_var = float(cumulative[dims - 1])
            print(f"  explained variance {explained_var:.4f}")

        for model_name in pending:
            best_so_far = run_one(
                config.EMBEDDING_NAME, dims, model_name,
                parts["train"], labels["train"], parts["val"], labels["val"],
                parts["test"], labels["test"],
                explained_var=explained_var, best_so_far=best_so_far,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["tfidf", "embeddings", "all"], default="all")
    parser.add_argument("--models", default=",".join(config.MODELS))
    parser.add_argument("--dims", default=",".join(str(d) for d in config.PCA_DIMS))
    args = parser.parse_args()

    model_names = args.models.split(",")
    dims_list = [int(d) for d in args.dims.split(",")]

    labels = load_labels()
    done = evaluate.done_runs()

    if args.part in ("tfidf", "all"):
        run_tfidf(labels, model_names, done)
    if args.part in ("embeddings", "all"):
        run_embeddings(labels, model_names, dims_list, done)


if __name__ == "__main__":
    main()
