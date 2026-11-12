"""Build the two feature sets: TF-IDF tokens and bge-m3 embeddings.

Run from the repo root:
    python -m src.features
"""
import time

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, data


def thai_tokens(text):
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def tokens_path(split):
    return config.PROCESSED_DIR / f"tokens_{split}.parquet"


def build_tokens(split):
    """Tokenize once and cache. newmm on 40k reviews takes minutes."""
    path = tokens_path(split)
    if path.exists():
        return pd.read_parquet(path)

    df = data.load_split(split)
    joined = []
    for text in df["text"]:
        joined.append(" ".join(thai_tokens(text)))

    # Thai has no word spaces, so a space is safe as a token separator and lets
    # TfidfVectorizer handle the ngrams itself with a plain str.split tokenizer.
    out = pd.DataFrame({"id": df["id"], "tokens": joined})
    out.to_parquet(path, index=False)
    return out


def load_tokens(split):
    df = build_tokens(split)
    expected = data.load_split(split)["id"].to_numpy()
    if not np.array_equal(df["id"].to_numpy(), expected):
        raise ValueError(f"Token cache for {split} is out of sync with the parquet split")
    return df["tokens"].tolist()


def build_tfidf(train_tokens):
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES,
    )
    vectorizer.fit(train_tokens)
    return vectorizer


def embedding_path(split):
    return config.EMBEDDING_DIR / f"{config.EMBEDDING_NAME}_{split}.npy"


def id_path(split):
    return config.EMBEDDING_DIR / f"{config.EMBEDDING_NAME}_{split}_ids.npy"


def load_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config.EMBEDDING_MODEL, model_kwargs={"torch_dtype": "float16"}
    )
    model.max_seq_length = config.EMBEDDING_MAX_LENGTH
    return model


def encode(model, texts, batch_size=None):
    import torch

    if batch_size is None:
        batch_size = config.EMBEDDING_BATCH_SIZE

    # encode() already sorts by length internally and restores the original
    # order, so short reviews do not get padded up to the longest one.
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    except torch.cuda.OutOfMemoryError:
        fallback = config.EMBEDDING_FALLBACK_BATCH_SIZE
        if batch_size <= fallback:
            raise
        print(f"Out of VRAM at batch {batch_size}, retrying at {fallback}")
        torch.cuda.empty_cache()
        vectors = model.encode(
            texts,
            batch_size=fallback,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    return vectors.astype(np.float32)


def embed_split(model, split):
    if embedding_path(split).exists():
        print(f"{split}: embeddings already on disk, skipping")
        return

    df = data.load_split(split)
    started = time.perf_counter()
    vectors = encode(model, df["text"].tolist())
    elapsed = time.perf_counter() - started

    config.EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path(split), vectors)
    np.save(id_path(split), df["id"].to_numpy())
    print(f"{split}: {vectors.shape} in {elapsed / 60:.1f} min")


def load_embeddings(split):
    vectors = np.load(embedding_path(split))
    ids = np.load(id_path(split))
    expected = data.load_split(split)["id"].to_numpy()
    if not np.array_equal(ids, expected):
        raise ValueError(f"Embedding ids for {split} do not match the parquet split")
    return vectors


def estimate_runtime(model):
    """Time a small batch so the full run time is known before committing to it."""
    sample = data.load_split("train")["text"].head(200).tolist()
    started = time.perf_counter()
    model.encode(sample, batch_size=config.EMBEDDING_BATCH_SIZE, normalize_embeddings=True)
    elapsed = time.perf_counter() - started

    total_rows = sum(len(data.load_split(split)) for split in config.SPLITS)
    estimate = elapsed / len(sample) * total_rows
    print(f"200 reviews in {elapsed:.1f}s -> about {estimate / 60:.0f} min for all {total_rows:,}")


def sanity_check():
    for split in config.SPLITS:
        vectors = load_embeddings(split)
        norms = np.linalg.norm(vectors, axis=1)
        print(
            f"{split:<6} shape={vectors.shape} nan={int(np.isnan(vectors).sum())} "
            f"norm min={norms.min():.4f} max={norms.max():.4f}"
        )

    # A review should be closest to something that shares its sentiment, otherwise
    # the vectors carry nothing useful for the classifiers.
    train = data.load_split("train")
    val = data.load_split("val")
    train_vectors = load_embeddings("train")
    val_vectors = load_embeddings("val")

    print()
    negatives = val.index[val["label"] == "negative"][:3]
    for i in negatives:
        similarities = train_vectors @ val_vectors[i]
        nearest = int(np.argmax(similarities))
        print(f"cos={similarities[nearest]:.3f}  query({val.loc[i, 'label']}): {val.loc[i, 'text'][:70]}")
        print(f"            nearest({train.iloc[nearest]['label']}): {train.iloc[nearest]['text'][:70]}")


def main():
    for split in config.SPLITS:
        started = time.perf_counter()
        build_tokens(split)
        print(f"{split}: tokens ready in {time.perf_counter() - started:.0f}s")

    model = load_model()
    estimate_runtime(model)
    for split in config.SPLITS:
        embed_split(model, split)

    print()
    sanity_check()


if __name__ == "__main__":
    main()
