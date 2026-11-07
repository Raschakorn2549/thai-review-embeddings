"""Download, clean and split the Wongnai review dataset.

Run from the repo root:
    python -m src.data
"""
import io
import re
import urllib.request
import zipfile

import pandas as pd
from pythainlp.util import normalize
from sklearn.model_selection import train_test_split

from src import config

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Same non-digit character 4+ times in a row -> keep 3 ("มากกกกกก" -> "มากกก").
# Digits are skipped so prices like 1000 stay intact.
REPEAT_PATTERN = re.compile(r"([^\d\s])\1{3,}")
SPACE_PATTERN = re.compile(r"\s+")


def download_raw():
    if config.RAW_FILE.exists():
        print(f"Raw file already exists: {config.RAW_FILE}")
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset...")
    with urllib.request.urlopen(config.DATASET_URL) as response:
        zip_bytes = response.read()

    # The zip also has test_file.csv, but it has no labels, so only the train file is kept.
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open("w_review_train.csv") as src, open(config.RAW_FILE, "wb") as dst:
            dst.write(src.read())
    print(f"Saved {config.RAW_FILE}")


def load_raw():
    # No header, ";" separated, reviews can span multiple lines inside quotes
    return pd.read_csv(config.RAW_FILE, sep=";", header=None, names=["text", "star"])


def clean_text(text):
    text = URL_PATTERN.sub(" ", text)
    text = REPEAT_PATTERN.sub(r"\1\1\1", text)
    text = normalize(text)
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()


def build_dataset():
    df = load_raw()
    print(f"Loaded {len(df)} reviews")

    n_before = len(df)
    df = df.drop_duplicates(subset="text")
    print(f"Dropped {n_before - len(df)} duplicate reviews")

    df = df.rename(columns={"text": "raw_text"})
    cleaned = []
    for text in df["raw_text"]:
        cleaned.append(clean_text(text))
    df["text"] = cleaned

    empty = df["text"].str.len() == 0
    if empty.any():
        print(f"Dropped {empty.sum()} reviews that were empty after cleaning")
        df = df[~empty]

    df["label"] = df["star"].map(config.STAR_TO_LABEL)
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df[["id", "text", "raw_text", "star", "label"]]


def split_dataset(df):
    # Stratify on star (not label) so each split keeps the original star mix too
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["star"], random_state=config.SEED
    )
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["star"], random_state=config.SEED
    )
    return {"train": train, "val": val, "test": test}


def load_split(name):
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")


def main():
    download_raw()
    df = build_dataset()
    splits = split_dataset(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part = part.sort_values("id").reset_index(drop=True)
        part.to_parquet(config.PROCESSED_DIR / f"{name}.parquet", index=False)

    print()
    print(f"{'split':<6} {'rows':>6}  " + "  ".join(f"{label:>9}" for label in config.LABELS))
    for name, part in splits.items():
        shares = part["label"].value_counts(normalize=True)
        row = "  ".join(f"{shares.get(label, 0):>9.1%}" for label in config.LABELS)
        print(f"{name:<6} {len(part):>6}  {row}")


if __name__ == "__main__":
    main()
