"""Fetch bge-m3 from ModelScope into data/models/bge-m3.

For networks where huggingface.co is blocked. BAAI publishes the same model on
ModelScope, but as pytorch_model.bin rather than safetensors. src/features.py
loads data/models/bge-m3 automatically once this has run.

    python tools/fetch_bge_m3.py
"""
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

BASE = "https://www.modelscope.cn/api/v1/models/BAAI/bge-m3"
REVISION = "master"

# Everything SentenceTransformer needs to build the model from a local folder.
# 1_Pooling/config.json sits in a subfolder, so it never shows up in the
# top-level listing and has to be fetched by path.
REQUIRED = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
]


def list_files():
    response = requests.get(f"{BASE}/repo/files", params={"Revision": REVISION}, timeout=60)
    response.raise_for_status()
    files = response.json()["Data"]["Files"]
    return {item["Path"]: item for item in files}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


ATTEMPTS = 10


def fetch_range(path, target, expected_size, start):
    """Pull one stretch of a file, appending from `start`. Returns bytes on disk."""
    headers = {"Range": f"bytes={start}-"} if start else {}
    big = expected_size > 10e6

    with requests.get(f"{BASE}/repo", params={"Revision": REVISION, "FilePath": path},
                      headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()

        # A server that ignores Range answers 200 and resends from byte zero
        resuming = response.status_code == 206
        mode = "ab" if resuming else "wb"
        written = start if resuming else 0

        with open(target, mode) as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if big:
                    percent = written / expected_size * 100 if expected_size else 0
                    print(f"\r  get    {path}  {written / 1e6:7.0f} / {expected_size / 1e6:.0f} MB"
                          f"  {percent:5.1f}%", end="", flush=True)
    return written


def download(path, target, expected_size):
    """Download with resume. Large files over this link drop part way through."""
    target.parent.mkdir(parents=True, exist_ok=True)
    big = expected_size > 10e6

    for attempt in range(1, ATTEMPTS + 1):
        have = target.stat().st_size if target.exists() else 0
        if expected_size and have == expected_size:
            if attempt == 1:
                print(f"  have   {path}")
            else:
                print(f"\n  done   {path}")
            return
        if expected_size and have > expected_size:
            # Longer than advertised means a bad earlier write, so start over
            target.unlink()
            have = 0

        try:
            written = fetch_range(path, target, expected_size, have)
            if not expected_size:
                print(f"  get    {path}  ({written:,} bytes)")
                return
        except (requests.RequestException, OSError) as error:
            if attempt == ATTEMPTS:
                raise
            if big:
                print()
            print(f"  retry  {path} ({attempt}/{ATTEMPTS}): {type(error).__name__}")

    raise RuntimeError(f"Gave up on {path} after {ATTEMPTS} attempts")


def main():
    target_dir = config.LOCAL_MODEL_DIR
    print(f"Target: {target_dir}\n")

    listing = list_files()
    unverified = []

    for path in REQUIRED:
        item = listing.get(path, {})
        download(path, target_dir / path, item.get("Size", 0))

    print("\nVerifying checksums...")
    for path in REQUIRED:
        item = listing.get(path)
        local = target_dir / path
        expected = item.get("Sha256") if item else None
        if not expected:
            unverified.append(path)
            print(f"  ?      {path}  (no sha256 published, {local.stat().st_size:,} bytes)")
            continue
        actual = sha256_of(local)
        if actual != expected:
            print(f"  FAIL   {path}\n         expected {expected}\n         got      {actual}")
            return 1
        print(f"  ok     {path}")

    if unverified:
        print("\nNot checksum-verified (not listed by the API):")
        for path in unverified:
            print(f"  {path}")

    print("\nAll listed files verified. Next: python -m src.features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
