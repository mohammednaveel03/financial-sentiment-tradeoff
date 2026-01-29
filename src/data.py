import os
import re
import zipfile
import urllib.request
from typing import Tuple

import pandas as pd


DATA_DIR = os.path.join("data")
ZIP_PATH = os.path.join(DATA_DIR, "FinancialPhraseBank-v1.0.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "FinancialPhraseBank-v1.0")

# Stable public mirror on GitHub (zip is inside the repo)
ZIP_URL = "https://github.com/neoyipeng2018/FinancialPhraseBank-v1.0/raw/main/FinancialPhraseBank-v1.0.zip"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_phrasebank(force: bool = False) -> str:
    """
    Downloads and extracts FinancialPhraseBank into ./data/.
    Returns extract directory.
    """
    _ensure_dir(DATA_DIR)

    if force and os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading dataset zip to: {ZIP_PATH}")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    if force and os.path.exists(EXTRACT_DIR):
        # remove extracted files
        for root, dirs, files in os.walk(EXTRACT_DIR, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(EXTRACT_DIR)

    if not os.path.exists(EXTRACT_DIR):
        _ensure_dir(EXTRACT_DIR)
        print(f"Extracting zip into: {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)

    return EXTRACT_DIR


def _find_sentences_file(extract_dir: str, agree: str = "75") -> str:
    """
    Finds Sentences_75Agree.txt (or other agree level) anywhere under extract_dir.
    """
    target = f"sentences_{agree}agree.txt".lower()

    candidates = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower() == target:
                return os.path.join(root, f)
            # keep fallback candidates
            if f.lower().startswith("sentences_") and f.lower().endswith("agree.txt"):
                candidates.append(os.path.join(root, f))

    if candidates:
        # If exact agree not found, pick first and warn
        print("WARNING: Exact agree file not found. Using:", candidates[0])
        return candidates[0]

    raise FileNotFoundError(f"Could not find any Sentences_*Agree.txt inside {extract_dir}")


def load_phrasebank_df(agree: str = "75") -> pd.DataFrame:
    """
    Returns DataFrame with columns: text, label (0=neg,1=neu,2=pos)
    """
    extract_dir = download_phrasebank(force=False)
    path = _find_sentences_file(extract_dir, agree=agree)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    # Robust parse: try split by '@' first, else by tab
    texts = []
    labels = []

    label_map = {"negative": 0, "neutral": 1, "positive": 2}

    for ln in lines:
        if "@" in ln:
            parts = ln.rsplit("@", 1)
        else:
            parts = re.split(r"\t+", ln)
            if len(parts) > 2:
                parts = [parts[0], parts[-1]]

        if len(parts) != 2:
            continue

        text = parts[0].strip()
        lab = parts[1].strip().lower()

        if lab not in label_map:
            continue

        texts.append(text)
        labels.append(label_map[lab])

    df = pd.DataFrame({"text": texts, "label": labels})
    if len(df) == 0:
        raise RuntimeError(
            f"Parsed dataset is EMPTY. File format may have changed. "
            f"Open {path} and check how labels are stored."
        )

    return df


def train_val_test_split(
    df: pd.DataFrame, seed: int = 42, test_size: float = 0.2, val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )

    # val_size is fraction of the ORIGINAL dataset; convert to fraction of train_df
    val_fraction_of_train = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_df, test_size=val_fraction_of_train, random_state=seed, stratify=train_df["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
