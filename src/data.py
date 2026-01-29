import re
import zipfile
from typing import Tuple

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.model_selection import train_test_split

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _parse_phrasebank_lines(lines):
    sents, labels = [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Many PhraseBank files are: label@sentence
        if "@" in line:
            a, b = line.split("@", 1)
            label = a.strip().lower()
            sent = b.strip()
        # Some variants are tab separated: label \t sentence
        elif "\t" in line:
            a, b = line.split("\t", 1)
            label = a.strip().lower()
            sent = b.strip()
        else:
            continue

        if label in LABEL2ID:
            labels.append(LABEL2ID[label])
            sents.append(sent)

    df = pd.DataFrame({"sentence": sents, "label": labels})
    return df


def load_phrasebank_75agree(seed: int = 42) -> pd.DataFrame:
    """
    Robust loader:
    - looks inside takala/financial_phrasebank dataset repo
    - downloads either a ZIP containing PhraseBank or a 75Agree txt directly
    - parses into sentence/label DataFrame
    """
    repo_id = "takala/financial_phrasebank"
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    # 1) try to find a .zip (most robust)
    zip_candidates = [f for f in files if f.lower().endswith(".zip")]
    if zip_candidates:
        zip_file = zip_candidates[0]
        zip_path = hf_hub_download(repo_id=repo_id, filename=zip_file, repo_type="dataset")
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            # look for 75Agree txt inside zip
            candidates = [n for n in names if re.search(r"75.*agree", n, re.IGNORECASE) and n.lower().endswith(".txt")]
            if not candidates:
                raise RuntimeError(f"ZIP found but no 75Agree txt inside. Example files: {names[:30]}")
            target = candidates[0]
            raw = z.read(target)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            lines = text.splitlines()
        df = _parse_phrasebank_lines(lines)
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 2) fallback: search for any txt that looks like 75Agree
    txt_candidates = [f for f in files if f.lower().endswith(".txt") and re.search(r"75.*agree", f, re.IGNORECASE)]
    if not txt_candidates:
        raise RuntimeError(f"Could not locate PhraseBank 75Agree. Files include: {files[:50]}")

    txt_path = hf_hub_download(repo_id=repo_id, filename=txt_candidates[0], repo_type="dataset")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    df = _parse_phrasebank_lines(lines)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_df(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
