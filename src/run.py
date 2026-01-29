import os
import time
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .data import load_phrasebank_df, train_val_test_split


LABEL_ID_TO_NAME = {0: "negative", 1: "neutral", 2: "positive"}


def build_rag_index(train_texts, train_labels):
    """
    Simple retrieval: embed train texts -> KNN index.
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embedder.encode(train_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(embs)

    return embedder, knn, np.array(train_labels), np.array(train_texts)


def rag_augment(texts, embedder, knn, train_labels, train_texts, k=3):
    """
    For each input text, retrieve k nearest train samples and append short evidence.
    """
    q = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    distances, idx = knn.kneighbors(q, n_neighbors=k)

    augmented = []
    for i, t in enumerate(texts):
        hits = idx[i]
        evidence = []
        for j in hits:
            lab = LABEL_ID_TO_NAME[int(train_labels[j])]
            # keep evidence short to avoid very long sequences
            ev = train_texts[j]
            if len(ev) > 180:
                ev = ev[:180] + "..."
            evidence.append(f"[{lab}] {ev}")
        aug = t + "\n\nEvidence:\n" + "\n".join(evidence)
        augmented.append(aug)
    return augmented


@torch.no_grad()
def eval_model(model_name, test_texts, test_labels, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3-class head expected; we use pretrained sentiment fine-tuned models to stand out
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    preds = []
    latencies = []

    bs = 16
    for i in tqdm(range(0, len(test_texts), bs), desc=f"Eval {model_name}"):
        batch = test_texts[i : i + bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

        start = time.perf_counter()
        out = model(**enc)
        torch.cuda.synchronize() if device.startswith("cuda") and torch.cuda.is_available() else None
        end = time.perf_counter()

        logits = out.logits.detach().cpu().numpy()
        batch_preds = logits.argmax(axis=1).tolist()
        preds.extend(batch_preds)

        # latency per sample
        latencies.append((end - start) / len(batch))

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    latency_ms = float(np.mean(latencies) * 1000.0)
    return acc, f1, latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--agree", type=str, default="75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rag_k", type=int, default=0, help="0 disables retrieval augmentation. Use 3 to enable.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load + split
    df = load_phrasebank_df(agree=args.agree)
    train_df, val_df, test_df = train_val_test_split(df, seed=args.seed)

    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    # Optional RAG augmentation
    if args.rag_k and args.rag_k > 0:
        embedder, knn, train_labels, train_texts = build_rag_index(train_df["text"].tolist(), train_df["label"].tolist())
        test_texts = rag_augment(test_texts, embedder, knn, train_labels, train_texts, k=args.rag_k)

    # Models that stand out (finance domain)
    models = [
        ("ProsusAI/finbert", "FinBERT"),
        ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT-SST2"),
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa-Twitter"),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Test size:", len(test_texts))

    rows = []
    for model_id, short_name in models:
        acc, f1, latency_ms = eval_model(model_id, test_texts, test_labels, device=device)
        rows.append(
            {
                "model_id": model_id,
                "model_name": short_name,
                "agree": args.agree,
                "rag_k": args.rag_k,
                "accuracy": acc,
                "macro_f1": f1,
                "latency_ms_per_sample": latency_ms,
            }
        )

    out_csv = os.path.join(args.output_dir, "results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
