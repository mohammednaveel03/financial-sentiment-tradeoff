import numpy as np
from sentence_transformers import SentenceTransformer


def build_knn_index(train_texts):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = embedder.encode(list(train_texts), normalize_embeddings=True, show_progress_bar=True)
    return embedder, np.array(embs, dtype=np.float32)


def retrieve_context(embedder, train_embs, train_texts, query_texts, k=3):
    q_embs = embedder.encode(list(query_texts), normalize_embeddings=True, show_progress_bar=False)
    q_embs = np.array(q_embs, dtype=np.float32)

    # cosine sim because normalized embeddings
    sims = q_embs @ train_embs.T
    idx = np.argsort(-sims, axis=1)[:, :k]

    contexts = []
    for i in range(len(query_texts)):
        retrieved = [train_texts[j] for j in idx[i]]
        ctx = " [SEP] ".join(retrieved)
        contexts.append(ctx)
    return contexts


def make_rag_inputs(train_df, test_df, k=3):
    train_texts = list(train_df["sentence"])
    embedder, train_embs = build_knn_index(train_texts)

    test_texts = list(test_df["sentence"])
    ctx = retrieve_context(embedder, train_embs, train_texts, test_texts, k=k)

    rag_texts = [f"{test_texts[i]} [SEP] {ctx[i]}" for i in range(len(test_texts))]
    return rag_texts
