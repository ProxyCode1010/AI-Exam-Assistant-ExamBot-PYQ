"""
STEP 2: embeddings/builder.py
Converts cleaned questions into embeddings and builds a FAISS index.
Run: python -m embeddings.builder
"""

import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QUESTIONS_PATH  = "data/questions_clean.json"
INDEX_PATH      = "embeddings/faiss.index"
META_PATH       = "embeddings/metadata.pkl"


def build_index():
    # ── Load questions ──────────────────────────────────────────────────────────
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions: list[str] = json.load(f)
    print(f"[builder] Loaded {len(questions)} questions")

    # ── Load model ──────────────────────────────────────────────────────────────
    print(f"[builder] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Embed ───────────────────────────────────────────────────────────────────
    print("[builder] Encoding questions (this may take ~30s first time)…")
    embeddings: np.ndarray = model.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    )
    print(f"[builder] Embedding shape: {embeddings.shape}")   # (N, 384)

    # ── FAISS index (inner-product = cosine on normalised vectors) ──────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    print(f"[builder] FAISS index total vectors: {index.ntotal}")

    # ── Save ────────────────────────────────────────────────────────────────────
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"questions": questions, "model": EMBEDDING_MODEL}, f)

    print(f"[builder] Saved index  → {INDEX_PATH}")
    print(f"[builder] Saved meta   → {META_PATH}")
    return index, questions, model


if __name__ == "__main__":
    build_index()