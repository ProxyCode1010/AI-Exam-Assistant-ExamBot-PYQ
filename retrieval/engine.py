"""
STEP 3: retrieval/engine.py
Retrieval engine — given a topic string, returns top-k similar PYQs.
Used by the API. Can also be tested standalone.
Run: python -m retrieval.engine
"""

import os
import pickle
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "embeddings/faiss.index"
META_PATH  = "embeddings/metadata.pkl"
TOP_K      = int(os.getenv("TOP_K", "8"))


@dataclass
class RetrievedQuestion:
    question: str
    score: float    # cosine similarity 0-1


class RetrievalEngine:
    def __init__(self):
        print("[engine] Loading FAISS index…")
        self.index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

        self.questions: list[str] = meta["questions"]
        self.model_name: str      = meta["model"]

        print(f"[engine] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"[engine] Ready. {len(self.questions)} questions indexed.")

    def retrieve(self, topic: str, top_k: int = TOP_K) -> list[RetrievedQuestion]:
        """Embed topic and return top-k nearest PYQs by cosine similarity."""
        query_vec = self.model.encode(
            [topic],
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedQuestion(
                question=self.questions[idx],
                score=round(float(score), 4),
            ))

        return results


# ── Singleton (loaded once per process) ────────────────────────────────────────
_engine: RetrievalEngine | None = None


def get_engine() -> RetrievalEngine:
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
    return _engine


# ── Standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = get_engine()
    topic = "Hibernate"
    print(f"\nTop {TOP_K} PYQs for topic: '{topic}'\n")
    for i, r in enumerate(engine.retrieve(topic), 1):
        print(f"  {i:2}. [{r.score:.3f}] {r.question}")