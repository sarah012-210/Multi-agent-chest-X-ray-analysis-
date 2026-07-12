"""
RAG Agent: embeds the medical knowledge base with sentence-transformers,
indexes it with FAISS, and retrieves the most relevant knowledge chunks
for a given query (typically the list of flagged conditions from the
Image Agent, or a follow-up question from the Chat Agent).
"""
import json
import os
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, KNOWLEDGE_BASE_JSON, TOP_K_RETRIEVAL


class RAGAgent:
    def __init__(self, knowledge_base_path: str = KNOWLEDGE_BASE_JSON,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME):
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        self.texts = [entry["text"] for entry in self.knowledge_base]
        self.conditions = [entry["condition"] for entry in self.knowledge_base]

        self.embedder = SentenceTransformer(embedding_model_name)
        self._build_index()

    def _build_index(self):
        embeddings = self.embedder.encode(self.texts, convert_to_numpy=True,
                                           normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim (normalized)
        self.index.add(embeddings.astype(np.float32))

    def retrieve_by_query(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[dict]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True,
                                          normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "condition": self.conditions[idx],
                "text": self.texts[idx],
                "score": float(score),
            })
        return results

    def retrieve_for_conditions(self, condition_names: List[str]) -> List[dict]:
        """Direct lookup by exact condition name — used right after the
        Image Agent flags specific diseases, since we already know exactly
        which knowledge entries are relevant (no need for semantic search)."""
        results = []
        for name in condition_names:
            for entry in self.knowledge_base:
                if entry["condition"] == name:
                    results.append(entry)
                    break
        return results


if __name__ == "__main__":
    agent = RAGAgent()
    hits = agent.retrieve_by_query("fluid in the lungs from heart failure")
    for hit in hits:
        print(f"[{hit['score']:.3f}] {hit['condition']}: {hit['text'][:80]}...")
