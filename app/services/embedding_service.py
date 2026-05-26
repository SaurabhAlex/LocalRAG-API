from __future__ import annotations

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return emb.astype("float32")

