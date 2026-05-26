from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VectorSearchResult:
    chunk_indices: List[int]
    scores: List[float]


class VectorStoreService:
    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir
        os.makedirs(self.faiss_dir, exist_ok=True)

    def _doc_dir(self, document_id: str) -> str:
        return os.path.join(self.faiss_dir, document_id)

    def _faiss_path(self, document_id: str) -> str:
        return os.path.join(self._doc_dir(document_id), "index.faiss")

    def _meta_path(self, document_id: str) -> str:
        return os.path.join(self._doc_dir(document_id), "metadata.json")

    def index_exists(self, document_id: str) -> bool:
        return os.path.exists(self._faiss_path(document_id)) and os.path.exists(self._meta_path(document_id))

    def create_or_update_index(self, *, document_id: str, embeddings: np.ndarray, chunks: List[str]) -> None:
        import faiss

        os.makedirs(self._doc_dir(document_id), exist_ok=True)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks length must match embeddings")

        # Use cosine similarity via IndexFlatIP with normalized vectors.
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, self._faiss_path(document_id))
        with open(self._meta_path(document_id), "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f)

    def search(self, *, document_id: str, query_embedding: np.ndarray, top_k: int) -> Tuple[VectorSearchResult, List[str]]:
        import faiss

        if not self.index_exists(document_id):
            raise FileNotFoundError("index not found")

        index = faiss.read_index(self._faiss_path(document_id))
        with open(self._meta_path(document_id), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        chunks: List[str] = metadata["chunks"]

        q = query_embedding.astype("float32")
        faiss.normalize_L2(q)

        scores, indices = index.search(q, top_k)
        chunk_indices = indices[0].tolist()
        scores_list = scores[0].tolist()

        # Filter -1 (FAISS can return -1 if not enough vectors)
        filtered = [(i, s) for i, s in zip(chunk_indices, scores_list) if i != -1]
        chunk_indices = [i for i, _ in filtered]
        scores_list = [s for _, s in filtered]

        retrieved_chunks = [chunks[i] for i in chunk_indices]
        return VectorSearchResult(chunk_indices=chunk_indices, scores=scores_list), retrieved_chunks

