"""
Piri — FAISS Vektör Store Modülü
Chunk embedding'lerini indeksler ve hızlı benzerlik araması yapar.
"""
import json
import os
import numpy as np
import faiss
from typing import List, Dict


class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Inner Product (normalize edilmiş vektörlerle = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = []  # chunk_id → metadata mapping

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Embedding'leri ve metadata'ları indekse ekler.

        Args:
            embeddings: (n, dimension) float32 numpy array
            metadata_list: Her embedding için metadata dict listesi
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embedding ve metadata sayısı eşleşmiyor")

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)
        print(f"[Piri] İndekse {len(embeddings)} chunk eklendi. Toplam: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        En benzer chunk'ları bulur.

        Args:
            query_embedding: (1, dimension) float32 numpy array
            top_k: Kaç sonuç döndürülecek
            score_threshold: Minimum benzerlik skoru (0-1)

        Returns:
            [{"text": ..., "source": ..., "score": ..., ...}, ...]
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results

    def save(self, directory: str):
        """İndeksi ve metadata'yı diske kaydeder."""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[Piri] Vektör store kaydedildi: {directory} ({self.index.ntotal} chunk)")

    def load(self, directory: str) -> bool:
        """İndeksi ve metadata'yı diskten yükler."""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print(f"[Piri] Vektör store bulunamadı: {directory}")
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.dimension = self.index.d
        print(f"[Piri] Vektör store yüklendi: {self.index.ntotal} chunk")
        return True

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
