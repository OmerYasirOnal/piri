"""
Piri — Embedding Modülü
Multilingual embedding ile metin → vektör dönüşümü.
intfloat/multilingual-e5-small: 384 boyutlu, 100+ dil, Türkçe optimize.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Multilingual E5 — Türkçe dahil 100+ dilde güçlü
DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Piri] Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.is_e5 = "e5" in model_name.lower()
        print(f"[Piri] Embedding boyutu: {self.dimension}")

    def _prepare_texts(self, texts: List[str], prefix: str = "passage") -> List[str]:
        """
        E5 modelleri 'query: ' ve 'passage: ' prefix'i gerektirir.
        Bu prefix'ler retrieval kalitesini önemli ölçüde artırır.
        """
        if self.is_e5:
            return [f"{prefix}: {t}" for t in texts]
        return texts

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Metin listesini vektörlere dönüştürür (doküman/passage embedding).
        Returns: (n_texts, dimension) boyutunda numpy array
        """
        prepared = self._prepare_texts(texts, prefix="passage")
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Tek bir sorguyu vektöre dönüştürür (query embedding)."""
        prepared = self._prepare_texts([query], prefix="query")
        embedding = self.model.encode(
            prepared,
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
