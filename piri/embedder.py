"""
Piri — Embedding Modülü
Sentence-Transformers ile metin → vektör dönüşümü.
all-MiniLM-L6-v2: 384 boyutlu, ~80MB, hızlı.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Hafif ama etkili embedding modeli
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Piri] Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Piri] Embedding boyutu: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Metin listesini vektörlere dönüştürür.
        Returns: (n_texts, dimension) boyutunda numpy array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # Cosine similarity için normalize
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Tek bir sorguyu vektöre dönüştürür."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)
