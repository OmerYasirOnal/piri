"""
Piri — Cross-Encoder Reranker Modülü
İlk retrieval sonuçlarını daha akıllı bir modelle yeniden sıralar.

Strateji: Bi-encoder ile geniş aday listesi getir (top-20),
sonra cross-encoder ile en kaliteli 3-5 chunk'ı seç.
Bu iki aşamalı yaklaşım retrieval doğruluğunu %25-40 artırır.
"""
from typing import List, Dict, Optional
import numpy as np


class Reranker:
    """Cross-encoder tabanlı reranker."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """
        Args:
            model_name: Multilingual cross-encoder modeli.
                - mmarco-mMiniLMv2-L12-H384-v1: 100+ dil, Türkçe iyi
                - ms-marco-MiniLM-L6-v2: Sadece İngilizce ama çok hızlı
        """
        print(f"[Piri] Reranker yükleniyor: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print(f"[Piri] Reranker hazır.")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Chunk'ları cross-encoder ile yeniden sıralar.

        Args:
            query: Kullanıcı sorusu
            chunks: Bi-encoder'dan gelen aday chunk listesi
            top_k: Kaç chunk döndürülecek

        Returns:
            Yeniden sıralanmış chunk listesi (en alakalı önce)
        """
        if not chunks:
            return []

        # Cross-encoder için (query, passage) çiftleri oluştur
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Cross-encoder skorları (daha doğru ama daha yavaş)
        ce_scores = self.model.predict(pairs)

        # Orijinal bi-encoder skorlarıyla birleştir (hybrid scoring)
        for i, chunk in enumerate(chunks):
            chunk["ce_score"] = float(ce_scores[i])
            bi_score = chunk.get("score", 0.0)
            # Hybrid: %70 cross-encoder + %30 bi-encoder
            chunk["hybrid_score"] = 0.7 * float(ce_scores[i]) + 0.3 * bi_score

        # Hybrid skora göre sırala
        ranked = sorted(chunks, key=lambda x: x["hybrid_score"], reverse=True)

        # Top-k döndür, score'u hybrid_score ile güncelle
        result = ranked[:top_k]
        for chunk in result:
            chunk["score"] = chunk["hybrid_score"]

        return result


class DummyReranker:
    """Reranker modeli yüklenemezse kullanılan fallback."""

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        # Sadece orijinal skorlara göre sırala
        ranked = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[:top_k]


def create_reranker(model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> "Reranker":
    """Reranker oluştur, yüklenemezse DummyReranker döndür."""
    try:
        return Reranker(model_name)
    except Exception as e:
        print(f"[Piri] Reranker yüklenemedi ({e}), fallback kullanılıyor.")
        return DummyReranker()
