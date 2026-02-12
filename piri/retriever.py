"""
Piri — Retriever Modülü (Gelişmiş)
İki aşamalı retrieval: Bi-encoder → Cross-encoder reranking.
"""
from typing import List, Dict, Optional
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker=None,
    ):
        self.embedder = embedder
        self.store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.10,
        use_reranking: bool = True,
    ) -> List[Dict]:
        """
        İki aşamalı retrieval:
        1. Bi-encoder ile geniş aday listesi getir (top_k * 4)
        2. Cross-encoder ile rerank et, en iyi top_k'yı döndür

        Args:
            query: Kullanıcı sorusu
            top_k: Kaç chunk getirilecek
            score_threshold: Minimum benzerlik
            use_reranking: Reranking kullanılsın mı
        """
        # Aşama 1: Geniş aday listesi (reranking için daha fazla getir)
        candidate_k = top_k * 4 if (self.reranker and use_reranking) else top_k
        candidate_k = min(candidate_k, self.store.total_chunks)

        query_embedding = self.embedder.embed_query(query)
        candidates = self.store.search(
            query_embedding,
            top_k=candidate_k,
            score_threshold=score_threshold,
        )

        if not candidates:
            return []

        # Aşama 2: Cross-encoder reranking
        if self.reranker and use_reranking and len(candidates) > top_k:
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ) -> Dict:
        """
        Sorgu için zengin bağlam oluşturur.

        Returns:
            {
                "context": birleştirilmiş metin,
                "sources": kaynak listesi,
                "chunks": ham chunk listesi,
                "num_retrieved": bulunan chunk sayısı
            }
        """
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return {
                "context": "",
                "sources": [],
                "chunks": [],
                "num_retrieved": 0,
            }

        # Bağlamı oluştur - en yüksek skordan başla
        context_parts = []
        sources = set()
        total_chars = 0

        for chunk in chunks:
            text = chunk["text"]
            if total_chars + len(text) > max_context_chars:
                remaining = max_context_chars - total_chars
                if remaining > 50:
                    context_parts.append(text[:remaining] + "...")
                break
            context_parts.append(text)
            sources.add(chunk["source"])
            total_chars += len(text)

        context = "\n\n---\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sorted(sources),
            "chunks": chunks,
            "num_retrieved": len(chunks),
        }
