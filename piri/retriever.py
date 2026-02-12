"""
Piri — Retriever Modülü
Sorguya en uygun chunk'ları bulur ve bağlam oluşturur.
"""
from typing import List, Dict
from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.15,
    ) -> List[Dict]:
        """
        Sorguya en benzer chunk'ları döndürür.

        Args:
            query: Kullanıcı sorusu
            top_k: Kaç chunk getirilecek
            score_threshold: Minimum benzerlik (düşük = geniş, yüksek = dar)
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.store.search(
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        return results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        max_context_chars: int = 2000,
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
                # Kalan alanı doldur
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
