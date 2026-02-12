"""
Piri — Lightweight LLM + RAG Engine by AKIS Platform (v3)

Bilgi denizinde harita çıkaran küçük ama güçlü yapay zeka motoru.

v3 İyileştirmeler:
- Relevance threshold + pozitif skorlu kaynak filtreleme
- Akıllı web-search fallback (Wikipedia + DuckDuckGo)
- Metadata/artifact agresif temizlik
- Premium dark UI
"""

__version__ = "3.0.0"
__author__ = "AKIS Platform"

from .engine import PiriEngine

__all__ = ["PiriEngine"]
