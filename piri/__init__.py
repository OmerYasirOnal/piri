"""
Piri — Lightweight LLM + RAG Engine by AKIS Platform (v2)

Bilgi denizinde harita çıkaran küçük ama güçlü yapay zeka motoru.

v2 İyileştirmeler:
- Multilingual embedding (Türkçe optimize)
- Cross-encoder reranking
- OpenAI API backend desteği
- Gelişmiş prompt engineering
- Post-processing pipeline
"""

__version__ = "2.0.0"
__author__ = "AKIS Platform"

from .engine import PiriEngine

__all__ = ["PiriEngine"]
