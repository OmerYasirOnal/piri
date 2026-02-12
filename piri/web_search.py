"""
Piri — Web Search Modülü
DuckDuckGo üzerinden ücretsiz web araması yapar,
sonuçları Piri'nin bilgi tabanına ekler.

API key gerektirmez.
"""
import re
from typing import List, Dict, Optional
from datetime import datetime


def search_web(
    query: str,
    max_results: int = 5,
    region: str = "tr-tr",
) -> List[Dict]:
    """
    DuckDuckGo ile web araması yapar.

    Args:
        query: Arama sorgusu
        max_results: Maksimum sonuç sayısı
        region: Bölge (tr-tr = Türkiye)

    Returns:
        [{title, url, body, source}] listesi
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                    "source": _extract_domain(r.get("href", "")),
                })

        print(f"[Piri] Web araması: '{query}' → {len(results)} sonuç bulundu.")
        return results

    except Exception as e:
        print(f"[Piri] Web araması hatası: {e}")
        return []


def search_and_compile(
    query: str,
    max_results: int = 5,
    region: str = "tr-tr",
) -> Dict:
    """
    Web araması yapar ve sonuçları bilgi tabanına eklenebilir
    formatta derler.

    Returns:
        {
            "query": arama sorgusu,
            "results": sonuç listesi,
            "compiled_text": birleştirilmiş metin (ingest için),
            "source_name": kaynak adı,
            "timestamp": zaman damgası
        }
    """
    results = search_web(query, max_results=max_results, region=region)

    if not results:
        return {
            "query": query,
            "results": [],
            "compiled_text": "",
            "source_name": "",
            "timestamp": datetime.now().isoformat(),
        }

    # Sonuçları tek bir metin olarak derle
    parts = [f"# Web Araması: {query}\n"]
    parts.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for i, r in enumerate(results, 1):
        parts.append(f"## [{i}] {r['title']}")
        parts.append(f"Kaynak: {r['source']} ({r['url']})")
        if r["body"]:
            parts.append(r["body"])
        parts.append("")

    compiled_text = "\n".join(parts)

    # Kaynak adı oluştur
    safe_query = re.sub(r'[^\w\s-]', '', query)[:40].strip().replace(' ', '_')
    source_name = f"web_{safe_query}.md"

    return {
        "query": query,
        "results": results,
        "compiled_text": compiled_text,
        "source_name": source_name,
        "timestamp": datetime.now().isoformat(),
        "result_count": len(results),
    }


def _extract_domain(url: str) -> str:
    """URL'den domain adını çıkarır."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url[:50]
