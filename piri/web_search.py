"""
Piri — Web Search Modülü v3 (Wikipedia + DuckDuckGo Hibrit)

Arama stratejisi:
1. Wikipedia API (Türkçe) — güvenilir, hızlı, CAPTCHA yok, zengin içerik
2. DuckDuckGo — genel web sonuçları (Wikipedia'da bulunmazsa)

Her iki kaynak da birleştirilerek Piri'nin bilgi tabanına eklenir.
"""
import re
import requests
from typing import List, Dict
from datetime import datetime
from urllib.parse import quote


WIKI_UA = "PiriBot/3.0 (AKIS Platform; https://github.com/AKISPlatform/piri)"
REQUEST_TIMEOUT = 10

# ─── Sorgu Genişletme (Query Expansion) ──────────────────────

# Yaygın Türkçe kısaltmalar ve eş anlamlılar
ABBREVIATIONS = {
    "fsm": "Fatih Sultan Mehmet",
    "ysm": "Yavuz Sultan Selim",
    "ata": "Mustafa Kemal Atatürk",
    "ataturk": "Mustafa Kemal Atatürk",
    "tc": "Türkiye Cumhuriyeti",
    "ab": "Avrupa Birliği",
    "nato": "NATO Kuzey Atlantik Antlaşması Örgütü",
    "tbmm": "Türkiye Büyük Millet Meclisi",
    "osm": "Osmanlı İmparatorluğu",
    "ist": "İstanbul",
    "ank": "Ankara",
    "izm": "İzmir",
}


def expand_query(query: str) -> str:
    """
    Kısaltmaları ve belirsiz terimleri genişletir.
    'fsm kaç yaşında öldü' → 'Fatih Sultan Mehmet kaç yaşında öldü'
    """
    words = query.split()
    expanded = []
    changed = False

    for w in words:
        low = w.lower().strip('?.,!:;')
        if low in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[low])
            changed = True
        else:
            expanded.append(w)

    result = " ".join(expanded)
    if changed:
        print(f"[Piri] Sorgu genişletildi: '{query}' → '{result}'")
    return result


# ─── Wikipedia API ────────────────────────────────────────────

def search_wikipedia(
    query: str,
    max_results: int = 3,
    lang: str = "tr",
) -> List[Dict]:
    """
    Wikipedia API ile arama yapar ve sayfa içeriklerini çeker.
    Tamamen ücretsiz, API key gerektirmez, CAPTCHA yok.
    """
    headers = {"User-Agent": WIKI_UA}
    results = []

    try:
        # 1. Arama yap
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote(query)}"
            f"&format=json&srlimit={max_results}&utf8=1"
        )
        r = requests.get(search_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []

        data = r.json()
        search_results = data.get("query", {}).get("search", [])

        if not search_results:
            return []

        print(f"[Piri] Wikipedia araması: '{query}' → {len(search_results)} sonuç")

        # 2. Her sonucun tam içeriğini çek
        for sr in search_results:
            title = sr["title"]
            snippet = re.sub(r'<[^>]+>', '', sr.get("snippet", ""))

            # Sayfa extract (paragraf bazlı)
            content = _get_wiki_content(title, lang, headers)
            wiki_url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

            results.append({
                "title": title,
                "url": wiki_url,
                "body": content or snippet,
                "source": f"{lang}.wikipedia.org",
            })

        return results

    except Exception as e:
        print(f"[Piri] Wikipedia hatası: {e}")
        return []


def _get_wiki_content(title: str, lang: str, headers: dict, max_chars: int = 3000) -> str:
    """Wikipedia sayfasının metin içeriğini çeker."""
    try:
        url = (
            f"https://{lang}.wikipedia.org/w/api.php"
            f"?action=query&titles={quote(title)}"
            f"&prop=extracts&explaintext=1&format=json&utf8=1"
            f"&exsectionformat=plain"
        )
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return ""

        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            extract = page.get("extract", "")
            if extract:
                # Max karakter sınırı, cümle sınırında kes
                if len(extract) > max_chars:
                    cut = extract[:max_chars]
                    last = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
                    if last > max_chars * 0.5:
                        extract = cut[:last + 1]
                    else:
                        extract = cut
                return extract

    except Exception as e:
        print(f"[Piri] Wiki sayfa hatası ({title}): {e}")

    return ""


# ─── DuckDuckGo Fallback ─────────────────────────────────────

def search_ddg(
    query: str,
    max_results: int = 5,
) -> List[Dict]:
    """DuckDuckGo ile web araması (fallback)."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="tr-tr", max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                    "source": _extract_domain(r.get("href", "")),
                })

        print(f"[Piri] DuckDuckGo: '{query}' → {len(results)} sonuç")
        return results

    except Exception as e:
        print(f"[Piri] DuckDuckGo hatası: {e}")
        return []


# ─── Birleşik Arama ──────────────────────────────────────────

def search_web(
    query: str,
    max_results: int = 5,
    lang: str = "tr",
) -> List[Dict]:
    """
    Hibrit arama: Wikipedia + DuckDuckGo
    Kısaltmaları otomatik genişletir, Wikipedia her zaman önce denenir.
    """
    # Sorguyu genişlet (kısaltmalar çöz)
    query = expand_query(query)

    all_results = []
    seen_urls = set()

    # 1. Wikipedia (her zaman dene — en iyi kaynak)
    wiki_results = search_wikipedia(query, max_results=min(3, max_results), lang=lang)
    for r in wiki_results:
        if r["url"] not in seen_urls:
            all_results.append(r)
            seen_urls.add(r["url"])

    # 2. DuckDuckGo (ek sonuçlar)
    remaining = max_results - len(all_results)
    if remaining > 0:
        ddg_results = search_ddg(query, max_results=remaining + 2)
        for r in ddg_results:
            if r["url"] not in seen_urls and len(all_results) < max_results:
                all_results.append(r)
                seen_urls.add(r["url"])

    print(f"[Piri] Toplam: {len(all_results)} sonuç (Wiki: {len(wiki_results)}, DDG: {len(all_results) - len(wiki_results)})")
    return all_results


def search_and_compile(
    query: str,
    max_results: int = 5,
    lang: str = "tr",
) -> Dict:
    """
    Ara → Sayfa içeriklerini çek → Bilgi tabanına eklenebilir formatta derle.
    """
    results = search_web(query, max_results=max_results, lang=lang)

    if not results:
        return {
            "query": query,
            "results": [],
            "compiled_text": "",
            "source_name": "",
            "timestamp": datetime.now().isoformat(),
        }

    # Sonuçları temiz metin olarak derle (metadata YOK — sadece içerik)
    parts = []
    for r in results:
        if r["title"]:
            parts.append(r["title"])
        if r["body"]:
            parts.append(r["body"])
        parts.append("")  # paragraf ayracı

    compiled_text = "\n\n".join(parts).strip()

    # Kaynak adı
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
