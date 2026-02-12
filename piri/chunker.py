"""
Piri — Doküman Chunking Modülü
Metinleri anlamlı, örtüşen (overlapping) parçalara ayırır.
"""
import os
import re
from typing import List, Dict


def load_document(file_path: str) -> str:
    """Tek bir dosyayı okur."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_all_documents(directory: str) -> List[Dict[str, str]]:
    """
    Bir klasördeki tüm .txt ve .md dosyalarını yükler.
    Her doküman: {"source": dosya_adı, "content": metin}
    """
    documents = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.endswith((".txt", ".md")):
                fpath = os.path.join(root, fname)
                content = load_document(fpath)
                if content.strip():
                    documents.append({
                        "source": fname,
                        "path": fpath,
                        "content": content,
                    })
    return documents


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    respect_sentences: bool = True,
) -> List[str]:
    """
    Metni parçalara ayırır.

    Args:
        text: Kaynak metin
        chunk_size: Her chunk'ın maksimum karakter sayısı
        chunk_overlap: Chunk'lar arası örtüşme (bağlam kaybını önler)
        respect_sentences: True ise cümle sınırlarına saygı gösterir
    """
    if not text.strip():
        return []

    if respect_sentences:
        # Cümle bazlı bölme (. ! ? ve satır sonu)
        sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        sentences = [text]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Eğer tek cümle bile chunk_size'dan büyükse, zorla böl
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Uzun cümleyi karakter bazlı böl
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                chunks.append(sentence[i:i + chunk_size].strip())
            continue

        # Mevcut chunk'a sığıyor mu?
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Overlap: önceki chunk'ın son kısmını al
            if chunk_overlap > 0 and current_chunk:
                overlap_text = current_chunk[-chunk_overlap:]
                # Kelime sınırına kadar geri git
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
    """
    Doküman listesini chunk'lara ayırır.
    Her chunk: {"text": parça, "source": dosya_adı, "chunk_id": idx}
    """
    all_chunks = []
    global_id = 0

    for doc in documents:
        text_chunks = chunk_text(
            doc["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": global_id,
                "text": chunk,
                "source": doc["source"],
                "doc_chunk_idx": i,
            })
            global_id += 1

    return all_chunks
