# ingest.py — Piri Knowledge Base İndeksleme
"""
Dokümanları chunk'lara ayırır, embedding'lerini hesaplar
ve FAISS vektör store'a kaydeder.

AKIS Platform — Piri Engine

Kullanım:
    python ingest.py                              # Varsayılan
    python ingest.py --input ./my_docs            # Özel klasör
    python ingest.py --chunk-size 1024 --output ./vs  # Özel ayarlar
"""
import argparse
import time
from piri.chunker import load_all_documents, chunk_documents
from piri.embedder import Embedder
from piri.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Piri — Knowledge Base İndeksleme")
    parser.add_argument(
        "--input", "-i",
        default="knowledge_base",
        help="Doküman klasörü (varsayılan: knowledge_base)",
    )
    parser.add_argument(
        "--output", "-o",
        default="vector_store",
        help="Vektör store çıktı klasörü (varsayılan: vector_store)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk boyutu karakter cinsinden (varsayılan: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Chunk örtüşme boyutu (varsayılan: 64)",
    )
    args = parser.parse_args()

    start = time.time()

    print(f"\n{'='*60}")
    print(f"  Piri — Knowledge Base İndeksleme")
    print(f"  AKIS Platform")
    print(f"{'='*60}")
    print(f"  Kaynak: {args.input}")
    print(f"  Chunk boyutu: {args.chunk_size} karakter, Overlap: {args.chunk_overlap}")

    # 1. Dokümanları yükle
    documents = load_all_documents(args.input)
    print(f"\n[1/4] {len(documents)} doküman yüklendi:")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['content'])} karakter)")

    # 2. Chunk'la
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"\n[2/4] {len(chunks)} chunk oluşturuldu")
    for doc in documents:
        doc_chunks = [c for c in chunks if c["source"] == doc["source"]]
        print(f"  - {doc['source']}: {len(doc_chunks)} chunk")

    # 3. Embedding'leri hesapla
    print(f"\n[3/4] Embedding'ler hesaplanıyor...")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"  Embedding boyutu: {embeddings.shape}")

    # 4. Vektör store'a kaydet
    print(f"\n[4/4] Vektör store oluşturuluyor...")
    store = VectorStore(dimension=embedder.dimension)

    metadata = []
    for chunk in chunks:
        metadata.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
        })

    store.add(embeddings, metadata)
    store.save(args.output)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Piri — İndeksleme tamamlandı!")
    print(f"  Toplam chunk: {len(chunks)}")
    print(f"  Toplam süre: {elapsed:.1f} saniye")
    print(f"  Çıktı: {args.output}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
