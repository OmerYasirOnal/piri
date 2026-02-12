<p align="center">
  <h1 align="center">Piri</h1>
  <p align="center">
    <strong>Lightweight LLM + RAG Engine by AKIS Platform</strong>
  </p>
  <p align="center">
    <em>Bilgi denizinde harita çıkaran küçük ama güçlü yapay zeka motoru</em>
  </p>
</p>

<p align="center">
  <a href="#kurulum">Kurulum</a> •
  <a href="#hızlı-başlangıç">Hızlı Başlangıç</a> •
  <a href="#api-referansı">API</a> •
  <a href="#mimari">Mimari</a> •
  <a href="#değerlendirme">Değerlendirme</a> •
  <a href="#katkıda-bulunma">Katkıda Bulunma</a>
</p>

---

## Piri Nedir?

**Piri**, [AKIS Platform](https://github.com/akis-platform) ekosisteminin açık kaynak yapay zeka motorudur. Adını, bilinmeyen sularda harita çıkaran efsanevi Osmanlı kartografı **Piri Reis**'ten alır — tıpkı onun gibi, Piri de bilgi tabanlarında arama yaparak soruların cevabını bulur.

### Öne Çıkan Özellikler

| Özellik | Açıklama |
|---------|----------|
| **Hafif LLM** | Qwen2.5-0.5B-Instruct (~500M parametre), CPU üzerinde çalışır |
| **RAG Pipeline** | Retrieve → Augment → Generate; FAISS vektör araması |
| **5 Boyutlu Kalite Ölçümü** | Faithfulness, Relevance, Context Precision, Coverage, Coherence |
| **Türkçe Odaklı** | Türkçe system prompt'lar, Türkçe doküman desteği |
| **Modüler Tasarım** | Her bileşen bağımsız kullanılabilir |
| **Kolay Fine-tuning** | ChatML formatında özel veriyle eğitim desteği |
| **REST API** | FastAPI ile production-ready endpoint'ler |

---

## Kurulum

### Gereksinimler

- Python 3.9+
- 2GB+ RAM (model yükleme için)
- İnternet bağlantısı (ilk model indirme için)

### Pip ile Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/akis-platform/piri.git
cd piri

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

---

## Hızlı Başlangıç

### 1. Knowledge Base İndeksle

```bash
# Varsayılan knowledge_base/ klasörünü indeksle
python ingest.py

# Özel klasör belirt
python ingest.py --input ./my_docs --chunk-size 512
```

### 2. API'yi Başlat

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Sorgula

```bash
# RAG ile soru-cevap
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG pipeline nasıl çalışır?"}'

# Serbest metin üretimi
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Yapay zeka nedir?"}'
```

### 4. Kalite Değerlendirmesi

```bash
# Tüm test setini değerlendir
python evaluate.py

# Tek soru değerlendir
python evaluate.py --question "Transformer mimarisi nedir?"

# JSON formatında çıktı
python evaluate.py --json
```

---

## API Referansı

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | API durumu ve bilgileri |
| `/generate` | POST | Serbest metin üretimi (RAG olmadan) |
| `/rag/query` | POST | RAG ile soru-cevap |
| `/rag/search` | POST | Semantik arama (üretim yapmaz) |
| `/rag/ingest` | POST | Yeni doküman indeksle |
| `/rag/evaluate` | POST | Sorgu + 5 boyutlu kalite değerlendirme |
| `/rag/stats` | GET | RAG sistem istatistikleri |

### Örnek: RAG Sorgusu

```json
// POST /rag/query
{
  "question": "LoRA fine-tuning nasıl çalışır?",
  "top_k": 3,
  "max_new_tokens": 200,
  "temperature": 0.7
}
```

**Yanıt:**
```json
{
  "answer": "LoRA, büyük dil modellerinin ağırlık matrislerine...",
  "sources": ["llm_training_techniques.md"],
  "num_sources": 3,
  "retrieval_scores": [
    {"source": "llm_training_techniques.md", "score": 0.8234}
  ]
}
```

---

## Mimari

```
┌─────────────────────────────────────────────────────────┐
│                     Piri API (FastAPI)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ /generate│   │/rag/query│   │  /rag/evaluate   │   │
│  └────┬─────┘   └────┬─────┘   └────────┬─────────┘   │
│       │              │                   │              │
│       ▼              ▼                   ▼              │
│  ┌─────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │   LLM   │   │   RAG    │   │   Evaluator      │   │
│  │Generator│   │  Engine  │   │ (5-dim scoring)  │   │
│  └─────────┘   └────┬─────┘   └──────────────────┘   │
│                      │                                  │
│         ┌────────────┼────────────┐                    │
│         ▼            ▼            ▼                    │
│    ┌─────────┐ ┌──────────┐ ┌──────────┐             │
│    │Retriever│ │ Embedder │ │ Chunker  │             │
│    └────┬────┘ └──────────┘ └──────────┘             │
│         │                                              │
│         ▼                                              │
│    ┌──────────┐                                        │
│    │  FAISS   │                                        │
│    │VectorDB  │                                        │
│    └──────────┘                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Bileşenler

| Modül | Dosya | Açıklama |
|-------|-------|----------|
| **Engine** | `piri/engine.py` | RAG orkestrasyon — Retrieve → Augment → Generate |
| **Chunker** | `piri/chunker.py` | Doküman bölümleme (cümle-aware, overlap) |
| **Embedder** | `piri/embedder.py` | all-MiniLM-L6-v2 ile metin→vektör (384d) |
| **Retriever** | `piri/retriever.py` | Semantik arama + bağlam oluşturma |
| **VectorStore** | `piri/vector_store.py` | FAISS IndexFlatIP, persist/load |
| **Evaluator** | `piri/evaluator.py` | 5 boyutlu kalite değerlendirme |

---

## Değerlendirme

Piri, RAG yanıtlarını 5 boyutta değerlendirir:

| Metrik | Ağırlık | Açıklama |
|--------|---------|----------|
| **Faithfulness** | %30 | Cevap bağlama ne kadar sadık? (halüsinasyon tespiti) |
| **Relevance** | %25 | Cevap soruyu ne kadar yanıtlıyor? |
| **Context Precision** | %15 | Doğru chunk'lar mı getirildi? |
| **Coverage** | %15 | Bağlamdaki bilgi cevaba ne kadar yansımış? |
| **Coherence** | %15 | Cevap tutarlı ve okunabilir mi? |

```
  GENEL SKOR: 0.72/1.00  [GOOD]
  ████████████████░░░░ 72%

  Faithfulness         0.85    30%    FAITHFUL
  Relevance            0.78    25%    RELEVANT
  Context Precision    0.65    15%    GOOD_RETRIEVAL
  Coverage             0.58    15%    COMPREHENSIVE
  Coherence            0.61    15%    COHERENT
```

---

## Fine-tuning

Piri, özel veriyle fine-tune edilebilir:

```bash
# ChatML formatında eğitim verisi hazırla (data/rag_train.jsonl)
# Her satır: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

# Eğitimi başlat
python train.py --epochs 2 --batch-size 1 --lr 1e-5

# Sadece modeli indir (eğitim yapmadan)
python train.py --skip-train
```

---

## AKIS Platform Entegrasyonu

Piri, AKIS Platform'un dördüncü agent'ıdır:

| Agent | Görev | Teknoloji |
|-------|-------|-----------|
| **Scribe** | Doküman üretimi | TypeScript + MCP |
| **Trace** | Hata analizi | TypeScript + MCP |
| **Proto** | Prototipleme | TypeScript + MCP |
| **Piri** | Bilgi motoru + RAG | Python + FAISS |

Piri, diğer agent'lardan bağımsız olarak da kullanılabilir — tamamen açık kaynak ve standalone çalışır.

---

## Proje Yapısı

```
piri/
├── main.py              # FastAPI uygulaması
├── train.py             # Fine-tuning scripti
├── ingest.py            # Knowledge base indeksleme
├── evaluate.py          # Kalite değerlendirme
├── requirements.txt     # Python bağımlılıkları
├── pyproject.toml       # Paket yapılandırması
├── LICENSE              # MIT Lisansı
├── piri/                # Core paket
│   ├── __init__.py
│   ├── engine.py        # RAG Engine
│   ├── chunker.py       # Doküman chunking
│   ├── embedder.py      # Text embedding
│   ├── retriever.py     # Semantik arama
│   ├── vector_store.py  # FAISS vektör store
│   └── evaluator.py     # 5-boyutlu değerlendirme
├── knowledge_base/      # Örnek bilgi tabanı
│   └── ...
└── data/                # Eğitim verileri
    └── ...
```

---

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Detaylar için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: yeni özellik ekle'`)
4. Push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

---

## Lisans

MIT License — detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

<p align="center">
  <strong>Piri</strong> — AKIS Platform tarafından geliştirilmiştir.<br>
  <em>"Haritası olmayana her rüzgâr ters eser." — Piri Reis</em>
</p>
