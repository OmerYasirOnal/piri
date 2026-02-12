# RAG (Retrieval-Augmented Generation) Literatür Araştırması

## RAG Nedir?

RAG (Retrieval-Augmented Generation), Lewis et al. (2020) tarafından tanıtılan bir tekniktir. Temel fikir basittir: üretici modele (generator) cevap üretmeden önce ilgili bilgileri bir bilgi tabanından çekip bağlam olarak vermek. Bu, modelin halüsinasyonunu azaltır ve güncel bilgiye erişimini sağlar.

RAG pipeline'ı üç ana adımdan oluşur:
1. Indexing (İndeksleme): Dokümanları chunk'la, embedding'lere çevir, vektör veritabanına kaydet
2. Retrieval (Getirme): Kullanıcı sorgusunu embed et, en benzer chunk'ları bul
3. Generation (Üretim): Bulunan bağlamı prompt'a ekle, modelden cevap üret

## Temel RAG Makaleleri

### Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
İlk RAG makalesi. BART modeli ile DPR (Dense Passage Retrieval) birleştirilmiştir. Knowledge-intensive görevlerde (TriviaQA, Natural Questions) state-of-the-art sonuçlar elde edilmiştir. Önemli bulgu: RAG, parametrik bilgi ile non-parametrik bilgiyi başarılı şekilde birleştirir.

### Karpukhin et al. (2020) - "Dense Passage Retrieval for Open-Domain Question Answering"
DPR (Dense Passage Retrieval) yaklaşımını tanıtmıştır. Geleneksel TF-IDF/BM25 sparse retrieval yerine, BERT tabanlı dense embeddings kullanır. Open-domain QA'da sparse retrieval'a göre %10-20 daha iyi sonuç verir.

### Izacard & Grave (2021) - "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"
Fusion-in-Decoder (FiD) yaklaşımı. Birden fazla retrieve edilmiş pasajı decoder'ın cross-attention mekanizmasıyla birleştirir. 100 pasaj kullanarak Natural Questions'da %51.4 exact match skoru elde edilmiştir.

### Borgeaud et al. (2022) - "Improving Language Models by Retrieving from Trillions of Tokens"
RETRO (Retrieval-Enhanced Transformer). 2 trilyon token üzerinde retrieval yaparak 25 kat daha küçük modellerin büyük modellere denk performans göstermesini sağlamıştır. Chunked cross-attention mekanizması kullanır.

## İleri RAG Teknikleri

### Naive RAG vs Advanced RAG vs Modular RAG

Naive RAG: Basit index → retrieve → generate pipeline'ı. Sorunları: düşük retrieval kalitesi, context window limitleri, kayıp bilgi.

Advanced RAG: Pre-retrieval ve post-retrieval optimizasyonlar ekler.
- Pre-retrieval: Query rewriting, query expansion, HyDE (Hypothetical Document Embedding)
- Post-retrieval: Re-ranking, compression, filtering

Modular RAG: Pipeline'ı modüler bileşenlere ayırır. Her bileşen bağımsız olarak optimize edilebilir ve değiştirilebilir.

### Chunking Stratejileri

Etkili chunking, RAG kalitesinin temel belirleyicisidir.

Fixed-size chunking: Sabit karakter veya token sayısıyla böl. Basit ama bağlam kaybına yol açabilir.
Semantic chunking: Cümle veya paragraf sınırlarına saygı gösterir. Anlamsal bütünlüğü korur.
Recursive chunking: Önce büyük bölümlere, sonra küçük parçalara ayır.
Overlap: Chunk'lar arası örtüşme ekleyerek bağlam kaybını azalt. Genellikle %10-20 overlap önerilir.

Önerilen chunk boyutları:
- Genel amaçlı: 256-512 token
- Teknik dokümantasyon: 512-1024 token
- Soru-cevap: 128-256 token

### Embedding Modelleri Karşılaştırması

Embedding modeli seçimi RAG performansını doğrudan etkiler.

MTEB (Massive Text Embedding Benchmark) liderlik tablosu:
- text-embedding-3-large (OpenAI): Yüksek kalite, API tabanlı
- voyage-large-2 (Voyage AI): MTEB'de üst sıralarda
- all-MiniLM-L6-v2 (Sentence-Transformers): Açık kaynak, hızlı, 384 boyut, iyi denge
- bge-large-en-v1.5 (BAAI): Açık kaynak, yüksek kalite, 1024 boyut
- e5-large-v2 (Microsoft): Açık kaynak, instruction-tuned

### Re-Ranking

İlk retrieval sonuçlarını daha sofistike bir modelle yeniden sıralama. Cross-encoder modelleri (ör. ms-marco-MiniLM-L-6-v2) bu amaçla kullanılır. Re-ranking, retrieval kalitesini %10-30 artırabilir.

### HyDE (Hypothetical Document Embedding)

Gao et al. (2022) tarafından önerilen teknik. Kullanıcı sorgusuna doğrudan embedding uygulamak yerine, önce LLM'den hipotetik bir cevap üretilir. Bu hipotetik cevabın embedding'i, doküman embedding'leriyle karşılaştırılır. Sorgu-doküman boşluğunu (query-document gap) kapatır.

### CRAG (Corrective RAG)

Yan et al. (2024) tarafından önerilen teknik. Retrieve edilen dokümanların kalitesini değerlendirir. Eğer dokümanlar yetersizse, web araması gibi alternatif kaynaklara başvurur. Self-correction mekanizması ile halüsinasyonu azaltır.

### Self-RAG

Asai et al. (2023) tarafından önerilen teknik. Model, ne zaman retrieval yapacağına kendisi karar verir. Reflection token'ları kullanarak çıktı kalitesini değerlendirir. İstek üzerine retrieval: her zaman retrieve etmek yerine, gerektiğinde retrieve et.

## RAG Değerlendirme Metrikleri

### Retrieval Kalitesi
- Precision@K: İlk K sonuçtan kaçı gerçekten ilgili
- Recall@K: Tüm ilgili dokümanlardan kaçı ilk K'da bulundu
- MRR (Mean Reciprocal Rank): İlk doğru sonucun sıralaması
- NDCG (Normalized Discounted Cumulative Gain): Sıralama kalitesi

### Üretim Kalitesi
- Faithfulness: Cevap, verilen bağlama sadık mı?
- Answer Relevance: Cevap soruyla ne kadar ilgili?
- Context Relevance: Retrieve edilen bağlam soruyla ne kadar ilgili?
- Hallucination Rate: Bağlamda olmayan bilgi üretme oranı

### Değerlendirme Framework'leri
- RAGAS (RAG Assessment): Faithfulness, answer relevancy, context precision/recall
- TruLens: LLM-as-judge yaklaşımıyla RAG değerlendirme
- LangSmith: LangChain ekosisteminde debug ve değerlendirme

## RAG Optimizasyon İpuçları

1. Chunk boyutunu görev tipine göre ayarla (küçük chunk = hassas retrieval, büyük chunk = daha fazla bağlam)
2. Overlap kullanarak chunk sınırlarında bilgi kaybını önle
3. Metadata filtreleme ekleyerek arama alanını daralt
4. Hybrid search kullan: sparse (BM25) + dense (embedding) birlikte daha iyi sonuç verir
5. Query expansion ile kullanıcı sorgusunu zenginleştir
6. Re-ranking ile ilk sonuçları iyileştir
7. Context window'u verimli kullan: gereksiz chunk'ları filtrele
8. Embedding modelini domain'ine göre fine-tune et
9. Evaluation pipeline kur: otomatik kalite izleme
10. Chunking stratejisini doküman türüne göre adapte et
