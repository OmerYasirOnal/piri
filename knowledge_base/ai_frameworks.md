# Yapay Zeka Framework'leri ve Kütüphaneleri

## PyTorch

PyTorch, Meta (Facebook) tarafından geliştirilen açık kaynaklı bir derin öğrenme framework'üdür. Dinamik hesaplama grafı (eager execution) kullanır, bu da hata ayıklamayı ve araştırmayı kolaylaştırır.

Temel özellikler:
- Autograd: Otomatik türev hesaplama motoru
- torch.nn: Sinir ağı katmanları ve modülleri
- torch.optim: SGD, Adam, AdamW gibi optimizerlar
- DataLoader: Verimli veri yükleme ve batching
- TorchScript: Modelleri production'a deploy etmek için JIT compiler
- torch.compile: Python kodunu optimize edilmiş makine koduna çevirir (PyTorch 2.0+)
- MPS (Metal Performance Shaders): Apple Silicon GPU desteği

PyTorch kullanım alanları: araştırma, doğal dil işleme, bilgisayarlı görü, güçlendirmeli öğrenme.

## Hugging Face Transformers

Hugging Face Transformers, önceden eğitilmiş transformer modelleriyle çalışmak için en popüler kütüphanedir. 200.000'den fazla model Hugging Face Hub'da mevcuttur.

Temel bileşenler:
- AutoModel: Model adından otomatik olarak doğru mimariyi yükler
- AutoTokenizer: Modele uygun tokenizer'ı otomatik seçer
- Pipeline: Yüksek seviyeli inference API'si (text-generation, sentiment-analysis, translation vb.)
- Trainer: Eğitim döngüsünü yöneten sınıf
- TrainingArguments: Eğitim hiperparametrelerini tanımlar
- PEFT (Parameter-Efficient Fine-Tuning): LoRA, QLoRA gibi verimli fine-tuning teknikleri

Desteklenen görevler: text-generation, text-classification, question-answering, summarization, translation, token-classification, image-classification, object-detection, automatic-speech-recognition.

## LangChain

LangChain, LLM uygulamaları geliştirmek için kullanılan bir framework'tür. Zincir (chain) yapısıyla karmaşık LLM iş akışları oluşturulabilir.

Temel modüller:
- Models: LLM ve Chat modeli soyutlamaları
- Prompts: Prompt template ve yönetimi
- Chains: Sıralı LLM çağrıları zinciri
- Agents: Araç kullanabilen otonom agent'lar
- Memory: Konuşma geçmişi yönetimi
- Retrieval: RAG (Retrieval-Augmented Generation) bileşenleri
- Document Loaders: PDF, HTML, Markdown ve diğer format desteği
- Text Splitters: Doküman chunking stratejileri
- Vector Stores: FAISS, Chroma, Pinecone, Weaviate entegrasyonları

## LlamaIndex

LlamaIndex (eski adıyla GPT Index), veri ve LLM'ler arasında köprü kuran bir framework'tür. Özellikle RAG uygulamaları için optimize edilmiştir.

Temel kavramlar:
- Documents: Veri kaynakları (dosya, API, veritabanı)
- Nodes: Chunk'lanmış veri parçaları
- Indices: Veri organizasyon yapıları (vector, list, tree, keyword)
- Query Engine: Sorgu işleme ve cevap üretme
- Response Synthesizer: Birden fazla kaynaktan cevap sentezleme

Avantajları: Karmaşık veri yapılarını destekler, hierarchical indexing yapabilir, multi-modal verilerle çalışır.

## FAISS (Facebook AI Similarity Search)

FAISS, Meta AI tarafından geliştirilen, büyük ölçekli vektör benzerlik araması için optimize edilmiş bir kütüphanedir.

İndeks türleri:
- IndexFlatL2: Tam doğruluk, brute-force L2 uzaklığı
- IndexFlatIP: Tam doğruluk, inner product (cosine similarity ile kullanılır)
- IndexIVFFlat: Inverted file indeksi, yaklaşık arama, daha hızlı
- IndexHNSW: Hierarchical navigable small world graph, yüksek hız ve doğruluk
- IndexPQ: Product quantization, bellek tasarrufu

CPU ve GPU desteği mevcuttur. Milyarlarca vektör üzerinde milisaniye seviyesinde arama yapabilir.

## Sentence-Transformers

Sentence-Transformers, cümle ve paragraf seviyesinde embedding üretmek için kullanılan bir kütüphanedir. SBERT (Sentence-BERT) mimarisine dayanır.

Popüler modeller:
- all-MiniLM-L6-v2: 384 boyut, hızlı, genel amaçlı (en iyi hız/kalite dengesi)
- all-mpnet-base-v2: 768 boyut, yüksek kalite, daha yavaş
- paraphrase-multilingual-MiniLM-L12-v2: Çok dilli destek, 50+ dil
- multi-qa-MiniLM-L6-cos-v1: Soru-cevap için optimize edilmiş

Kullanım alanları: semantik arama, metin benzerliği, kümeleme, bilgi erişimi, paraphrase tespiti.

## Weights & Biases (W&B)

W&B, makine öğrenmesi deneylerini takip etmek, görselleştirmek ve paylaşmak için kullanılan bir platformdur.

Temel özellikler:
- Experiment Tracking: Hiperparametreler, metrikler, artifaktlar
- Sweeps: Otomatik hiperparametre optimizasyonu
- Artifacts: Veri seti ve model versiyonlama
- Reports: İnteraktif deneysel raporlar
- Tables: Veri görselleştirme ve analiz

## ONNX Runtime

ONNX (Open Neural Network Exchange) Runtime, modelleri platform bağımsız olarak optimize edip çalıştırmak için kullanılır.

Avantajları:
- Framework bağımsız: PyTorch, TensorFlow modellerini ONNX formatına çevir
- Optimizasyon: Graph optimization, operator fusion, quantization
- Cross-platform: CPU, GPU, mobile, edge cihazlarda çalışır
- Yüksek performans: Genellikle orijinal framework'ten 2-5x daha hızlı inference
