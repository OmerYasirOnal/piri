# LLM Eğitim Teknikleri ve Optimizasyonları

## Pre-training (Ön Eğitim)

Pre-training, büyük miktarda etiketsiz metin üzerinde dil modelinin temel dil anlama yeteneklerini kazanması sürecidir.

### Eğitim Hedefleri (Training Objectives)

Causal Language Modeling (CLM): Bir sonraki token'ı tahmin et. GPT ailesi bu yaklaşımı kullanır. Autoregressive bir yapıdır — soldan sağa üretir.

Masked Language Modeling (MLM): Rastgele maskelenen token'ları tahmin et. BERT ailesi bu yaklaşımı kullanır. Bidirectional context kullanır.

Seq2Seq: Encoder-decoder yapısı. Girdi dizisini çıktı dizisine çevir. T5, BART bu yaklaşımı kullanır.

### Tokenization Yöntemleri

BPE (Byte Pair Encoding): En sık görülen karakter çiftlerini iteratif olarak birleştirir. GPT-2, GPT-3, GPT-4 tarafından kullanılır. Sennrich et al. (2016) tarafından NLP'ye uyarlanmıştır.

WordPiece: BERT tarafından kullanılır. BPE'ye benzer ama birleştirme kriteri farklıdır — likelihood artışına göre seçim yapar.

SentencePiece: Dile bağımsız tokenizer. Unigram ve BPE algoritmalarını destekler. T5, ALBERT tarafından kullanılır.

Tiktoken: OpenAI'ın hızlı BPE implementasyonu. GPT-4, GPT-3.5 için kullanılır.

## Fine-Tuning Yöntemleri

### Full Fine-Tuning
Tüm model parametrelerini günceller. En yüksek kalite ama en yüksek kaynak tüketimi. Küçük modeller (DistilGPT-2, GPT-2 small) için uygundur.

### LoRA (Low-Rank Adaptation)
Hu et al. (2021) tarafından önerilmiştir. Orijinal ağırlıkları dondurur, düşük rank'li adaptasyon matrisleri ekler. Parametre sayısını %0.1-1 seviyesine düşürür. Eğitim hızını 3-5x artırır, bellek kullanımını 3x azaltır.

LoRA hiperparametreleri:
- rank (r): Adaptasyon matrislerinin rank'ı. r=8 veya r=16 genellikle yeterli.
- alpha: Scaling faktörü. Genellikle alpha = 2 * rank.
- target_modules: Hangi katmanlara uygulanacak. Genellikle attention katmanları (q_proj, v_proj).

### QLoRA (Quantized LoRA)
Dettmers et al. (2023) tarafından önerilmiştir. 4-bit quantized model üzerine LoRA uygular. 65B parametre modeli tek bir 48GB GPU'da fine-tune etmeyi mümkün kılar. NF4 (NormalFloat4) quantization kullanır.

### Prefix Tuning
Li & Liang (2021). Modelin her katmanına öğrenilebilir prefix vektörleri ekler. Model ağırlıklarına dokunmaz, sadece prefix'leri öğrenir.

### Adapter Layers
Houlsby et al. (2019). Transformer katmanları arasına küçük adapter modülleri ekler. Her görev için farklı adapter'lar kullanılabilir. Temel model paylaşılır, sadece adapter'lar görev-özeldir.

## RLHF (Reinforcement Learning from Human Feedback)

Ouyang et al. (2022) InstructGPT makalesinde detaylandırılmıştır.

RLHF üç aşamadan oluşur:
1. SFT (Supervised Fine-Tuning): İnsan tarafından yazılmış yüksek kaliteli örneklerle fine-tune
2. Reward Model Training: İnsan tercihlerinden (A > B karşılaştırmaları) bir ödül modeli eğit
3. PPO (Proximal Policy Optimization): Ödül modelini kullanarak dil modelini optimize et

## DPO (Direct Preference Optimization)

Rafailov et al. (2023) tarafından önerilmiştir. RLHF'nin reward model + PPO aşamasını tek bir kayıp fonksiyonuyla birleştirir. Daha basit, daha kararlı ve daha verimli. Ayrı bir reward model eğitmeye gerek kalmaz.

## Quantization (Niceleme)

Quantization, model ağırlıklarını daha düşük bit hassasiyetine dönüştürerek bellek kullanımını ve inference hızını iyileştirir.

Türleri:
- INT8: 8-bit integer quantization. ~2x bellek tasarrufu, minimal kalite kaybı.
- INT4: 4-bit integer quantization. ~4x bellek tasarrufu, kabul edilebilir kalite kaybı.
- GPTQ: Post-training quantization. OPT-175B gibi büyük modelleri 4-bit'e indirger.
- AWQ (Activation-aware Weight Quantization): Aktivasyonlara göre önemli ağırlıkları korur.
- GGUF: llama.cpp formatı. CPU inference için optimize.

## Distillation (Bilgi Damıtma)

Hinton et al. (2015) tarafından önerilmiştir. Büyük bir "öğretmen" modelin bilgisini küçük bir "öğrenci" modele aktarır.

DistilBERT: BERT'in %60 daha küçük, %60 daha hızlı versiyonu. Performansın %97'sini korur.
DistilGPT-2: GPT-2'nin damıtılmış versiyonu. 82M parametre (GPT-2: 124M). CPU'da etkili çalışır.

## Eğitim Optimizasyonları

### Gradient Accumulation
Küçük batch'leri biriktirerek büyük effective batch size elde et. Sınırlı GPU belleğinde büyük batch etkisi yaratır.

### Mixed Precision Training (Micikevicius et al., 2018)
FP16/BF16 ile eğitim, FP32 ile gradient güncelleme. 2x bellek tasarrufu, 2-3x hız artışı.

### Gradient Checkpointing
İleri geçişteki ara aktivasyonları saklamak yerine geri geçişte yeniden hesapla. %30-50 bellek tasarrufu, %20-30 hız kaybı.

### DeepSpeed ZeRO
Microsoft DeepSpeed kütüphanesi. Model state'ini (optimizer, gradients, parameters) birden fazla GPU'ya dağıtır. ZeRO Stage 1/2/3 ile artan bellek tasarrufu.

### Flash Attention (Dao et al., 2022)
IO-aware attention implementasyonu. Standart attention'dan 2-4x daha hızlı. Bellek kullanımını O(N^2)'den O(N)'e düşürür.
