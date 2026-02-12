# AI Agent Mimarileri

## Agent Nedir?

AI agent, bir LLM'in çevresiyle etkileşime girerek otonom kararlar almasını ve görevler gerçekleştirmesini sağlayan bir yazılım mimarisidir. Temel fark: chatbot sadece cevap verir, agent ise plan yapar, araç kullanır ve hedefe ulaşana kadar iteratif olarak çalışır.

## Temel Agent Desenleri

### ReAct (Yao et al., 2022)
Reasoning + Acting birleşimi. Her adımda:
- Thought: Ne yapacağımı düşünüyorum
- Action: Bir araç çağırıyorum
- Observation: Aracın sonucunu görüyorum
- Thought: Sonucu değerlendirip bir sonraki adımı planlıyorum

Bu döngü hedefe ulaşılana kadar devam eder.

### Plan-and-Execute
İki aşamalı yaklaşım:
1. Planlama: LLM tam bir plan oluşturur (adımlar listesi)
2. Yürütme: Her adım sırayla çalıştırılır

Avantajı: Büyük resmi görerek daha tutarlı planlar üretir.
Dezavantajı: Plan hata içeriyorsa düzeltme zor olabilir.

### Plan-Execute-Reflect (Shinn et al., 2023 - Reflexion)
Plan-and-Execute'a reflection (yansıtma) adımı ekler:
1. Plan yap
2. Planı uygula
3. Sonucu değerlendir
4. Gerekirse planı güncelle ve tekrarla

Bu yaklaşım %20-30 daha iyi sonuçlar üretir.

### LATS (Language Agent Tree Search)
Zhou et al. (2023). Monte Carlo Tree Search (MCTS) prensiplerini agent'lara uygular. Birden fazla olası yolu keşfeder ve en iyi yolu seçer. Hata durumunda geriye sarma (backtracking) yapabilir.

## Multi-Agent Sistemler

### Crew AI Yaklaşımı
Farklı roller ve uzmanlıklara sahip birden fazla agent'ın birlikte çalışması:
- Researcher Agent: Bilgi toplar ve analiz eder
- Writer Agent: İçerik üretir
- Reviewer Agent: Kalite kontrolü yapar
- Manager Agent: Görev dağılımını koordine eder

### AutoGen (Microsoft)
Birden fazla LLM agent'ının konuşma tabanlı iş birliği yapması. Agent'lar birbirlerine mesaj gönderir ve birlikte çözüm üretir. Human-in-the-loop desteği ile insan onayı alınabilir.

### LangGraph
LangChain ekosisteminde graf tabanlı agent orkestrasyon. Durum makinesi (state machine) yaklaşımı. Döngüsel (cyclic) iş akışları destekler. Checkpoint ve geri alma desteği.

## Agent Araçları (Tools)

Agent'lar LLM'in tek başına yapamayacağı görevleri araçlarla gerçekleştirir:

Yaygın araç türleri:
- Web Search: Google, Bing, DuckDuckGo API
- Code Execution: Python REPL, JavaScript sandbox
- File Operations: Okuma, yazma, dosya sistemi navigasyonu
- API Calls: REST API, GraphQL sorguları
- Database: SQL sorguları, CRUD işlemleri
- Calculator: Matematiksel hesaplamalar
- Browser: Web sayfası navigasyonu, scraping

## Agent Bellek Yönetimi

### Kısa Süreli Bellek (Working Memory)
Mevcut konuşma bağlamı. Context window ile sınırlı. Son N mesaj veya son N token.

### Uzun Süreli Bellek (Long-Term Memory)
Önceki etkileşimlerden öğrenilen bilgiler:
- Vector Store: Semantik arama ile geçmiş etkileşimleri bul
- Summary Memory: Konuşmaları özetleyerek sakla
- Entity Memory: Bahsedilen varlıklar (kişiler, projeler) hakkında bilgi tut
- Knowledge Graph: İlişkileri graf yapısında sakla

### Episodik Bellek
Spesifik deneyimleri (başarılı/başarısız görevler) sakla ve benzer durumlardan ders çıkar.

## Agent Güvenilirliği

### Halüsinasyon Önleme
- Grounding: Cevapları kanıtlarla destekle
- Citation: Her iddia için kaynak göster
- Confidence scoring: Belirsiz olduğunda açıkça belirt
- Retrieval-augmented: Bilgi tabanından doğrula

### Hata Yönetimi
- Retry with backoff: Başarısız araç çağrılarını tekrar dene
- Fallback strategies: Alternatif yollar belirle
- Human escalation: Karmaşık durumları insana yönlendir
- Error recovery: Hata sonrası güvenli duruma dön

### Değerlendirme
- Task completion rate: Görev tamamlama oranı
- Step efficiency: Hedefe ulaşmak için gereken adım sayısı
- Tool use accuracy: Araçları doğru kullanma oranı
- Safety violations: Güvenlik ihlali sayısı

## AKIS Platform Agent Mimarisi

AKIS Platform'da üç temel agent bulunur:

### Scribe Agent
Repository analizi yaparak dokümantasyon üretir. En gelişmiş system prompt'a sahiptir. Repository grounding ve anti-hallucination kuralları içerir. README, API referansı, mimari doküman ve kurulum rehberi üretebilir.

### Trace Agent
Kod yollarını analiz ederek test planları oluşturur. Test senaryolarını belirler, coverage matrisi çıkarır ve yapılandırılmış test planları üretir.

### Proto Agent
MVP iskelet kodu ve proje yapısı üretir. Boilerplate kod, yapılandırma dosyaları ve temel implementasyonlar oluşturur.

Her agent Plan → Generate → Reflect → Validate pipeline'ını takip eder. Bu pipeline, prompt kalitesine doğrudan bağlıdır.
