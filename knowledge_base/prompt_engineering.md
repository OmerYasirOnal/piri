# Prompt Engineering: Kapsamlı Rehber

## Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting, Wei et al. (2022) tarafından tanıtılan bir tekniktir. Model, nihai cevaba ulaşmadan önce adım adım düşünme sürecini gösterir. Bu teknik, matematiksel muhakeme görevlerinde %20-40 iyileşme sağlamıştır.

CoT'nin iki ana varyantı vardır:
- Zero-Shot CoT: Prompt'a "Adım adım düşünelim" eklenir. Kojima et al. (2022) tarafından önerilmiştir.
- Few-Shot CoT: Birkaç örnek çözüm adımı gösterilir. Bu varyant genellikle daha iyi performans verir.

CoT özellikle şu görevlerde etkilidir: aritmetik muhakeme, sembolik mantık, çok adımlı problem çözme ve karar verme süreçleri.

## Few-Shot Learning

Few-shot learning, modele birkaç örnek göstererek istenen çıktı formatını ve kalitesini öğretme tekniğidir. Brown et al. (2020) GPT-3 makalesinde bu yaklaşımı detaylı olarak tanımlamıştır.

Zhou et al. (2023) araştırmasına göre, iyi seçilmiş few-shot örnekler %50-200 arası iyileşme sağlayabilir. Örnek seçimi kritiktir — çeşitli, temsili ve görev-uyumlu örnekler seçilmelidir.

Few-shot prompting stratejileri:
- Pozitif örnekler: İdeal çıktıları göster
- Negatif örnekler: "Bunu yapMA" örnekleri göster
- Diverse örnekler: Farklı durumları kapsayan örnekler kullan
- Template-aligned: Örneklerin formatı istenilen çıktı formatıyla aynı olmalı

## Structured Reasoning

ReAct (Yao et al., 2022) framework'ü, reasoning (düşünme) ve acting (eylem) adımlarını birleştirir. Model şu döngüyü takip eder: Thought → Action → Observation → Thought.

Structured reasoning blokları şu aşamaları içerir:
1. Thinking: Problemi analiz et, yaklaşımı planla
2. Planning: Adımları sırala, bağımlılıkları belirle
3. Execution: Planı uygula
4. Reflection: Sonucu değerlendir, hataları tespit et

## Self-Consistency (Wang et al., 2022)

Self-consistency tekniği, aynı soruya birden fazla cevap üretir ve çoğunluk oylaması (majority voting) ile en tutarlı cevabı seçer. Bu yöntem, CoT ile birlikte kullanıldığında %10-20 ek iyileşme sağlar.

Uygulama adımları:
1. Aynı prompt'u yüksek temperature ile birden fazla kez çalıştır
2. Her çıktının nihai cevabını çıkar
3. En sık tekrarlanan cevabı seç

## Context Framing ve Grounding

Jimenez et al. (2024) SWE-bench çalışmasında, bağlam çerçevelemenin (context framing) performansta 3-5 kat fark yarattığını göstermiştir. Etkili context framing şunları içerir:

- Bilgi önceliklendirme: En ilgili bilgiyi önce ver
- Yapısal organizasyon: Markdown başlıkları ve listeler kullan
- Kaynak etiketleme: Her bilgi parçasının kaynağını belirt
- Noise filtreleme: İlgisiz bilgiyi çıkar

## Adversarial Reflection

Shinn et al. (2023) Reflexion makalesinde, modelin kendi çıktısını eleştirel olarak değerlendirmesinin %20-30 iyileşme sağladığını göstermiştir.

Etkili reflection prompt'u şu boyutları değerlendirmelidir:
- Doğruluk (Correctness): Her teknik iddia doğru mu?
- Tamlık (Completeness): Görevin tüm kapsamı karşılanmış mı?
- Tutarlılık (Consistency): İsimlendirme ve format tutarlı mı?
- Temellendirilmişlik (Groundedness): Her iddia kanıta dayalı mı?
- Kullanılabilirlik (Usability): Hedef kitle için hemen kullanılabilir mi?

## Persona Engineering

Derin persona tanımları, sığ tanımlara göre çok daha kaliteli çıktılar üretir. İyi bir persona tanımı şunları içermelidir:
- Uzmanlık alanı ve deneyim yılı
- Spesifik beceriler ve araç seti
- Karar verme tarzı ve öncelikleri
- Kalite standartları ve kırmızı çizgiler

Örnek: "You are a Principal Software Engineering Strategist with 20 years of experience" bir "You are an AI assistant" tanımından çok daha etkilidir.

## Output Schema Enforcement

Yapılandırılmış çıktı zorlama (output schema enforcement), modelin belirli bir JSON/XML şemasına uygun çıktı üretmesini sağlar. Bu teknik:
- Zod gibi runtime validation kütüphaneleriyle birlikte kullanılır
- Prompt'ta şema açıklaması ve örnek çıktı gösterilir
- Geçersiz çıktılar otomatik olarak tespit edilir ve yeniden üretilir

## Prompt Kalitesi Değerlendirme Kriterleri

İyi bir prompt şu özelliklere sahiptir:
- Atomiklik: Her adım tek bir işlemde tamamlanabilir
- Doğrulanabilirlik: Her adımın başarı kriteri bellidir
- Tamlık: Görevin tüm yönleri kapsanır
- Verimlilik: Gereksiz veya döngüsel adım yoktur
- Spesifiklik: Belirsiz talimatlar yerine somut yönergeler içerir
