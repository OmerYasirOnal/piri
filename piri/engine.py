"""
Piri Engine — Akıllı RAG Orkestrasyon Modülü (v3)

v3 İyileştirmeler:
1. Relevance threshold — alakasız chunk filtresi
2. Auto web-search fallback — bilgi yoksa internetten öğren
3. Smart query routing — RAG → Web → Dürüst cevap
4. Multilingual embedding + Cross-encoder reranking
5. OpenAI API backend desteği (opsiyonel)
"""
import os
import re
from typing import Dict, Optional, List
from transformers import pipeline as hf_pipeline, AutoTokenizer

from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .reranker import create_reranker
from .chunker import load_all_documents, chunk_documents

# ─── Model Ayarları ───────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_PATH = "./model"

# ─── Relevance Threshold ─────────────────────────────────────
# Reranker skoru bu değerin altındaysa bilgi tabanında alakalı içerik YOK demek.
# Cross-encoder skorları: >2 çok iyi, 0-2 orta, <0 alakasız
RELEVANCE_THRESHOLD = 0.0


# ─── Prompt'lar — OpenAI için (detaylı, kapsamlı) ────────────

OPENAI_RAG_SYSTEM = """Sen Piri adında bilgili bir yapay zeka asistanısın. AKIS Platform tarafından geliştirildin.

## Görevin
Sana verilen bağlam bilgisini kullanarak soruları doğru, kapsamlı ve anlaşılır şekilde Türkçe yanıtla.

## Kuralların
1. MUTLAKA Türkçe yanıt ver.
2. Yalnızca bağlamda bulunan bilgilere dayan. Bağlamda olmayan bilgiyi UYDURMA.
3. Cevabını yapılandır: Önce kısa bir özet, sonra detaylar.
4. Teknik terimleri açıkla, basit dil kullan.
5. Emin olmadığın bilgiyi "Bu konuda bağlamda yeterli bilgi bulunmuyor" diye belirt.
6. Cevabını tamamla, yarım bırakma."""

OPENAI_SIMPLE_SYSTEM = """Sen Piri adında bilgili bir yapay zeka asistanısın.
Soruları açık, net ve doğru şekilde Türkçe yanıtla. Cevabını yapılandır ve tamamla."""


# ─── Prompt'lar — Local model için (basit, net, kısa) ────────

LOCAL_RAG_SYSTEM = "Sen Piri, Türkçe cevap veren bir asistan. Verilen bilgileri kullanarak soruyu yanıtla."

LOCAL_SIMPLE_SYSTEM = "Sen Piri, Türkçe cevap veren bir asistan."


def resolve_model_path(model_path: str) -> str:
    """Fine-tuned model varsa onu, yoksa HuggingFace modelini döndür."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        return model_path
    return DEFAULT_MODEL


# ─── Extractive Fallback ─────────────────────────────────────

def _clean_line(line: str) -> str:
    """Tek bir satırdan metadata/artifact temizler."""
    # URL'leri kaldır
    line = re.sub(r'https?://\S+', '', line)
    # Markdown başlıklarını kaldır
    line = re.sub(r'^#{1,4}\s+', '', line)
    # [1], [2] gibi referans numaralarını kaldır
    line = re.sub(r'\[(\d+)\]\s*', '', line)
    # Parantez içindeki URL'leri kaldır
    line = re.sub(r'\([^)]*https?://[^)]*\)', '', line)
    return line.strip()


def _is_junk(line: str) -> bool:
    """Bu satır metadata/çöp mü?"""
    low = line.lower().strip()
    # Çok kısa
    if len(low) < 15:
        return True
    # Metadata kalıpları
    junk = [
        'web araması:', 'kaynak:', 'tarih:', '```', '---',
        'http://', 'https://', '## [', 'source:', 'date:',
    ]
    if any(low.startswith(j) for j in junk):
        return True
    # Tamamen URL olan satır
    if re.match(r'^[\w\-]+\.\w+\s*$', low):
        return True
    # Ağırlıklı İngilizce (Türkçe harfler yoksa ve İngilizce kelimeler çoksa)
    tr_chars = set('çğıöşüÇĞİÖŞÜ')
    en_markers = {'the', 'and', 'for', 'with', 'you', 'are', 'that', 'this', 'from', 'have'}
    words = low.split()
    if len(words) > 5:
        has_tr = any(c in line for c in tr_chars)
        en_count = sum(1 for w in words if w in en_markers)
        if not has_tr and en_count >= 3:
            return True
    return False


def extract_best_answer(context: str, question: str) -> str:
    """
    v3: Bağlamdan en alakalı cümleleri çıkarır. Agresif temizlik.
    Metadata, URL, İngilizce, tekrar = hepsi filtrelenir.
    """
    if not context:
        return ""

    # Satır bazlı böl ve temizle
    raw_lines = re.split(r'\n---\n|\n\n|\n', context)
    sentences = []
    seen = set()
    for line in raw_lines:
        line = _clean_line(line)
        if not line or _is_junk(line):
            continue
        # Tekrar kontrolü
        norm = re.sub(r'\s+', ' ', line.lower())[:80]
        if norm in seen:
            continue
        seen.add(norm)
        sentences.append(line)

    if not sentences:
        return ""

    # Soru kelimelerini çıkar
    stopwords = {
        "ne", "nasıl", "nedir", "ve", "ile", "bir", "bu", "da", "de",
        "mi", "mı", "için", "olan", "var", "mı", "mu", "mü", "dir",
        "ise", "olarak", "gibi", "kadar", "daha", "en", "çok", "her",
        "hakkında", "bilgi", "ver", "neler", "hangi", "neden", "kim",
        "kaç", "oldu", "olan", "yaşında", "öldü",
    }
    q_words = set()
    for w in question.lower().split():
        cw = re.sub(r'[^\wçğıöşü]', '', w)
        if cw not in stopwords and len(cw) > 2:
            q_words.add(cw)

    # Cümleleri puanla
    scored = []
    for s in sentences:
        s_low = s.lower()
        sc = 0
        matched = 0
        for w in q_words:
            if w in s_low:
                sc += 3; matched += 1
            elif len(w) > 3 and any(w[:4] in pw for pw in s_low.split() if len(pw) > 3):
                sc += 1; matched += 1
        if q_words and matched / len(q_words) > 0.5:
            sc += 3
        # Bilgi yoğunluğu
        if len(s) > 80: sc += 1
        if '.' in s: sc += 0.5
        scored.append((sc, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # En iyi 2 cümle, skor > 0 olanlar
    best = [s for sc, s in scored if sc > 0][:2]
    if not best:
        best = [scored[0][1]] if scored else []

    result = "\n\n".join(best)

    # 800 karakterde cümle sınırında kes
    if len(result) > 800:
        for p in ".!?":
            pos = result[:800].rfind(p)
            if pos > 200:
                result = result[:pos + 1]
                break

    return result.strip()


def quality_check(text: str, context: str = "", question: str = "") -> float:
    """
    Cevap kalitesi heuristic skoru (0-1).
    Düşük skor = kötü cevap, extractive fallback kullan.

    Sıkı kontrol: 0.5B model genellikle anlamsız Türkçe ürettiği için
    bağlamla tutarlılık ÇOK önemli.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    score = 0.3  # Düşük başlangıç (local model genelde kötü)

    words = text.lower().split()
    if not words:
        return 0.0

    # 1. Bağlamla tutarlılık (EN ÖNEMLİ)
    if context:
        ctx_words = set(context.lower().split())
        answer_words = set(words)
        # Cevaptaki kelimelerin kaçı bağlamda var?
        overlap = len(answer_words & ctx_words) / max(len(answer_words), 1)
        if overlap > 0.5:
            score += 0.3
        elif overlap > 0.3:
            score += 0.15
        else:
            score -= 0.2  # Bağlamla ilgisiz = muhtemelen halüsinasyon

    # 2. Soru kelimelerinin cevapta geçmesi
    if question:
        q_words = set(w for w in question.lower().split() if len(w) > 3)
        ans_lower = text.lower()
        q_in_answer = sum(1 for w in q_words if w in ans_lower)
        if q_words and q_in_answer / len(q_words) > 0.3:
            score += 0.1

    # 3. Tekrar kontrolü (ÇOK ÖNEMLİ — 0.5B çok tekrar eder)
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.5:
        score -= 0.3  # Çok fazla tekrar
    elif unique_ratio < 0.65:
        score -= 0.1

    # 4. Anlamsız yapılar
    # "Bir genellidir ki" tipi anlamsız cümleler
    nonsense_patterns = [
        "genellidir ki", "daha fazlasıyla", "genelleştirilmiş",
        "açıklayacağı üzere", "belirsizlik veya", "gerekirse",
        "etkileşime geçtiği", "temsilcisel deneyimi",
    ]
    nonsense_count = sum(1 for p in nonsense_patterns if p in text.lower())
    if nonsense_count > 2:
        score -= 0.3

    # 5. Markdown overkill (model bazen random markdown üretir)
    markdown_ratio = text.count('#') + text.count('*') + text.count('`')
    if len(text) > 0 and markdown_ratio / len(text) > 0.05:
        score -= 0.15

    return max(0.0, min(1.0, score))


# ─── Post-Processing ─────────────────────────────────────────

def clean_response(text: str, aggressive: bool = False) -> str:
    """
    Model çıktısını temizler.
    Asla cevabı silmez veya "oluşturulamadı" mesajı göstermez.
    Sadece artifact'leri ve tekrarları temizler.
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # 1. ChatML ve özel token temizliği
    artifacts = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|end|>", "<|assistant|>", "<|user|>", "<|system|>",
    ]
    for tag in artifacts:
        text = text.replace(tag, "")

    # "assistant\n" sadece satır başında temizle
    if text.startswith("assistant\n"):
        text = text[len("assistant\n"):]
    if text.startswith("assistant"):
        text = text[len("assistant"):]

    text = text.strip()

    # 2. Satır bazlı tekrar temizliği
    lines = text.split("\n")
    cleaned_lines = []
    seen_normalized = set()

    for line in lines:
        stripped = line.strip()

        # Boş satır yönetimi
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Tekrar tespiti (aynı satırı iki kez gösterme)
        normalized = re.sub(r'\s+', ' ', stripped.lower())
        if len(normalized) > 20 and normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines).strip()

    # 3. Son cümle tamamlama — çok kısa kesme
    if text and len(text) > 50 and text[-1] not in ".!?:)\"'":
        last_punct = -1
        for punct in ".!?":
            pos = text.rfind(punct)
            if pos > last_punct:
                last_punct = pos
        if last_punct > len(text) * 0.4:
            text = text[:last_punct + 1]

    return text.strip()


# ─── OpenAI Backend ──────────────────────────────────────────

class OpenAIBackend:
    """OpenAI API kullanarak yüksek kaliteli cevaplar üretir."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.is_local = False
        print(f"[Piri] OpenAI backend aktif: {model}")

    def generate(self, messages: list, max_tokens: int = 500, temperature: float = 0.3) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI API hatası: {str(e)}"


# ─── Local Backend ───────────────────────────────────────────

class LocalBackend:
    """Yerel Qwen modeli ile cevap üretir — basitleştirilmiş prompt ile."""

    def __init__(self, model_path: str):
        resolved = resolve_model_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.generator = hf_pipeline(
            "text-generation",
            model=resolved,
            tokenizer=self.tokenizer,
            device=-1,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = resolved
        self.is_local = True
        print(f"[Piri] Local backend aktif: {resolved}")

    def generate(self, messages: list, max_tokens: int = 300, temperature: float = 0.4) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=max(temperature, 0.1),
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.4,     # Tekrar GÜÇLÜ engelle
            no_repeat_ngram_size=4,     # 4-gram tekrarı yasak
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = output[0]["generated_text"]
        answer = full_text[len(prompt):].strip()
        return answer if answer else full_text


# ─── Ana Engine ──────────────────────────────────────────────

class PiriEngine:
    """
    Piri RAG Engine v3 — Retrieve → Rerank → Augment → Generate → Clean
    """

    def __init__(
        self,
        model_path: str = LOCAL_MODEL_PATH,
        vector_store_path: str = "./vector_store",
        embedding_model: str = None,
        use_reranking: bool = None,
        openai_model: str = None,
    ):
        # .env yükleme
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Ayarları ortam değişkenlerinden al
        if embedding_model is None:
            embedding_model = os.environ.get("PIRI_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        if use_reranking is None:
            use_reranking = os.environ.get("PIRI_RERANKING", "true").lower() == "true"
        model_path = os.environ.get("PIRI_MODEL_PATH", model_path)

        # Embedding modeli
        self.embedder = Embedder(model_name=embedding_model)

        # Vektör store
        self.store = VectorStore(dimension=self.embedder.dimension)
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            self.store.load(vector_store_path)

        # Reranker
        reranker = None
        if use_reranking:
            reranker = create_reranker()

        # Retriever
        self.retriever = Retriever(self.embedder, self.store, reranker=reranker)

        # Backend seçimi
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key and len(api_key) > 10:
            model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.backend = OpenAIBackend(model=model)
            self.backend_type = "openai"
        else:
            self.backend = LocalBackend(model_path)
            self.backend_type = "local"

        self.model_name = (
            self.backend.model if self.backend_type == "openai"
            else self.backend.model_name
        )
        self.vector_store_path = vector_store_path
        print(f"[Piri] Engine v3 hazır. Backend: {self.backend_type}")
        print(f"[Piri] İndekste {self.store.total_chunks} chunk var.")

    def _get_prompts(self):
        """Backend'e uygun prompt seti döndürür."""
        if self.backend_type == "openai":
            return OPENAI_RAG_SYSTEM, OPENAI_SIMPLE_SYSTEM
        return LOCAL_RAG_SYSTEM, LOCAL_SIMPLE_SYSTEM

    def ingest(self, directory: str, chunk_size: int = 512, chunk_overlap: int = 64) -> int:
        """Klasördeki dokümanları indeksle."""
        print(f"[Piri] Dokümanlar yükleniyor: {directory}")
        documents = load_all_documents(directory)
        if not documents:
            return 0

        print(f"[Piri] {len(documents)} doküman bulundu.")
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"[Piri] {len(chunks)} chunk oluşturuldu.")

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        metadata = [
            {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]
        self.store.add(embeddings, metadata)
        self.store.save(self.vector_store_path)
        return len(chunks)

    def learn_text(
        self,
        text: str,
        source_name: str = "kullanici_dokumani",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> Dict:
        """
        Ham metin veya doküman içeriğinden öğrenir.
        Dosya yüklemeden doğrudan metin olarak beslenebilir.

        Args:
            text: Öğrenilecek metin içeriği
            source_name: Kaynak adı (dosya adı veya kullanıcı etiketi)
            chunk_size: Chunk boyutu
            chunk_overlap: Overlap miktarı

        Returns:
            {"chunks_added": int, "total_chunks": int, "source": str}
        """
        from .chunker import chunk_text

        if not text or not text.strip():
            return {"chunks_added": 0, "total_chunks": self.store.total_chunks, "source": source_name}

        # Metni chunk'la
        text_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not text_chunks:
            return {"chunks_added": 0, "total_chunks": self.store.total_chunks, "source": source_name}

        print(f"[Piri] '{source_name}' → {len(text_chunks)} chunk oluşturuldu.")

        # Embedding oluştur
        embeddings = self.embedder.embed_texts(text_chunks)

        # Metadata oluştur
        base_id = self.store.total_chunks
        metadata = [
            {
                "text": chunk,
                "source": source_name,
                "chunk_id": base_id + i,
            }
            for i, chunk in enumerate(text_chunks)
        ]

        # Store'a ekle ve kaydet
        self.store.add(embeddings, metadata)
        self.store.save(self.vector_store_path)

        print(f"[Piri] '{source_name}' öğrenildi. Toplam: {self.store.total_chunks} chunk.")

        return {
            "chunks_added": len(text_chunks),
            "total_chunks": self.store.total_chunks,
            "source": source_name,
        }

    def _check_relevance(self, chunks: List[Dict]) -> bool:
        """
        Retrieval sonuçlarının gerçekten alakalı olup olmadığını kontrol eder.
        En iyi chunk'ın skoru RELEVANCE_THRESHOLD altındaysa → alakasız.
        """
        if not chunks:
            return False
        best_score = max(c.get("score", -999) for c in chunks)
        return best_score >= RELEVANCE_THRESHOLD

    def _filter_relevant_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Sadece threshold üstü skorlu chunk'ları döndürür."""
        return [c for c in chunks if c.get("score", -999) >= RELEVANCE_THRESHOLD]

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
        auto_web_search: bool = True,
    ) -> Dict:
        """
        Akıllı RAG Sorgu — 3 aşamalı pipeline:

        Aşama 1: RAG → Bilgi tabanında ara. Alakalı sonuç varsa cevapla.
        Aşama 2: Web → Bilgi yoksa otomatik web araması yap, öğren, tekrar sor.
        Aşama 3: Dürüst → Hiçbir yerde bulamazsa "bilmiyorum" de.

        Artık ASLA alakasız chunk döndürmez. Negatif skorlu sonuçları filtreler.
        """
        rag_prompt, simple_prompt = self._get_prompts()

        # ── Aşama 1: RAG Retrieval ──────────────────
        retrieval = self.retriever.build_context(question, top_k=top_k, max_context_chars=4000)
        chunks = retrieval.get("chunks", [])
        is_relevant = self._check_relevance(chunks)

        if is_relevant and retrieval["context"]:
            # Alakalı bilgi bulundu → Cevap üret
            relevant_chunks = self._filter_relevant_chunks(chunks)
            
            if self.backend_type == "openai":
                # OpenAI: Tam generative RAG
                messages = [
                    {"role": "system", "content": rag_prompt},
                    {"role": "user", "content": (
                        f"Aşağıdaki bağlam bilgisini kullanarak soruyu Türkçe yanıtla.\n\n"
                        f"## Bağlam\n{retrieval['context']}\n\n"
                        f"## Soru\n{question}"
                    )},
                ]
                raw = self.backend.generate(messages=messages, max_tokens=max_new_tokens, temperature=temperature)
                answer = clean_response(raw)
            else:
                # Local model: Extractive RAG — doğrudan bilgi tabanından
                # Sadece alakalı chunk'ların metinlerini kullan
                relevant_text = "\n\n---\n\n".join(c["text"] for c in relevant_chunks) if relevant_chunks else retrieval["context"]
                answer = extract_best_answer(relevant_text, question)

            return self._build_response(
                answer=answer,
                retrieval=retrieval,
                top_k=top_k,
                method="rag",
            )

        # ── Aşama 2: Otomatik Web Arama → Öğren → Tekrar Sor ──
        if auto_web_search:
            print(f"[Piri] RAG'da alakalı bilgi bulunamadı (en iyi skor: {max((c.get('score', -999) for c in chunks), default=-999):.2f}). Web araması yapılıyor...")
            web_result = self._auto_web_search(question)
            if web_result:
                return web_result

        # ── Aşama 3: Dürüst Cevap ──────────────────
        return self._build_response(
            answer=f"Bu konuda bilgi tabanımda ve web'de yeterli bilgi bulamadım. "
                   f"Lütfen sorunuzu farklı kelimelerle deneyin veya 'Web Ara' sekmesinden "
                   f"manuel olarak arama yapabilirsiniz.",
            retrieval={"context": "", "sources": [], "chunks": [], "num_retrieved": 0},
            top_k=top_k,
            method="no_info",
        )

    def _auto_web_search(self, question: str) -> Optional[Dict]:
        """
        Otomatik web araması yapar, öğrenir ve tekrar sorgular.
        Hata durumunda None döner.
        """
        try:
            from .web_search import search_and_compile
            
            search_result = search_and_compile(query=question, max_results=5)
            if not search_result["results"] or not search_result["compiled_text"]:
                return None

            # Öğren
            learn_result = self.learn_text(
                text=search_result["compiled_text"],
                source_name=search_result["source_name"],
                chunk_size=512,
            )

            if learn_result["chunks_added"] == 0:
                return None

            print(f"[Piri] Web'den {learn_result['chunks_added']} chunk öğrenildi. Tekrar sorguluyorum...")

            # Tekrar RAG sorgusu (ama bu sefer auto_web_search=False → sonsuz döngü engelle)
            result = self.query(question, top_k=3, auto_web_search=False)
            
            # Web arama bilgilerini ekle
            result["web_searched"] = True
            result["web_results"] = [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "snippet": r["body"][:200],
                    "source": r["source"],
                }
                for r in search_result["results"][:5]
            ]
            result["chunks_learned"] = learn_result["chunks_added"]
            result["method"] = "auto_web"
            return result

        except Exception as e:
            print(f"[Piri] Web arama hatası: {e}")
            return None

    def _build_response(self, answer: str, retrieval: Dict, top_k: int, method: str = "rag") -> Dict:
        """Standart response formatı oluşturur. Sadece alakalı kaynakları döndürür."""
        chunks = retrieval.get("chunks", [])
        # Sadece pozitif skorlu chunk'ları göster
        good_chunks = [c for c in chunks if c.get("score", -999) >= RELEVANCE_THRESHOLD]
        good_sources = list(dict.fromkeys(c["source"] for c in good_chunks))
        return {
            "answer": answer,
            "sources": good_sources if good_sources else retrieval.get("sources", [])[:1],
            "context_preview": (
                retrieval["context"][:500] + "..."
                if retrieval.get("context") and len(retrieval["context"]) > 500
                else retrieval.get("context", "")
            ),
            "num_sources": len(good_chunks),
            "retrieval_scores": [
                {"source": c["source"], "score": round(c.get("score", 0.0), 2)}
                for c in good_chunks[:top_k]
            ],
            "model": self.model_name,
            "backend": self.backend_type,
            "method": method,
        }

    def search_only(self, query: str, top_k: int = 5) -> Dict:
        """Sadece semantik arama, üretim yok."""
        chunks = self.retriever.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                    "source": c["source"],
                    "score": round(c.get("score", 0.0), 4),
                }
                for c in chunks
            ],
            "total_found": len(chunks),
        }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Serbest metin üretimi.
        Local model çok kötü Türkçe ürettiği için akıllı yönlendirme:
        1. Basit selamlaşmalar → hazır Türkçe cevap
        2. Bilgi sorusu → RAG'a yönlendir, bulamazsa web'de ara
        3. Gerçek üretim → modele sor (sadece OpenAI backend'de güvenilir)
        """
        prompt_lower = prompt.lower().strip()
        
        # Basit selamlaşmalar — local model bile bunları bozmasın
        greetings = {
            "selam": "Selam! Ben Piri, AKIS Platform'un yapay zeka asistanıyım. Size nasıl yardımcı olabilirim?",
            "merhaba": "Merhaba! Ben Piri. Bana bir soru sorabilir, web'de arama yapabilir veya dosya yükleyerek yeni bilgiler öğretebilirsiniz.",
            "naber": "İyiyim, teşekkürler! Ben Piri, bilgi denizinde harita çıkaran yapay zeka. Size nasıl yardımcı olabilirim?",
            "nasılsın": "Teşekkürler, iyiyim! Ben Piri, AKIS Platform asistanıyım. Size nasıl yardımcı olabilirim?",
            "hey": "Hey! Ben Piri. Sormak istediğiniz bir şey var mı?",
            "sa": "Aleyküm selam! Ben Piri, size nasıl yardımcı olabilirim?",
            "slm": "Selam! Ben Piri, AKIS Platform asistanıyım. Buyurun, nasıl yardımcı olabilirim?",
        }
        
        for key, response in greetings.items():
            if prompt_lower in (key, key + ".", key + "!") or prompt_lower.startswith(key + " "):
                return {"text": response, "model": self.model_name, "backend": self.backend_type, "engine": "Piri", "method": "greeting"}
        
        # Bilgi sorusu algılama — RAG'a yönlendir
        info_keywords = ["nedir", "nasıl", "ne zaman", "nerede", "kimdir", "kaç", "hakkında", "anlat", "açıkla", "bilgi"]
        is_info_question = any(kw in prompt_lower for kw in info_keywords)
        
        if is_info_question and self.store.total_chunks > 0:
            # RAG pipeline'a yönlendir (web arama dahil)
            rag_result = self.query(prompt, top_k=3, auto_web_search=True)
            if rag_result.get("method") != "no_info":
                return {
                    "text": rag_result["answer"],
                    "model": self.model_name,
                    "backend": self.backend_type,
                    "engine": "Piri",
                    "method": "rag_redirect",
                    "sources": rag_result.get("sources", []),
                }
        
        # Gerçek üretim — modele sor
        _, simple_prompt = self._get_prompts()
        messages = [
            {"role": "system", "content": simple_prompt},
            {"role": "user", "content": prompt},
        ]
        raw = self.backend.generate(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer = clean_response(raw, aggressive=False)
        
        # Local model çıktı kalite kontrolü
        if self.backend.is_local and answer:
            # Saçmalık tespiti: çok kısa, tekrarlayan veya anlamsız
            words = answer.split()
            if len(words) < 3:
                answer = f"Bu konuda yeterli bilgi üretemiyorum. 'Soru' sekmesinden RAG ile sorabilir veya 'Web Ara' ile internetten araştırabilirim."
            elif len(set(words)) / max(len(words), 1) < 0.3:
                answer = f"Bu konuda net bir cevap üretemiyorum. Lütfen 'Web Ara' sekmesinden araştırmamı isteyin."
        
        return {
            "text": answer,
            "model": self.model_name,
            "backend": self.backend_type,
            "engine": "Piri",
            "method": "generate",
        }

    def get_stats(self) -> Dict:
        """Sistem istatistikleri."""
        return {
            "engine": "Piri",
            "version": "3.0.0",
            "model": self.model_name,
            "backend": self.backend_type,
            "total_chunks": self.store.total_chunks,
            "embedding_model": self.embedder.model.get_sentence_embedding_dimension(),
            "embedding_dimension": self.embedder.dimension,
            "reranking": self.retriever.reranker is not None,
            "vector_store_path": self.vector_store_path,
            "context_limit_chars": 4000,
        }
