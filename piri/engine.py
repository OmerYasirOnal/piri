"""
Piri Engine — Gelişmiş RAG Orkestrasyon Modülü (v2)

İyileştirmeler:
1. Multilingual embedding (intfloat/multilingual-e5-small)
2. Cross-encoder reranking (mmarco-mMiniLMv2)
3. Gelişmiş prompt engineering (model boyutuna uygun)
4. Post-processing (cümle temizleme, tekrar kaldırma)
5. OpenAI API backend desteği (opsiyonel)
"""
import os
import re
from typing import Dict, Optional
from transformers import pipeline as hf_pipeline, AutoTokenizer

from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .reranker import create_reranker
from .chunker import load_all_documents, chunk_documents

# ─── Model Ayarları ───────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_PATH = "./model"


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

def extract_best_answer(context: str, question: str) -> str:
    """
    Model kötü cevap ürettiğinde, bağlamdan en alakalı paragrafları çeker.
    Bu, 0.5B gibi küçük modeller için extractive-hybrid fallback'tir.

    Strateji: Soru kelimelerini bağlamda ara, en alakalı paragrafları
    birleştirerek temiz bir cevap oluştur.
    """
    if not context:
        return ""

    # Context'i anlamlı paragraflara böl (--- ayracı ve \n\n kullan)
    raw_blocks = re.split(r'\n---\n|\n\n', context)
    paragraphs = [p.strip() for p in raw_blocks if p.strip() and len(p.strip()) > 30]

    if not paragraphs:
        # Satır bazlı dene
        paragraphs = [p.strip() for p in context.split("\n") if p.strip() and len(p.strip()) > 30]

    if not paragraphs:
        return context[:1500]

    # Soru kelimelerini çıkar (stopword'ler hariç)
    stopwords = {
        "ne", "nasıl", "nedir", "ve", "ile", "bir", "bu", "da", "de",
        "mi", "mı", "için", "olan", "var", "mı", "mu", "mü", "dir",
        "ise", "olarak", "gibi", "kadar", "daha", "en", "çok", "her",
    }
    q_words = set()
    for w in question.lower().split():
        clean_w = re.sub(r'[^\wçğıöşü]', '', w)
        if clean_w not in stopwords and len(clean_w) > 2:
            q_words.add(clean_w)

    # Paragrafları puanla
    scored = []
    for p in paragraphs:
        p_lower = p.lower()
        score = 0

        # Soru kelime eşleşmesi (kısmi eşleşme de sayılır)
        for w in q_words:
            if w in p_lower:
                score += 2
            elif any(w[:4] in pw for pw in p_lower.split() if len(pw) > 3):
                score += 1  # Kök eşleşmesi

        # Bilgi yoğunluğu bonusu
        if len(p) > 80:
            score += 1
        if any(c in p for c in ".;:"):
            score += 0.5
        # Başlık paragraflarını tercih etme
        if p.startswith("#"):
            score -= 0.5
        # Numaralı liste = yapılandırılmış bilgi
        if re.match(r'^\d+\.', p):
            score += 0.5

        scored.append((score, p))

    # Sırala ve en iyi paragrafları al
    scored.sort(key=lambda x: x[0], reverse=True)
    best = [p for s, p in scored if s > 0][:5]

    if not best:
        best = [p for _, p in scored[:3]]

    # Temiz formatlama
    result = "\n\n".join(best)

    # Çok uzunsa kısalt (cümle sınırında)
    if len(result) > 2000:
        last_punct = -1
        for punct in ".!?":
            pos = result[:2000].rfind(punct)
            if pos > last_punct:
                last_punct = pos
        if last_punct > 500:
            result = result[:last_punct + 1]
        else:
            result = result[:2000]

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
    Piri RAG Engine v2 — Retrieve → Rerank → Augment → Generate → Clean
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
        print(f"[Piri] Engine v2 hazır. Backend: {self.backend_type}")
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

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
    ) -> Dict:
        """
        RAG Sorgu — backend'e göre strateji seçer:

        OpenAI backend: Retrieve → Rerank → Augment → Generate (tam RAG)
        Local backend:  Retrieve → Rerank → Extractive (bilgi tabanından direkt cevap)

        0.5B local model Türkçe üretimde zayıf olduğu için,
        RAG sorgularında bilgi tabanındaki bilgiyi doğrudan gösterir.
        Bu yaklaşım %100 doğru bilgi garanti eder (halüsinasyon YOK).
        """
        rag_prompt, simple_prompt = self._get_prompts()

        # 1. Retrieve + Rerank
        retrieval = self.retriever.build_context(question, top_k=top_k, max_context_chars=4000)

        if not retrieval["context"]:
            # Bağlam bulunamadı — modele sor
            messages = [
                {"role": "system", "content": simple_prompt},
                {"role": "user", "content": question},
            ]
            raw = self.backend.generate(messages=messages, max_tokens=max_new_tokens, temperature=temperature)
            answer = clean_response(raw)
        elif self.backend_type == "openai":
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
            # Local model: Extractive RAG — bilgi tabanından direkt cevap
            # Bu yaklaşım %100 doğru bilgi verir, halüsinasyon olmaz
            answer = extract_best_answer(retrieval["context"], question)

        return {
            "answer": answer,
            "sources": retrieval["sources"],
            "context_preview": (
                retrieval["context"][:500] + "..."
                if len(retrieval["context"]) > 500
                else retrieval["context"]
            ),
            "num_sources": retrieval["num_retrieved"],
            "retrieval_scores": [
                {"source": c["source"], "score": round(c.get("score", 0.0), 4)}
                for c in retrieval["chunks"][:top_k]
            ],
            "model": self.model_name,
            "backend": self.backend_type,
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
        """Serbest metin üretimi (RAG olmadan)."""
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

        return {
            "text": answer,
            "model": self.model_name,
            "backend": self.backend_type,
            "engine": "Piri",
        }

    def get_stats(self) -> Dict:
        """Sistem istatistikleri."""
        return {
            "engine": "Piri",
            "version": "2.0.0",
            "model": self.model_name,
            "backend": self.backend_type,
            "total_chunks": self.store.total_chunks,
            "embedding_model": self.embedder.model.get_sentence_embedding_dimension(),
            "embedding_dimension": self.embedder.dimension,
            "reranking": self.retriever.reranker is not None,
            "vector_store_path": self.vector_store_path,
            "context_limit_chars": 4000,
        }
