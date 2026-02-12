"""
Piri Engine — Ana RAG orkestrasyon modülü.
Retrieval + Augmented Generation pipeline'ını yönetir.

Model: Qwen2.5-0.5B-Instruct (32K context window)
"""
import os
from typing import Dict
from transformers import pipeline as hf_pipeline, AutoTokenizer

from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .chunker import load_all_documents, chunk_documents

# Varsayılan model: fine-tuned varsa onu kullan, yoksa HuggingFace'den indir
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_PATH = "./model"

PIRI_SYSTEM_PROMPT = (
    "Sen bilgili bir yapay zeka asistanısın. "
    "Verilen bağlam bilgisini kullanarak soruları doğru ve kapsamlı yanıtla. "
    "Yalnızca bağlamda bulunan bilgilere dayan. "
    "Bağlamda olmayan bilgiyi uydurma."
)


def resolve_model_path(model_path: str) -> str:
    """Fine-tuned model varsa onu, yoksa HuggingFace modelini döndür."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        return model_path
    return DEFAULT_MODEL


class PiriEngine:
    """
    Piri RAG Engine — Retrieve → Augment → Generate

    AKIS Platform'un bilgi motoru. Dokümanları indeksler,
    sorulara bağlam-odaklı cevaplar üretir.
    """

    def __init__(
        self,
        model_path: str = LOCAL_MODEL_PATH,
        vector_store_path: str = "./vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # Embedding modeli
        self.embedder = Embedder(model_name=embedding_model)

        # Vektör store
        self.store = VectorStore(dimension=self.embedder.dimension)
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            self.store.load(vector_store_path)

        # Retriever
        self.retriever = Retriever(self.embedder, self.store)

        # Generator — Qwen2.5-0.5B-Instruct
        resolved_path = resolve_model_path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        self.generator = hf_pipeline(
            "text-generation",
            model=resolved_path,
            tokenizer=self.tokenizer,
            device=-1,
        )

        # Pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_name = resolved_path
        self.vector_store_path = vector_store_path
        print(f"[Piri] Engine hazır. Model: {resolved_path}")
        print(f"[Piri] İndekste {self.store.total_chunks} chunk var.")

    def ingest(
        self,
        directory: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """Bir klasördeki dokümanları indeksler."""
        print(f"[Piri] Dokümanlar yükleniyor: {directory}")
        documents = load_all_documents(directory)
        if not documents:
            print("[Piri] Hiç doküman bulunamadı!")
            return 0

        print(f"[Piri] {len(documents)} doküman bulundu.")

        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"[Piri] {len(chunks)} chunk oluşturuldu.")

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        metadata = []
        for chunk in chunks:
            metadata.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            })

        self.store.add(embeddings, metadata)
        self.store.save(self.vector_store_path)

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict:
        """
        RAG sorgusu: Retrieve → Augment → Generate

        Qwen2.5-0.5B-Instruct: 32K context, ChatML format
        Context limiti 4000 karakter
        """
        # 1. Retrieve
        retrieval = self.retriever.build_context(
            question,
            top_k=top_k,
            max_context_chars=4000,
        )

        # 2. Augment — ChatML formatında prompt oluştur
        if retrieval["context"]:
            augmented_prompt = self._build_rag_prompt(question, retrieval["context"])
        else:
            augmented_prompt = self._build_simple_prompt(question)

        # 3. Generate
        output = self.generator(
            augmented_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = output[0]["generated_text"]

        # Prompt kısmını çıkar, sadece cevabı al
        answer = generated_text[len(augmented_prompt):].strip()
        if not answer:
            answer = generated_text

        return {
            "answer": answer,
            "full_text": generated_text,
            "sources": retrieval["sources"],
            "context_preview": (
                retrieval["context"][:500] + "..."
                if len(retrieval["context"]) > 500
                else retrieval["context"]
            ),
            "num_sources": retrieval["num_retrieved"],
            "retrieval_scores": [
                {"source": c["source"], "score": round(c["score"], 4)}
                for c in retrieval["chunks"][:top_k]
            ],
            "model": self.model_name,
        }

    def search_only(self, query: str, top_k: int = 5) -> Dict:
        """Sadece arama yapar, üretim yapmaz."""
        chunks = self.retriever.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                    "source": c["source"],
                    "score": round(c["score"], 4),
                }
                for c in chunks
            ],
            "total_found": len(chunks),
        }

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """ChatML formatında RAG prompt oluştur."""
        messages = [
            {"role": "system", "content": PIRI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Bağlam bilgisi:\n{context}\n\nSoru: {question}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_simple_prompt(self, question: str) -> str:
        """Bağlam olmadan basit soru prompt'u."""
        messages = [
            {"role": "system", "content": PIRI_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_stats(self) -> Dict:
        """Sistem istatistikleri."""
        return {
            "engine": "Piri",
            "version": "1.0.0",
            "model": self.model_name,
            "total_chunks": self.store.total_chunks,
            "embedding_dimension": self.embedder.dimension,
            "vector_store_path": self.vector_store_path,
            "context_limit_chars": 4000,
        }
