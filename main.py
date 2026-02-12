# main.py — Piri API Server
"""
Piri — Lightweight LLM + RAG Engine by AKIS Platform

FastAPI tabanlı REST API. Metin üretimi, RAG soru-cevap,
semantik arama ve 5 boyutlu kalite değerlendirme sunar.

Kullanım:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer

# ─── Model Çözümleme ─────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_PATH = "./model"


def resolve_model():
    """Fine-tuned model varsa onu, yoksa HuggingFace modelini kullan."""
    if os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        return LOCAL_MODEL_PATH
    return DEFAULT_MODEL


MODEL_PATH = resolve_model()

app = FastAPI(
    title="Piri — AKIS Platform LLM + RAG Engine",
    version="1.0.0",
    description=(
        "Piri: Bilgi denizinde harita çıkaran küçük ama güçlü yapay zeka motoru. "
        "Qwen2.5-0.5B-Instruct + Retrieval-Augmented Generation sistemi. "
        "AKIS Platform tarafından geliştirilmiştir."
    ),
)

# ─── Model Yükleme ───────────────────────────────────────────
print(f"[Piri] Model yükleniyor: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
generator = pipeline(
    "text-generation",
    model=MODEL_PATH,
    tokenizer=tokenizer,
    device=-1,  # CPU
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[Piri] Model hazır: {MODEL_PATH}")

# ─── RAG Engine Yükleme ──────────────────────────────────────
piri_engine = None
piri_evaluator = None
if os.path.exists("vector_store/index.faiss"):
    from piri import PiriEngine
    from piri.evaluator import PiriEvaluator
    piri_engine = PiriEngine(
        model_path=MODEL_PATH,
        vector_store_path="./vector_store",
    )
    piri_evaluator = PiriEvaluator(piri_engine.embedder)
    print("[Piri] RAG Engine + Evaluator yüklendi.")
else:
    print("[Piri] Uyarı: vector_store bulunamadı. RAG devre dışı.")
    print("[Piri] Önce 'python ingest.py' çalıştırın.")


# ─── Request/Response Modelleri ───────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    use_chat_template: bool = True


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.7


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class IngestRequest(BaseModel):
    directory: str = "knowledge_base"
    chunk_size: int = 512
    chunk_overlap: int = 64


class EvaluateRequest(BaseModel):
    question: str
    top_k: int = 3
    max_new_tokens: int = 200


# ─── Endpoint'ler ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "engine": "Piri",
        "tagline": "Bilgi denizinde harita çıkaran yapay zeka motoru",
        "version": "1.0.0",
        "by": "AKIS Platform",
        "model": MODEL_PATH,
        "endpoints": {
            "docs": "/docs",
            "generate": "POST /generate — Serbest metin üretimi",
            "rag_query": "POST /rag/query — RAG ile soru-cevap",
            "rag_search": "POST /rag/search — Bilgi tabanında arama",
            "rag_evaluate": "POST /rag/evaluate — Sorgu + kalite değerlendirme",
            "rag_ingest": "POST /rag/ingest — Yeni doküman indeksle",
            "rag_stats": "GET /rag/stats — RAG istatistikleri",
        },
        "rag_status": "aktif" if piri_engine else "devre dışı",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    """Serbest metin üretimi (RAG olmadan). ChatML template destekli."""
    if request.use_chat_template:
        messages = [
            {"role": "system", "content": "Sen yardımsever bir yapay zeka asistanısın."},
            {"role": "user", "content": request.prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = request.prompt

    output = generator(
        formatted_prompt,
        max_new_tokens=request.max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        temperature=request.temperature,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = output[0]["generated_text"]
    answer = full_text[len(formatted_prompt):].strip()

    return {
        "text": answer if answer else full_text,
        "model": MODEL_PATH,
        "engine": "Piri",
    }


@app.post("/rag/query")
def rag_query(request: RAGQueryRequest):
    """RAG ile soru-cevap. Bilgi tabanından bağlam bulur, modele verir."""
    if not piri_engine:
        raise HTTPException(
            status_code=503,
            detail="[Piri] RAG sistemi aktif değil. Önce 'python ingest.py' çalıştırın.",
        )

    result = piri_engine.query(
        question=request.question,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    return result


@app.post("/rag/search")
def rag_search(request: RAGSearchRequest):
    """Bilgi tabanında semantik arama. Üretim yapmaz."""
    if not piri_engine:
        raise HTTPException(
            status_code=503,
            detail="[Piri] RAG sistemi aktif değil. Önce 'python ingest.py' çalıştırın.",
        )

    return piri_engine.search_only(
        query=request.query,
        top_k=request.top_k,
    )


@app.post("/rag/ingest")
def rag_ingest(request: IngestRequest):
    """Yeni dokümanları indeksle."""
    global piri_engine

    if not os.path.exists(request.directory):
        raise HTTPException(
            status_code=404,
            detail=f"[Piri] Klasör bulunamadı: {request.directory}",
        )

    if not piri_engine:
        from piri import PiriEngine
        piri_engine = PiriEngine(
            model_path=MODEL_PATH,
            vector_store_path="./vector_store",
        )

    num_chunks = piri_engine.ingest(
        directory=request.directory,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    return {
        "message": f"[Piri] {num_chunks} chunk başarıyla indekslendi.",
        "directory": request.directory,
        "total_chunks_in_store": piri_engine.store.total_chunks,
    }


@app.post("/rag/evaluate")
def rag_evaluate(request: EvaluateRequest):
    """
    RAG sorgusu + 5 boyutlu kalite değerlendirmesi.

    Metrikler: Faithfulness, Relevance, Context Precision, Coverage, Coherence
    """
    if not piri_engine or not piri_evaluator:
        raise HTTPException(
            status_code=503,
            detail="[Piri] RAG sistemi aktif değil. Önce 'python ingest.py' çalıştırın.",
        )

    # 1. RAG sorgusu
    rag_result = piri_engine.query(
        question=request.question,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
    )

    # 2. Retrieve edilen chunk'ları al
    retrieval = piri_engine.retriever.build_context(
        request.question, top_k=request.top_k,
    )

    # 3. Değerlendir
    evaluation = piri_evaluator.evaluate(
        question=request.question,
        answer=rag_result["answer"],
        context=retrieval["context"],
        chunks=retrieval["chunks"],
    )

    return {
        "question": request.question,
        "answer": rag_result["answer"],
        "sources": rag_result["sources"],
        "evaluation": {
            "overall_score": evaluation["overall_score"],
            "verdict": evaluation["verdict"],
            "metrics": evaluation["metrics"],
        },
        "report": evaluation["report"],
        "model": MODEL_PATH,
        "engine": "Piri",
    }


@app.get("/rag/stats")
def rag_stats():
    """Piri RAG sistemi istatistikleri."""
    if not piri_engine:
        return {
            "engine": "Piri",
            "status": "devre dışı",
            "message": "Önce 'python ingest.py' çalıştırın.",
        }

    stats = piri_engine.get_stats()
    stats["status"] = "aktif"
    return stats
