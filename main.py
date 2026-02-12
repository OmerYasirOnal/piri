# main.py — Piri API Server v2
"""
Piri — Lightweight LLM + RAG Engine by AKIS Platform

v2: Gelişmiş kalite — multilingual embedding, reranking,
    OpenAI backend, prompt engineering, post-processing.

Kullanım:
    uvicorn main:app --host 0.0.0.0 --port 8000

    # OpenAI backend ile (çok daha kaliteli):
    # .env dosyasına OPENAI_API_KEY=sk-... ekleyin
"""
import os
import tempfile
import hashlib
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# ─── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Piri — AKIS Platform LLM + RAG Engine",
    version="2.0.0",
    description=(
        "Piri v2: Bilgi denizinde harita çıkaran yapay zeka motoru. "
        "Multilingual embedding, cross-encoder reranking, OpenAI backend desteği. "
        "AKIS Platform tarafından geliştirilmiştir."
    ),
)

# ─── Static Files ─────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─── Piri Engine Yükleme ─────────────────────────────────────
piri_engine = None
piri_evaluator = None

if os.path.exists("vector_store/index.faiss"):
    from piri import PiriEngine
    from piri.evaluator import PiriEvaluator
    piri_engine = PiriEngine(
        vector_store_path="./vector_store",
    )
    piri_evaluator = PiriEvaluator(piri_engine.embedder)
    print("[Piri] Engine + Evaluator yüklendi.")
else:
    # Engine'i RAG olmadan da yükle (generate için)
    from piri import PiriEngine
    try:
        piri_engine = PiriEngine(
            vector_store_path="./vector_store",
        )
        print("[Piri] Engine yüklendi (RAG devre dışı, generate aktif).")
    except Exception as e:
        print(f"[Piri] Engine yüklenemedi: {e}")


# ─── Request/Response Modelleri ───────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 3
    max_new_tokens: int = 300
    temperature: float = 0.3


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
    max_new_tokens: int = 300


class LearnTextRequest(BaseModel):
    text: str
    source_name: str = "kullanici_dokumani"
    chunk_size: int = 512


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5
    auto_learn: bool = True
    top_k: int = 3


# ─── Endpoint'ler ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"engine": "Piri", "ui": "static/index.html bulunamadı"}


@app.get("/api/info")
def api_info():
    return {
        "engine": "Piri",
        "version": "2.0.0",
        "by": "AKIS Platform",
        "model": piri_engine.model_name if piri_engine else "yüklenmedi",
        "backend": piri_engine.backend_type if piri_engine else "yok",
        "rag_status": "aktif" if (piri_engine and piri_engine.store.total_chunks > 0) else "devre dışı",
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    """Serbest metin üretimi (RAG olmadan)."""
    if not piri_engine:
        raise HTTPException(status_code=503, detail="Piri Engine yüklenemedi.")

    return piri_engine.generate_text(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )


@app.post("/rag/query")
def rag_query(request: RAGQueryRequest):
    """RAG ile soru-cevap. Bilgi tabanından bağlam bulur, rerank eder, modele verir."""
    if not piri_engine or piri_engine.store.total_chunks == 0:
        raise HTTPException(
            status_code=503,
            detail="RAG sistemi aktif değil. Önce 'python ingest.py' çalıştırın.",
        )

    return piri_engine.query(
        question=request.question,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )


@app.post("/rag/search")
def rag_search(request: RAGSearchRequest):
    """Bilgi tabanında semantik arama. Üretim yapmaz."""
    if not piri_engine or piri_engine.store.total_chunks == 0:
        raise HTTPException(
            status_code=503,
            detail="RAG sistemi aktif değil. Önce 'python ingest.py' çalıştırın.",
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
            detail=f"Klasör bulunamadı: {request.directory}",
        )

    if not piri_engine:
        from piri import PiriEngine
        piri_engine = PiriEngine(vector_store_path="./vector_store")

    num_chunks = piri_engine.ingest(
        directory=request.directory,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    return {
        "message": f"{num_chunks} chunk başarıyla indekslendi.",
        "directory": request.directory,
        "total_chunks_in_store": piri_engine.store.total_chunks,
    }


@app.post("/rag/evaluate")
def rag_evaluate(request: EvaluateRequest):
    """RAG sorgusu + 5 boyutlu kalite değerlendirmesi."""
    if not piri_engine or not piri_evaluator:
        raise HTTPException(
            status_code=503,
            detail="RAG/Evaluator aktif değil. Önce 'python ingest.py' çalıştırın.",
        )

    rag_result = piri_engine.query(
        question=request.question,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
    )

    retrieval = piri_engine.retriever.build_context(
        request.question, top_k=request.top_k,
    )

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
        "model": piri_engine.model_name,
        "backend": piri_engine.backend_type,
        "engine": "Piri v2",
    }


@app.post("/rag/learn")
def rag_learn(request: LearnTextRequest):
    """
    Metin yapıştırarak Piri'ye öğret.
    Herhangi bir metni doğrudan gönderip bilgi tabanına ekleyebilirsiniz.
    """
    global piri_engine, piri_evaluator

    if not piri_engine:
        from piri import PiriEngine
        piri_engine = PiriEngine(vector_store_path="./vector_store")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    if len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Metin en az 20 karakter olmalı.")

    result = piri_engine.learn_text(
        text=request.text,
        source_name=request.source_name,
        chunk_size=request.chunk_size,
    )

    # Evaluator yoksa oluştur
    if piri_evaluator is None and piri_engine.store.total_chunks > 0:
        from piri.evaluator import PiriEvaluator
        piri_evaluator = PiriEvaluator(piri_engine.embedder)

    return {
        "message": f"'{request.source_name}' başarıyla öğrenildi!",
        "chunks_added": result["chunks_added"],
        "total_chunks": result["total_chunks"],
        "source": result["source"],
        "char_count": len(request.text),
    }


@app.post("/rag/web-search")
def rag_web_search(request: WebSearchRequest):
    """
    Web'de ara → Öğren → Cevapla.

    1. DuckDuckGo ile internette arama yapar
    2. Sonuçları Piri'nin bilgi tabanına ekler (auto_learn=true ise)
    3. Yeni öğrenilen bilgiyle soruyu cevaplar

    API key gerektirmez, ücretsiz çalışır.
    """
    global piri_engine, piri_evaluator

    from piri.web_search import search_and_compile

    # 1. Web'de ara
    search_result = search_and_compile(
        query=request.query,
        max_results=request.max_results,
    )

    if not search_result["results"]:
        return {
            "query": request.query,
            "web_results": [],
            "learned": False,
            "answer": "Web aramasında sonuç bulunamadı.",
            "sources": [],
        }

    # 2. Sonuçları öğren (bilgi tabanına ekle)
    learned = False
    chunks_added = 0
    if request.auto_learn and piri_engine and search_result["compiled_text"]:
        learn_result = piri_engine.learn_text(
            text=search_result["compiled_text"],
            source_name=search_result["source_name"],
            chunk_size=512,
        )
        chunks_added = learn_result["chunks_added"]
        learned = chunks_added > 0

        # Evaluator yoksa oluştur
        if piri_evaluator is None and piri_engine.store.total_chunks > 0:
            from piri.evaluator import PiriEvaluator
            piri_evaluator = PiriEvaluator(piri_engine.embedder)

    # 3. Yeni öğrenilen bilgiyle cevapla
    answer = ""
    sources = []
    if piri_engine and piri_engine.store.total_chunks > 0:
        rag_result = piri_engine.query(
            question=request.query,
            top_k=request.top_k,
        )
        answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

    return {
        "query": request.query,
        "web_results": [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["body"][:200],
                "source": r["source"],
            }
            for r in search_result["results"]
        ],
        "learned": learned,
        "chunks_added": chunks_added,
        "total_chunks": piri_engine.store.total_chunks if piri_engine else 0,
        "answer": answer,
        "sources": sources,
        "source_name": search_result["source_name"],
    }


@app.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
    chunk_size: int = Form(512),
):
    """
    Dosya yükleyerek Piri'ye öğret.
    Desteklenen formatlar: .txt, .md, .csv, .json, .py, .js, .html, .xml, .yaml, .yml
    """
    global piri_engine, piri_evaluator

    if not piri_engine:
        from piri import PiriEngine
        piri_engine = PiriEngine(vector_store_path="./vector_store")

    # Dosya uzantısı kontrolü
    allowed_extensions = {
        ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
        ".html", ".xml", ".yaml", ".yml", ".log", ".rst",
        ".tex", ".sql", ".sh", ".bash", ".r", ".java", ".go",
        ".c", ".cpp", ".h", ".hpp", ".css", ".scss",
    }
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {ext}. Desteklenenler: {', '.join(sorted(allowed_extensions))}",
        )

    # Dosyayı oku
    try:
        raw_bytes = await file.read()
        # UTF-8 dene, olmazsa latin-1
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dosya okunamadı: {str(e)}")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Dosya boş.")

    # Dosya boyutu limiti (5MB metin)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Dosya çok büyük (maks. 5MB metin).")

    # Source name
    final_source = source_name or file.filename or "yuklenilen_dosya"

    result = piri_engine.learn_text(
        text=content,
        source_name=final_source,
        chunk_size=chunk_size,
    )

    # Evaluator yoksa oluştur
    if piri_evaluator is None and piri_engine.store.total_chunks > 0:
        from piri.evaluator import PiriEvaluator
        piri_evaluator = PiriEvaluator(piri_engine.embedder)

    return {
        "message": f"'{final_source}' başarıyla yüklendi ve öğrenildi!",
        "filename": file.filename,
        "file_size": len(raw_bytes),
        "char_count": len(content),
        "chunks_added": result["chunks_added"],
        "total_chunks": result["total_chunks"],
        "source": result["source"],
    }


@app.get("/rag/stats")
def rag_stats():
    """Piri RAG sistemi istatistikleri."""
    if not piri_engine:
        return {"engine": "Piri", "status": "yüklenemedi"}

    stats = piri_engine.get_stats()
    stats["status"] = "aktif" if piri_engine.store.total_chunks > 0 else "devre dışı"
    return stats
