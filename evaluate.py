# evaluate.py — Piri RAG Kalite Değerlendirme
"""
Örnek sorguları Piri RAG pipeline'dan geçirir, sonuçları değerlendirir
ve detaylı rapor üretir.

AKIS Platform — Piri Engine

Kullanım:
    python evaluate.py                     # Varsayılan test seti
    python evaluate.py --question "Sorum?" # Tek soru
    python evaluate.py --json              # JSON formatında çıktı
"""
import argparse
import json
import os
import time
from piri.embedder import Embedder
from piri.vector_store import VectorStore
from piri.retriever import Retriever
from piri.evaluator import PiriEvaluator
from transformers import pipeline as hf_pipeline, AutoTokenizer

# Varsayılan model
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_MODEL_PATH = "./model"

PIRI_SYSTEM_PROMPT = (
    "Sen bilgili bir yapay zeka asistanısın. "
    "Verilen bağlam bilgisini kullanarak soruları doğru ve kapsamlı yanıtla. "
    "Yalnızca bağlamda bulunan bilgilere dayan. "
    "Bağlamda olmayan bilgiyi uydurma."
)

# Varsayılan test soruları
DEFAULT_QUESTIONS = [
    "Chain-of-Thought prompting nedir ve nasıl çalışır?",
    "LoRA fine-tuning tekniği nasıl çalışır?",
    "RAG pipeline hangi adımlardan oluşur?",
    "FAISS nedir ve hangi amaçla kullanılır?",
    "Halüsinasyon nedir ve nasıl önlenir?",
    "AKIS Platform'daki agent'lar nelerdir?",
    "Quantum bilgisayarlar yapay zekayı nasıl etkiler?",  # Bilerek KB dışı
]


def resolve_model(model_arg: str) -> str:
    """Fine-tuned model varsa onu, yoksa HuggingFace modelini kullan."""
    config_path = os.path.join(model_arg, "config.json")
    if os.path.exists(config_path):
        return model_arg
    return DEFAULT_MODEL


def run_rag_query(retriever, generator, tokenizer, question, top_k=3, max_new_tokens=200):
    """Piri RAG sorgusu — ChatML template ile."""
    retrieval = retriever.build_context(question, top_k=top_k, max_context_chars=4000)

    if retrieval["context"]:
        messages = [
            {"role": "system", "content": PIRI_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Bağlam bilgisi:\n{retrieval['context']}\n\nSoru: {question}",
            },
        ]
    else:
        messages = [
            {"role": "system", "content": PIRI_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = output[0]["generated_text"]
    answer = full_text[len(prompt):].strip() if retrieval["context"] else full_text

    return {
        "question": question,
        "answer": answer,
        "full_text": full_text,
        "context": retrieval["context"],
        "chunks": retrieval["chunks"],
        "sources": retrieval["sources"],
    }


def main():
    parser = argparse.ArgumentParser(description="Piri — RAG Kalite Değerlendirme")
    parser.add_argument("--question", "-q", help="Tek bir soru değerlendir")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieval top-k")
    parser.add_argument("--json", action="store_true", help="JSON formatında çıktı")
    parser.add_argument("--model", default=LOCAL_MODEL_PATH, help="Model yolu")
    args = parser.parse_args()

    model_path = resolve_model(args.model)

    print(f"\n{'='*60}")
    print(f"  Piri — RAG Kalite Değerlendirme")
    print(f"  AKIS Platform")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"\nBileşenler yükleniyor...")

    embedder = Embedder()

    store = VectorStore(dimension=embedder.dimension)
    if not store.load("./vector_store"):
        print("[Piri] Hata: Vector store bulunamadı. Önce 'python ingest.py' çalıştırın.")
        return

    retriever = Retriever(embedder, store)
    evaluator = PiriEvaluator(embedder)

    print(f"[Piri] Model yükleniyor: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generator = hf_pipeline(
        "text-generation", model=model_path, tokenizer=tokenizer, device=-1,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    print(f"\n{len(questions)} soru değerlendirilecek...\n")

    all_results = []
    total_start = time.time()

    for i, question in enumerate(questions):
        print(f"{'─'*60}")
        print(f"[{i+1}/{len(questions)}] {question}")
        print(f"{'─'*60}")

        t0 = time.time()
        rag_result = run_rag_query(
            retriever, generator, tokenizer, question,
            top_k=args.top_k,
        )
        query_time = time.time() - t0

        t1 = time.time()
        evaluation = evaluator.evaluate(
            question=rag_result["question"],
            answer=rag_result["answer"],
            context=rag_result["context"],
            chunks=rag_result["chunks"],
        )
        eval_time = time.time() - t1

        result = {
            "question": question,
            "answer": rag_result["answer"][:300],
            "sources": rag_result["sources"],
            "evaluation": evaluation,
            "timing": {
                "query_ms": round(query_time * 1000),
                "eval_ms": round(eval_time * 1000),
            },
        }
        all_results.append(result)

        if not args.json:
            print(evaluation["report"])
            print(f"  Süre: Sorgu {query_time*1000:.0f}ms + Değerlendirme {eval_time*1000:.0f}ms")
            print()

    # Özet
    summary = None
    if len(questions) > 1:
        scores = [r["evaluation"]["overall_score"] for r in all_results]
        verdicts = [r["evaluation"]["verdict"] for r in all_results]

        faith_scores = [r["evaluation"]["metrics"]["faithfulness"]["score"] for r in all_results]
        relev_scores = [r["evaluation"]["metrics"]["relevance"]["score"] for r in all_results]
        coh_scores = [r["evaluation"]["metrics"]["coherence"]["score"] for r in all_results]

        summary = {
            "engine": "Piri",
            "model": model_path,
            "total_questions": len(questions),
            "overall_mean": round(float(sum(scores) / len(scores)), 4),
            "overall_min": round(float(min(scores)), 4),
            "overall_max": round(float(max(scores)), 4),
            "faithfulness_mean": round(float(sum(faith_scores) / len(faith_scores)), 4),
            "relevance_mean": round(float(sum(relev_scores) / len(relev_scores)), 4),
            "coherence_mean": round(float(sum(coh_scores) / len(coh_scores)), 4),
            "verdict_distribution": {v: verdicts.count(v) for v in set(verdicts)},
            "total_time_s": round(time.time() - total_start, 1),
        }

        if not args.json:
            print("\n" + "=" * 60)
            print("  PIRI — GENEL ÖZET")
            print("=" * 60)
            print(f"  Engine:             Piri v1.0.0")
            print(f"  Model:              {summary['model']}")
            print(f"  Toplam soru:        {summary['total_questions']}")
            print(f"  Ortalama skor:      {summary['overall_mean']:.2f}")
            print(f"  Min / Max:          {summary['overall_min']:.2f} / {summary['overall_max']:.2f}")
            print(f"  Faithfulness ort:   {summary['faithfulness_mean']:.2f}")
            print(f"  Relevance ort:      {summary['relevance_mean']:.2f}")
            print(f"  Coherence ort:      {summary['coherence_mean']:.2f}")
            print(f"  Verdict dağılımı:   {summary['verdict_distribution']}")
            print(f"  Toplam süre:        {summary['total_time_s']}s")
            print("=" * 60)

    if args.json:
        output = {
            "engine": "Piri",
            "results": all_results,
            "summary": summary,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
