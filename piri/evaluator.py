"""
Piri — RAG Değerlendirme Modülü (Evaluator)

5 boyutlu kalite ölçümü:
1. Faithfulness   — Cevap bağlama ne kadar sadık? (halüsinasyon tespiti)
2. Relevance      — Cevap soruyu ne kadar yanıtlıyor?
3. Context Prec.  — Doğru chunk'lar mı getirildi?
4. Coverage       — Bağlamdaki bilgi cevaba ne kadar yansımış?
5. Coherence      — Cevap tutarlı ve okunabilir mi?

Tüm metrikler 0.0 – 1.0 arasında (1.0 = mükemmel).
"""
import re
import numpy as np
from typing import List, Dict
from .embedder import Embedder


# ────────────────────────────────────────────────────────────────
#  Yardımcı fonksiyonlar
# ────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Metni cümlelere ayır."""
    if not text.strip():
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _extract_ngrams(text: str, n: int = 2) -> List[str]:
    """N-gram'ları çıkar."""
    words = re.findall(r'\w+', text.lower())
    if len(words) < n:
        return words
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektör arası cosine similarity."""
    dot = np.dot(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ────────────────────────────────────────────────────────────────
#  Ana Evaluator Sınıfı
# ────────────────────────────────────────────────────────────────

class PiriEvaluator:
    """Piri RAG pipeline kalite değerlendirici."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    # ── 1. FAITHFULNESS (Halüsinasyon Ölçümü) ──────────────────

    def faithfulness(
        self,
        answer: str,
        context: str,
        threshold: float = 0.35,
    ) -> Dict:
        """
        Cevabın bağlama ne kadar sadık olduğunu ölçer.

        Algoritma:
        1. Cevabı cümlelere ayır
        2. Bağlamı cümlelere ayır
        3. Her cevap cümlesi için bağlamdaki en yakın cümleyle
           cosine similarity hesapla
        4. Threshold altındaki cümleler = HALÜSİNASYON
        """
        answer_sents = _split_sentences(answer)
        context_sents = _split_sentences(context)

        if not answer_sents:
            return {
                "score": 0.0,
                "verdict": "EMPTY",
                "hallucinated_sentences": [],
                "grounded_sentences": [],
                "details": [],
                "explanation": "Cevap boş veya geçerli cümle içermiyor.",
            }

        if not context_sents:
            return {
                "score": 0.0,
                "verdict": "NO_CONTEXT",
                "hallucinated_sentences": answer_sents,
                "grounded_sentences": [],
                "details": [],
                "explanation": "Bağlam boş — tüm cevap potansiyel halüsinasyon.",
            }

        # Embedding'leri hesapla
        answer_embs = self.embedder.embed_texts(answer_sents)
        context_embs = self.embedder.embed_texts(context_sents)

        grounded = []
        hallucinated = []
        details = []

        for i, (sent, emb) in enumerate(zip(answer_sents, answer_embs)):
            similarities = [
                _cosine_similarity(emb, ctx_emb)
                for ctx_emb in context_embs
            ]
            max_sim = max(similarities)
            best_match_idx = int(np.argmax(similarities))
            is_grounded = max_sim >= threshold

            detail = {
                "sentence": sent,
                "max_similarity": round(max_sim, 4),
                "best_match": context_sents[best_match_idx][:100] + "..." if len(context_sents[best_match_idx]) > 100 else context_sents[best_match_idx],
                "grounded": is_grounded,
            }
            details.append(detail)

            if is_grounded:
                grounded.append(sent)
            else:
                hallucinated.append(sent)

        score = len(grounded) / len(answer_sents) if answer_sents else 0.0

        if score >= 0.9:
            verdict = "FAITHFUL"
        elif score >= 0.7:
            verdict = "MOSTLY_FAITHFUL"
        elif score >= 0.5:
            verdict = "PARTIALLY_FAITHFUL"
        else:
            verdict = "HIGH_HALLUCINATION"

        return {
            "score": round(score, 4),
            "verdict": verdict,
            "hallucinated_count": len(hallucinated),
            "grounded_count": len(grounded),
            "total_sentences": len(answer_sents),
            "hallucinated_sentences": hallucinated,
            "grounded_sentences": grounded,
            "details": details,
            "explanation": (
                f"{len(answer_sents)} cümleden {len(grounded)} tanesi bağlama dayalı, "
                f"{len(hallucinated)} tanesi potansiyel halüsinasyon."
            ),
        }

    # ── 2. RELEVANCE (Cevap Uygunluğu) ─────────────────────────

    def relevance(self, question: str, answer: str) -> Dict:
        """Cevabın soruyu ne kadar yanıtladığını ölçer."""
        if not answer.strip():
            return {
                "score": 0.0,
                "verdict": "EMPTY",
                "semantic_similarity": 0.0,
                "keyword_overlap": 0.0,
                "explanation": "Cevap boş.",
            }

        # Semantik benzerlik
        q_emb = self.embedder.embed_query(question)
        a_emb = self.embedder.embed_query(answer)
        semantic_sim = _cosine_similarity(q_emb, a_emb)

        # Anahtar kelime örtüşmesi
        stopwords = {
            "bir", "ve", "ile", "bu", "da", "de", "için", "ne", "mi", "mu",
            "mı", "mü", "olan", "olarak", "gibi", "daha", "çok", "en",
            "the", "is", "are", "a", "an", "and", "or", "of", "in", "to",
            "what", "how", "which", "nedir", "nasıl", "hangi", "kadar",
        }
        q_words = set(re.findall(r'\w+', question.lower())) - stopwords
        a_words = set(re.findall(r'\w+', answer.lower())) - stopwords

        if q_words:
            keyword_overlap = len(q_words & a_words) / len(q_words)
        else:
            keyword_overlap = 0.0

        score = 0.7 * semantic_sim + 0.3 * keyword_overlap

        if score >= 0.6:
            verdict = "RELEVANT"
        elif score >= 0.4:
            verdict = "PARTIALLY_RELEVANT"
        else:
            verdict = "NOT_RELEVANT"

        return {
            "score": round(score, 4),
            "verdict": verdict,
            "semantic_similarity": round(semantic_sim, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "matched_keywords": sorted(q_words & a_words),
            "missing_keywords": sorted(q_words - a_words),
            "explanation": (
                f"Semantik benzerlik: {semantic_sim:.2%}, "
                f"Anahtar kelime örtüşmesi: {keyword_overlap:.2%}."
            ),
        }

    # ── 3. CONTEXT PRECISION (Retrieval Kalitesi) ───────────────

    def context_precision(
        self,
        question: str,
        chunks: List[Dict],
        relevance_threshold: float = 0.3,
    ) -> Dict:
        """Getirilen chunk'ların soruyla ne kadar ilgili olduğunu ölçer."""
        if not chunks:
            return {
                "score": 0.0,
                "verdict": "NO_CHUNKS",
                "relevant_count": 0,
                "total_count": 0,
                "explanation": "Hiç chunk retrieve edilmedi.",
            }

        scores = [c.get("score", 0.0) for c in chunks]
        relevant = [s for s in scores if s >= relevance_threshold]

        precision = len(relevant) / len(scores)
        avg_score = np.mean(scores)
        score_std = np.std(scores)

        # MRR
        mrr = 0.0
        for i, s in enumerate(scores):
            if s >= relevance_threshold:
                mrr = 1.0 / (i + 1)
                break

        composite = (precision * 0.4) + (float(avg_score) * 0.4) + (mrr * 0.2)

        if composite >= 0.5:
            verdict = "GOOD_RETRIEVAL"
        elif composite >= 0.3:
            verdict = "MODERATE_RETRIEVAL"
        else:
            verdict = "POOR_RETRIEVAL"

        return {
            "score": round(composite, 4),
            "verdict": verdict,
            "precision": round(precision, 4),
            "relevant_count": len(relevant),
            "total_count": len(scores),
            "mrr": round(mrr, 4),
            "score_distribution": {
                "mean": round(float(avg_score), 4),
                "std": round(float(score_std), 4),
                "min": round(float(min(scores)), 4),
                "max": round(float(max(scores)), 4),
            },
            "per_chunk": [
                {
                    "source": c.get("source", "?"),
                    "score": round(c.get("score", 0.0), 4),
                    "relevant": c.get("score", 0.0) >= relevance_threshold,
                }
                for c in chunks
            ],
            "explanation": (
                f"{len(chunks)} chunk'tan {len(relevant)} tanesi ilgili "
                f"(precision: {precision:.0%}). "
                f"Ortalama skor: {avg_score:.4f}."
            ),
        }

    # ── 4. COVERAGE (Kapsam Ölçümü) ────────────────────────────

    def coverage(self, answer: str, context: str) -> Dict:
        """Bağlamdaki bilginin cevaba ne kadar yansıdığını ölçer."""
        if not answer.strip() or not context.strip():
            return {
                "score": 0.0,
                "verdict": "EMPTY",
                "ngram_coverage": 0.0,
                "semantic_coverage": 0.0,
                "explanation": "Cevap veya bağlam boş.",
            }

        ctx_bigrams = set(_extract_ngrams(context, n=2))
        ans_bigrams = set(_extract_ngrams(answer, n=2))

        if ctx_bigrams:
            ngram_cov = len(ctx_bigrams & ans_bigrams) / len(ctx_bigrams)
        else:
            ngram_cov = 0.0

        ctx_sents = _split_sentences(context)
        ans_sents = _split_sentences(answer)

        covered = 0
        if ctx_sents and ans_sents:
            ctx_embs = self.embedder.embed_texts(ctx_sents)
            ans_embs = self.embedder.embed_texts(ans_sents)

            for ctx_emb in ctx_embs:
                max_sim = max(
                    _cosine_similarity(ctx_emb, a_emb)
                    for a_emb in ans_embs
                )
                if max_sim >= 0.4:
                    covered += 1
            semantic_cov = covered / len(ctx_sents)
        else:
            semantic_cov = 0.0

        score = 0.4 * ngram_cov + 0.6 * semantic_cov

        if score >= 0.5:
            verdict = "COMPREHENSIVE"
        elif score >= 0.3:
            verdict = "PARTIAL_COVERAGE"
        else:
            verdict = "LOW_COVERAGE"

        return {
            "score": round(score, 4),
            "verdict": verdict,
            "ngram_coverage": round(ngram_cov, 4),
            "semantic_coverage": round(semantic_cov, 4),
            "context_sentences": len(ctx_sents),
            "covered_sentences": int(covered) if ctx_sents and ans_sents else 0,
            "explanation": (
                f"Bağlamdaki {len(ctx_sents)} cümleden "
                f"{int(covered) if ctx_sents and ans_sents else 0} tanesi cevaba yansımış. "
                f"N-gram kapsam: {ngram_cov:.0%}, Semantik kapsam: {semantic_cov:.0%}."
            ),
        }

    # ── 5. COHERENCE (Tutarlılık) ───────────────────────────────

    def coherence(self, answer: str) -> Dict:
        """Cevabın tutarlılığını ve okunabilirliğini ölçer."""
        if not answer.strip():
            return {
                "score": 0.0,
                "verdict": "EMPTY",
                "explanation": "Cevap boş.",
            }

        words = re.findall(r'\w+', answer.lower())
        sentences = _split_sentences(answer)

        # 1. Tekrar tespiti (trigram repetition)
        if len(words) >= 3:
            trigrams = _extract_ngrams(answer, n=3)
            unique_trigrams = set(trigrams)
            repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0.0
        else:
            repetition_ratio = 0.0

        repetition_score = max(0.0, 1.0 - repetition_ratio * 3)

        # 2. Type-Token Ratio
        if words:
            ttr = len(set(words)) / len(words)
        else:
            ttr = 0.0

        # 3. Cümle çeşitliliği
        avg_sent_sim = 0.0
        if len(sentences) >= 2:
            sent_embs = self.embedder.embed_texts(sentences)
            sim_sum = 0.0
            pair_count = 0
            for i in range(len(sent_embs)):
                for j in range(i + 1, len(sent_embs)):
                    sim_sum += _cosine_similarity(sent_embs[i], sent_embs[j])
                    pair_count += 1
            avg_sent_sim = sim_sum / pair_count if pair_count else 0.0
            diversity = max(0.0, 1.0 - avg_sent_sim)
        else:
            diversity = 0.5

        # 4. Yapısal kalite
        avg_sent_len = 0
        if sentences:
            avg_sent_len = np.mean([len(s.split()) for s in sentences])
            if 8 <= avg_sent_len <= 25:
                structure_score = 1.0
            elif 5 <= avg_sent_len <= 35:
                structure_score = 0.7
            else:
                structure_score = 0.3
        else:
            structure_score = 0.0

        score = (
            repetition_score * 0.35 +
            ttr * 0.20 +
            diversity * 0.25 +
            structure_score * 0.20
        )

        if score >= 0.6:
            verdict = "COHERENT"
        elif score >= 0.4:
            verdict = "PARTIALLY_COHERENT"
        else:
            verdict = "INCOHERENT"

        return {
            "score": round(score, 4),
            "verdict": verdict,
            "repetition_score": round(repetition_score, 4),
            "repetition_ratio": round(repetition_ratio, 4),
            "type_token_ratio": round(ttr, 4),
            "diversity": round(diversity, 4),
            "structure_score": round(structure_score, 4),
            "stats": {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "unique_words": len(set(words)),
                "avg_sentence_length": round(float(avg_sent_len), 1),
            },
            "explanation": (
                f"{len(words)} kelime, {len(sentences)} cümle. "
                f"Tekrar oranı: {repetition_ratio:.0%}, "
                f"Bilgi yoğunluğu: {ttr:.0%}, "
                f"Çeşitlilik: {diversity:.0%}."
            ),
        }

    # ── TOPLAM DEĞERLENDİRME ───────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        chunks: List[Dict],
    ) -> Dict:
        """
        Tüm boyutları ölçer ve genel bir kalite skoru üretir.

        Returns:
            overall_score: 0.0-1.0 ağırlıklı toplam
            verdict: EXCELLENT / GOOD / ACCEPTABLE / POOR / VERY_POOR
            metrics: her boyutun detaylı sonucu
            report: insan-okunabilir rapor
        """
        faith = self.faithfulness(answer, context)
        relev = self.relevance(question, answer)
        ctx_prec = self.context_precision(question, chunks)
        cov = self.coverage(answer, context)
        coh = self.coherence(answer)

        weights = {
            "faithfulness": 0.30,
            "relevance": 0.25,
            "context_precision": 0.15,
            "coverage": 0.15,
            "coherence": 0.15,
        }

        overall = (
            faith["score"] * weights["faithfulness"] +
            relev["score"] * weights["relevance"] +
            ctx_prec["score"] * weights["context_precision"] +
            cov["score"] * weights["coverage"] +
            coh["score"] * weights["coherence"]
        )

        if overall >= 0.75:
            verdict = "EXCELLENT"
        elif overall >= 0.6:
            verdict = "GOOD"
        elif overall >= 0.45:
            verdict = "ACCEPTABLE"
        elif overall >= 0.3:
            verdict = "POOR"
        else:
            verdict = "VERY_POOR"

        report = self._build_report(
            question, answer, faith, relev, ctx_prec, cov, coh,
            overall, verdict, weights,
        )

        return {
            "overall_score": round(overall, 4),
            "verdict": verdict,
            "weights": weights,
            "metrics": {
                "faithfulness": {
                    "score": faith["score"],
                    "verdict": faith["verdict"],
                    "explanation": faith["explanation"],
                },
                "relevance": {
                    "score": relev["score"],
                    "verdict": relev["verdict"],
                    "explanation": relev["explanation"],
                },
                "context_precision": {
                    "score": ctx_prec["score"],
                    "verdict": ctx_prec["verdict"],
                    "explanation": ctx_prec["explanation"],
                },
                "coverage": {
                    "score": cov["score"],
                    "verdict": cov["verdict"],
                    "explanation": cov["explanation"],
                },
                "coherence": {
                    "score": coh["score"],
                    "verdict": coh["verdict"],
                    "explanation": coh["explanation"],
                },
            },
            "detailed": {
                "faithfulness": faith,
                "relevance": relev,
                "context_precision": ctx_prec,
                "coverage": cov,
                "coherence": coh,
            },
            "report": report,
        }

    def _build_report(
        self, question, answer, faith, relev, ctx_prec, cov, coh,
        overall, verdict, weights,
    ) -> str:
        """İnsan-okunabilir değerlendirme raporu."""
        bar = lambda score: "█" * int(score * 20) + "░" * (20 - int(score * 20))

        lines = [
            "=" * 60,
            "  PIRI — RAG KALİTE DEĞERLENDİRME RAPORU",
            "=" * 60,
            f"  Soru: {question[:80]}{'...' if len(question) > 80 else ''}",
            f"  Cevap: {answer[:80]}{'...' if len(answer) > 80 else ''}",
            "-" * 60,
            "",
            f"  GENEL SKOR: {overall:.2f}/1.00  [{verdict}]",
            f"  {bar(overall)} {overall:.0%}",
            "",
            "  BOYUT BAZLI SONUÇLAR:",
            f"  {'Metrik':<22} {'Skor':>6}  {'Ağırlık':>8}  Durum",
            f"  {'─'*22} {'─'*6}  {'─'*8}  {'─'*20}",
            f"  Faithfulness         {faith['score']:>5.2f}   {weights['faithfulness']:>6.0%}    {faith['verdict']}",
            f"  Relevance            {relev['score']:>5.2f}   {weights['relevance']:>6.0%}    {relev['verdict']}",
            f"  Context Precision    {ctx_prec['score']:>5.2f}   {weights['context_precision']:>6.0%}    {ctx_prec['verdict']}",
            f"  Coverage             {cov['score']:>5.2f}   {weights['coverage']:>6.0%}    {cov['verdict']}",
            f"  Coherence            {coh['score']:>5.2f}   {weights['coherence']:>6.0%}    {coh['verdict']}",
            "",
            "  DETAYLAR:",
            f"  • {faith['explanation']}",
            f"  • {relev['explanation']}",
            f"  • {ctx_prec['explanation']}",
            f"  • {cov['explanation']}",
            f"  • {coh['explanation']}",
        ]

        if faith.get("hallucinated_sentences"):
            lines.append("")
            lines.append("  ⚠ HALÜSİNASYON TESPİTİ:")
            for sent in faith["hallucinated_sentences"][:3]:
                lines.append(f"    → \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
