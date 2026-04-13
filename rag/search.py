"""
RAG Search - Query expansion, Classification, Enhanced search pipeline
"""
import re

import numpy as np

from .config import WEIGHT_PROFILES, CONFIDENCE_THRESHOLD, EXACT_MATCH_BOOST, EXACT_WORD_BOOST
from .ollama import _ollama_ping, _ollama_call_simple, semantic_boost_query


# ── Query expansion ──────────────────────────────────────────
# Ерөнхий abbreviation → full form толь бичиг.
_ABBR: dict[str, str] = {
    # Академик / ажлын байр
    "cv":    "curriculum vitae resume",
    "gpa":   "grade point average score",
    "phd":   "doctor of philosophy doctorate",
    "mba":   "master of business administration",
    "hr":    "human resources personnel",
    "kpi":   "key performance indicator metric",
    "roi":   "return on investment profit",
    "qa":    "quality assurance testing",
    "r&d":   "research and development",
    # Технологи
    "ai":    "artificial intelligence",
    "ml":    "machine learning",
    "llm":   "large language model",
    "nlp":   "natural language processing",
    "api":   "application programming interface",
    "db":    "database",
    "ui":    "user interface",
    "ux":    "user experience",
    "os":    "operating system",
    "gpu":   "graphics processing unit",
    "cpu":   "central processing unit",
    "ram":   "random access memory",
    # Санхүү / бизнес
    "ceo":   "chief executive officer",
    "cto":   "chief technology officer",
    "b2b":   "business to business",
    "b2c":   "business to consumer",
    "ipo":   "initial public offering",
    # Хэлний шалгалт / сертификат
    "jlpt":  "japanese language proficiency test N1 N2 N3 N4 N5 япон хэл",
    "n1":    "JLPT N1 japanese language proficiency advanced",
    "n2":    "JLPT N2 japanese language proficiency",
    "n3":    "JLPT N3 japanese language proficiency intermediate",
    "n4":    "JLPT N4 japanese language proficiency",
    "n5":    "JLPT N5 japanese language proficiency basic",
    "toefl": "test of english foreign language TOEFL score",
    "ielts": "international english language testing system IELTS band",
    "topik": "test of proficiency korean language TOPIK солонгос хэл",
    "hsk":   "hanyu shuiping kaoshi chinese proficiency HSK хятад хэл",
    # Монгол товчилсон үгс
    "ббсб":  "банк бус санхүүгийн байгууллага",
    "ааноат": "аж ахуйн нэгж байгууллагын орлогын албан татвар",
    "хаан":  "хаан банк",
    "тэш":   "тусгай эрхийн шаардлага",
}

# ── Domain-specific expansion ─────────────────────────────────
_DOMAIN_EXPAND: dict[str, list[str]] = {
    "jlpt": ["N1", "N2", "N3", "N4", "N5", "japanese language", "япон хэл", "日本語能力試験"],
    "японы": ["jlpt", "N1", "N2", "N3", "N4", "N5", "japanese", "nihongo"],
    "япон хэл": ["jlpt", "N1", "N2", "N3", "N4", "N5", "japanese language proficiency"],
    "toefl": ["english", "англи хэл", "score", "iBT", "PBT"],
    "ielts": ["english", "англи хэл", "band", "academic", "general"],
    "topik": ["korean", "солонгос хэл", "level", "한국어"],
    "hsk":   ["chinese", "хятад хэл", "level", "汉语"],
    "банк":  ["хүү", "зээл", "данс", "гүйлгээ", "шилжүүлэг"],
    "зээл":  ["хүү", "банк", "loan", "interest", "төлбөр"],
}


def expand_query(query: str) -> list[str]:
    """
    Асуултыг өргөтгөнө:
      1. Бүтэн асуулт (exact)
      2. Abbreviation-г full form-оор солих
      3. Domain-specific expansion (JLPT → N1, N2, N3...)
      4. Хоёр үгийн хослол шалгах
    """
    q_lower = query.lower().strip()
    tokens = re.findall(r"\w+", q_lower, re.UNICODE)
    expanded = [q_lower]

    # Аббревиатур → full form
    for tok in tokens:
        full = _ABBR.get(tok)
        if full and full not in expanded:
            expanded.append(full)
        # Монгол дотор Англи abbreviation байж болно
        full2 = _ABBR.get(tok.lower())
        if full2 and full2 not in expanded:
            expanded.append(full2)

    # Domain-specific expansion (JLPT → N1, N2, N3, N4, N5, япон хэл...)
    for tok in tokens:
        domain_terms = _DOMAIN_EXPAND.get(tok.lower())
        if domain_terms:
            for term in domain_terms:
                if term.lower() not in expanded:
                    expanded.append(term)

    # Хоёр үгийн хослол шалгах (жнь "япон хэл")
    for i, t1 in enumerate(tokens[:-1]):
        combo_key = f"{t1} {tokens[i + 1]}"
        domain_terms = _DOMAIN_EXPAND.get(combo_key)
        if domain_terms:
            for term in domain_terms:
                if term.lower() not in expanded:
                    expanded.append(term)
        # _ABBR-д бас шалгана
        combo = _ABBR.get(combo_key)
        if combo and combo not in expanded:
            expanded.append(combo)

    return expanded


# ── Query classification ─────────────────────────────────────
def classify_query_type(query: str) -> str:
    """
    Query-ийн төрлийг тодорхойлж, зохих weight profile сонгоно.
    - factual: нэр, огноо, тоо (яг тодорхой мэдээлэл)
    - semantic: ойлголт, утга, тайлбар (семантик хайлт)
    - exact: "...", code, тодорхой phrase
    - mixed: холимог
    """
    q = query.lower().strip()

    # Exact match patterns
    if '"' in query or "'" in query:
        return "exact"
    if re.search(r'\b(яг|exact|literal|код|code)\b', q, re.UNICODE):
        return "exact"

    # Factual patterns - тоо, огноо, нэр
    if re.search(r'\d{4}|\d+%|\$\d+|₮\d+', q):  # огноо, хувь, мөнгө
        return "factual"
    if re.search(r'\b(хэд|хэзээ|хэн|аль|ямар|хаана|when|where|who|how many|which)\b', q, re.UNICODE):
        return "factual"

    # Semantic patterns - ойлголт, тайлбар
    if re.search(r'\b(яагаад|юу|тайлбарла|ялгаа|учир|why|what|explain|difference|meaning|concept)\b', q, re.UNICODE):
        return "semantic"
    if len(q.split()) > 8:  # урт асуулт = семантик
        return "semantic"

    return "mixed"


def get_dynamic_weights(query: str) -> dict:
    """Query төрөлд тохирсон жингүүдийг буцаана."""
    qtype = classify_query_type(query)
    return WEIGHT_PROFILES.get(qtype, WEIGHT_PROFILES["mixed"])


# ── Sub-query decomposition ──────────────────────────────────
def decompose_query(model: str, query: str) -> list[str]:
    """
    Sub-query decomposition: нарийн асуултыг хэсэг болгож задална.
    """
    # Хялбар тохиолдол: "болон", "ба", "and", "or" гэх мэт
    simple_splits = re.split(r'\s+(?:болон|ба|бол|and|or|vs\.?|versus)\s+', query, flags=re.IGNORECASE)
    if len(simple_splits) > 1 and all(len(s.strip()) > 3 for s in simple_splits):
        return [s.strip() for s in simple_splits if s.strip()]

    # LLM-ээр задлах
    if not _ollama_ping():
        return [query]

    prompt = (
        "Break down this search query into 2-4 simpler sub-queries if it contains multiple questions.\n"
        "If it's already simple, return it as-is.\n"
        "Output ONLY the sub-queries, one per line. No numbering, no explanation.\n\n"
        f"Query: {query}\n\nSub-queries:"
    )
    result = _ollama_call_simple(model, prompt, timeout=10)
    if not result:
        return [query]

    subqueries = [sq.strip().strip('-•').strip() for sq in result.split('\n') if sq.strip()]
    subqueries = [sq for sq in subqueries if len(sq) > 3 and sq.lower() != query.lower()]

    return subqueries[:4] if subqueries else [query]


# ── Confidence scoring ───────────────────────────────────────
def calculate_confidence_score(chunk: dict, query: str, weights: dict) -> float:
    """
    Answer confidence scoring: хариултын найдвартай байдлыг тооцно.
    """
    score = chunk.get("score", 0.0)
    v_score = chunk.get("vector_score", 0.0)
    b_score = chunk.get("bm25_score", 0.0)

    # Үндсэн оноо
    confidence = score

    # Exact match bonus - яг таарсан үгстэй бол найдвартай
    exact_bonus = chunk.get("exact_bonus", 0.0)
    if exact_bonus > 0:
        confidence += 0.10

    # Олон арга зүйгээр олдсон бол найдвартай
    match_methods = sum([
        v_score > 0.3,
        b_score > 0.3,
        chunk.get("_kw", False),
        chunk.get("_rel", 0) > 0.3
    ])
    if match_methods >= 2:
        confidence += 0.15
    if match_methods >= 3:
        confidence += 0.10

    # Reranker оноотой бол
    if "rerank_score" in chunk and chunk["rerank_score"] > 0.5:
        confidence += 0.10

    # Файл нэрд таарсан бол
    if chunk.get("_rel", 0) > 0.5:
        confidence += 0.05

    # Query үгүүд текстэд хэр олон удаа гарч байгаа (word boundary)
    q_words = set(re.findall(r'\w+', query.lower(), re.UNICODE))
    text_lower = chunk.get("text", "").lower()
    exact_hits = sum(1 for w in q_words if len(w) > 1 and re.search(rf'\b{re.escape(w)}\b', text_lower, re.UNICODE))
    if exact_hits >= 3:
        confidence += 0.08
    elif exact_hits >= 2:
        confidence += 0.05

    return min(confidence, 1.0)


# ── Enhanced search pipeline ─────────────────────────────────
def enhanced_search_pipeline(store, query: str, model: str, top_k: int,
                             use_reranker: bool, debug: bool) -> tuple[list, dict]:
    """
    Сайжруулсан хайлтын pipeline:
    1. Query classification → dynamic weights
    2. AI semantic boost
    3. Sub-query decomposition
    4. Multi-source search
    5. Confidence scoring
    """
    meta = {"query_type": "", "sub_queries": [], "enhanced": "", "weights": {}}

    # 1. Query classification
    qtype = classify_query_type(query)
    weights = WEIGHT_PROFILES.get(qtype, WEIGHT_PROFILES["mixed"])
    meta["query_type"] = qtype
    meta["weights"] = weights

    if debug:
        print(f"  📊 Query type: {qtype} | weights: bm25={weights['bm25']:.2f} vec={weights['vector']:.2f} kw={weights['keyword']:.2f}")

    # 2. AI Semantic boost (LLM байвал)
    use_llm = bool(model and model != "(олдсонгүй)" and _ollama_ping())
    enhanced_data = {}
    if use_llm:
        enhanced_data = semantic_boost_query(model, query)
        if enhanced_data.get("enhanced_query") and enhanced_data["enhanced_query"] != query:
            meta["enhanced"] = enhanced_data["enhanced_query"]
            if debug:
                print(f"  🧠 Semantic: {enhanced_data.get('intent', '')} | +{enhanced_data.get('keywords', [])}")

    # 3. Sub-query decomposition
    sub_queries = [query]
    if use_llm and len(query.split()) > 5:
        sub_queries = decompose_query(model, query)
        if len(sub_queries) > 1:
            meta["sub_queries"] = sub_queries
            if debug:
                print(f"  🔀 Sub-queries: {sub_queries}")

    # 4. Multi-source search with dynamic weights
    all_results = []
    seen_texts = set()

    encoded_sub_queries = np.atleast_2d(store.engine.encode(sub_queries, is_query=True))

    for sq, qe in zip(sub_queries, encoded_sub_queries):
        # Vector хайлт
        qe = np.asarray(qe, dtype="float32").reshape(1, -1)
        vs, vi = store.faiss_idx.search(qe, min(top_k * 15, len(store.chunks)))
        vmx = max(float(vs[0][0]), 1e-9)

        for score, idx in zip(vs[0], vi[0]):
            if idx < 0:
                continue
            chunk = store.chunks[int(idx)]
            text_key = chunk["text"][:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            v_score = float(score) / vmx

            # BM25 score
            bm_results = dict(store.bm25.search(sq, top_k=len(store.chunks)))
            b_score = bm_results.get(int(idx), 0.0)
            bmx = max(bm_results.values(), default=1e-9)
            b_score = b_score / bmx if bmx > 0 else 0.0

            # Keyword score with exact match boost
            kw_score = 0.0
            exact_bonus = 0.0
            sq_lower = sq.lower()
            text_lower = chunk["text"].lower()

            # Бүтэн query яг таарсан бол маш өндөр оноо
            if sq_lower in text_lower:
                kw_score = 1.0
                exact_bonus += EXACT_MATCH_BOOST

            # Үг бүрийг word boundary-тай шалгах
            words = [w for w in re.findall(r'\w+', sq_lower, re.UNICODE) if len(w) > 1]
            if words:
                exact_word_hits = 0
                partial_hits = 0
                for w in words:
                    # Word boundary match - үг яг таарсан эсэх
                    if re.search(rf'\b{re.escape(w)}\b', text_lower, re.UNICODE):
                        exact_word_hits += 1
                    elif w in text_lower:
                        partial_hits += 0.5

                word_match_ratio = (exact_word_hits + partial_hits) / len(words)
                if kw_score < word_match_ratio:
                    kw_score = word_match_ratio

                # Үг бүр яг таарсан бол нэмэлт оноо
                if exact_word_hits == len(words):
                    exact_bonus += EXACT_WORD_BOOST
                elif exact_word_hits >= len(words) * 0.7:
                    exact_bonus += EXACT_WORD_BOOST * 0.5

            # Dynamic weighted score
            combined = (
                weights["vector"] * v_score +
                weights["bm25"] * b_score +
                weights["keyword"] * kw_score +
                exact_bonus  # Exact match bonus
            )

            result = {
                **chunk,
                "score": combined,
                "vector_score": v_score,
                "bm25_score": b_score,
                "keyword_score": kw_score,
                "exact_bonus": exact_bonus,
                "_subquery": sq if len(sub_queries) > 1 else None
            }
            all_results.append(result)

    # Enhanced query-ээс нэмэлт хайлт
    if enhanced_data.get("keywords"):
        for kw in enhanced_data["keywords"][:3]:
            extra = store.search(kw, top_k=3, use_reranker=False, debug=False, expand_query_fn=expand_query)
            for c in extra:
                text_key = c["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    c = dict(c)
                    c["score"] *= 0.8  # бага зэрэг penalty
                    c["_enhanced"] = True
                    all_results.append(c)

    # Synonyms-аас хайлт
    if enhanced_data.get("synonyms"):
        for syn in enhanced_data["synonyms"][:2]:
            extra = store.search(syn, top_k=2, use_reranker=False, debug=False, expand_query_fn=expand_query)
            for c in extra:
                text_key = c["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    c = dict(c)
                    c["score"] *= 0.7
                    c["_synonym"] = True
                    all_results.append(c)

    # Sort by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # 5. Confidence scoring
    for r in all_results:
        r["confidence"] = calculate_confidence_score(r, query, weights)

    # Reranker
    candidates = all_results[:top_k * 3]
    if use_reranker and store.reranker.available and len(candidates) > 1:
        q_ascii = sum(1 for ch in query if ord(ch) < 128) / max(len(query), 1)
        if q_ascii > 0.7:
            candidates = store.reranker.rerank(query, candidates, top_k)
            # Rerank хийсний дараа confidence дахин тооцох
            for r in candidates:
                r["confidence"] = calculate_confidence_score(r, query, weights)

    return candidates[:top_k], meta
