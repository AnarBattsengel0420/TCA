"""
RAG Ollama - LLM integration (Ollama local models)
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import OLLAMA_HOST, OLLAMA_TIMEOUT


# ── Ollama discovery ────────────────────────────────────────
def _find_ollama():
    """Ollama executable олох"""
    p = shutil.which("ollama")
    if p:
        return p
    if sys.platform == "win32":
        for c in [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe",
                  Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"]:
            if c.exists():
                return str(c)
    return None


def get_ollama_models():
    """Боломжит Ollama model-ийн жагсаалт авах"""
    models = []
    # Method 1: ollama library
    try:
        import ollama
        r = ollama.list()
        ml = r.get("models", []) if isinstance(r, dict) else getattr(r, "models", []) or []
        for m in ml:
            n = m.get("model", m.get("name", "")) if isinstance(m, dict) else getattr(m, "model", "")
            if n:
                models.append(n)
        if models:
            return models
    except Exception:
        pass

    # Method 2: CLI
    try:
        exe = _find_ollama()
        if exe:
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            r = subprocess.run([exe, "list"], capture_output=True, text=True, timeout=10, **flags)
            for line in r.stdout.strip().split("\n")[1:]:
                p = line.split()
                if p and p[0].lower() != "name":
                    models.append(p[0])
            if models:
                return models
    except Exception:
        pass

    # Method 3: HTTP API
    try:
        import urllib.request
        with urllib.request.urlopen(
                urllib.request.Request(f"{OLLAMA_HOST}/api/tags"), timeout=5) as resp:
            for m in json.loads(resp.read()).get("models", []):
                n = m.get("model") or m.get("name", "")
                if n:
                    models.append(n)
    except Exception:
        pass
    return models


def _ollama_ping():
    """Ollama ажиллаж байгаа эсэхийг шалгах"""
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3).close()
        return True
    except Exception:
        return False


def _ollama_call_simple(model: str, prompt: str, timeout: int = 15) -> str:
    """
    Ollama-г нэг асуулт/хариултад ашиглана (streaming биш).
    Алдаа гарвал хоосон string буцаана.
    """
    exe = _find_ollama()
    if exe:
        try:
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            proc = subprocess.Popen(
                [exe, "run", model, "--nowordwrap"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", **flags)
            out, _ = proc.communicate(input=prompt, timeout=timeout)
            return out.strip()
        except Exception:
            pass
    import socket
    import urllib.request
    try:
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=data,
                                      headers={"Content-Type": "application/json"})
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read()).get("response", "").strip()
        finally:
            socket.setdefaulttimeout(old)
    except Exception:
        return ""


# ── Query rewriting ─────────────────────────────────────────
def rewrite_query(model: str, question: str) -> str:
    """
    Query rewriting — LLM өөрөө хайлтын query бичнэ.
    """
    if not _ollama_ping():
        return question
    prompt = (
        "Та хайлтын системийн query-г оновчтой болгодог мэргэжилтэн юм.\n"
        "Дараах дүрмийг дагаж, хэрэглэгчийн асуултыг илүү сайн хайлтын query болгон өөрчил:\n"
        "- Монгол, Англи хэл холилдсон байж болно → хоёуланг нь харгалзан үз\n"
        "- Товчлол, аббревиатурыг задлаарай (жишээ: JLPT → Japanese Language Proficiency Test N1 N2 N3 N4 N5)\n"
        "- Чухал нэр, код, техникийн нэр томъёо, тоо, огноо, байгууллагын нэр зэргийг хэвээр үлдээ\n"
        "- Синоним, холбогдох ойролцоо үгсийг нэм (гэхдээ хэт их бүү хий)\n"
        "- 6–14 үгтэй, keyword баялаг, тодорхой query гарга\n"
        "- Хэрэглэгчийн асуултын гол санаа, зорилгыг бүрэн илэрхийл\n"
        "- Зөвхөн нэг мөр query-г хэвлэ. Өөр юу ч бичвэл болохгүй "
        "(тайлбар, \"Search query:\", \"Here is...\" гэх мэт)\n\n"
        "Жишээ:\n"
        "User: \"миний jlpt level ямар вэ\"\n"
        "→ \"JLPT level N1 N2 N3 N4 N5 Japanese Language Proficiency Test score result\"\n"
        "User: \"2025 оны татварын хөнгөлөлт\"\n"
        "→ \"2025 он татварын хөнгөлөлт хөнгөлөлттэй татвар Монгол Улс\"\n\n"
        f"User question: {question}\n"
        "Search query:"
    )
    result = _ollama_call_simple(model, prompt, timeout=12)
    if not result or len(result) > 200 or "\n" in result[:20]:
        return question
    return result.strip('"\'')


def grade_chunks(model: str, question: str, chunks: list, max_chunks: int = 8) -> list:
    """
    Relevance grader — AI chunk бүрийг YES/NO-оор шалгана.
    """
    if not _ollama_ping() or not chunks:
        return chunks

    graded = []
    for chunk in chunks[:max_chunks]:
        src = chunk.get("source", "?")
        preview = chunk.get("text", "")[:300].replace("\n", " ")
        prompt = (
            "You are a relevance grader. "
            "Determine if the document chunk is relevant to the user question.\n"
            "Answer with ONLY 'YES' or 'NO'.\n\n"
            f"Question: {question}\n\n"
            f"Document chunk ({src}):\n{preview}\n\n"
            "Is this chunk relevant? Answer YES or NO:"
        )
        answer = _ollama_call_simple(model, prompt, timeout=10).upper()
        if "NO" in answer and "YES" not in answer:
            continue
        graded.append(chunk)

    return graded if graded else chunks


def semantic_boost_query(model: str, query: str, context_hint: str = "") -> dict:
    """
    AI Semantic Boost: LLM-ээр query-г илүү сайн ойлгож, enrichment хийнэ.
    """
    if not _ollama_ping():
        return {"enhanced_query": query, "keywords": [], "intent": "unknown", "synonyms": []}

    prompt = (
        "Analyze this search query and provide semantic enrichment.\n"
        "Output JSON only, no other text.\n\n"
        f"Query: {query}\n"
        f"Context: {context_hint[:200] if context_hint else 'general document search'}\n\n"
        "Output format:\n"
        '{"intent": "brief intent description", "keywords": ["key1", "key2"], '
        '"synonyms": ["syn1", "syn2"], "enhanced_query": "improved search query"}'
    )
    result = _ollama_call_simple(model, prompt, timeout=12)

    try:
        match = re.search(r'\{[^{}]+\}', result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "enhanced_query": data.get("enhanced_query", query),
                "keywords": data.get("keywords", [])[:5],
                "intent": data.get("intent", "unknown"),
                "synonyms": data.get("synonyms", [])[:5]
            }
    except (json.JSONDecodeError, AttributeError):
        pass

    return {"enhanced_query": query, "keywords": [], "intent": "unknown", "synonyms": []}


def ask_stream(model, question, context, expand_query_fn=None):
    """
    LLM-д асуулт тавьж streaming хариулт авах.
    expand_query_fn: search.py-аас expand_query функц авна.
    """
    # Expansion мэдээллийг prompt-д оруулна
    exp_terms = expand_query_fn(question) if expand_query_fn else [question]
    exp_hint = ", ".join(exp_terms[1:4]) if len(exp_terms) > 1 else ""

    prompt = f"""You are an advanced AI assistant designed for an intelligent knowledge retrieval system.
Your goal is to answer the user's question by intelligently searching, evaluating, and reasoning \
over retrieved documents while also using your own general knowledge.
{f"Search terms used: {exp_hint}" if exp_hint else ""}
Respond in the SAME language as the question \
(Mongolian → Mongolian, Japanese → Japanese, English → English).

STEP 1 — Understand the Question
Carefully analyze the user's question. Identify the main topic, intent, \
and important keywords. Think about what kind of information would be needed.

STEP 2 — Generate Better Search Queries
Mentally rewrite the question into 3–5 improved search queries using synonyms \
and related terminology. Use these to evaluate the retrieved documents below.

STEP 3 — Evaluate Retrieved Documents
For each document excerpt below:
- KEEP it if it contains useful information related to the question.
- IGNORE it if it is unrelated or about a completely different topic.
Be strict about relevance.

STEP 4 — Select Useful Information
From relevant documents: extract only the useful parts, remove unrelated text, \
summarize instead of copying raw text. Combine multiple documents logically.

STEP 5 — Use General Knowledge
If retrieved documents are incomplete or partially helpful: \
combine useful parts with your own knowledge.
If none are relevant: ignore them and answer using your own knowledge.

STEP 6 — Reason Before Answering
Think about the relationship between the question and the documents. \
Identify the most reliable information. Avoid irrelevant details.

STEP 7 — Generate the Final Answer
- Directly answer the user's question
- Be concise but informative
- Avoid copying long raw text
- Explain concepts clearly
- Use markdown for code blocks and bullet lists when helpful
- If the question cannot be answered, clearly say the information is insufficient.

=== RETRIEVED DOCUMENTS ===
{context}
===========================

USER QUESTION: {question}

ANSWER:"""
    if not _ollama_ping():
        yield "❌ Ollama ажиллахгүй байна.\n`ollama serve` командыг тусдаа терминалд ажиллуулна уу."
        return

    exe = _find_ollama()
    if exe:
        try:
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            proc = subprocess.Popen([exe, "run", model, "--nowordwrap"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1, **flags)
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                if line:
                    yield line
            proc.wait(timeout=OLLAMA_TIMEOUT)
            return
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            yield f"\n\n⏱️ Timeout ({OLLAMA_TIMEOUT}с). `OLLAMA_TIMEOUT` утгыг нэмнэ үү."
            return
        except Exception:
            pass

    import socket
    import urllib.request
    try:
        data = json.dumps({"model": model, "prompt": prompt, "stream": True, "options": {"num_predict": 2048}}).encode()
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=data,
                                      headers={"Content-Type": "application/json"})
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(OLLAMA_TIMEOUT)
        try:
            with urllib.request.urlopen(req) as resp:
                for raw in resp:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        d = json.loads(raw)
                        if d.get("error"):
                            yield f"\n\n❌ {d['error']}"
                            return
                        if d.get("response"):
                            yield d["response"]
                        if d.get("done"):
                            return
                    except json.JSONDecodeError:
                        continue
        finally:
            socket.setdefaulttimeout(old)
    except Exception as e:
        err = str(e).lower()
        if "timed out" in err or "timeout" in err:
            yield f"\n\n⏱️ Timeout. `OLLAMA_TIMEOUT={OLLAMA_TIMEOUT}` → `1200` | жижиг загвар: `qwen2.5:0.5b`"
        else:
            yield f"\n\nOllama алдаа: {e}"


def multi_step_search(store, question: str, rewritten: str,
                      top_k: int, use_reranker: bool, debug: bool) -> list:
    """
    Multi-step retrieval — 2 удаа хайж нэгтгэнэ.
    """
    # Round 1: original
    r1 = store.search(question, top_k=top_k, use_reranker=False, debug=debug)

    # Round 2: rewritten query
    r2 = []
    if rewritten.lower() != question.lower():
        r2 = store.search(rewritten, top_k=top_k, use_reranker=False, debug=False)

    # Нэгтгэх — давхардлыг text[:80]-аар шалгана
    seen = {c["text"][:80] for c in r1}
    merged = list(r1)
    for c in r2:
        if c["text"][:80] not in seen:
            seen.add(c["text"][:80])
            c = dict(c)
            c["score"] *= 0.9
            c["_r2"] = True
            merged.append(c)

    merged.sort(key=lambda x: x["score"], reverse=True)

    # Reranker нэг удаа нэгтгэсэн үр дүн дээр ажиллуулна
    if use_reranker and store.reranker.available and len(merged) > 1:
        q_ascii = sum(1 for ch in question if ord(ch) < 128) / max(len(question), 1)
        if q_ascii > 0.7:
            merged = store.reranker.rerank(question, merged, top_k)
            return merged

    return merged[:top_k]
