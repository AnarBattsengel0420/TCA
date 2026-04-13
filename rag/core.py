"""
RAG Core - EmbedEngine, BM25Index, Reranker, HybridStore
"""
import os
import re
import time
import pickle
import threading
from pathlib import Path

import numpy as np

from .config import (
    CACHE_DIR, LOCAL_MODEL_DIR, EMBED_MODELS, DEFAULT_EMBED,
    RERANKER_HF, EMBED_MAX_LEN, DEFAULT_TOP_K,
    BM25_WEIGHT, VECTOR_WEIGHT, KEYWORD_BOOST,
    _embed_st_path, _embed_onnx_path, _reranker_path
)
from .utils import _DEVICE, _DEVICE_TYPE, _VRAM_GB, _opt_batch, _opt_threads, _gpu_batch, _gpu_str
from .readers import EmbeddingCache, FileTracker, build_chunks_parallel


# ── Embed engine ─────────────────────────────────────────────
class EmbedEngine:
    """Embedding model wrapper (ONNX + PyTorch)"""
    def __init__(self, key=DEFAULT_EMBED):
        self.key = key
        self.info = EMBED_MODELS.get(key, EMBED_MODELS[DEFAULT_EMBED])
        self._model = self._tok = self.backend = None
        self._use_prefix = self.info.get("use_prefix", False)

    def _load(self):
        if self._model:
            return
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        onnx_dir = _embed_onnx_path(self.key)
        st_dir = _embed_st_path(self.key)

        # ONNX (CPU-д хамгийн хурдан)
        if _DEVICE_TYPE == "cpu" and onnx_dir.exists():
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = _opt_threads()
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                q8 = onnx_dir / "model_q8.onnx"
                fname = "model_q8.onnx" if q8.exists() else "model.onnx"

                providers = ort.get_available_providers()

                # Vulkan EP (GPU-д хамгийн түрүүнд оролдоно)
                if "VulkanExecutionProvider" in providers:
                    try:
                        self._model = ORTModelForFeatureExtraction.from_pretrained(
                            str(onnx_dir), file_name=fname, session_options=opts,
                            provider="VulkanExecutionProvider")
                        self.backend = ("onnx_q8" if q8.exists() else "onnx") + "+vulkan"
                        self._tok = AutoTokenizer.from_pretrained(str(onnx_dir))
                        print(f"  Embed: {self.backend} ({self.key}) ⚡ Vulkan")
                        return
                    except Exception as e:
                        print(f"  Vulkan алдаа: {e}")

                # DirectML EP (Windows GPU fallback)
                if "DmlExecutionProvider" in providers:
                    try:
                        self._model = ORTModelForFeatureExtraction.from_pretrained(
                            str(onnx_dir), file_name=fname, session_options=opts,
                            provider="DmlExecutionProvider")
                        self.backend = ("onnx_q8" if q8.exists() else "onnx") + "+dml"
                        self._tok = AutoTokenizer.from_pretrained(str(onnx_dir))
                        print(f"  Embed: {self.backend} ({self.key}) ⚡ DirectML")
                        return
                    except Exception as e:
                        print(f"  DirectML алдаа: {e}")

                # OpenVINO EP (Intel CPU-д нэмэлт хурдасгал)
                if "OpenVINOExecutionProvider" in providers:
                    try:
                        self._model = ORTModelForFeatureExtraction.from_pretrained(
                            str(onnx_dir), file_name=fname, session_options=opts,
                            provider="OpenVINOExecutionProvider",
                            provider_options={"device_type": "CPU", "num_of_threads": _opt_threads()})
                        self.backend = ("onnx_q8" if q8.exists() else "onnx") + "+openvino"
                        self._tok = AutoTokenizer.from_pretrained(str(onnx_dir))
                        print(f"  Embed: {self.backend} ({self.key}) ⚡ OpenVINO")
                        return
                    except Exception as e:
                        print(f"  OpenVINO алдаа: {e}")

                self._model = ORTModelForFeatureExtraction.from_pretrained(
                    str(onnx_dir), file_name=fname, session_options=opts)
                self.backend = "onnx_q8" if q8.exists() else "onnx"
                self._tok = AutoTokenizer.from_pretrained(str(onnx_dir))
                print(f"  Embed: {self.backend} ({self.key})")
                return
            except Exception as e:
                print(f"  ONNX алдаа: {e} → PyTorch")

        if st_dir.exists():
            self._load_st(self.key)
            return

        # Fallback
        for k in list(EMBED_MODELS):
            if _embed_st_path(k).exists():
                self.key = k
                self.info = EMBED_MODELS[k]
                self._use_prefix = self.info.get("use_prefix", False)
                self._load_st(k)
                return
        raise RuntimeError("Model олдсонгүй!\npython App.py --download")

    def _load_st(self, key):
        from sentence_transformers import SentenceTransformer
        device = str(_DEVICE) if _DEVICE_TYPE != "cpu" else "cpu"
        st_dir = _embed_st_path(key)
        try:
            self._model = SentenceTransformer(str(st_dir), device=device)
            self.backend = f"pytorch+{_DEVICE_TYPE}"
            self._model.encode(["test"], convert_to_numpy=True, normalize_embeddings=True)
            ml = " 🌐" if self.info.get("multilingual") else ""
            print(f"  Embed: {self.backend} ({key}){ml}")
        except Exception as e:
            if _DEVICE_TYPE != "cpu":
                print(f"  GPU алдаа: {e} → CPU")
                self._model = SentenceTransformer(str(st_dir), device="cpu")
                self.backend = "pytorch+cpu"

    def encode(self, texts, progress_cb=None, is_query=False):
        self._load()
        if self._use_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        bs = _gpu_batch(_opt_batch()) if _DEVICE_TYPE != "cpu" else _opt_batch()
        total = len(texts)
        start = time.perf_counter()

        if self.backend.startswith("pytorch"):
            all_embs = []
            for i in range(0, total, bs):
                batch = texts[i:i + bs]
                try:
                    emb = self._model.encode(batch, show_progress_bar=False, batch_size=len(batch),
                                             convert_to_numpy=True, normalize_embeddings=True)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        half = max(len(batch) // 2, 1)
                        emb = np.vstack([self._model.encode(batch[j:j + half], show_progress_bar=False,
                                                            batch_size=half, convert_to_numpy=True,
                                                            normalize_embeddings=True)
                                         for j in range(0, len(batch), half)])
                    else:
                        raise
                all_embs.append(emb)
                if progress_cb:
                    progress_cb(min(i + bs, total), total, start)
            return np.vstack(all_embs).astype("float32")

        # ONNX
        all_embs = []
        for i in range(0, total, bs):
            enc = self._tok(texts[i:i + bs], padding=True, truncation=True,
                            max_length=EMBED_MAX_LEN, return_tensors="np")
            out = self._model(**dict(enc))
            m3 = np.expand_dims(enc["attention_mask"], -1).astype("float32")
            emb = (out.last_hidden_state * m3).sum(1) / np.maximum(m3.sum(1), 1e-9)
            n = np.linalg.norm(emb, axis=1, keepdims=True)
            all_embs.append((emb / np.maximum(n, 1e-9)).astype("float32"))
            if progress_cb:
                progress_cb(min(i + bs, total), total, start)
        return np.concatenate(all_embs, axis=0)


# ── BM25 ────────────────────────────────────────────────────
class BM25Index:
    """BM25 хайлтын индекс"""
    def __init__(self):
        self._index = self._tfidf_data = None
        self._corpus = []
        self._use_bm25 = False

    @staticmethod
    def _tokenize(t):
        return re.findall(r'\w+', t.lower(), re.UNICODE)

    def build(self, texts):
        self._corpus = [self._tokenize(t) for t in texts]
        if not self._corpus:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._index = BM25Okapi(self._corpus)
            self._use_bm25 = True
        except ImportError:
            import math
            n = len(self._corpus)
            df = {}
            for toks in self._corpus:
                for t in set(toks):
                    df[t] = df.get(t, 0) + 1
            self._tfidf_data = (n, df)

    def search(self, query, top_k=20):
        tokens = self._tokenize(query)
        if not tokens or not self._corpus:
            return []
        if self._use_bm25:
            scores = self._index.get_scores(tokens)
        else:
            import math
            n, df = self._tfidf_data
            scores = np.zeros(len(self._corpus))
            for t in tokens:
                d = df.get(t, 0)
                if not d:
                    continue
                idf = math.log((n - d + 0.5) / (d + 0.5) + 1)
                for i, doc in enumerate(self._corpus):
                    tf = doc.count(t)
                    if tf:
                        scores[i] += tf * idf
        idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# ── Reranker ────────────────────────────────────────────────
class Reranker:
    """Cross-encoder reranker"""
    def __init__(self):
        self._model = None
        self._tried = self._on_gpu = False
        self.available = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        rp = _reranker_path()
        if not rp.exists():
            print("  Reranker байхгүй (--download)")
            return
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import CrossEncoder
            device = str(_DEVICE) if _DEVICE_TYPE != "cpu" else "cpu"
            self._model = CrossEncoder(str(rp), device=device)
            self.available = True
            self._on_gpu = _DEVICE_TYPE != "cpu"
            print(f"  Reranker: {'GPU' if self._on_gpu else 'CPU'}")
        except Exception as e:
            print(f"  Reranker алдаа: {e}")

    def rerank(self, query, chunks, top_k=3):
        self._load()
        if not self.available or not chunks:
            return chunks[:top_k]
        bs = _gpu_batch(32) if self._on_gpu else 32
        try:
            scores = self._model.predict([(query, c["text"]) for c in chunks],
                                          batch_size=bs, show_progress_bar=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                scores = self._model.predict([(query, c["text"]) for c in chunks],
                                              batch_size=8, show_progress_bar=False)
            else:
                raise
        for i, c in enumerate(chunks):
            c["rerank_score"] = float(scores[i])
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ── File relevance score ─────────────────────────────────────
def _file_relevance_score(chunk: dict, query: str, expansions: list) -> float:
    """
    Chunk-ийн файл нэр болон текст нь асуулттай хэр холбоотойг шалгана.
    """
    src = chunk.get("source", "").lower()
    text = chunk.get("text", "").lower()
    score = 0.0

    all_terms = [query.lower()] + [e.lower() for e in expansions]

    for term in all_terms:
        words = re.findall(r"\w+", term, re.UNICODE)
        for w in words:
            if len(w) < 2:
                continue
            # Файл нэрд байгаа → өндөр нөлөө
            if w in src:
                score += 0.4
            # Текстэд байгаа → дунд нөлөө
            if w in text:
                score += 0.15

    return min(score, 1.0)


# ── Hybrid store ─────────────────────────────────────────────
class HybridStore:
    """Vector + BM25 + Keyword хайлтын индекс"""
    def __init__(self, key=DEFAULT_EMBED):
        self.engine = EmbedEngine(key)
        self.bm25 = BM25Index()
        self.reranker = Reranker()
        self.cache = EmbeddingCache()
        self.faiss_idx = None
        self.chunks = []
        self._folder = ""

    @staticmethod
    def _cs_path(fid):
        return CACHE_DIR / "chunk_stores" / f"{fid}.pkl"

    def _load_cs(self, fid):
        p = self._cs_path(fid)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save_cs(self, fid, store):
        p = self._cs_path(fid)
        p.parent.mkdir(parents=True, exist_ok=True)

        def _w():
            try:
                tmp = p.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(store, f)
                tmp.replace(p)
            except Exception:
                pass
        threading.Thread(target=_w, daemon=True).start()

    def build(self, docs, progress=None, folder_id=""):
        import faiss
        self._folder = str(Path(docs[0]["path"]).parent) if docs else ""
        tracker = FileTracker(folder_id) if folder_id else None
        changed_docs = docs
        if tracker:
            changed_docs, skip_n = tracker.filter_changed(docs)
            if skip_n:
                print(f"  өөрчлөгдөөгүй: {skip_n}")

        existing = {d["path"] for d in docs}
        file_chunks = self._load_cs(folder_id) if folder_id else {}
        if changed_docs:
            if progress:
                progress(0.30, desc=f"Chunk: {len(changed_docs)} файл")
            file_chunks.update(build_chunks_parallel(changed_docs))
        for path in list(file_chunks):
            if path not in existing:
                del file_chunks[path]
        if folder_id:
            self._save_cs(folder_id, file_chunks)

        self.chunks, all_texts = [], []
        for d in docs:
            for ch in file_chunks.get(d["path"], []):
                self.chunks.append(ch)
                all_texts.append(ch["text"])
        if not all_texts:
            return 0

        if progress:
            progress(0.33, desc="BM25 индекс")
        self.bm25.build(all_texts)

        tag = f"[{_gpu_str()}]"
        if progress:
            progress(0.38, desc=f"Embedding {tag}: {len(all_texts)} chunk")
        t0 = time.perf_counter()

        def _pf(done, total, s):
            if progress:
                el = max(time.perf_counter() - s, 1e-6)
                progress(0.38 + 0.42 * done / max(total, 1), desc=f"Embedding {done}/{total} | {done / el:.0f}/с")

        embeddings = self.cache.encode_with_cache(
            all_texts,
            lambda texts, progress_cb=None: self.engine.encode(texts, progress_cb=progress_cb, is_query=False),
            _pf)
        el = time.perf_counter() - t0
        print(f"  Embedding: {el:.1f}с | {len(all_texts) / max(el, 0.001):.0f} chunk/с {tag}")

        if progress:
            progress(0.85, desc="FAISS индекс")
        dim, n = embeddings.shape[1], len(all_texts)
        if n > 5000:
            nl = min(int(np.sqrt(n)), 256)
            idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nl, faiss.METRIC_INNER_PRODUCT)
            idx.train(embeddings)
            idx.add(embeddings)
            idx.nprobe = max(min(nl // 6, 24), 6)
        else:
            idx = faiss.IndexFlatIP(dim)
            idx.add(embeddings)
        self.faiss_idx = idx
        if progress:
            progress(1.0, desc="✅ Бэлэн!")
        return n

    def save(self, path):
        import faiss
        if not self.faiss_idx:
            raise RuntimeError("Хадгалах индекс байхгүй")
        faiss.write_index(self.faiss_idx, path + ".faiss")
        with open(path + ".meta", "wb") as f:
            pickle.dump({"chunks": self.chunks, "key": self.engine.key,
                         "texts": [c["text"] for c in self.chunks],
                         "folder": self._folder, "saved_at": time.strftime("%Y-%m-%d %H:%M")}, f)
        print(f"  💾 {path}")

    def load(self, path):
        try:
            import faiss
            self.faiss_idx = faiss.read_index(path + ".faiss")
            with open(path + ".meta", "rb") as f:
                meta = pickle.load(f)
            self.chunks = meta["chunks"]
            self.engine = EmbedEngine(meta.get("key", DEFAULT_EMBED))
            self._folder = meta.get("folder", "")
            self.bm25.build(meta.get("texts", [c["text"] for c in self.chunks]))
            ml = " 🌐" if self.engine.info.get("multilingual") else ""
            print(f"  ✅ {len(self.chunks):,} chunk [{self.engine.key}{ml}]")
            return True
        except Exception as e:
            print(f"  ❌ {e}")
            return False

    def _keyword_search(self, query, top_k):
        q = query.lower().strip()
        STOP = {"вэ", "бэ", "юу", "яу", "гэж", "гэх", "мэт", "байна", "бол",
                "нь", "ын", "ийн", "ийг", "ний", "дэх", "дээр", "дотор", "аа", "ээ"}
        words = [w for w in re.findall(r'\w+', q, re.UNICODE) if w not in STOP]
        if not words:
            return []
        max_s = 3.0 + sum(2.0 if len(w) >= 4 else 1.0 for w in words)
        scored = []
        for i, ch in enumerate(self.chunks):
            tl = ch["text"].lower()
            s = (3.0 if q in tl else 0.0) + sum(2.0 if len(w) >= 4 else 1.0 for w in words if w in tl)
            if s > 0:
                scored.append((i, s / max_s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**self.chunks[i], "score": s, "vector_score": 0.0, "bm25_score": 0.0, "_kw": True}
                for i, s in scored[:top_k]]

    def search(self, query, top_k=DEFAULT_TOP_K, use_reranker=True, debug=False, expand_query_fn=None):
        """
        Hybrid search (Vector + BM25 + Keyword).
        expand_query_fn: search.py-аас expand_query функц
        """
        if not self.faiss_idx or not self.chunks:
            return []
        cand_k = min(max(top_k * 20, 100), len(self.chunks))

        # ── Query expansion: abbreviation/synonym/орчуулга ─────
        expansions = expand_query_fn(query) if expand_query_fn else [query]
        if debug and len(expansions) > 1:
            print(f"  🔄 Query expansion: {expansions[:4]}")

        # Vector: expanded query-уудыг нэг удаа batch encode хийнэ
        expanded_queries = expansions[:4] or [query]
        all_qe = np.atleast_2d(self.engine.encode(expanded_queries, is_query=True))
        qe = all_qe.mean(axis=0, keepdims=True).astype("float32")

        vs, vi = self.faiss_idx.search(qe, cand_k)
        vmx = max(float(vs[0][0]), 1e-9)
        vh = {int(i): float(s) / vmx for s, i in zip(vs[0], vi[0]) if i >= 0}

        # BM25: бүх expanded term-ийг хайна
        bh: dict = {}
        for exp_q in expansions[:5]:
            for idx, sc in self.bm25.search(exp_q, top_k=cand_k):
                bh[idx] = max(bh.get(idx, 0.0), sc)
        bmx = max(bh.values(), default=1e-9)
        bh = {i: s / bmx for i, s in bh.items()}

        vb = []
        for idx in set(vh) | set(bh):
            v, b = vh.get(idx, 0.0), bh.get(idx, 0.0)
            vb.append({**self.chunks[idx], "score": VECTOR_WEIGHT * v + BM25_WEIGHT * b,
                       "vector_score": v, "bm25_score": b})
        vb.sort(key=lambda x: x["score"], reverse=True)

        kw_results = self._keyword_search(query, top_k * 4)
        kw_map = {c["text"][:120]: c for c in kw_results}
        seen, merged = {}, []
        for c in vb[:cand_k]:
            k80 = c["text"][:120]
            if kw_c := kw_map.pop(k80, None):
                c = dict(c)
                c["score"] = min(c["score"] + kw_c["score"] * 0.7, 1.0)
                c["_kw"] = True
            seen[k80] = True
            merged.append(c)
        for k, c in kw_map.items():
            if k not in seen:
                c = dict(c)
                c["score"] = min(c["score"] + KEYWORD_BOOST, 1.0)
                merged.append(c)

        # ── File relevance filter + boost ────────────────────
        for c in merged:
            rel = _file_relevance_score(c, query, expansions)
            if rel > 0:
                c["score"] = min(c["score"] + rel * 0.3, 1.0)
                c["_rel"] = rel
        merged.sort(key=lambda x: x["score"], reverse=True)

        if debug:
            print(f"\n  🔍 [{query[:40]}]")
            for c in merged[:5]:
                tag = "🔑" if c.get("_kw") else "📐"
                print(f"  {tag} {c['source']} s={c['score']:.3f} v={c['vector_score']:.2f} b={c['bm25_score']:.2f}")

        candidates = merged[:cand_k]
        if use_reranker and self.reranker.available and len(candidates) > 1:
            if sum(1 for ch in query if ord(ch) < 128) / max(len(query), 1) > 0.7:
                return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def search_top_files(self, query, n_files=5, expand_query_fn=None):
        if not self.faiss_idx or not self.chunks:
            return []
        cand_k = min(max(n_files * 20, 100), len(self.chunks))
        qe = self.engine.encode([query], is_query=True)
        vs, vi = self.faiss_idx.search(qe, cand_k)
        vmx = max(float(vs[0][0]), 1e-9)
        vh = {int(i): float(s) / vmx for s, i in zip(vs[0], vi[0]) if i >= 0}
        br = self.bm25.search(query, top_k=cand_k)
        bmx = max((s for _, s in br), default=1e-9)
        bh = {i: s / bmx for i, s in br}
        file_best = {}
        for idx in set(vh) | set(bh):
            score = VECTOR_WEIGHT * vh.get(idx, 0) + BM25_WEIGHT * bh.get(idx, 0)
            src = self.chunks[idx]["source"]
            if src not in file_best or score > file_best[src]:
                file_best[src] = score
        for c in self._keyword_search(query, cand_k):
            src = c["source"]
            file_best[src] = min(file_best.get(src, 0) + c["score"] * 0.6, 1.0)
        return [f for f, _ in sorted(file_best.items(), key=lambda x: x[1], reverse=True)[:n_files]]
