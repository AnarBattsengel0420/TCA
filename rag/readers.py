"""
RAG Readers - File readers, Chunking, Caching
"""
import os
import re
import hashlib
import pickle
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from .config import (
    CACHE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
    MAX_FILE_MB, FILE_TIMEOUT_SEC, IGNORE_DIRS, SKIP_EXTENSIONS
)
from .utils import clean_text, safe_encode


# ── Файл уншигч ─────────────────────────────────────────────
def read_txt(p):
    """TXT файл уншиж текст буцаах"""
    for enc in ["utf-8", "utf-16", "cp1251", "latin-1"]:
        try:
            return clean_text(Path(p).read_text(encoding=enc))
        except Exception:
            continue
    return ""


def read_pdf(p):
    """PDF файл уншиж текст буцаах (fitz эсвэл pypdf)"""
    # PyMuPDF (fitz) эхлээд оролдоно
    try:
        import fitz
        doc = fitz.open(p)
        pages = []
        for page in doc:
            t = page.get_text("text")
            if len(t.strip()) < 10:
                try:
                    lines = [" ".join(s.get("text", "") for s in ln.get("spans", []))
                             for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
                             for ln in blk.get("lines", [])]
                    t = "\n".join(l for l in lines if l.strip())
                except Exception:
                    pass
            if t.strip():
                pages.append(t)
        doc.close()
        result = clean_text("\n\n".join(pages))
        if len(result.strip()) >= 10:
            return result
    except ImportError:
        pass
    except Exception:
        pass

    # pypdf fallback
    try:
        with open(p, "rb") as f:
            if f.read(4) != b"%PDF":
                return ""
        from pypdf import PdfReader
        pages = []
        for pg in PdfReader(p, strict=False).pages:
            try:
                t = pg.extract_text() or ""
                if t.strip():
                    pages.append(t)
            except Exception:
                pass
        result = clean_text("\n".join(pages))
        if len(result.strip()) >= 10:
            return result
    except Exception:
        pass

    # Зурган PDF placeholder
    name = Path(p).stem.replace("_", " ").replace("-", " ")
    try:
        import fitz
        doc = fitz.open(p)
        n = len(doc)
        doc.close()
        if n > 0:
            return f"[Зурган PDF] {name} | {n} хуудас"
    except Exception:
        pass
    return f"[PDF уншигдсангүй] {name}"


def read_docx(p):
    """DOCX файл уншиж текст буцаах"""
    try:
        from docx import Document
        return clean_text("\n".join(pg.text for pg in Document(p).paragraphs))
    except Exception as e:
        return f"[DOCX: {e}]"


def read_csv(p):
    """CSV файл уншиж текст буцаах"""
    try:
        import pandas as pd
        return clean_text(pd.read_csv(p).to_string(index=False))
    except Exception as e:
        return f"[CSV: {e}]"


def read_xlsx(p):
    """Excel файл уншиж текст буцаах"""
    try:
        import pandas as pd
        return clean_text(pd.read_excel(p).to_string(index=False))
    except Exception as e:
        return f"[XLSX: {e}]"


# Reader mapping
READERS = {
    ".txt": read_txt, ".md": read_txt, ".log": read_txt, ".rst": read_txt,
    ".pdf": read_pdf, ".docx": read_docx,
    ".csv": read_csv, ".tsv": read_csv,
    ".xlsx": read_xlsx, ".xls": read_xlsx,
}


# ── File utilities ─────────────────────────────────────────
def _file_hash(p):
    """Файлын MD5 hash тооцоолох"""
    h = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(8192), b""):
                h.update(blk)
    except Exception:
        pass
    return h.hexdigest()


def _read_one(fp):
    """Нэг файл уншиж result буцаах"""
    ext = fp.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return {"skip": True, "reason": "кодын файл", "name": fp.name}
    if ext not in READERS:
        return {"skip": True, "reason": f"дэмжигдэхгүй ({ext})", "name": fp.name}
    try:
        sz = fp.stat().st_size
        if sz == 0:
            return {"skip": True, "reason": "хоосон", "name": fp.name}
        if sz > MAX_FILE_MB * 1024 * 1024:
            return {"skip": True, "reason": f"том ({sz // 1024 // 1024}MB)", "name": fp.name}
        txt = (READERS[ext](str(fp)) or "").strip()
        if not txt:
            r = "PDF: текст олдсонгүй" if ext == ".pdf" else "хоосон агуулга"
            return {"skip": True, "reason": r, "name": fp.name}
        if len(txt) < MIN_CHUNK_LEN:
            return {"skip": True, "reason": f"богино ({len(txt)}с)", "name": fp.name}
        return {"path": str(fp), "name": fp.name, "text": txt, "size": len(txt), "hash": _file_hash(str(fp))}
    except PermissionError:
        return {"skip": True, "reason": "эрх байхгүй", "name": fp.name}
    except Exception as e:
        return {"skip": True, "reason": str(e)[:80], "name": fp.name}


def load_documents(folder, progress=None):
    """Фолдероос бүх документ ачаалах"""
    folder = folder.strip().strip('"\'')
    fp = Path(folder)
    if not fp.exists():
        return [], [f"Фолдер олдсонгүй: {folder}"]
    files = [f for f in fp.rglob("*")
             if f.is_file()
             and not any(p.name in IGNORE_DIRS for p in f.parents)
             and f.suffix.lower() not in SKIP_EXTENSIONS
             and f.suffix.lower() in READERS]
    if not files:
        exts = {f.suffix.lower() for f in fp.rglob("*") if f.is_file()}
        return [], [f"Файл байхгүй. Өргөтгөлүүд: {', '.join(sorted(exts)[:10])}"]
    docs, skipped, done, total = [], [], 0, len(files)
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8, total)) as ex:
        futs = {ex.submit(_read_one, f): f for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            done += 1
            if progress:
                progress(done / total * 0.30, desc=f"📄 {done}/{total} — {f.name[:35]}")
            try:
                r = fut.result(timeout=FILE_TIMEOUT_SEC)
                if r is None:
                    skipped.append(f"{f.name} — алдаа")
                elif r.get("skip"):
                    skipped.append(f"{r['name']} — {r['reason']}")
                else:
                    docs.append(r)
            except Exception as e:
                skipped.append(f"{f.name} — {str(e)[:50]}")
    return docs, skipped


# ── Chunking ─────────────────────────────────────────────────
_SEPS = [". ", ".\n", "!\n", "?\n", "\n\n", "\n", "; ", ", ", " "]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Текстийг chunk болгон хуваах"""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if len(text) >= MIN_CHUNK_LEN else []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            for sep in _SEPS:
                pos = text.rfind(sep, start + overlap, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        c = text[start:end].strip()
        if len(c) >= MIN_CHUNK_LEN:
            chunks.append(c)
        start = end - overlap
    return chunks


def _chunk_doc(d):
    """Нэг document-ийг chunk болгох (multiprocessing-д зориулсан)"""
    return d["path"], d["name"], chunk_text(d["text"])


def build_chunks_parallel(docs):
    """Олон document-ийг зэрэг chunk болгох"""
    if not docs:
        return {}
    file_chunks = {}
    try:
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(docs), 8)) as ex:
            for fut in as_completed({ex.submit(_chunk_doc, d): d for d in docs}):
                try:
                    path, name, chunks = fut.result(timeout=60)
                    file_chunks[path] = [{"text": c, "source": name, "path": path} for c in chunks]
                except Exception:
                    pass
    except Exception:
        for d in docs:
            path, name, chunks = _chunk_doc(d)
            file_chunks[path] = [{"text": c, "source": name, "path": path} for c in chunks]
    return file_chunks


# ── Embedding кэш ────────────────────────────────────────────
def _chunk_hash(t):
    """Chunk текстийн hash"""
    return hashlib.md5(safe_encode(t)).hexdigest()


class EmbeddingCache:
    """Embedding-үүдийг кэшлэх"""
    def __init__(self):
        self.dir = CACHE_DIR / "embeddings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._store, self._dirty, self._lock = {}, False, threading.Lock()
        p = self.dir / "emb_v9.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    self._store = pickle.load(f)
            except Exception:
                pass

    def save(self):
        """Кэшийг файлд хадгалах (background)"""
        def _w():
            with self._lock:
                if not self._dirty:
                    return
                try:
                    p = self.dir / "emb_v9.pkl"
                    tmp = p.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump(self._store, f)
                    tmp.replace(p)
                    self._dirty = False
                except Exception:
                    pass
        threading.Thread(target=_w, daemon=True).start()

    def encode_with_cache(self, texts, encode_fn, progress_fn=None):
        """Кэш ашиглан encode хийх"""
        hashes = [_chunk_hash(t) for t in texts]
        results = [None] * len(texts)
        need_idx, need_txt, seen = [], [], {}
        for i, h in enumerate(hashes):
            c = self._store.get(h)
            if c is not None:
                results[i] = c
            elif h in seen:
                results[i] = ("dup", seen[h])
            else:
                seen[h] = len(need_idx)
                need_idx.append(i)
                need_txt.append(texts[i])
        cached = sum(1 for r in results if r is not None and not isinstance(r, tuple))
        print(f"  кэш:{cached} dedup:{sum(1 for r in results if isinstance(r, tuple))} шинэ:{len(need_txt)}")
        if need_txt:
            new = encode_fn(need_txt, progress_cb=lambda d, t, s: progress_fn(d, t, s) if progress_fn else None)
            with self._lock:
                for j, idx in enumerate(need_idx):
                    results[idx] = new[j]
                    self._store[hashes[idx]] = new[j]
                self._dirty = True
            self.save()
        for i, r in enumerate(results):
            if isinstance(r, tuple) and r[0] == "dup":
                results[i] = results[need_idx[r[1]]]
        return np.stack(results).astype("float32")


# ── File tracker ─────────────────────────────────────────────
class FileTracker:
    """Файлын өөрчлөлтийг хянах"""
    def __init__(self, fid):
        self._p = CACHE_DIR / "trackers" / f"{fid}.pkl"
        self._p.parent.mkdir(parents=True, exist_ok=True)
        self._h = {}
        if self._p.exists():
            try:
                with open(self._p, "rb") as f:
                    self._h = pickle.load(f)
            except Exception:
                pass

    def save(self):
        """Tracker хадгалах"""
        with open(self._p, "wb") as f:
            pickle.dump(self._h, f)

    def filter_changed(self, docs):
        """Өөрчлөгдсөн файлуудыг шүүх"""
        changed, skipped = [], 0
        for d in docs:
            if self._h.get(d["path"]) == d["hash"]:
                skipped += 1
            else:
                changed.append(d)
                self._h[d["path"]] = d["hash"]
        self.save()
        return changed, skipped
